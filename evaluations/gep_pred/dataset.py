import torch
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40)) 
import numpy as np
import torchvision.transforms as transforms
import glob
from PIL import Image
import pandas as pd
import scprep as scp
from PIL import ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True 
Image.MAX_IMAGE_PIXELS = None
import random
from typing import List

class GaussianBlur:
    """Gaussian blur augmentation from SimCLR."""
    def __init__(self, sigma: list = [0.1, 2.0]):
        self.sigma = sigma
    
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

# ============================================================================
# Gene Sentence Utilities
# ============================================================================

# Load pretraining gene list (7816 common overlap genes)
PRETRAINING_GENES_PATH = "./pt_dataset/common_overlap_genes.txt"
with open(PRETRAINING_GENES_PATH, 'r') as f:
    PRETRAINING_GENES = set([line.strip() for line in f.readlines()])
print(f"Loaded {len(PRETRAINING_GENES)} pretraining genes")


def expression_to_gene_sentence(
    expression: np.ndarray,
    gene_names: List[str],
    top_k: int = 100
) -> str:
    """
    Convert expression vector to gene sentence (pretraining format).

    Only includes genes that were in pretraining (7816 common overlap genes).

    Args:
        expression: [num_genes] expression values
        gene_names: list of gene names corresponding to expression indices
        top_k: number of top genes to include

    Returns:
        space-separated gene names sorted by expression (descending)
    """
    # Get indices sorted by expression (descending)
    sorted_indices = np.argsort(expression)[::-1]

    # Filter to only include genes from pretraining
    top_genes = []
    for idx in sorted_indices:
        gene = gene_names[idx]
        if gene in PRETRAINING_GENES:
            top_genes.append(gene)
            if len(top_genes) >= top_k:
                break

    # Join with space (matching pretraining format)
    return ' '.join(top_genes)


class PatientWiseCVDataset(torch.utils.data.Dataset):
    """
    Patient-Wise Leave-One-Out Cross-Validation Dataset.

    - HER2ST: 8 patients (A-H) -> 8-fold CV
    - CSCC: 4 patients (P2, P5, P9, P10) -> 4-fold CV
    - HLT: 4 samples (A1, B1, C1, D1) -> 4-fold CV

    Each fold uses one patient as test, remaining patients as train.
    Uses pre-generated:
    - ST-sentences: gene sentences (CSV with id, sentence)
    - ST-patches: image patches (PNG files)
    - ST-cnts-normalized: normalized expression (TSV, 250 HEG, CPM 1e6 + log1p)
    - ST-cnts-normalized-hvg: normalized expression (TSV, 250 HVG, CPM 1e6 + log1p)
    """

    def __init__(
        self,
        dataset: str = 'her2st',  # 'her2st', 'cscc', or 'hlt'
        fold: int = 0,
        train: bool = True,
        use_augmentation: bool = True,
        gene_type: str = 'heg'  # 'heg' or 'hvg'
    ):
        """
        Args:
            dataset: 'her2st', 'cscc', or 'hlt'
            fold: which patient to use as test set (0 to n_patients-1)
            train: if True, use all other patients for training; else use this patient for test
            use_augmentation: whether to apply data augmentation
            gene_type: 'heg' (highly expressed genes) or 'hvg' (highly variable genes)
        """
        super().__init__()

        self.dataset = dataset
        self.fold = fold
        self.train = train
        self.use_augmentation = use_augmentation
        self.gene_type = gene_type

        # Validate gene_type
        if gene_type not in ['heg', 'hvg']:
            raise ValueError(f"gene_type must be 'heg' or 'hvg', got: {gene_type}")

        # Base directory
        base_dir = './ft_dataset/gep_pred'

        # Set expression directory and gene list based on gene_type
        exp_suffix = 'ST-cnts-normalized' if gene_type == 'heg' else 'ST-cnts-normalized-hvg'
        gene_list_suffix = f'{gene_type}_cut_250.txt'

        if dataset == 'her2st':
            self.sentence_dir = os.path.join(base_dir, 'her2st/ST-sentences')
            self.patch_dir = os.path.join(base_dir, 'her2st/ST-patches')
            self.exp_dir = os.path.join(base_dir, f'her2st/{exp_suffix}')
            self.gene_list_path = os.path.join(base_dir, f'her2st_{gene_list_suffix}')
            # HER2ST: patients A-H (first letter of sample name)
            self.patients = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            self.get_patient = lambda name: name[0]  # A1 -> A
        elif dataset == 'cscc':
            self.sentence_dir = os.path.join(base_dir, 'cscc/ST-sentences')
            self.patch_dir = os.path.join(base_dir, 'cscc/ST-patches')
            self.exp_dir = os.path.join(base_dir, f'cscc/{exp_suffix}')
            self.gene_list_path = os.path.join(base_dir, f'cscc_{gene_list_suffix}')
            # CSCC: patients P2, P5, P9, P10 (prefix before _ST_)
            self.patients = ['P2', 'P5', 'P9', 'P10']
            self.get_patient = lambda name: name.split('_')[0]  # P2_ST_rep1 -> P2
        elif dataset == 'hlt':
            self.sentence_dir = os.path.join(base_dir, 'hlt/ST-sentences')
            self.patch_dir = os.path.join(base_dir, 'hlt/ST-patches')
            self.exp_dir = os.path.join(base_dir, f'hlt/{exp_suffix}')
            self.gene_list_path = os.path.join(base_dir, f'hlt_{gene_list_suffix}')
            # HLT: 4 samples (A1, B1, C1, D1) - each sample is a separate patient
            self.patients = ['A1', 'B1', 'C1', 'D1']
            self.get_patient = lambda name: name  # A1 -> A1
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Must be 'her2st', 'cscc', or 'hlt'")

        self.n_folds = len(self.patients)

        # Load gene list (250 HEG)
        with open(self.gene_list_path, 'r') as f:
            self.gene_list = [line.strip() for line in f if line.strip()]

        # Get test patient
        self.test_patient = self.patients[fold]

        # Load all spots and filter by patient
        print(f"Loading {dataset} data for patient-wise CV ({gene_type.upper()})...")
        print(f"  Gene type: {gene_type.upper()} (250 genes)")
        print(f"  Patients: {self.patients}")
        print(f"  Test patient (fold {fold}): {self.test_patient}")
        self.all_spots = self._load_all_spots()

        # Filter spots by train/test patient
        if train:
            self.spots = [s for s in self.all_spots if self.get_patient(s['sample_name']) != self.test_patient]
        else:
            self.spots = [s for s in self.all_spots if self.get_patient(s['sample_name']) == self.test_patient]

        print(f"  {'Train' if train else 'Test'}: {len(self.spots)} spots")

        # Image transforms
        if use_augmentation and train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.4),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ])
            
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

    def _load_all_spots(self):
        """Load all spot data from prepared files."""
        all_spots = []

        # Get sample names from sentence files
        sentence_files = [f for f in os.listdir(self.sentence_dir) if f.endswith('.csv') and f != 'stats.csv']
        sample_names = [f.replace('.csv', '') for f in sentence_files]
        sample_names.sort()

        for sample_name in sample_names:
            # Load sentences
            sentence_path = os.path.join(self.sentence_dir, f'{sample_name}.csv')
            sentence_df = pd.read_csv(sentence_path)

            # Load normalized expression
            exp_path = os.path.join(self.exp_dir, f'{sample_name}.tsv')
            exp_df = pd.read_csv(exp_path, sep='\t', index_col=0)

            # Process each spot
            for _, row in sentence_df.iterrows():
                spot_id = row['id']
                sentence = row['sentence']

                parts = spot_id.split('_')
                spot_coord = parts[-1] 

                # Get expression for this spot
                if spot_coord in exp_df.index:
                    expression = exp_df.loc[spot_coord].values.astype(np.float32)
                else:
                    continue  

                # Patch path
                patch_path = os.path.join(self.patch_dir, sample_name, f'{spot_coord}.png')
                if not os.path.exists(patch_path):
                    continue  

                all_spots.append({
                    'spot_id': spot_id,
                    'sample_name': sample_name,
                    'spot_coord': spot_coord,
                    'sentence': sentence,
                    'expression': expression,
                    'patch_path': patch_path
                })

        return all_spots

    def __len__(self):
        return len(self.spots)

    def __getitem__(self, index):
        spot = self.spots[index]

        # Load image patch
        img = Image.open(spot['patch_path']).convert('RGB')
        img = self.transform(img)

        # Expression
        expression = torch.tensor(spot['expression'], dtype=torch.float32)

        item = {
            'image': img,
            'gene_sentence': spot['sentence'],
            'expression': expression,
            'sample_name': spot['sample_name'],
            'spot_id': spot['spot_id']
        }

        return item

    def get_num_folds(self):
        """Return number of folds (= number of patients)."""
        return self.n_folds

    def get_patient_name(self):
        """Return the test patient name for this fold."""
        return self.test_patient


def collate_fn_patient_cv(batch):
    """Collate function for patient-wise CV dataset."""
    images = torch.stack([item['image'] for item in batch])
    gene_sentences = [item['gene_sentence'] for item in batch]
    expressions = torch.stack([item['expression'] for item in batch])
    sample_names = [item['sample_name'] for item in batch]
    spot_ids = [item['spot_id'] for item in batch]

    return {
        'images': images,
        'gene_sentences': gene_sentences,
        'expressions': expressions,
        'sample_names': sample_names,
        'spot_ids': spot_ids
    }
