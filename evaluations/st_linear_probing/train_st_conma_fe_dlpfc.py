import os
import sys
import glob
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.pt_load_inference import MultimodalInference

# ============================================================================
# Configuration
# ============================================================================

SEEDS = [42, 123, 456, 789, 101]

# Sample ID conversion (DLPFC ID -> MISC ID in image files)
SAMPLE_ID_CONVERT = {
    151676: "MISC1",
    151675: "MISC2",
    151674: "MISC3",
    151673: "MISC4",
    151672: "MISC5",
    151671: "MISC6",
    151670: "MISC7",
    151669: "MISC8",
    151510: "MISC9",
    151509: "MISC10",
    151508: "MISC11",
    151507: "MISC12"
}

# Reverse mapping
MISC_TO_DLPFC = {v: k for k, v in SAMPLE_ID_CONVERT.items()}

# Replicate pair groupings (6 replicates, 2 samples each)
REPLICATE_PAIRS = {
    "replicate1": [151507, 151508],  # MISC12, MISC11
    "replicate2": [151509, 151510],  # MISC10, MISC9
    "replicate3": [151669, 151670],  # MISC8, MISC7
    "replicate4": [151671, 151672],  # MISC6, MISC5
    "replicate5": [151673, 151674],  # MISC4, MISC3
    "replicate6": [151675, 151676],  # MISC2, MISC1
}

# Label mapping
LABEL_MAP = {
    "Layer_1": 0,
    "Layer_2": 1,
    "Layer_3": 2,
    "Layer_4": 3,
    "Layer_5": 4,
    "Layer_6": 5,
    "WM": 6
}
NUM_CLASSES = len(LABEL_MAP)
LABEL_NAMES = list(LABEL_MAP.keys())

# Paths
IMAGE_DIR = "./pt_dataset/st_images"
GENE_CSV_PATH = "./pt_dataset/st_sentences/top100_sentences.csv"
ANNOTATION_DIR = "./ft_dataset/DLPFC/DLPFC_annotations"
CHECKPOINT_PATH = "./checkpoints/st_conma_pythia410m_12layers_3aug_clip/checkpoint_epoch_12.pt"

# Model config
VISION_MODEL_NAME = "pathoduet"
GENE_MODEL_NAME = "pythia_410m"
VISION_DIM = 768
GENE_DIM = 1024
PROJ_DIM = 768
MAX_SEQ_LEN = 512
NUM_LAYERS = 12
DEVICE = "cuda:3"

# Training config
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 10000
PATIENCE = 10
VAL_RATIO = 0.2  # 80:20 train:val split

# Output
OUTPUT_DIR = "./results/st_linear_probing/st_conma_fe_dlpfc"


# ============================================================================
# Dataset
# ============================================================================

class DLPFCEmbeddingDataset(Dataset):
    """Dataset for DLPFC embeddings with labels"""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ============================================================================
# Linear Classifier
# ============================================================================

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        # Weight initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        return self.linear(x)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_annotations(sample_id: int) -> Dict[str, str]:
    """Load annotations for a DLPFC sample"""
    annotation_file = os.path.join(ANNOTATION_DIR, f"{sample_id}_truth.txt")
    annotations = {}

    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                spot_id, label = parts
                annotations[spot_id] = label

    return annotations


def extract_embeddings_for_sample(
    model: MultimodalInference,
    df_gene: pd.DataFrame,
    sample_id: int,
    annotations: Dict[str, str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract fused embeddings for a single DLPFC sample"""

    misc_id = SAMPLE_ID_CONVERT[sample_id]

    # Filter gene data for this sample
    df_sample = df_gene[df_gene['id'].str.startswith(f"{misc_id}_")]

    if len(df_sample) == 0:
        print(f"Warning: No gene data found for {sample_id} ({misc_id})")
        return np.array([]), np.array([]), []

    # Filter spots that have annotations and images
    valid_image_paths = []
    valid_sentences = []
    valid_labels = []
    valid_spot_ids = []
    valid_gene_ids = []

    for _, row in df_sample.iterrows():
        gene_id = row['id']  # format: MISC1_AAACAAGTATCTCCCA-1
        spot_id = gene_id.replace(f"{misc_id}_", "")

        # Check annotation
        if spot_id not in annotations:
            continue

        label_str = annotations[spot_id]
        if label_str not in LABEL_MAP:
            continue

        # Check image exists
        image_path = os.path.join(IMAGE_DIR, f"{gene_id}.png")
        if not os.path.exists(image_path):
            continue

        valid_image_paths.append(image_path)
        valid_sentences.append(row['sentence'])
        valid_labels.append(LABEL_MAP[label_str])
        valid_spot_ids.append(spot_id)
        valid_gene_ids.append(gene_id)

    if len(valid_sentences) == 0:
        print(f"Warning: No valid annotated spots for {sample_id}")
        return np.array([]), np.array([]), []

    # Extract fused embeddings
    embeddings = model.encode_fused_batch(
        images=valid_image_paths,
        gene_sentences=valid_sentences,
        gene_ids=valid_gene_ids,
        batch_size=32,
        return_numpy=True,
        save_csv=False
    )

    labels = np.array(valid_labels)

    return embeddings, labels, valid_spot_ids


def load_all_data(model: MultimodalInference, df_gene: pd.DataFrame) -> Dict[int, Dict]:
    """Load all DLPFC data with fused embeddings and labels"""

    all_data = {}

    for sample_id in SAMPLE_ID_CONVERT.keys():
        print(f"\nLoading {sample_id} ({SAMPLE_ID_CONVERT[sample_id]})...")

        annotations = load_annotations(sample_id)
        embeddings, labels, spot_ids = extract_embeddings_for_sample(
            model, df_gene, sample_id, annotations
        )

        if len(embeddings) > 0:
            all_data[sample_id] = {
                'embeddings': embeddings,
                'labels': labels,
                'spot_ids': spot_ids
            }
            print(f"  Loaded {len(labels)} spots")

            # Print label distribution
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                print(f"    {LABEL_NAMES[u]}: {c}")

    return all_data


def save_embeddings(all_data: Dict[int, Dict], output_dir: str):
    """Save extracted embeddings to .npz file for reuse"""
    os.makedirs(output_dir, exist_ok=True)
    embeddings_path = os.path.join(output_dir, "embeddings.npz")

    # Prepare data for saving
    save_dict = {}
    for sample_id, data in all_data.items():
        save_dict[f'{sample_id}_embeddings'] = data['embeddings']
        save_dict[f'{sample_id}_labels'] = data['labels']
        save_dict[f'{sample_id}_spot_ids'] = np.array(data['spot_ids'], dtype=object)

    np.savez(embeddings_path, **save_dict)
    print(f"Embeddings saved to: {embeddings_path}")


def load_embeddings(output_dir: str) -> Dict[int, Dict]:
    """Load pre-extracted embeddings from .npz file"""
    embeddings_path = os.path.join(output_dir, "embeddings.npz")

    if not os.path.exists(embeddings_path):
        return None

    print(f"Loading embeddings from: {embeddings_path}")
    loaded = np.load(embeddings_path, allow_pickle=True)

    all_data = {}
    for sample_id in SAMPLE_ID_CONVERT.keys():
        emb_key = f'{sample_id}_embeddings'
        if emb_key in loaded:
            all_data[sample_id] = {
                'embeddings': loaded[f'{sample_id}_embeddings'],
                'labels': loaded[f'{sample_id}_labels'],
                'spot_ids': loaded[f'{sample_id}_spot_ids'].tolist()
            }
            print(f"  Loaded {sample_id}: {len(all_data[sample_id]['labels'])} spots")

    return all_data


def save_predictions(
    spot_ids: List[str],
    predictions: np.ndarray,
    sample_id: int,
    output_dir: str
):
    """Save predictions in truth file format: barcode\tlabel"""
    pred_file = os.path.join(output_dir, f"{sample_id}_pred.txt")

    with open(pred_file, 'w') as f:
        for spot_id, pred in zip(spot_ids, predictions):
            label_name = LABEL_NAMES[pred]
            f.write(f"{spot_id}\t{label_name}\n")

    print(f"    Saved predictions to {pred_file}")


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for embeddings, labels in dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float, float]:
    """Evaluate model - returns loss, balanced accuracy, macro F1"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            outputs = model(embeddings)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, balanced_acc, macro_f1


def train_with_early_stopping(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    device: str,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE
) -> Tuple[nn.Module, Dict]:
    """Train linear classifier with early stopping"""

    model = LinearProbe(input_dim, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_balanced_acc = 0.0
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_balanced_acc': [], 'val_macro_f1': []}

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_balanced_acc, val_macro_f1 = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_balanced_acc'].append(val_balanced_acc)
        history['val_macro_f1'].append(val_macro_f1)

        # Early stopping based on balanced accuracy (higher is better)
        if val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"val_balanced_acc={val_balanced_acc:.4f}, val_macro_f1={val_macro_f1:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    return model, history


# ============================================================================
# Cross-Validation
# ============================================================================

def replicate_cross_validation(
    all_data: Dict[int, Dict],
    device: str,
    output_dir: str,
    seed: int = 42
) -> Dict:
    """Perform replicate-wise 6-fold cross-validation"""

    os.makedirs(output_dir, exist_ok=True)

    results = {
        'fold_results': [],
        'all_predictions': [],
        'all_labels': [],
        'per_sample_results': {}
    }

    replicate_names = list(REPLICATE_PAIRS.keys())

    for fold_idx, test_replicate in enumerate(replicate_names):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}: Test replicate = {test_replicate}")
        print(f"{'='*60}")

        # Get test samples
        test_sample_ids = REPLICATE_PAIRS[test_replicate]

        # Get train samples (other replicates)
        train_sample_ids = []
        for replicate, samples in REPLICATE_PAIRS.items():
            if replicate != test_replicate:
                train_sample_ids.extend(samples)

        print(f"Test samples: {test_sample_ids}")
        print(f"Train samples: {train_sample_ids}")

        # Collect training data
        train_embeddings = []
        train_labels = []

        for sample_id in train_sample_ids:
            if sample_id in all_data:
                train_embeddings.append(all_data[sample_id]['embeddings'])
                train_labels.append(all_data[sample_id]['labels'])

        train_embeddings = np.concatenate(train_embeddings, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        print(f"Total training spots: {len(train_labels)}")

        # Stratified split for train/val
        train_emb, val_emb, train_lbl, val_lbl = train_test_split(
            train_embeddings, train_labels,
            test_size=VAL_RATIO,
            stratify=train_labels,
            random_state=seed
        )

        print(f"Train: {len(train_lbl)}, Val: {len(val_lbl)}")

        # Create dataloaders
        train_dataset = DLPFCEmbeddingDataset(train_emb, train_lbl)
        val_dataset = DLPFCEmbeddingDataset(val_emb, val_lbl)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Train
        input_dim = train_embeddings.shape[1]
        print(f"\nTraining linear classifier (input_dim={input_dim})...")

        model, history = train_with_early_stopping(
            train_loader, val_loader, input_dim, device
        )

        # Evaluate on test samples
        fold_test_preds = []
        fold_test_labels = []

        for test_sample_id in test_sample_ids:
            if test_sample_id not in all_data:
                continue

            test_emb = all_data[test_sample_id]['embeddings']
            test_lbl = all_data[test_sample_id]['labels']

            test_dataset = DLPFCEmbeddingDataset(test_emb, test_lbl)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            test_loss, test_balanced_acc, test_macro_f1 = evaluate(
                model, test_loader, nn.CrossEntropyLoss(), device
            )

            # Get predictions
            model.eval()
            sample_preds = []
            with torch.no_grad():
                for emb, _ in test_loader:
                    emb = emb.to(device)
                    outputs = model(emb)
                    preds = outputs.argmax(dim=1)
                    sample_preds.extend(preds.cpu().numpy())

            sample_preds = np.array(sample_preds)

            # Per-sample results
            results['per_sample_results'][test_sample_id] = {
                'balanced_accuracy': test_balanced_acc,
                'macro_f1': test_macro_f1,
                'num_spots': len(test_lbl)
            }

            print(f"\n  {test_sample_id} ({SAMPLE_ID_CONVERT[test_sample_id]}): "
                  f"balanced_acc={test_balanced_acc:.4f}, macro_f1={test_macro_f1:.4f}, n={len(test_lbl)}")

            # Save predictions in truth file format (only for seed 42)
            if seed == 42:
                test_spot_ids = all_data[test_sample_id]['spot_ids']
                save_predictions(test_spot_ids, sample_preds, test_sample_id, output_dir)

            fold_test_preds.extend(sample_preds)
            fold_test_labels.extend(test_lbl)

        # Fold-level metrics
        fold_balanced_acc = balanced_accuracy_score(fold_test_labels, fold_test_preds)
        fold_macro_f1 = f1_score(fold_test_labels, fold_test_preds, average='macro')

        print(f"\nFold {fold_idx + 1} Overall: balanced_acc={fold_balanced_acc:.4f}, macro_f1={fold_macro_f1:.4f}")

        results['fold_results'].append({
            'fold': fold_idx + 1,
            'test_replicate': test_replicate,
            'balanced_accuracy': fold_balanced_acc,
            'macro_f1': fold_macro_f1,
            'num_test_spots': len(fold_test_labels)
        })

        results['all_predictions'].extend(fold_test_preds)
        results['all_labels'].extend(fold_test_labels)

        # Save fold model
        fold_model_path = os.path.join(output_dir, f"fold{fold_idx+1}_model.pt")
        torch.save(model.state_dict(), fold_model_path)

    # Overall metrics
    overall_balanced_acc = balanced_accuracy_score(results['all_labels'], results['all_predictions'])
    overall_macro_f1 = f1_score(results['all_labels'], results['all_predictions'], average='macro')

    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"Balanced Accuracy: {overall_balanced_acc:.4f}")
    print(f"Macro F1: {overall_macro_f1:.4f}")

    # Per-fold summary
    print("\nPer-fold results:")
    for fold_result in results['fold_results']:
        print(f"  Fold {fold_result['fold']} ({fold_result['test_replicate']}): "
              f"balanced_acc={fold_result['balanced_accuracy']:.4f}, macro_f1={fold_result['macro_f1']:.4f}")

    # Mean and std across folds
    fold_balanced_accs = [r['balanced_accuracy'] for r in results['fold_results']]
    fold_macro_f1s = [r['macro_f1'] for r in results['fold_results']]

    print(f"\nMean +/- Std across folds:")
    print(f"  Balanced Accuracy: {np.mean(fold_balanced_accs):.4f} +/- {np.std(fold_balanced_accs):.4f}")
    print(f"  Macro F1: {np.mean(fold_macro_f1s):.4f} +/- {np.std(fold_macro_f1s):.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        results['all_labels'],
        results['all_predictions'],
        target_names=LABEL_NAMES
    ))

    # Save results
    results['overall'] = {
        'balanced_accuracy': overall_balanced_acc,
        'macro_f1': overall_macro_f1,
        'balanced_accuracy_mean': np.mean(fold_balanced_accs),
        'balanced_accuracy_std': np.std(fold_balanced_accs),
        'macro_f1_mean': np.mean(fold_macro_f1s),
        'macro_f1_std': np.std(fold_macro_f1s),
        'num_total_spots': len(results['all_labels'])
    }

    results_path = os.path.join(output_dir, "results.json")

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*60)
    print("DLPFC Linear Probing with ST-ConMa Fused Encoder")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Vision model: {VISION_MODEL_NAME} ({VISION_DIM}d)")
    print(f"  Gene model: {GENE_MODEL_NAME} ({GENE_DIM}d)")
    print(f"  Fused embedding dim: {VISION_DIM}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Max seq len: {MAX_SEQ_LEN}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Device: {DEVICE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max epochs: {MAX_EPOCHS}")
    print(f"  Early stopping patience: {PATIENCE}")
    print(f"  Val ratio: {VAL_RATIO}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Seeds: {SEEDS}")

    # Try to load cached embeddings first
    print("\n" + "="*60)
    print("Loading Embeddings...")
    print("="*60)

    all_data = load_embeddings(OUTPUT_DIR)

    if all_data is None:
        print("No cached embeddings found. Extracting new embeddings...")

        # Load multimodal model
        print("\nLoading Multimodal Model (ST-ConMa-text)...")
        model = MultimodalInference(
            checkpoint_path=CHECKPOINT_PATH,
            vision_model_name=VISION_MODEL_NAME,
            gene_model_name=GENE_MODEL_NAME,
            device=DEVICE,
            vision_dim=VISION_DIM,
            gene_dim=GENE_DIM,
            proj_dim=PROJ_DIM,
            max_seq_len=MAX_SEQ_LEN,
            proj_hidden_dim=3072,
            proj_layers=2,
            num_cross_layers=3,
            num_heads=12,
            loss_type='clip',
            temperature=0.07,
            learnable_temperature=False,
            igc_weight=1.0,
            iic_weight=1.0,
            igm_weight=1.0,
            igc_loss_type='clip',
            iic_loss_type='clip',
            igc_temperature=None,
            iic_temperature=None,
            num_layers=NUM_LAYERS
        )

        # Load gene CSV
        print("\nLoading Gene Data...")
        df_gene = pd.read_csv(GENE_CSV_PATH, dtype={'id': str, 'sentence': str})
        print(f"Total gene sequences loaded: {len(df_gene)}")

        # Extract embeddings
        print("\nExtracting Fused Embeddings...")
        all_data = load_all_data(model, df_gene)

        # Save embeddings for future use
        save_embeddings(all_data, OUTPUT_DIR)
    else:
        print("Using cached embeddings.")

    # Run cross-validation for each seed
    all_seed_results = []

    for seed in SEEDS:
        set_seed(seed)
        print("\n" + "="*60)
        print(f"Running with Seed: {seed}")
        print("="*60)

        results = replicate_cross_validation(all_data, DEVICE, OUTPUT_DIR, seed=seed)
        # Collect per-sample results (convert sample_id keys to str for JSON)
        per_sample = {}
        for sample_id, sample_res in results['per_sample_results'].items():
            per_sample[str(sample_id)] = {
                'balanced_accuracy': sample_res['balanced_accuracy'],
                'macro_f1': sample_res['macro_f1'],
                'num_spots': sample_res['num_spots']
            }

        all_seed_results.append({
            'seed': seed,
            'overall_balanced_acc': results['overall']['balanced_accuracy'],
            'overall_macro_f1': results['overall']['macro_f1'],
            'fold_balanced_accs': [r['balanced_accuracy'] for r in results['fold_results']],
            'fold_macro_f1s': [r['macro_f1'] for r in results['fold_results']],
            'per_sample_results': per_sample
        })

    # Print summary across all seeds
    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL SEEDS")
    print("="*60)

    print("\nPer-seed results:")
    for res in all_seed_results:
        print(f"  Seed {res['seed']}: balanced_acc={res['overall_balanced_acc']:.4f}, macro_f1={res['overall_macro_f1']:.4f}")

    # Calculate mean and std across seeds
    all_balanced_accs = [r['overall_balanced_acc'] for r in all_seed_results]
    all_macro_f1s = [r['overall_macro_f1'] for r in all_seed_results]

    print(f"\nMean +/- Std across {len(SEEDS)} seeds:")
    print(f"  Balanced Accuracy: {np.mean(all_balanced_accs):.4f} +/- {np.std(all_balanced_accs):.4f}")
    print(f"  Macro F1: {np.mean(all_macro_f1s):.4f} +/- {np.std(all_macro_f1s):.4f}")

    # Aggregate per-sample results across seeds
    all_sample_ids = sorted(SAMPLE_ID_CONVERT.keys())
    per_sample_summary = {}

    for sample_id in all_sample_ids:
        sid = str(sample_id)
        sample_balanced_accs = []
        sample_macro_f1s = []

        for res in all_seed_results:
            if sid in res['per_sample_results']:
                sample_balanced_accs.append(res['per_sample_results'][sid]['balanced_accuracy'])
                sample_macro_f1s.append(res['per_sample_results'][sid]['macro_f1'])

        if sample_balanced_accs:
            per_sample_summary[sid] = {
                'sample_id': sample_id,
                'misc_id': SAMPLE_ID_CONVERT[sample_id],
                'num_spots': all_seed_results[0]['per_sample_results'][sid]['num_spots'],
                'balanced_accuracy_per_seed': sample_balanced_accs,
                'macro_f1_per_seed': sample_macro_f1s,
                'mean_balanced_accuracy': float(np.mean(sample_balanced_accs)),
                'std_balanced_accuracy': float(np.std(sample_balanced_accs)),
                'mean_macro_f1': float(np.mean(sample_macro_f1s)),
                'std_macro_f1': float(np.std(sample_macro_f1s))
            }

    # Print per-sample summary
    print(f"\nPer-sample results (mean +/- std across {len(SEEDS)} seeds):")
    for sid, info in per_sample_summary.items():
        print(f"  {sid} ({info['misc_id']}): "
              f"balanced_acc={info['mean_balanced_accuracy']:.4f} +/- {info['std_balanced_accuracy']:.4f}, "
              f"macro_f1={info['mean_macro_f1']:.4f} +/- {info['std_macro_f1']:.4f}, "
              f"n={info['num_spots']}")

    # Save summary results
    summary_results = {
        'seeds': SEEDS,
        'per_seed_results': all_seed_results,
        'per_sample_summary': per_sample_summary,
        'mean_balanced_acc': float(np.mean(all_balanced_accs)),
        'std_balanced_acc': float(np.std(all_balanced_accs)),
        'mean_macro_f1': float(np.mean(all_macro_f1s)),
        'std_macro_f1': float(np.std(all_macro_f1s))
    }

    summary_path = os.path.join(OUTPUT_DIR, "summary_all_seeds.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
