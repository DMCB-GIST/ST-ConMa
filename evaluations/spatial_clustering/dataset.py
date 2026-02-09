"""
Fine-tuning Dataset for ST-ConMa on DLPFC and Human Breast Cancer samples
Loads data from generate_sentences.py and extract_patches.py outputs
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from typing import Dict, List


# DLPFC sample IDs
DLPFC_SAMPLES = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]

# Human Breast Cancer
BREAST_CANCER_SAMPLE = "Human_breast_cancer"


class DLPFCFineTuningDataset(Dataset):
    """
    Dataset for fine-tuning on DLPFC samples.

    Loads data from generate_sentences.py and extract_patches.py outputs:
    - CSV: {data_dir}/{sample_id}/top100_sentences.csv (columns: id, sentence)
    - Images: {data_dir}/{sample_id}/st_images/{sample_id}_{barcode}.png
    """

    def __init__(
        self,
        data_dir: str,
        sample_ids: List[str],
        use_filtered_images: bool = False,
        transform=None
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.image_folder = 'st_images_filter' if use_filtered_images else 'st_images'

        # Collect all samples
        self.samples = []

        for sample_id in sample_ids:
            sample_dir = os.path.join(data_dir, sample_id)

            # Load CSV
            csv_path = os.path.join(sample_dir, 'top100_sentences.csv')
            if not os.path.exists(csv_path):
                print(f"Warning: CSV not found for {sample_id}: {csv_path}")
                continue

            df = pd.read_csv(csv_path, dtype={'id': str, 'sentence': str})

            # Image directory
            img_dir = os.path.join(sample_dir, self.image_folder)
            if not os.path.exists(img_dir):
                print(f"Warning: Image directory not found for {sample_id}: {img_dir}")
                continue

            # Match CSV entries with images
            for _, row in df.iterrows():
                spot_id = row['id']
                image_path = os.path.join(img_dir, f"{spot_id}.png")

                if os.path.exists(image_path):
                    self.samples.append({
                        'id': spot_id,
                        'sentence': row['sentence'],
                        'image_path': image_path
                    })

        print(f"Loaded {len(self.samples)} samples from {len(sample_ids)} DLPFC slides")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict:
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'sentence': sample['sentence'],
            'id': sample['id']
        }


def collate_fn(batch):
    """Collate function for fine-tuning dataloader."""
    return {
        'images': torch.stack([b['image'] for b in batch]),
        'gene_sequences': [b['sentence'] for b in batch],
        'ids': [b['id'] for b in batch]
    }


def create_dlpfc_dataloader(
    data_dir: str,
    sample_ids: List[str],
    use_filtered_images: bool = False,
    transform=None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Create dataloader for DLPFC fine-tuning.

    Args:
        data_dir: Base directory containing DLPFC samples
        sample_ids: List of sample IDs to include
        use_filtered_images: If True, use FFT-filtered images
        transform: Image transform
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
    """
    dataset = DLPFCFineTuningDataset(
        data_dir=data_dir,
        sample_ids=sample_ids,
        use_filtered_images=use_filtered_images,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


class BreastCancerFineTuningDataset(Dataset):
    """
    Dataset for fine-tuning on Human Breast Cancer sample.

    Loads data from generate_sentences.py and extract_patches.py outputs:
    - CSV: {data_dir}/top100_sentences.csv (columns: id, sentence)
    - Images: {data_dir}/st_images/{sample_id}_{barcode}.png
    """

    def __init__(
        self,
        data_dir: str,
        use_filtered_images: bool = False,
        transform=None
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.image_folder = 'st_images_filter' if use_filtered_images else 'st_images'

        # Collect all samples
        self.samples = []

        # Load CSV
        csv_path = os.path.join(data_dir, 'top100_sentences.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, dtype={'id': str, 'sentence': str})

        # Image directory
        img_dir = os.path.join(data_dir, self.image_folder)
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        # Match CSV entries with images
        for _, row in df.iterrows():
            spot_id = row['id']
            image_path = os.path.join(img_dir, f"{spot_id}.png")

            if os.path.exists(image_path):
                self.samples.append({
                    'id': spot_id,
                    'sentence': row['sentence'],
                    'image_path': image_path
                })

        print(f"Loaded {len(self.samples)} samples from Human Breast Cancer")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict:
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'sentence': sample['sentence'],
            'id': sample['id']
        }


def create_breast_cancer_dataloader(
    data_dir: str,
    use_filtered_images: bool = False,
    transform=None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Create dataloader for Human Breast Cancer fine-tuning.

    Args:
        data_dir: Directory containing Human_breast_cancer data
        use_filtered_images: If True, use FFT-filtered images
        transform: Image transform
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
    """
    dataset = BreastCancerFineTuningDataset(
        data_dir=data_dir,
        use_filtered_images=use_filtered_images,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    return dataloader
