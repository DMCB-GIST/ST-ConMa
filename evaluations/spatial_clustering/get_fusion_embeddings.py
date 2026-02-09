"""
Extract fusion embeddings from trained ST_AlignmentModel using cross-attention.

This script loads a trained checkpoint via MultimodalInference and extracts
fused (cross-attention) embeddings for each spot, saving them as .npy files.

Usage:
    python get_fusion_embeddings.py \
        --data_dir /path/to/ft_dataset/DLPFC \
        --checkpoint /path/to/checkpoint.pt \
        --output_name fusion_embeddings.npy
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Add paths - insert at beginning to override local utils.py
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils.pt_load_inference import MultimodalInference


# DLPFC sample IDs
DLPFC_SAMPLES = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]

# Dataset directories
DLPFC_DIR = './ft_dataset/spatial_clustering/DLPFC'
BREAST_CANCER_DIR = './ft_dataset/spatial_clustering/human_breast_cancer'


class DLPFCEmbeddingDataset(Dataset):
    """
    Dataset for extracting embeddings from DLPFC samples.
    Returns data in spot order (sorted by id) to match STAIG format.
    """

    def __init__(
        self,
        data_dir: str,
        sample_id: str,
        use_filtered_images: bool = False,
        transform=None
    ):
        self.data_dir = data_dir
        self.sample_id = sample_id
        self.transform = transform
        self.image_folder = 'st_images_filter' if use_filtered_images else 'st_images'

        sample_dir = os.path.join(data_dir, sample_id)

        # Load CSV
        csv_path = os.path.join(sample_dir, 'top100_sentences.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, dtype={'id': str, 'sentence': str})

        # Image directory
        img_dir = os.path.join(sample_dir, self.image_folder)
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        # Collect samples (sorted by id for consistent ordering)
        self.samples = []
        for _, row in df.iterrows():
            spot_id = row['id']
            image_path = os.path.join(img_dir, f"{spot_id}.png")

            if os.path.exists(image_path):
                self.samples.append({
                    'id': spot_id,
                    'sentence': row['sentence'],
                    'image_path': image_path
                })

        # Sort by id (numeric part for proper ordering)
        try:
            self.samples.sort(key=lambda x: int(x['id'].split('_')[-1]) if '_' in x['id'] else int(x['id']))
        except ValueError:
            # If ids are not numeric, sort alphabetically
            self.samples.sort(key=lambda x: x['id'])

        print(f"  Sample {sample_id}: {len(self.samples)} spots")

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


class BreastCancerEmbeddingDataset(Dataset):
    """
    Dataset for extracting embeddings from Human Breast Cancer sample.
    Returns data in spot order (sorted by id) to match STAIG format.
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

        # Load CSV
        csv_path = os.path.join(data_dir, 'top100_sentences.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, dtype={'id': str, 'sentence': str})

        # Image directory
        img_dir = os.path.join(data_dir, self.image_folder)
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        # Collect samples (sorted by id for consistent ordering)
        self.samples = []
        for _, row in df.iterrows():
            spot_id = row['id']
            image_path = os.path.join(img_dir, f"{spot_id}.png")

            if os.path.exists(image_path):
                self.samples.append({
                    'id': spot_id,
                    'sentence': row['sentence'],
                    'image_path': image_path
                })

        # Sort by id (numeric part for proper ordering)
        try:
            self.samples.sort(key=lambda x: int(x['id'].split('_')[-1]) if '_' in x['id'] else int(x['id']))
        except ValueError:
            # If ids are not numeric, sort alphabetically
            self.samples.sort(key=lambda x: x['id'])

        print(f"  Human Breast Cancer: {len(self.samples)} spots")

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
    """Collate function for embedding extraction."""
    return {
        'images': torch.stack([b['image'] for b in batch]),
        'gene_sentences': [b['sentence'] for b in batch],
        'ids': [b['id'] for b in batch]
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract fusion embeddings from ST_AlignmentModel')

    # Data
    parser.add_argument('--dataset', type=str, default='dlpfc',
                       choices=['dlpfc', 'human_breast_cancer'],
                       help='Dataset to extract embeddings from: dlpfc or human_breast_cancer')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Base directory containing samples (auto-set based on --dataset if not provided)')
    parser.add_argument('--sample_ids', type=str, default=None,
                       help='Comma-separated sample IDs (e.g., 151507,151508) - DLPFC only')
    parser.add_argument('--all', action='store_true',
                       help='Process all 12 DLPFC samples')
    parser.add_argument('--use_filtered_images', action='store_true',
                       help='Use FFT-filtered images')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length for C2S model')

    # Model
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint directory (auto-set based on --dataset if not provided)')
    parser.add_argument('--vision_model', type=str, default='pathoduet')
    parser.add_argument('--vision_dim', type=int, default=768)
    parser.add_argument('--gene_model', type=str, default='pythia_410m')
    parser.add_argument('--gene_dim', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--proj_dim', type=int, default=768)
    parser.add_argument('--proj_hidden_dim', type=int, default=3072)
    parser.add_argument('--proj_layers', type=int, default=2)
    parser.add_argument('--num_cross_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=12)

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (auto-set based on --dataset if not provided)')
    parser.add_argument('--output_name', type=str, default='embeddings.npy',
                       help='Name of output .npy file')

    # System
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)

    # Set defaults based on dataset
    if args.dataset == 'human_breast_cancer':
        if args.data_dir is None:
            args.data_dir = BREAST_CANCER_DIR
        if args.checkpoint is None:
            args.checkpoint = './results/spatial_clustering/st_conma/human_breast_cancer/checkpoints'
        if args.output_dir is None:
            args.output_dir = './results/spatial_clustering/st_conma/human_breast_cancer/fusion_embeddings'
        sample_ids = ['human_breast_cancer']
    else:
        if args.data_dir is None:
            args.data_dir = DLPFC_DIR
        if args.checkpoint is None:
            args.checkpoint = './results/spatial_clustering/st_conma/dlpfc/checkpoints'
        if args.output_dir is None:
            args.output_dir = './results/spatial_clustering/st_conma/dlpfc/fusion_embeddings'
        # Parse sample IDs for DLPFC
        if args.all:
            sample_ids = DLPFC_SAMPLES
        elif args.sample_ids:
            sample_ids = [s.strip() for s in args.sample_ids.split(',')]
        else:
            raise ValueError("Please specify --all or --sample_ids for DLPFC dataset")

    dataset_name = "Human Breast Cancer" if args.dataset == 'human_breast_cancer' else "DLPFC"
    print("=" * 80)
    print(f"Extract Fusion Embeddings from ST_AlignmentModel ({dataset_name})")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Checkpoint Directory: {args.checkpoint}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Sample IDs: {sample_ids}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Output Name: {args.output_name}")
    print("=" * 80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create MultimodalInference (without checkpoint; we load per-sample checkpoints below)
    print("\nLoading model via MultimodalInference...")

    inference = MultimodalInference(
        checkpoint_path=None,
        vision_model_name=args.vision_model,
        gene_model_name=args.gene_model,
        device=args.device,
        vision_dim=args.vision_dim,
        gene_dim=args.gene_dim,
        proj_dim=args.proj_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_layers=args.proj_layers,
        num_cross_layers=args.num_cross_layers,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_length,
        loss_type='clip',
        temperature=1.0,
        learnable_temperature=False,
        igc_weight=1.0,
        iic_weight=0.0,
        igm_weight=1.0,
    )

    transform = inference.transform

    # Process each sample
    print("\nExtracting embeddings...")

    for sample_id in sample_ids:
        print(f"\nProcessing sample {sample_id}...")

        # Load checkpoint (different paths for DLPFC vs breast_cancer)
        if args.dataset == 'human_breast_cancer':
            checkpoint_path = os.path.join(args.checkpoint, 'checkpoint_epoch_6.pt')
        else:
            checkpoint_path = os.path.join(args.checkpoint, sample_id, 'checkpoint_epoch_6.pt')

        if not os.path.exists(checkpoint_path):
            print(f"  Checkpoint not found: {checkpoint_path}, skipping...")
            continue

        print(f"  Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        inference.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        inference.model.eval()

        try:
            # Create dataset based on dataset type
            if args.dataset == 'human_breast_cancer':
                dataset = BreastCancerEmbeddingDataset(
                    data_dir=args.data_dir,
                    use_filtered_images=args.use_filtered_images,
                    transform=transform
                )
            else:
                dataset = DLPFCEmbeddingDataset(
                    data_dir=args.data_dir,
                    sample_id=sample_id,
                    use_filtered_images=args.use_filtered_images,
                    transform=transform
                )

            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,  # Important: keep order
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False
            )

            # Extract embeddings
            all_embeddings = []
            all_ids = []

            for batch in tqdm(dataloader, desc=f"  Extracting {sample_id}"):
                # Use MultimodalInference.encode_fused (returns numpy by default)
                fused = inference.encode_fused(
                    images=batch['images'],
                    gene_sentences=batch['gene_sentences'],
                    return_numpy=False
                )
                # L2 normalize to match original output
                fused = F.normalize(fused, dim=-1)
                all_embeddings.append(fused.cpu().numpy())
                all_ids.extend(batch['ids'])

            # Concatenate
            all_embeddings = np.vstack(all_embeddings)
            print(f"  Embedding shape: {all_embeddings.shape}")

            # Save embeddings
            if args.output_dir:
                # Save to specified output directory with sample_id prefix
                output_path = os.path.join(args.output_dir, f"{sample_id}_{args.output_name}")
            else:
                # Save to sample folder
                output_path = os.path.join(args.data_dir, sample_id, args.output_name)
            np.save(output_path, all_embeddings)
            print(f"  Saved to {output_path}")

            # Clear GPU cache to prevent memory accumulation
            torch.cuda.empty_cache()

        except FileNotFoundError as e:
            print(f"  Skipping {sample_id}: {e}")
            continue

    print("\n" + "=" * 80)
    print("Embedding extraction completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
