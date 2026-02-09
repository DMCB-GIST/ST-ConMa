#!/usr/bin/env python
"""
Fine-tuning Script for ST_AlignmentModel on DLPFC samples.

Uses outputs from generate_sentences.py and extract_patches.py.
Uses IGC (Image-Gene Contrastive) and IGM (Image-Gene Matching) losses only.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import argparse
import warnings
import time
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Add paths - insert at beginning to override local utils.py
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils.model import get_vision_model, get_c2s_model
from utils.multimodal import ST_AlignmentModel
from dataset import DLPFC_SAMPLES, create_dlpfc_dataloader, create_breast_cancer_dataloader

# Dataset directories
DLPFC_DIR = './ft_dataset/spatial_clustering/DLPFC'
BREAST_CANCER_DIR = './ft_dataset/spatial_clustering/human_breast_cancer'


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune ST_AlignmentModel on DLPFC')

    # Data
    parser.add_argument('--dataset', type=str, default='dlpfc',
                        choices=['dlpfc', 'breast_cancer'],
                        help='Dataset to train on: dlpfc or breast_cancer')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Base directory containing samples (auto-set based on --dataset if not provided)')
    parser.add_argument('--sample_ids', type=str, default=None,
                        help='Comma-separated sample IDs. Default: all 12 samples')
    parser.add_argument('--use_filtered_images', action='store_true',
                        help='Use FFT-filtered images')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length for C2S model')

    # Model architecture
    parser.add_argument('--vision_model', type=str, default='pathoduet',
                        choices=['uni2_h', 'pathoduet', 'phikon'])
    parser.add_argument('--vision_dim', type=int, default=768)
    parser.add_argument('--gene_model', type=str, default='pythia_410m',
                        choices=['pythia_410m', 'pythia_1b'])
    parser.add_argument('--gene_dim', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--proj_dim', type=int, default=768)
    parser.add_argument('--proj_hidden_dim', type=int, default=3072)
    parser.add_argument('--proj_layers', type=int, default=2)
    parser.add_argument('--num_cross_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=12)

    # Loss configuration (IGC + IGM only)
    parser.add_argument('--igc_weight', type=float, default=1.0,
                        help='Weight for Image-Gene Contrastive loss')
    parser.add_argument('--igm_weight', type=float, default=1.0,
                        help='Weight for Image-Gene Matching loss')
    parser.add_argument('--igc_loss_type', type=str, default='clip',
                        choices=['clip', 'infonce'])
    parser.add_argument('--igc_temperature', type=float, default=1.0)

    # Training (no scheduler, no gradient accumulation, no grad clipping)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--use_mixed_precision', action='store_true')

    # Checkpoint
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                        help='Path to pretrained checkpoint to fine-tune from')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')

    # System
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Per-sample training mode
    parser.add_argument('--per_sample', action='store_true',
                        help='Train separately on each sample and save individual checkpoints')

    return parser.parse_args()


def save_checkpoint(epoch, model, optimizer, metrics, args, filename):
    """Save training checkpoint."""
    os.makedirs(args.output_dir, exist_ok=True)
    filepath = os.path.join(args.output_dir, filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'args': vars(args)
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def train_one_epoch(epoch, model, dataloader, optimizer, scaler, device, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_igc_loss = 0
    total_igm_loss = 0
    num_batches = 0
    epoch_start_time = time.time()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch in pbar:
        images = batch['images'].to(device)
        gene_sequences = batch['gene_sequences']

        optimizer.zero_grad()

        # Forward pass
        if args.use_mixed_precision and scaler is not None:
            with autocast():
                outputs = model(
                    images=images,
                    gene_sentences=gene_sequences,
                    images_aug1=images,  # Use same images for IGC (no augmentation)
                    max_length=args.max_length
                )
                loss = outputs['total_loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                images=images,
                gene_sentences=gene_sequences,
                images_aug1=images,  # Use same images for IGC (no augmentation)
                max_length=args.max_length
            )
            loss = outputs['total_loss']

            loss.backward()
            optimizer.step()

        # Track losses
        total_loss += outputs['total_loss'].item()
        if 'igc_loss' in outputs:
            total_igc_loss += outputs['igc_loss'].item()
        if 'igm_loss' in outputs:
            total_igm_loss += outputs['igm_loss'].item()
        num_batches += 1

        # Update progress bar
        postfix = {'loss': f'{total_loss / num_batches:.4f}'}
        if total_igc_loss > 0:
            postfix['igc'] = f'{total_igc_loss / num_batches:.4f}'
        if total_igm_loss > 0:
            postfix['igm'] = f'{total_igm_loss / num_batches:.4f}'
        pbar.set_postfix(postfix)

    epoch_time = time.time() - epoch_start_time

    metrics = {
        'loss': total_loss / num_batches,
        'igc_loss': total_igc_loss / num_batches if total_igc_loss > 0 else 0,
        'igm_loss': total_igm_loss / num_batches if total_igm_loss > 0 else 0,
        'epoch_time': epoch_time
    }

    return metrics


def train_single_sample(sample_id, args, device, transform, tokenizer, gene_encoder, vision_encoder, pretrained_state_dict=None):
    """Train on a single sample and save checkpoint."""
    print(f"\n{'='*80}")
    print(f"Training on sample: {sample_id}")
    print(f"{'='*80}")

    # Create dataloader based on dataset type
    if args.dataset == 'breast_cancer':
        dataloader = create_breast_cancer_dataloader(
            data_dir=args.data_dir,
            use_filtered_images=args.use_filtered_images,
            transform=transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )
    else:
        dataloader = create_dlpfc_dataloader(
            data_dir=args.data_dir,
            sample_ids=[sample_id],
            use_filtered_images=args.use_filtered_images,
            transform=transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )
    print(f"  Training spots: {len(dataloader.dataset)}")

    # Create fresh model
    model = ST_AlignmentModel(
        vision_encoder=vision_encoder,
        gene_encoder=gene_encoder,
        tokenizer=tokenizer,
        vision_dim=args.vision_dim,
        gene_dim=args.gene_dim,
        proj_dim=args.proj_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_layers=args.proj_layers,
        num_cross_layers=args.num_cross_layers,
        num_heads=args.num_heads,
        freeze_encoders=False,
        loss_type=args.igc_loss_type,
        temperature=args.igc_temperature,
        learnable_temperature=False,
        igc_weight=args.igc_weight,
        iic_weight=0.0,
        igm_weight=args.igm_weight,
        device=device
    )
    model = model.to(device)

    # Load pretrained weights
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        print(f"  Loaded pretrained weights")

    # Create optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Setup mixed precision
    scaler = GradScaler() if args.use_mixed_precision else None

    # Training loop
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            epoch=epoch,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args
        )

        print(f"  Epoch {epoch}: Loss={metrics['loss']:.4f}, IGC={metrics['igc_loss']:.4f}, IGM={metrics['igm_loss']:.4f}")

    # Save final checkpoint with sample_id in its own directory
    sample_output_dir = os.path.join(args.output_dir, sample_id)
    os.makedirs(sample_output_dir, exist_ok=True)
    save_checkpoint(
        epoch=args.epochs,
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        args=args,
        filename=os.path.join(sample_id, f'checkpoint_epoch_{args.epochs}.pt')
    )

    return metrics


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    device = torch.device(args.device)

    # Set data directory based on dataset if not provided
    if args.data_dir is None:
        if args.dataset == 'breast_cancer':
            args.data_dir = BREAST_CANCER_DIR
        else:
            args.data_dir = DLPFC_DIR

    # Parse sample IDs based on dataset
    if args.dataset == 'breast_cancer':
        sample_ids = ['Human_breast_cancer']  # Single sample for breast cancer
    elif args.sample_ids:
        sample_ids = [s.strip() for s in args.sample_ids.split(',')]
    else:
        sample_ids = DLPFC_SAMPLES

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_name = "Human Breast Cancer" if args.dataset == 'human_breast_cancer' else "DLPFC"
    print("=" * 80)
    print(f"Fine-tuning ST_AlignmentModel on {dataset_name}")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Random Seed: {args.seed}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Sample IDs: {sample_ids}")
    print(f"Per-sample training: {args.per_sample}")
    print(f"Pretrained: {args.pretrained_checkpoint}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Vision Model: {args.vision_model} ({args.vision_dim} dim)")
    print(f"Gene Model: {args.gene_model} ({args.gene_dim} dim, {args.num_layers} layers)")
    print(f"Projection Dim: {args.proj_dim}")
    print(f"Cross-attention Layers: {args.num_cross_layers}")
    print(f"Loss: IGC={args.igc_weight} ({args.igc_loss_type}, temp={args.igc_temperature}), IGM={args.igm_weight}")
    print(f"Training: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Mixed Precision: {args.use_mixed_precision}")
    print("=" * 80)

    # ========================
    # Load Encoders (shared across all samples)
    # ========================
    print("\nLoading encoders...")

    transform, vision_encoder = get_vision_model(args.vision_model, device)
    tokenizer, gene_encoder = get_c2s_model(
        model_name=args.gene_model,
        device=device,
        num_layers=args.num_layers
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Vision Model: {args.vision_model} (dim={args.vision_dim})")
    print(f"Gene Model: C2S {args.gene_model} (dim={args.gene_dim}, num_layers={args.num_layers})")

    # Load pretrained checkpoint state dict (to be reused for each sample)
    pretrained_state_dict = None
    if args.pretrained_checkpoint:
        print(f"\nLoading pretrained checkpoint from {args.pretrained_checkpoint}")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
        pretrained_state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # ========================
    # Per-sample training mode
    # ========================
    if args.per_sample:
        print("\n" + "=" * 80)
        print(f"Per-sample training mode: Training on {len(sample_ids)} samples individually")
        print("=" * 80)

        all_metrics = {}
        for i, sample_id in enumerate(sample_ids):
            print(f"\n[{i+1}/{len(sample_ids)}] Processing sample {sample_id}")
            metrics = train_single_sample(
                sample_id=sample_id,
                args=args,
                device=device,
                transform=transform,
                tokenizer=tokenizer,
                gene_encoder=gene_encoder,
                vision_encoder=vision_encoder,
                pretrained_state_dict=pretrained_state_dict
            )
            all_metrics[sample_id] = metrics

    # ========================
    # Standard training mode (all samples together)
    # ========================
    else:
        print("\nCreating dataloader (all samples)...")

        if args.dataset == 'breast_cancer':
            dataloader = create_breast_cancer_dataloader(
                data_dir=args.data_dir,
                use_filtered_images=args.use_filtered_images,
                transform=transform,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True
            )
        else:
            dataloader = create_dlpfc_dataloader(
                data_dir=args.data_dir,
                sample_ids=sample_ids,
                use_filtered_images=args.use_filtered_images,
                transform=transform,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True
            )

        print(f"Training samples: {len(dataloader.dataset)}")

        # Create Model
        print("\nCreating ST_AlignmentModel...")

        model = ST_AlignmentModel(
            vision_encoder=vision_encoder,
            gene_encoder=gene_encoder,
            tokenizer=tokenizer,
            vision_dim=args.vision_dim,
            gene_dim=args.gene_dim,
            proj_dim=args.proj_dim,
            proj_hidden_dim=args.proj_hidden_dim,
            proj_layers=args.proj_layers,
            num_cross_layers=args.num_cross_layers,
            num_heads=args.num_heads,
            freeze_encoders=False,
            loss_type=args.igc_loss_type,
            temperature=args.igc_temperature,
            learnable_temperature=False,
            igc_weight=args.igc_weight,
            iic_weight=0.0,
            igm_weight=args.igm_weight,
            device=device
        )

        model = model.to(device)
        print("Model created successfully")

        # Load Pretrained Checkpoint
        if pretrained_state_dict is not None:
            model.load_state_dict(pretrained_state_dict)
            print("Loaded pretrained weights")

        # Create Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"\nTrainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Setup mixed precision
        scaler = GradScaler() if args.use_mixed_precision else None

        # Training Loop
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80)

        best_loss = float('inf')

        for epoch in range(1, args.epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*60}")

            metrics = train_one_epoch(
                epoch=epoch,
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                args=args
            )

            print(f"\nEpoch {epoch} Summary:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  IGC Loss: {metrics['igc_loss']:.4f}")
            print(f"  IGM Loss: {metrics['igm_loss']:.4f}")
            print(f"  Time: {metrics['epoch_time']:.2f}s")

            # Save checkpoint
            if epoch % args.save_every == 0:
                save_checkpoint(epoch, model, optimizer, metrics, args,
                              filename=f'checkpoint_epoch_{epoch}.pt')

        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Checkpoints saved to: {args.output_dir}")
        print("=" * 80)


if __name__ == '__main__':
    main()
