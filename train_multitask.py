"""
Pre-training script for ST-ConMa
Includes: IGC (Image-Gene Contrastive), IIC (Image-Image Contrastive), IGM (Image-Gene Matching)
"""

import os
import sys
import argparse
from tqdm import tqdm
import time
import warnings
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
from utils.model import get_vision_model, get_c2s_model
from utils.multimodal import ST_AlignmentModel, ST_AlignmentTrainer
from utils.dataset_load import create_dataloaders
from utils.augmentations import get_train_augmentation


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-task ST Pre-training')

    # Data
    parser.add_argument('--train_image_dir', type=str, required=True,
                       help='Training images directory')
    parser.add_argument('--train_gene_file', type=str, required=True,
                       help='Training gene sequences file (CSV)')
    parser.add_argument('--except_ft_samples', action='store_true',
                       help='Exclude downstream samples for evaluation')
    parser.add_argument('--use_augmentation', action='store_true',
                       help='Use data augmentation during training')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length for C2S model')

    # Model - Vision
    parser.add_argument('--vision_model', type=str, default='pathoduet',
                       choices=['pathoduet'],
                       help='Vision model name')
    parser.add_argument('--vision_dim', type=int, default=768,
                       help='Vision encoder output dimension')

    # Model - Gene (C2S)
    parser.add_argument('--gene_model', type=str, default='pythia_410m',
                       choices=['pythia_410m'],
                       help='C2S gene model name')
    parser.add_argument('--gene_dim', type=int, default=1024,
                       help='Gene encoder output dimension (pythia_410m: 1024)')
    parser.add_argument('--num_layers', type=int, default=12,
                       help='Number of transformer layers to keep in gene encoder (default: 12 layers)')
    parser.add_argument('--layer_index', type=int, default=None,
                       help='Layer index to extract from C2S model (None for last layer)')
    parser.add_argument('--random_init_gene', action='store_true',
                       help='Random initialize gene encoder instead of using pretrained weights')

    # Model - Projection
    parser.add_argument('--proj_dim', type=int, default=768,
                       help='Projection dimension')
    parser.add_argument('--proj_hidden_dim', type=int, default=3072,
                       help='Projection hidden dimension')
    parser.add_argument('--proj_layers', type=int, default=2,
                       help='Number of projection layers')

    # Model - Cross-attention (for IGM)
    parser.add_argument('--num_cross_layers', type=int, default=3,
                       help='Number of cross-attention layers')
    parser.add_argument('--num_heads', type=int, default=12,
                       help='Number of attention heads')

    # Loss weights
    parser.add_argument('--igc_weight', type=float, default=1.0,
                       help='Image-Gene Contrastive loss weight')
    parser.add_argument('--iic_weight', type=float, default=1.0,
                       help='Image-Image Contrastive loss weight')
    parser.add_argument('--igm_weight', type=float, default=1.0,
                       help='Image-Gene Matching loss weight')

    # Training
    parser.add_argument('--freeze_encoders', action='store_true',
                       help='Freeze encoder weights')
    parser.add_argument('--loss_type', type=str, default='clip',
                       choices=['clip', 'siglip'],
                       help='Contrastive loss type (default for both IGC and IIC)')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss (default for both IGC and IIC)')
    parser.add_argument('--learnable_temperature', action='store_true',
                       help='Make temperature learnable')

    # Separate loss configurations for IGC and IIC
    parser.add_argument('--igc_loss_type', type=str, default=None,
                       choices=['clip', 'siglip'],
                       help='Loss type specifically for IGC (overrides --loss_type)')
    parser.add_argument('--iic_loss_type', type=str, default=None,
                       choices=['clip', 'siglip', 'simclr', 'ntxent'],
                       help='Loss type specifically for IIC (overrides --loss_type). SimCLR uses NT-Xent loss.')
    parser.add_argument('--igc_temperature', type=float, default=None,
                       help='Temperature specifically for IGC (overrides --temperature)')
    parser.add_argument('--iic_temperature', type=float, default=None,
                       help='Temperature specifically for IIC (overrides --temperature)')
    parser.add_argument('--epochs', type=int, default=12,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')

    # Scheduler
    parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine',
                       choices=['cosine', 'linear_warmup_cosine', 'step', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                       help='Warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                       help='Minimum learning rate')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--use_mixed_precision', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./checkpoints/st_conma_pythia410m_12layers_3aug_clip',
                       help='Output directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=12,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    # DDP
    parser.add_argument('--use_ddp', action='store_true',
                       help='Use DistributedDataParallel')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for DDP')

    # Logging
    parser.add_argument('--log_every', type=int, default=1,
                       help='Log every N steps')

    # Wandb
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='ST-ConMa',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='pathoduet-pythia410m-12layers-3aug-clip',
                       help='Wandb run name')

    return parser.parse_args()


def setup_ddp(args):
    """Setup Distributed Data Parallel"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.use_ddp = False
        return args

    # Initialize process group
    dist.init_process_group(
        backend='nccl', 
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )

    # Set device
    torch.cuda.set_device(args.local_rank)
    args.device = f'cuda:{args.local_rank}'

    return args


def set_seed(seed: int, rank: int = 0, use_deterministic: bool = False):
    """
    Set random seed for reproducibility.

    Args:
        seed: base random seed
        rank: GPU rank (different augmentation per GPU)

    Note:
        - Model initialization uses same seed across GPUs (required for DDP)
        - Augmentation uses rank-specific seed for diversity
    """
    import random
    import numpy as np

    # Model weight initialization: same seed across all GPUs (required for DDP)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Data augmentation: different seed per GPU for diversity
    random.seed(seed + rank)
    np.random.seed(seed + rank)

    if use_deterministic:
        # WARNING: This slows down training significantly (20-30%)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[Rank {rank}] WARNING: Deterministic mode enabled - training will be slower!")
    else:
        # Use cudnn auto-tuner for best performance
        torch.backends.cudnn.benchmark = True

    os.environ['PYTHONHASHSEED'] = str(seed)


def save_checkpoint(epoch, model, optimizer, scheduler, args, filename='checkpoint.pt'):
    """Save checkpoint"""
    if args.rank != 0:
        return

    os.makedirs(args.output_dir, exist_ok=True)
    filepath = os.path.join(args.output_dir, filename)

    # Get model state dict
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, args):
    """Load checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    # Load model state
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"Resumed from epoch {start_epoch}")

    return start_epoch


def train_one_epoch(epoch, model, train_loader, trainer, args, global_step=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_updates = 0
    epoch_start_time = time.time()

    # Progress bar
    if args.rank == 0:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    else:
        pbar = train_loader

    for step, batch in enumerate(pbar):
        step_start_time = time.time()

        images_aug1 = batch['images_aug1']  # Unaug for IGC
        images_aug2 = batch['images_aug2']  # Aug for IIC
        images_aug3 = batch['images_aug3']  # Aug for IIC
        gene_sentences = batch['gene_sentences']

        # Train step
        metrics = trainer.train_step(
            gene_sentences=gene_sentences,
            images_aug1=images_aug1,
            images_aug2=images_aug2,
            images_aug3=images_aug3,
            max_length=args.max_length
        )

        # Only accumulate loss when weights are updated
        if metrics['updated']:
            total_loss += metrics['total_loss']
            num_updates += 1
            global_step += 1

            # Get current learning rate
            current_lr = trainer.optimizer.param_groups[0]['lr']
            step_time = time.time() - step_start_time

            # Logging
            if args.rank == 0:
                # Console logging
                if num_updates % args.log_every == 0:
                    postfix = {
                        'loss': f'{metrics["total_loss"]:.4f}',
                        'lr': f'{current_lr:.2e}'
                    }

                    # Add individual losses if available
                    if 'igc_loss' in metrics:
                        postfix['igc'] = f'{metrics["igc_loss"]:.4f}'
                    if 'iic_loss' in metrics:
                        postfix['iic'] = f'{metrics["iic_loss"]:.4f}'
                    if 'igm_loss' in metrics:
                        postfix['igm'] = f'{metrics["igm_loss"]:.4f}'

                    pbar.set_postfix(postfix)

                # Wandb logging
                if args.use_wandb:
                    log_dict = {
                        'train/loss': metrics['total_loss'],
                        'train/learning_rate': current_lr,
                        'train/step_time': step_time,
                        'train/epoch': epoch
                    }

                    # Add individual losses if available
                    if 'igc_loss' in metrics:
                        log_dict['train/igc_loss'] = metrics['igc_loss']
                    if 'iic_loss' in metrics:
                        log_dict['train/iic_loss'] = metrics['iic_loss']
                    if 'igm_loss' in metrics:
                        log_dict['train/igm_loss'] = metrics['igm_loss']

                    wandb.log(log_dict, step=global_step)

    # Calculate average loss and epoch time
    avg_loss = total_loss / max(num_updates, 1)
    epoch_time = time.time() - epoch_start_time

    # Synchronize across GPUs
    if args.use_ddp:
        avg_loss_tensor = torch.tensor(avg_loss, device=args.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

    return avg_loss, epoch_time, global_step


def main():
    """Main training function"""
    args = parse_args()

    # Setup DDP first to get rank information
    if args.use_ddp:
        args = setup_ddp(args)
    else:
        # Set default values for non-DDP mode
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    # Set random seed (with rank-specific seed for augmentation diversity)
    set_seed(args.seed, rank=args.rank, use_deterministic=False)

    # Print info
    if args.rank == 0:
        print("="*80)
        print("Multi-task ST Pre-training")
        print("="*80)
        print(f"Random Seed: {args.seed}")
        print(f"Device: {args.device}")
        print(f"World Size: {args.world_size}")
        print(f"Rank: {args.rank}")
        print(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps * args.world_size}")
        print(f"Loss Weights - IGC: {args.igc_weight}, IIC: {args.iic_weight}, IGM: {args.igm_weight}")
        print("="*80)

    # Initialize Wandb
    if args.use_wandb and args.rank == 0:
        wandb_api_key = os.environ.get('WANDB_API_KEY', None)
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        else:
            wandb.login()

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )

    # Set device
    device = torch.device(args.device)

    # HuggingFace Login
    hf_token = os.environ.get('HUGGINGFACE_TOKEN', None)
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        if args.rank == 0:
            print("Logged in to HuggingFace")

    # ========================
    # Load Encoders
    # ========================
    if args.rank == 0:
        print("\nLoading encoders...")

    # Vision encoder
    transform, vision_encoder = get_vision_model(
        args.vision_model,
        device
    )

    # Gene encoder
    tokenizer, gene_encoder = get_c2s_model(
        model_name=args.gene_model,
        device=device,
        num_layers=args.num_layers,
        random_init=args.random_init_gene
    )

    # Set pad_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.rank == 0:
        print(f"Vision Model: {args.vision_model} (dim={args.vision_dim})")
        num_layers_info = f"num_layers={args.num_layers}, " if args.num_layers else ""
        random_init_info = "random_init=True, " if args.random_init_gene else ""
        print(f"Gene Model: C2S {args.gene_model} (dim={args.gene_dim}, {num_layers_info}{random_init_info}layer_index={args.layer_index})")

    # ========================
    # Create Multimodal Model
    # ========================
    if args.rank == 0:
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
        freeze_encoders=args.freeze_encoders,
        loss_type=args.loss_type,
        temperature=args.temperature,
        learnable_temperature=args.learnable_temperature,
        igc_weight=args.igc_weight,
        iic_weight=args.iic_weight,
        igm_weight=args.igm_weight,
        igc_loss_type=args.igc_loss_type,
        iic_loss_type=args.iic_loss_type,
        igc_temperature=args.igc_temperature,
        iic_temperature=args.iic_temperature,
        device=device,
        layer_index=args.layer_index
    )

    model = model.to(device)

    # Wrap with DDP
    if args.use_ddp:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    if args.rank == 0:
        print("Model created successfully")

    # ========================
    # Create Dataloaders
    # ========================
    if args.rank == 0:
        print("\nCreating dataloaders...")

    # Get augmentation if requested (with model-specific normalization)
    augmentation = get_train_augmentation() if args.use_augmentation else None

    # For multitask learning with augmentation, we don't use a separate transform
    # The augmentation transforms already include ToTensor() and model-specific normalization
    transform = None

    train_loader = create_dataloaders(
        train_image_dir=args.train_image_dir,
        train_gene_file=args.train_gene_file,
        transform=transform,
        augmentation=augmentation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_ddp=args.use_ddp,
        except_ft_samples=args.except_ft_samples,
        use_multitask=True  # Enable multi-task learning
    )

    if args.rank == 0:
        print(f"Training samples: {len(train_loader.dataset)}")

    # ========================
    # Create Optimizer & Scheduler
    # ========================
    from utils.optimizer import get_optimizer

    optimizer_config = {
        'optimizer': args.optimizer,
        'lr': args.lr,
        'weight_decay': args.weight_decay
    }
    optimizer = get_optimizer(model.parameters(), optimizer_config)

    # Calculate total training steps for step-based scheduler
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_training_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    if args.rank == 0:
        print(f"\nScheduler Configuration:")
        print(f"  Scheduler type: {args.scheduler}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Total training steps: {total_training_steps}")

    # Create scheduler based on args.scheduler
    if args.scheduler == 'none':
        scheduler = None
        if args.rank == 0:
            print(f"  No scheduler - using constant learning rate: {args.lr}")
    else:
        # Use step-based cosine scheduler with warmup (default)
        from utils.scheduler import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
            eta_min=args.min_lr
        )
        if args.rank == 0:
            print(f"  Using cosine schedule with warmup")
            print(f"  Initial LR: {args.lr}, Min LR: {args.min_lr}")

    # ========================
    # Create Trainer
    # ========================
    trainer = ST_AlignmentTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_mixed_precision=args.use_mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm
    )

    # ========================
    # Resume from checkpoint
    # ========================
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, args)

    # ========================
    # Training Loop
    # ========================
    if args.rank == 0:
        print("\nStarting training...")

    global_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        if args.rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*60}")

        # Train one epoch
        avg_loss, epoch_time, global_step = train_one_epoch(
            epoch, model, train_loader, trainer, args, global_step
        )

        if args.rank == 0:
            print(f"Epoch {epoch} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")

            # Wandb logging
            if args.use_wandb:
                wandb.log({
                    'epoch/loss': avg_loss,
                    'epoch/time': epoch_time,
                    'epoch/number': epoch
                })

        # Save checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, args,
                          filename=f'checkpoint_epoch_{epoch}.pt')

    print("\nTraining completed!")

    # Cleanup
    if args.use_wandb and args.rank == 0:
        wandb.finish()

    if args.use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
