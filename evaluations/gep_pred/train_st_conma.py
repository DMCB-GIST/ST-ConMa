import os
# Disable tokenizers parallelism warning (must be set before importing transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PatientWiseCVDataset, collate_fn_patient_cv
from model import STConMaFinetuneFull


def compute_metrics(pred_exp, true_exp):
    """Compute PCC and MSE metrics."""
    if isinstance(pred_exp, torch.Tensor):
        pred_exp = pred_exp.cpu().numpy()
    if isinstance(true_exp, torch.Tensor):
        true_exp = true_exp.cpu().numpy()

    # Per-gene PCC (average across samples)
    gene_pccs = []
    for g in range(pred_exp.shape[1]):
        try:
            pcc, _ = pearsonr(pred_exp[:, g], true_exp[:, g])
            if not np.isnan(pcc):
                gene_pccs.append(pcc)
        except:
            pass
    gene_pccs = np.array(gene_pccs)
    mean_gene_pcc = np.mean(gene_pccs) if len(gene_pccs) > 0 else 0.0
    std_gene_pcc = np.std(gene_pccs) if len(gene_pccs) > 0 else 0.0

    # MSE
    total_mse = np.mean((pred_exp - true_exp) ** 2)
    gene_mse = np.mean((pred_exp - true_exp) ** 2, axis=0)

    # RMSE
    total_rmse = np.sqrt(total_mse)
    gene_rmse = np.sqrt(gene_mse)

    # MAE
    total_mae = np.mean(np.abs(pred_exp - true_exp))
    gene_mae = np.mean(np.abs(pred_exp - true_exp), axis=0)

    return {
        'gene_pcc': mean_gene_pcc,
        'gene_pcc_std': std_gene_pcc,
        'total_mse': total_mse,
        'gene_mse': gene_mse,
        'gene_mse_std': np.std(gene_mse),
        'total_rmse': total_rmse,
        'gene_rmse': gene_rmse,
        'gene_rmse_std': np.std(gene_rmse),
        'total_mae': total_mae,
        'gene_mae': gene_mae,
        'gene_mae_std': np.std(gene_mae),
    }


def train_one_epoch(model, dataloader, optimizer, device, epoch, scaler=None, use_igm=False):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0.0
    total_igc_loss = 0.0
    total_igm_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        images = batch['images'].to(device)
        gene_sentences = batch['gene_sentences']

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                result = model(images, gene_sentences, use_igm=use_igm)
                loss = result['loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            result = model(images, gene_sentences, use_igm=use_igm)
            loss = result['loss']
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_igc_loss += result['igc_loss'].item()
        if use_igm:
            total_igm_loss += result['igm_loss'].item()
        n_batches += 1

        if use_igm:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'igc': f'{result["igc_loss"].item():.4f}',
                'igm': f'{result["igm_loss"].item():.4f}'
            })
        else:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / n_batches
    avg_igc_loss = total_igc_loss / n_batches
    avg_igm_loss = total_igm_loss / n_batches if use_igm else 0.0

    return avg_loss, avg_igc_loss, avg_igm_loss


@torch.no_grad()
def build_gallery(model, dataloader, device, use_igm=False):
    """
    Build gene embedding gallery from training data.

    Args:
        model: trained model
        dataloader: training data loader
        device: device
        use_igm: if True, also cache gene sequences for IGM inference

    Returns:
        gallery_gene_emb: [N, proj_dim] normalized gene embeddings
        gallery_expressions: [N, n_genes] expression values
        gallery_gene_sentences: list of gene sentences (if use_igm)
        gallery_gene_seqs: [N, seq_len, gene_dim] gene sequences (if use_igm)
        gallery_gene_masks: [N, seq_len] attention masks (if use_igm)
    """
    model.eval()

    all_gene_emb = []
    all_expressions = []
    all_gene_sentences = []
    all_gene_seqs = []
    all_gene_masks = []

    for batch in tqdm(dataloader, desc='Building gallery'):
        gene_sentences = batch['gene_sentences']
        expressions = batch['expressions']

        with torch.cuda.amp.autocast():
            gene_emb = model.get_gene_embeddings(gene_sentences)

        all_gene_emb.append(gene_emb.float().cpu())
        all_expressions.append(expressions)
        all_gene_sentences.extend(gene_sentences)

        # If using IGM, also cache gene sequences
        if use_igm:
            with torch.cuda.amp.autocast():
                _, gene_seq, gene_mask = model.get_gene_sequences(gene_sentences)
            all_gene_seqs.append(gene_seq.float().cpu())
            all_gene_masks.append(gene_mask.cpu())

    all_gene_emb = torch.cat(all_gene_emb, dim=0)
    all_expressions = torch.cat(all_expressions, dim=0)

    if use_igm:
        all_gene_seqs = torch.cat(all_gene_seqs, dim=0)
        all_gene_masks = torch.cat(all_gene_masks, dim=0)
        return all_gene_emb, all_expressions, all_gene_sentences, all_gene_seqs, all_gene_masks
    else:
        return all_gene_emb, all_expressions, None, None, None


@torch.no_grad()
def evaluate(model, test_loader, train_loader, device, use_igm=False, top_k=128, temperature=0.07):
    """
    Evaluate model on test set.

    For each test image:
    1. Get image embedding
    2. Compute IGC similarity with all training gene embeddings
    3. If top_k is None: use full gallery with softmax weights
       Else: Select top-K candidates
    4. If use_igm (and top_k is not None): compute IGM probabilities for top-K, use as weights
       Else: use softmax(similarity/temperature) as weights
    5. Weighted sum of expressions as prediction

    Args:
        model: trained model
        test_loader: test data loader
        train_loader: train data loader (for gallery)
        device: device
        use_igm: whether to use IGM for reranking (ignored when top_k is None)
        top_k: number of top candidates to consider (None = use full gallery)
        temperature: temperature for softmax

    Returns:
        metrics: overall patient metrics
        all_pred_exp: all predictions
        all_true_exp: all ground truth
        sample_names: list of sample names for each spot
    """
    model.eval()

    # Build gallery from training data
    print("  Building gene embedding gallery...")
    gallery_gene_emb, gallery_expressions, gallery_gene_sentences, gallery_gene_seqs, gallery_gene_masks = \
        build_gallery(model, train_loader, device, use_igm=use_igm)

    gallery_gene_emb = gallery_gene_emb.to(device)
    gallery_expressions = gallery_expressions.to(device)
    # Note: gallery_gene_seqs and gallery_gene_masks stay on CPU to save GPU memory
    # Only top-K will be moved to GPU during evaluation

    gallery_size = gallery_gene_emb.shape[0]

    # Handle top_k=None (use full gallery)
    use_full_gallery = (top_k is None)
    if use_full_gallery:
        actual_top_k = gallery_size
    else:
        actual_top_k = min(top_k, gallery_size)

    all_pred_exp = []
    all_true_exp = []
    all_sample_names = []

    if use_full_gallery:
        print(f"  Using full gallery for prediction (gallery: {gallery_size})")
    elif use_igm:
        print(f"  Using IGM reranking for prediction (gallery: {gallery_size}, top-K: {actual_top_k})")
    else:
        print(f"  Using similarity-based prediction (gallery: {gallery_size}, top-K: {actual_top_k})")

    for batch in tqdm(test_loader, desc='Evaluating'):
        images = batch['images'].to(device)
        expressions = batch['expressions'].to(device)
        sample_names = batch['sample_names']
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast():
            # Get image embeddings
            img_emb = model.get_image_embeddings(images)  # [B, proj_dim], normalized

            # Compute similarity with all gallery
            sim = torch.matmul(img_emb, gallery_gene_emb.T)  # [B, N_gallery]

            if use_full_gallery:
                # Use full gallery: softmax over all similarities
                weights = F.softmax(sim / temperature, dim=1)  # [B, N_gallery]
                # Weighted sum prediction: [B, N] @ [N, n_genes] -> [B, n_genes]
                batch_pred = torch.matmul(weights, gallery_expressions.float())  # [B, n_genes]
            else:
                # Get top-K indices for each image
                top_k_sim, top_k_indices = torch.topk(sim, actual_top_k, dim=1)  # [B, K]

                batch_pred = []

                for i in range(batch_size):
                    indices = top_k_indices[i]  # [K] on GPU
                    top_expressions = gallery_expressions[indices]  # [K, n_genes]

                    if use_igm:
                        # Get gene sequences for top-K candidates (CPU -> GPU)
                        indices_cpu = indices.cpu()  # Move to CPU for indexing
                        top_gene_seqs = gallery_gene_seqs[indices_cpu].to(device)  # [K, seq_len, gene_dim]
                        top_gene_masks = gallery_gene_masks[indices_cpu].to(device)  # [K, seq_len]

                        # Compute IGM probabilities
                        single_image = images[i:i+1]  # [1, 3, H, W]
                        igm_probs = model.compute_igm_scores_batch(
                            image=single_image,
                            gene_sentences=None,  # Not needed, using pre-encoded sequences
                            gene_seqs=top_gene_seqs,
                            gene_attn_masks=top_gene_masks
                        )  # [K]

                        # Apply softmax to IGM probabilities for sharper weighting
                        weights = F.softmax(igm_probs / temperature, dim=0)  # [K]

                        # Free memory
                        del top_gene_seqs, top_gene_masks
                    else:
                        # Use softmax of similarities as weights
                        weights = F.softmax(top_k_sim[i] / temperature, dim=0)  # [K]

                    # Weighted sum prediction
                    pred = torch.matmul(weights.unsqueeze(0), top_expressions.float()).squeeze(0)  # [n_genes]
                    batch_pred.append(pred)

                batch_pred = torch.stack(batch_pred, dim=0)  # [B, n_genes]

        all_pred_exp.append(batch_pred.float())
        all_true_exp.append(expressions)
        all_sample_names.extend(sample_names)

    all_pred_exp = torch.cat(all_pred_exp, dim=0)
    all_true_exp = torch.cat(all_true_exp, dim=0)

    metrics = compute_metrics(all_pred_exp, all_true_exp)

    return metrics, all_pred_exp, all_true_exp, all_sample_names


def compute_per_sample_metrics(pred_exp, true_exp, sample_names):
    """
    Compute metrics for each sample (slide) separately.
    """
    if isinstance(pred_exp, torch.Tensor):
        pred_exp = pred_exp.cpu().numpy()
    if isinstance(true_exp, torch.Tensor):
        true_exp = true_exp.cpu().numpy()

    # Group by sample name
    unique_samples = sorted(list(set(sample_names)))
    per_sample_metrics = {}
    per_sample_pred = {}
    per_sample_true = {}

    for sample in unique_samples:
        indices = [i for i, s in enumerate(sample_names) if s == sample]

        sample_pred = pred_exp[indices]
        sample_true = true_exp[indices]

        metrics = compute_metrics(sample_pred, sample_true)

        per_sample_metrics[sample] = {
            'n_spots': len(indices),
            'gene_pcc': float(metrics['gene_pcc']),
            'gene_pcc_std': float(metrics['gene_pcc_std']),
            'total_mse': float(metrics['total_mse']),
            'gene_mse_std': float(metrics['gene_mse_std']),
            'total_rmse': float(metrics['total_rmse']),
            'gene_rmse_std': float(metrics['gene_rmse_std']),
            'total_mae': float(metrics['total_mae']),
            'gene_mae_std': float(metrics['gene_mae_std']),
        }

        per_sample_pred[sample] = sample_pred
        per_sample_true[sample] = sample_true

    return per_sample_metrics, per_sample_pred, per_sample_true


def main():
    parser = argparse.ArgumentParser(description='Patient-Wise CV Training')
    parser.add_argument('--dataset', type=str, choices=['her2st', 'cscc', 'hlt'], default='her2st')
    parser.add_argument('--fold', type=int, default=0, help='Fold (patient) to use as test')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/st_conma_pythia410m_12layers_3aug_clip/checkpoint_epoch_12.pt')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gene_type', type=str, choices=['heg', 'hvg'], default='heg')

    # IGM-specific arguments
    parser.add_argument('--use_igm', action='store_true',
                        help='Use IGM loss during training')
    parser.add_argument('--eval_use_igm', action='store_true',
                        help='Use IGM for evaluation (default: use similarity only even when --use_igm is set)')
    parser.add_argument('--num_cross_layers', type=int, default=3,
                        help='Number of cross-attention layers for IGM')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='Number of attention heads in cross-attention')
    parser.add_argument('--igc_weight', type=float, default=1.0,
                        help='Weight for IGC loss')
    parser.add_argument('--igm_weight', type=float, default=1.0,
                        help='Weight for IGM loss (only used when --use_igm)')
    parser.add_argument('--top_k', type=str, default='128',
                        help='Number of top candidates for evaluation (use "none" or "all" for full gallery)')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for softmax (when not using IGM)')

    parser.add_argument('--igm_negative_ratio', type=int, default=1,
                        help='Number of negatives per positive for IGM loss (default: 1 for 1:1 ratio)')
    parser.add_argument('--igm_hard_negative', action='store_true',
                        help='Use hard negative mining for IGM (select most similar non-matching genes)')

    args = parser.parse_args()

    # Handle checkpoint=none case
    if args.checkpoint and args.checkpoint.lower() == 'none':
        args.checkpoint = None

    # Handle top_k=none/all case (use full gallery)
    if args.top_k.lower() in ['none', 'all']:
        args.top_k = None
    else:
        args.top_k = int(args.top_k)

    # Set default output_dir based on checkpoint and use_igm
    if args.output_dir is None:
        if args.use_igm:
            model_name = 'st_conma_full' if args.checkpoint else 'st_conma_full_noPT'
        else:
            model_name = 'st_conma' if args.checkpoint else 'st_conma_noPT'
        args.output_dir = f'./results/gep_pred/{model_name}'

    # Construct full output path with gene_type
    output_base = os.path.join(args.output_dir, args.gene_type, args.dataset)

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    def seed_worker(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    # Create output directory
    os.makedirs(output_base, exist_ok=True)

    # Load datasets
    train_dataset = PatientWiseCVDataset(
        dataset=args.dataset,
        fold=args.fold,
        train=True,
        use_augmentation=True,
        gene_type=args.gene_type
    )
    test_dataset = PatientWiseCVDataset(
        dataset=args.dataset,
        fold=args.fold,
        train=False,
        use_augmentation=False,
        gene_type=args.gene_type
    )

    n_folds = train_dataset.get_num_folds()
    test_patient = train_dataset.test_patient
    unit_type = 'sample' if args.dataset == 'hlt' else 'patient'

    print(f"\n{'='*60}")
    print(f"{'Sample' if args.dataset == 'hlt' else 'Patient'}-Wise CV Training: {args.dataset}")
    print(f"Model: ST-ConMa {'Full (IGC + IGM)' if args.use_igm else '(IGC only)'}")
    print(f"Fold {args.fold}/{n_folds-1}: Test {unit_type} = {test_patient}")
    if args.use_igm:
        print(f"IGC weight: {args.igc_weight}, IGM weight: {args.igm_weight}")
        print(f"Cross-attention layers: {args.num_cross_layers}, Heads: {args.num_heads}")
    print(f"Evaluation: top-{args.top_k} {'+ IGM reranking' if args.eval_use_igm else '+ softmax'}")
    print(f"{'='*60}")

    if args.fold < 0 or args.fold >= n_folds:
        raise ValueError(f"fold must be 0-{n_folds-1} for {args.dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_patient_cv,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_patient_cv,
        worker_init_fn=seed_worker,
        generator=g
    )

    g_gallery = torch.Generator()
    g_gallery.manual_seed(args.seed)
    gallery_loader = DataLoader(
        PatientWiseCVDataset(
            dataset=args.dataset,
            fold=args.fold,
            train=True,
            use_augmentation=False,
            gene_type=args.gene_type
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_patient_cv,
        worker_init_fn=seed_worker,
        generator=g_gallery
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Genes ({args.gene_type.upper()}): {len(train_dataset.gene_list)}")

    # Load model
    print(f"\nLoading model (STConMaFinetuneFull, train_igm={args.use_igm}, eval_igm={args.eval_use_igm})...")
    model = STConMaFinetuneFull(
            pretrained_checkpoint=args.checkpoint,
            proj_dim=768,
            temperature=0.07,
            learnable_temperature=False,
            freeze_vision_encoder=False,
            freeze_gene_encoder=False,
            num_cross_layers=args.num_cross_layers,
            num_heads=args.num_heads,
            igc_weight=args.igc_weight,
            igm_weight=args.igm_weight if args.use_igm else 0.0,
            device=args.device
        )
    model = model.to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    print("\nStarting training...")

    for epoch in range(1, args.epochs + 1):
        train_loss, igc_loss, igm_loss = train_one_epoch(
            model, train_loader, optimizer, args.device, epoch, scaler,
            use_igm=args.use_igm
        )

        if args.use_igm:
            print(f"Epoch {epoch}/{args.epochs} - Loss: {train_loss:.4f} (IGC: {igc_loss:.4f}, IGM: {igm_loss:.4f})")
        else:
            print(f"Epoch {epoch}/{args.epochs} - Loss: {train_loss:.4f}")

    # Save trained model
    model_path = os.path.join(output_base, f'{unit_type}_{test_patient}_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'use_igm': args.use_igm,
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    # Final evaluation
    print("\n" + "="*60)
    print(f"Evaluating on test {unit_type}: {test_patient}")
    print("="*60)
    metrics, pred_exp, true_exp, sample_names = evaluate(
        model, test_loader, gallery_loader, args.device,
        use_igm=args.eval_use_igm,
        top_k=args.top_k,
        temperature=args.temperature
    )

    print("\n  Computing per-sample metrics...")
    per_sample_metrics, per_sample_pred, per_sample_true = compute_per_sample_metrics(
        pred_exp, true_exp, sample_names
    )

    # Print results
    print(f"\n=== Results ({unit_type.capitalize()} {test_patient}) ===")
    print(f"  Gene PCC: {metrics['gene_pcc']:.4f} +/- {metrics['gene_pcc_std']:.4f}")
    print(f"  Total MSE: {metrics['total_mse']:.4f}")
    print(f"  Total RMSE: {metrics['total_rmse']:.4f}")
    print(f"  Total MAE: {metrics['total_mae']:.4f}")

    if args.dataset != 'hlt':
        print(f"\n=== Per-Sample Results ({unit_type.capitalize()} {test_patient}) ===")
        for sample, sample_metrics in per_sample_metrics.items():
            print(f"  {sample}: Gene PCC={sample_metrics['gene_pcc']:.4f}, "
                  f"MSE={sample_metrics['total_mse']:.4f}, "
                  f"n_spots={sample_metrics['n_spots']}")

    # Save results
    if args.use_igm:
        model_name = 'st_conma_full'
    else:
        model_name = 'st_conma'

    results = {
        'dataset': args.dataset,
        'fold': args.fold,
        f'test_{unit_type}': test_patient,
        'n_folds': n_folds,
        'epochs': args.epochs,
        'gene_type': args.gene_type,
        'model': model_name,
        'use_igm': args.use_igm,
        'top_k': args.top_k,
        'igc_weight': args.igc_weight,
        'igm_weight': args.igm_weight if args.use_igm else 0.0,
        'igm_negative_ratio': args.igm_negative_ratio,
        'igm_hard_negative': args.igm_hard_negative,
        'num_cross_layers': args.num_cross_layers,
        'num_heads': args.num_heads,
        f'{unit_type}_metrics': {
            'gene_pcc': float(metrics['gene_pcc']),
            'gene_pcc_std': float(metrics['gene_pcc_std']),
            'total_mse': float(metrics['total_mse']),
            'total_rmse': float(metrics['total_rmse']),
            'total_mae': float(metrics['total_mae']),
        },
        'per_sample_metrics': per_sample_metrics if args.dataset != 'hlt' else {},
        'args': vars(args)
    }

    results_path = os.path.join(output_base, f'{args.dataset}_{unit_type}_{test_patient}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save per-sample predictions
    if args.dataset != 'hlt':
        sample_pred_dir = os.path.join(output_base, f'{unit_type}_{test_patient}_samples')
        os.makedirs(sample_pred_dir, exist_ok=True)

        for sample_name in per_sample_pred.keys():
            sample_pred_path = os.path.join(sample_pred_dir, f'{sample_name}_pred.npy')
            sample_true_path = os.path.join(sample_pred_dir, f'{sample_name}_true.npy')
            np.save(sample_pred_path, per_sample_pred[sample_name])
            np.save(sample_true_path, per_sample_true[sample_name])

    # Save test unit predictions
    if isinstance(pred_exp, torch.Tensor):
        pred_exp = pred_exp.cpu().numpy()
    if isinstance(true_exp, torch.Tensor):
        true_exp = true_exp.cpu().numpy()

    pred_path = os.path.join(output_base, f'{args.dataset}_{unit_type}_{test_patient}_pred.npy')
    true_path = os.path.join(output_base, f'{args.dataset}_{unit_type}_{test_patient}_true.npy')
    np.save(pred_path, pred_exp)
    np.save(true_path, true_exp)

    print(f"\nResults saved to: {results_path}")
    print(f"{unit_type.capitalize()}-level predictions: {pred_path}")
    print(f"  Shape: {pred_exp.shape} (n_spots, n_genes)")
    if args.dataset != 'hlt':
        print(f"Per-sample predictions: {sample_pred_dir}/")
        for sample_name in per_sample_pred.keys():
            print(f"  - {sample_name}: {per_sample_pred[sample_name].shape}")
    print("="*60)


if __name__ == '__main__':
    main()
