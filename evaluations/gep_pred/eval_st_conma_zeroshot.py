"""
ST-ConMa Zero-Shot Gene Expression Prediction.

Uses pretrained ST-ConMa model directly without any fine-tuning.
Patient-wise leave-one-out cross-validation.

Expression prediction:
- Build gallery from training patients (gene embeddings + expressions)
- Test: image embedding -> similarity with gallery -> softmax -> weighted sum

- HER2ST: 8 patients (A-H) -> 8-fold patient-wise CV
- CSCC: 4 patients (P2, P5, P9, P10) -> 4-fold patient-wise CV
- HLT: 4 samples (A1, B1, C1, D1) -> 4-fold sample-wise CV

Usage:
    # Single fold
    python eval_st_conma_zeroshot.py --dataset her2st --fold 0 --gene_type heg

    # All folds (run sequentially)
    for i in {0..7}; do python eval_st_conma_zeroshot.py --dataset her2st --fold $i --gene_type heg; done
    for i in {0..3}; do python eval_st_conma_zeroshot.py --dataset cscc --fold $i --gene_type heg; done
    for i in {0..3}; do python eval_st_conma_zeroshot.py --dataset hlt --fold $i --gene_type heg; done
"""

import os
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
from model import STConMaFinetune


# Default checkpoint
DEFAULT_CHECKPOINT = './checkpoints/st_conma_pythia410m_12layers_3aug_clip/checkpoint_epoch_12.pt'


def compute_metrics(pred_exp, true_exp):
    """Compute PCC and MSE metrics."""
    if isinstance(pred_exp, torch.Tensor):
        pred_exp = pred_exp.cpu().numpy()
    if isinstance(true_exp, torch.Tensor):
        true_exp = true_exp.cpu().numpy()

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

    total_mse = np.mean((pred_exp - true_exp) ** 2)
    gene_mse = np.mean((pred_exp - true_exp) ** 2, axis=0)
    total_rmse = np.sqrt(total_mse)
    gene_rmse = np.sqrt(gene_mse)
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


def compute_per_sample_metrics(pred_exp, true_exp, sample_names):
    """Compute metrics for each sample separately."""
    if isinstance(pred_exp, torch.Tensor):
        pred_exp = pred_exp.cpu().numpy()
    if isinstance(true_exp, torch.Tensor):
        true_exp = true_exp.cpu().numpy()

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


@torch.no_grad()
def build_gallery(model, dataloader, device):
    """Build gene embedding gallery from training data."""
    model.eval()
    all_gene_emb = []
    all_expressions = []

    for batch in tqdm(dataloader, desc='Building gallery'):
        gene_sentences = batch['gene_sentences']
        expressions = batch['expressions']

        with torch.amp.autocast('cuda'):
            gene_emb = model.get_gene_embeddings(gene_sentences)

        all_gene_emb.append(gene_emb.float().cpu())
        all_expressions.append(expressions)

    all_gene_emb = torch.cat(all_gene_emb, dim=0)
    all_expressions = torch.cat(all_expressions, dim=0)
    return all_gene_emb, all_expressions


@torch.no_grad()
def evaluate(model, test_loader, gallery_gene_emb, gallery_expressions, device, temperature=0.07):
    """Evaluate on test set using similarity-based prediction."""
    model.eval()
    gallery_gene_emb = gallery_gene_emb.to(device)
    gallery_expressions = gallery_expressions.to(device)

    gallery_size = gallery_gene_emb.shape[0]

    all_pred_exp = []
    all_true_exp = []
    all_sample_names = []

    print(f"  Evaluating (gallery: {gallery_size}, temperature: {temperature})")

    for batch in tqdm(test_loader, desc='Evaluating'):
        images = batch['images'].to(device)
        expressions = batch['expressions'].to(device)
        sample_names = batch['sample_names']

        with torch.amp.autocast('cuda'):
            # Get image embeddings
            img_emb = model.get_image_embeddings(images)

            # Compute similarity
            sim = torch.matmul(img_emb, gallery_gene_emb.T)

            # Softmax with temperature -> weighted sum
            weights = F.softmax(sim / temperature, dim=1)
            batch_pred = torch.matmul(weights, gallery_expressions.float())

        all_pred_exp.append(batch_pred.float().cpu())
        all_true_exp.append(expressions.cpu())
        all_sample_names.extend(sample_names)

    all_pred_exp = torch.cat(all_pred_exp, dim=0)
    all_true_exp = torch.cat(all_true_exp, dim=0)
    metrics = compute_metrics(all_pred_exp, all_true_exp)

    return metrics, all_pred_exp, all_true_exp, all_sample_names


def main():
    parser = argparse.ArgumentParser(description='ST-ConMa Zero-Shot Gene Expression Prediction')
    parser.add_argument('--dataset', type=str, choices=['her2st', 'cscc', 'hlt'], default='her2st')
    parser.add_argument('--fold', type=int, default=0, help='Fold (patient) to use as test')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output_dir', type=str, default='./results/gep_pred/st_conma_zeroshot')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gene_type', type=str, choices=['heg', 'hvg'], default='heg')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for softmax')

    args = parser.parse_args()

    output_base = os.path.join(args.output_dir, args.gene_type, args.dataset)
    os.makedirs(output_base, exist_ok=True)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # Dataset config
    DATASET_CONFIG = {
        'her2st': {
            'patients': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            'n_folds': 8
        },
        'cscc': {
            'patients': ['P2', 'P5', 'P9', 'P10'],
            'n_folds': 4
        },
        'hlt': {
            'patients': ['A1', 'B1', 'C1', 'D1'],
            'n_folds': 4
        }
    }

    config = DATASET_CONFIG[args.dataset]
    n_folds = config['n_folds']
    test_patient = config['patients'][args.fold]
    unit_type = 'sample' if args.dataset == 'hlt' else 'patient'

    print(f"\n{'='*60}")
    print(f"ST-ConMa Zero-Shot Gene Expression Prediction")
    print(f"Dataset: {args.dataset}, Fold {args.fold}/{n_folds-1}")
    print(f"Test {unit_type}: {test_patient}")
    print(f"Gene type: {args.gene_type}")
    print(f"{'='*60}")

    if args.fold < 0 or args.fold >= n_folds:
        raise ValueError(f"fold must be 0-{n_folds-1}")

    # Load model (no fine-tuning) - Use STConMaFinetune with pretrained checkpoint
    print(f"\nLoading ST-ConMa model from: {args.checkpoint}")
    model = STConMaFinetune(
        pretrained_checkpoint=args.checkpoint,
        proj_dim=768,
        temperature=0.07,
        learnable_temperature=False,
        freeze_vision_encoder=True,  # Freeze for zero-shot
        freeze_gene_encoder=True,    # Freeze for zero-shot
        device=args.device
    )
    model = model.to(args.device)
    model.eval()
    print("Using pretrained checkpoint (zero-shot, no fine-tuning)")

    # Load datasets (using 'dataset' parameter, not 'dataset_name')
    train_dataset = PatientWiseCVDataset(
        dataset=args.dataset,
        fold=args.fold,
        train=True,
        use_augmentation=False,
        gene_type=args.gene_type
    )
    test_dataset = PatientWiseCVDataset(
        dataset=args.dataset,
        fold=args.fold,
        train=False,
        use_augmentation=False,
        gene_type=args.gene_type
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_patient_cv,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_patient_cv,
    )

    print(f"\nGallery (train) samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Build gallery from training data
    print("\nBuilding gallery...")
    gallery_gene_emb, gallery_expressions = build_gallery(model, train_loader, args.device)

    # Evaluate
    print("\n=== Zero-Shot Evaluation ===")
    metrics, pred_exp, true_exp, sample_names = evaluate(
        model, test_loader, gallery_gene_emb, gallery_expressions, args.device, args.temperature
    )

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
        print(f"\n=== Per-Sample Results ===")
        for sample, sm in per_sample_metrics.items():
            print(f"  {sample}: PCC={sm['gene_pcc']:.4f}, MSE={sm['total_mse']:.4f}")

    # Save results
    results = {
        'dataset': args.dataset,
        'fold': args.fold,
        f'test_{unit_type}': test_patient,
        'n_folds': n_folds,
        'gene_type': args.gene_type,
        'model': 'st_conma_zeroshot',
        'fine_tuned': False,
        'temperature': args.temperature,
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

    # Save predictions
    if args.dataset != 'hlt':
        sample_pred_dir = os.path.join(output_base, f'{unit_type}_{test_patient}_samples')
        os.makedirs(sample_pred_dir, exist_ok=True)
        for sample_name in per_sample_pred.keys():
            np.save(os.path.join(sample_pred_dir, f'{sample_name}_pred.npy'), per_sample_pred[sample_name])
            np.save(os.path.join(sample_pred_dir, f'{sample_name}_true.npy'), per_sample_true[sample_name])

    if isinstance(pred_exp, torch.Tensor):
        pred_exp = pred_exp.cpu().numpy()
    if isinstance(true_exp, torch.Tensor):
        true_exp = true_exp.cpu().numpy()

    np.save(os.path.join(output_base, f'{args.dataset}_{unit_type}_{test_patient}_pred.npy'), pred_exp)
    np.save(os.path.join(output_base, f'{args.dataset}_{unit_type}_{test_patient}_true.npy'), true_exp)

    print(f"\nResults saved to: {results_path}")
    print("="*60)


if __name__ == '__main__':
    main()
