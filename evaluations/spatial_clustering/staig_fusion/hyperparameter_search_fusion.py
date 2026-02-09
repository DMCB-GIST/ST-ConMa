#!/usr/bin/env python
"""
Hyperparameter Search for STAIG Fusion on DLPFC dataset.

Based on the STAIG paper methodology:
- tau (temperature): 0.1 to 50
- num_epochs: 50 to 1000
- num_layers (GCN layers): 1 or 2
- num_neigh (k neighbors): 3, 5, or 7

Evaluates average ARI and NMI across all 12 DLPFC samples.
"""

import argparse
import os
os.chdir('./evaluations/spatial_clustering/staig_fusion')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings
warnings.filterwarnings('ignore')
import random
import gc
import torch
from staig.adata_processing import LoadSingle10xAdata
import numpy as np
from staig.staig import STAIG
import scanpy as sc
from sklearn import metrics
import json
from datetime import datetime
from itertools import product
import pandas as pd

# =====================
# Configuration
# =====================
OUTPUT_DIR = './results/spatial_clustering/st_conma/dlpfc/staig_fusion_hardneg_hpsearch'
INPUT_DIR = './ft_dataset/spatial_clustering/'
EMB_DIR = './results/spatial_clustering/st_conma/dlpfc/fusion_embeddings_hardneg'
SEED = 42

# DLPFC samples
SAMPLES_7_LAYERS = ['151507', '151508', '151509', '151510', '151673', '151674', '151675', '151676']
SAMPLES_5_LAYERS = ['151669', '151670', '151671', '151672']
ALL_SAMPLES = SAMPLES_7_LAYERS + SAMPLES_5_LAYERS


HYPERPARAMS = {
    'tau': [10, 20, 30, 40, 50],
    'num_epochs': [300, 350, 400, 450, 500],
    'num_layers': [1],
    'num_neigh': [5, 7],
    'k': [20, 40, 80],
    'img_pca_dim': [16, 32],
}

# Default fixed hyperparameters
DEFAULT_CONFIG = {
    'seed': 42,
    'learning_rate': 0.0005,
    'num_hidden': 64,
    'num_proj_hidden': 64,
    'activation': 'prelu',
    'base_model': 'GCNConv',
    'drop_feature_rate_1': 0.1,
    'drop_feature_rate_2': 0.2,
    'weight_decay': 0.00001,
    'num_gene': 3000,
    'k': 80
}


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_single_sample(sample_id, config, edge_temperature=1.0, img_pca_dim=16):
    """Run STAIG for a single sample with given config."""
    set_seed(SEED)

    # Determine number of clusters based on sample
    if sample_id in SAMPLES_7_LAYERS:
        num_clusters = 7
    else:
        num_clusters = 5

    config['num_clusters'] = num_clusters

    slide_path = os.path.join(INPUT_DIR, 'DLPFC', sample_id)

    args = argparse.Namespace(
        dataset='DLPFC',
        slide=sample_id,
        config=None,
        label=True,
    )

    # Suppress scanpy output
    sc.settings.verbosity = 0

    try:
        data = LoadSingle10xAdata(
            path=slide_path,
            n_neighbors=config['num_neigh'],
            npy_path=EMB_DIR,
            sample_name=sample_id,
            n_top_genes=config['num_gene'],
            image_emb=True,
            label=True,
            edge_temperature=edge_temperature,
            img_pca_dim=img_pca_dim
        ).run()

        staig = STAIG(args=args, config=config, single=False, refine=False)
        staig.adata = data
        staig.train()
        staig.eva()
        staig.cluster(True)

        ari = staig.adata.uns.get('ari', None)
        nmi = staig.adata.uns.get('nmi', None)

        # Clean up to prevent memory leak
        del staig.model
        del staig
        del data
        clear_memory()

        return {'ari': ari, 'nmi': nmi, 'success': True}

    except Exception as e:
        print(f"    Error in {sample_id}: {e}")
        clear_memory()
        return {'ari': None, 'nmi': None, 'success': False}


def evaluate_hyperparams(tau, num_epochs, num_layers, num_neigh, k, img_pca_dim=16, edge_temperature=1.0):
    """Evaluate a hyperparameter combination across all samples."""
    config = DEFAULT_CONFIG.copy()
    config['tau'] = tau
    config['num_epochs'] = num_epochs
    config['num_layers'] = num_layers
    config['num_neigh'] = num_neigh
    config['k'] = k

    results = []
    for sample_id in ALL_SAMPLES:
        result = run_single_sample(sample_id, config, edge_temperature, img_pca_dim)
        result['sample_id'] = sample_id
        results.append(result)

    # Calculate averages
    ari_values = [r['ari'] for r in results if r['ari'] is not None]
    nmi_values = [r['nmi'] for r in results if r['nmi'] is not None]

    return {
        'tau': tau,
        'num_epochs': num_epochs,
        'num_layers': num_layers,
        'num_neigh': num_neigh,
        'k': k,
        'img_pca_dim': img_pca_dim,
        'mean_ari': np.mean(ari_values) if ari_values else None,
        'std_ari': np.std(ari_values) if ari_values else None,
        'mean_nmi': np.mean(nmi_values) if nmi_values else None,
        'std_nmi': np.std(nmi_values) if nmi_values else None,
        'num_success': len(ari_values),
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter Search for STAIG Fusion')

    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for results')

    # Search mode
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'quick', 'custom'],
                        help='Search mode: full (all combinations), quick (reduced grid), custom')

    # Custom search space (comma-separated values)
    parser.add_argument('--tau_values', type=str, default=None,
                        help='Custom tau values (comma-separated)')
    parser.add_argument('--epoch_values', type=str, default=None,
                        help='Custom epoch values (comma-separated)')
    parser.add_argument('--layer_values', type=str, default=None,
                        help='Custom num_layers values (comma-separated)')
    parser.add_argument('--neigh_values', type=str, default=None,
                        help='Custom num_neigh values (comma-separated)')
    parser.add_argument('--k_values', type=str, default=None,
                        help='Custom k values for pseudo-label KMeans (comma-separated)')
    parser.add_argument('--img_pca_dim_values', type=str, default=None,
                        help='Custom img_pca_dim values (comma-separated, e.g., 16,32)')

    parser.add_argument('--edge_temperature', type=float, default=1.0,
                        help='Edge probability temperature')

    # Resume from checkpoint
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from previous results JSON')

    return parser.parse_args()


def get_search_space(args):
    """Get hyperparameter search space based on mode."""
    if args.mode == 'quick':
        return {
            'tau': [5, 10, 20, 35],
            'num_epochs': [200, 300, 400, 500],
            'num_layers': [1],
            'num_neigh': [5],
            'k': [80],
            'img_pca_dim': [16],
        }
    elif args.mode == 'custom':
        space = {}
        if args.tau_values:
            space['tau'] = [float(x) for x in args.tau_values.split(',')]
        else:
            space['tau'] = [10, 35]
        if args.epoch_values:
            space['num_epochs'] = [int(x) for x in args.epoch_values.split(',')]
        else:
            space['num_epochs'] = [300, 400]
        if args.layer_values:
            space['num_layers'] = [int(x) for x in args.layer_values.split(',')]
        else:
            space['num_layers'] = [1]
        if args.neigh_values:
            space['num_neigh'] = [int(x) for x in args.neigh_values.split(',')]
        else:
            space['num_neigh'] = [5]
        if args.k_values:
            space['k'] = [int(x) for x in args.k_values.split(',')]
        else:
            space['k'] = [80]
        if args.img_pca_dim_values:
            space['img_pca_dim'] = [int(x) for x in args.img_pca_dim_values.split(',')]
        else:
            space['img_pca_dim'] = [16]
        return space
    else:  # full
        return HYPERPARAMS


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get search space
    search_space = get_search_space(args)

    # Generate all combinations
    combinations = list(product(
        search_space['tau'],
        search_space['num_epochs'],
        search_space['num_layers'],
        search_space['num_neigh'],
        search_space['k'],
        search_space['img_pca_dim']
    ))

    total_combinations = len(combinations)

    print("=" * 80)
    print("STAIG Fusion Hyperparameter Search")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Edge Temperature: {args.edge_temperature}")
    print(f"\nSearch Space:")
    print(f"  tau: {search_space['tau']}")
    print(f"  num_epochs: {search_space['num_epochs']}")
    print(f"  num_layers: {search_space['num_layers']}")
    print(f"  num_neigh: {search_space['num_neigh']}")
    print(f"  k (KMeans clusters): {search_space['k']}")
    print(f"  img_pca_dim: {search_space['img_pca_dim']}")
    print(f"\nTotal combinations: {total_combinations}")
    print(f"Total runs: {total_combinations * len(ALL_SAMPLES)}")
    print("=" * 80)

    # Load previous results if resuming
    completed_combinations = set()
    all_results = []

    if args.resume and os.path.exists(args.resume):
        with open(args.resume, 'r') as f:
            prev_data = json.load(f)
            all_results = prev_data.get('results', [])
            for r in all_results:
                key = (r['tau'], r['num_epochs'], r['num_layers'], r['num_neigh'], r['k'], r.get('img_pca_dim', 16))
                completed_combinations.add(key)
            print(f"Resumed from {args.resume}: {len(completed_combinations)} combinations completed")

    # Run search
    best_result = None
    best_ari = -1

    start_time = datetime.now()

    for i, (tau, num_epochs, num_layers, num_neigh, k, img_pca_dim) in enumerate(combinations):
        key = (tau, num_epochs, num_layers, num_neigh, k, img_pca_dim)

        if key in completed_combinations:
            print(f"[{i+1}/{total_combinations}] Skipping (already completed): tau={tau}, epochs={num_epochs}, layers={num_layers}, neigh={num_neigh}, k={k}, img_pca_dim={img_pca_dim}")
            continue

        print(f"\n[{i+1}/{total_combinations}] Testing: tau={tau}, epochs={num_epochs}, layers={num_layers}, neigh={num_neigh}, k={k}, img_pca_dim={img_pca_dim}")

        result = evaluate_hyperparams(tau, num_epochs, num_layers, num_neigh, k, img_pca_dim, args.edge_temperature)
        all_results.append(result)

        if result['mean_ari'] is not None:
            print(f"  -> Mean ARI: {result['mean_ari']:.4f} +/- {result['std_ari']:.4f}")
            print(f"  -> Mean NMI: {result['mean_nmi']:.4f} +/- {result['std_nmi']:.4f}")

            if result['mean_ari'] > best_ari:
                best_ari = result['mean_ari']
                best_result = result
                print(f"  ** New best!")

        # Save intermediate results
        intermediate_path = os.path.join(args.output_dir, 'search_progress.json')
        with open(intermediate_path, 'w') as f:
            json.dump({
                'search_space': search_space,
                'completed': i + 1,
                'total': total_combinations,
                'best_result': best_result,
                'results': all_results
            }, f, indent=2)

    end_time = datetime.now()
    elapsed = end_time - start_time

    # Final summary
    print("\n" + "=" * 80)
    print("SEARCH COMPLETED")
    print("=" * 80)
    print(f"Total time: {elapsed}")
    print(f"Combinations tested: {len(all_results)}")

    if best_result:
        print(f"\nBest Hyperparameters:")
        print(f"  tau: {best_result['tau']}")
        print(f"  num_epochs: {best_result['num_epochs']}")
        print(f"  num_layers: {best_result['num_layers']}")
        print(f"  num_neigh: {best_result['num_neigh']}")
        print(f"  k: {best_result['k']}")
        print(f"  img_pca_dim: {best_result['img_pca_dim']}")
        print(f"\nBest Performance:")
        print(f"  Mean ARI: {best_result['mean_ari']:.4f} +/- {best_result['std_ari']:.4f}")
        print(f"  Mean NMI: {best_result['mean_nmi']:.4f} +/- {best_result['std_nmi']:.4f}")

    # Save final results
    final_results = {
        'search_space': search_space,
        'edge_temperature': args.edge_temperature,
        'total_combinations': total_combinations,
        'elapsed_time': str(elapsed),
        'best_hyperparameters': {
            'tau': best_result['tau'] if best_result else None,
            'num_epochs': best_result['num_epochs'] if best_result else None,
            'num_layers': best_result['num_layers'] if best_result else None,
            'num_neigh': best_result['num_neigh'] if best_result else None,
            'k': best_result['k'] if best_result else None,
            'img_pca_dim': best_result['img_pca_dim'] if best_result else None,
        },
        'best_performance': {
            'mean_ari': best_result['mean_ari'] if best_result else None,
            'std_ari': best_result['std_ari'] if best_result else None,
            'mean_nmi': best_result['mean_nmi'] if best_result else None,
            'std_nmi': best_result['std_nmi'] if best_result else None,
        },
        'results': all_results
    }

    final_path = os.path.join(args.output_dir, 'hyperparameter_search_results.json')
    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to: {final_path}")

    # Create summary DataFrame
    summary_data = []
    for r in all_results:
        summary_data.append({
            'tau': r['tau'],
            'num_epochs': r['num_epochs'],
            'num_layers': r['num_layers'],
            'num_neigh': r['num_neigh'],
            'k': r['k'],
            'img_pca_dim': r['img_pca_dim'],
            'mean_ari': r['mean_ari'],
            'std_ari': r['std_ari'],
            'mean_nmi': r['mean_nmi'],
            'std_nmi': r['std_nmi'],
        })

    df = pd.DataFrame(summary_data)
    df = df.sort_values('mean_ari', ascending=False)
    csv_path = os.path.join(args.output_dir, 'hyperparameter_search_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to: {csv_path}")

    print("\nTop 10 configurations by ARI:")
    print(df.head(10).to_string(index=False))

    print("\nDone!")


if __name__ == '__main__':
    main()
