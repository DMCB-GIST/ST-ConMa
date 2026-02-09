#!/usr/bin/env python
"""
Hyperparameter Search for STAIG Fusion on Human Breast Cancer dataset.

Based on the STAIG paper methodology with fixed parameters from HBC config:
- num_layers: 2 (fixed)
- drop_edge_rate_1: 0.1 (fixed)
- drop_edge_rate_2: 0.3 (fixed)
- tau: 3.0 (fixed, can be tuned)
- tau_decay: 0.00 (fixed)
- num_epochs: 580 (fixed, can be tuned)
- num_im_neigh: 2 (fixed)

Tuning parameters:
- tau: temperature
- num_epochs: training epochs
- num_neigh: k neighbors
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
OUTPUT_DIR = './results/spatial_clustering/st_conma/human_breast_cancer/staig_fusion_hardneg_hpsearch'
INPUT_DIR = './ft_dataset/spatial_clustering/'
EMB_DIR = './results/spatial_clustering/st_conma/human_breast_cancer/fusion_embeddings_hardneg'
SEED = 42

# Human Breast Cancer sample
SAMPLE_ID = 'human_breast_cancer'
NUM_CLUSTERS = 20


# Hyperparameters to search
HYPERPARAMS = {
    'tau': [10, 20, 30, 40, 50],
    'num_epochs': [300, 350, 400, 450, 500],
    'num_layers': [2],
    'num_neigh': [5, 7],
    'k': [20, 40, 80],
    'img_pca_dim': [16, 32],
}

# Fixed hyperparameters from HBC config (user specified)
DEFAULT_CONFIG = {
    'seed': 42,
    'learning_rate': 0.0005,
    'num_hidden': 64,
    'num_proj_hidden': 64,
    'activation': 'prelu',
    'base_model': 'GCNConv',
    'num_layers': 2, 
    'drop_edge_rate_1': 0.1,  
    'drop_edge_rate_2': 0.3,
    'drop_feature_rate_1': 0.1,
    'drop_feature_rate_2': 0.2,
    'tau_decay': 0.00,  
    'weight_decay': 0.00001,
    'num_gene': 3000,
    'num_im_neigh': 2,  
    'num_clusters': NUM_CLUSTERS,
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


def load_human_breast_cancer_data():
    """Load Human Breast Cancer ground truth."""
    data_dir = os.path.join(INPUT_DIR, 'human_breast_cancer')
    truth_path = os.path.join(data_dir, 'truth.txt')

    if os.path.exists(truth_path):
        truth = pd.read_csv(truth_path, sep='\t', header=None, names=['barcode', 'label'])
        truth = truth.set_index('barcode')
        return truth
    return None


def run_single_sample(config, edge_temperature=1.0, img_pca_dim=16):
    """Run STAIG for Human Breast Cancer with given config."""
    set_seed(SEED)

    slide_path = os.path.join(INPUT_DIR, 'human_breast_cancer')

    args = argparse.Namespace(
        dataset='human_breast_cancer',
        slide=SAMPLE_ID,
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
            sample_name=SAMPLE_ID,
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
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        clear_memory()
        return {'ari': None, 'nmi': None, 'success': False}


def evaluate_hyperparams(tau, num_epochs, num_neigh, k, img_pca_dim=16, edge_temperature=1.0):
    """Evaluate a hyperparameter combination."""
    config = DEFAULT_CONFIG.copy()
    config['tau'] = tau
    config['num_epochs'] = num_epochs
    config['num_neigh'] = num_neigh
    config['k'] = k

    result = run_single_sample(config, edge_temperature, img_pca_dim)

    return {
        'tau': tau,
        'num_epochs': num_epochs,
        'num_neigh': num_neigh,
        'k': k,
        'img_pca_dim': img_pca_dim,
        'ari': result['ari'],
        'nmi': result['nmi'],
        'success': result['success']
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter Search for STAIG Fusion on Human Breast Cancer')

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
            'tau': [2.0, 3.0, 5.0],
            'num_epochs': [400, 580],
            'num_neigh': [5],
            'k': [80],
            'img_pca_dim': [16],
        }
    elif args.mode == 'custom':
        space = {}
        if args.tau_values:
            space['tau'] = [float(x) for x in args.tau_values.split(',')]
        else:
            space['tau'] = [3.0, 5.0]
        if args.epoch_values:
            space['num_epochs'] = [int(x) for x in args.epoch_values.split(',')]
        else:
            space['num_epochs'] = [580]
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
        search_space['num_neigh'],
        search_space['k'],
        search_space['img_pca_dim']
    ))

    total_combinations = len(combinations)

    print("=" * 80)
    print("STAIG Fusion Hyperparameter Search - Human Breast Cancer")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Edge Temperature: {args.edge_temperature}")
    print(f"\nFixed Parameters (from HBC config):")
    print(f"  num_layers: {DEFAULT_CONFIG['num_layers']}")
    print(f"  drop_edge_rate_1: {DEFAULT_CONFIG['drop_edge_rate_1']}")
    print(f"  drop_edge_rate_2: {DEFAULT_CONFIG['drop_edge_rate_2']}")
    print(f"  tau_decay: {DEFAULT_CONFIG['tau_decay']}")
    print(f"  num_im_neigh: {DEFAULT_CONFIG['num_im_neigh']}")
    print(f"  num_clusters: {DEFAULT_CONFIG['num_clusters']}")
    print(f"\nSearch Space:")
    print(f"  tau: {search_space['tau']}")
    print(f"  num_epochs: {search_space['num_epochs']}")
    print(f"  num_neigh: {search_space['num_neigh']}")
    print(f"  k (KMeans clusters): {search_space['k']}")
    print(f"  img_pca_dim: {search_space['img_pca_dim']}")
    print(f"\nTotal combinations: {total_combinations}")
    print("=" * 80)

    # Load previous results if resuming
    completed_combinations = set()
    all_results = []

    if args.resume and os.path.exists(args.resume):
        with open(args.resume, 'r') as f:
            prev_data = json.load(f)
            all_results = prev_data.get('results', [])
            for r in all_results:
                key = (r['tau'], r['num_epochs'], r['num_neigh'], r['k'], r.get('img_pca_dim', 16))
                completed_combinations.add(key)
            print(f"Resumed from {args.resume}: {len(completed_combinations)} combinations completed")

    # Run search
    best_result = None
    best_ari = -1

    start_time = datetime.now()

    for i, (tau, num_epochs, num_neigh, k, img_pca_dim) in enumerate(combinations):
        key = (tau, num_epochs, num_neigh, k, img_pca_dim)

        if key in completed_combinations:
            print(f"[{i+1}/{total_combinations}] Skipping (already completed): tau={tau}, epochs={num_epochs}, neigh={num_neigh}, k={k}, img_pca_dim={img_pca_dim}")
            continue

        print(f"\n[{i+1}/{total_combinations}] Testing: tau={tau}, epochs={num_epochs}, neigh={num_neigh}, k={k}, img_pca_dim={img_pca_dim}")

        result = evaluate_hyperparams(tau, num_epochs, num_neigh, k, img_pca_dim, args.edge_temperature)
        all_results.append(result)

        if result['ari'] is not None:
            print(f"  -> ARI: {result['ari']:.4f}")
            print(f"  -> NMI: {result['nmi']:.4f}")

            if result['ari'] > best_ari:
                best_ari = result['ari']
                best_result = result
                print(f"  ** New best!")

        # Save intermediate results
        intermediate_path = os.path.join(args.output_dir, 'search_progress.json')
        with open(intermediate_path, 'w') as f:
            json.dump({
                'search_space': search_space,
                'fixed_config': {
                    'num_layers': DEFAULT_CONFIG['num_layers'],
                    'drop_edge_rate_1': DEFAULT_CONFIG['drop_edge_rate_1'],
                    'drop_edge_rate_2': DEFAULT_CONFIG['drop_edge_rate_2'],
                    'tau_decay': DEFAULT_CONFIG['tau_decay'],
                    'num_im_neigh': DEFAULT_CONFIG['num_im_neigh'],
                },
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
        print(f"  num_neigh: {best_result['num_neigh']}")
        print(f"  k: {best_result['k']}")
        print(f"  img_pca_dim: {best_result['img_pca_dim']}")
        print(f"\nBest Performance:")
        print(f"  ARI: {best_result['ari']:.4f}")
        print(f"  NMI: {best_result['nmi']:.4f}")

    # Save final results
    final_results = {
        'dataset': 'human_breast_cancer',
        'search_space': search_space,
        'fixed_config': {
            'num_layers': DEFAULT_CONFIG['num_layers'],
            'drop_edge_rate_1': DEFAULT_CONFIG['drop_edge_rate_1'],
            'drop_edge_rate_2': DEFAULT_CONFIG['drop_edge_rate_2'],
            'tau_decay': DEFAULT_CONFIG['tau_decay'],
            'num_im_neigh': DEFAULT_CONFIG['num_im_neigh'],
            'num_clusters': DEFAULT_CONFIG['num_clusters'],
        },
        'edge_temperature': args.edge_temperature,
        'total_combinations': total_combinations,
        'elapsed_time': str(elapsed),
        'best_hyperparameters': {
            'tau': best_result['tau'] if best_result else None,
            'num_epochs': best_result['num_epochs'] if best_result else None,
            'num_neigh': best_result['num_neigh'] if best_result else None,
            'k': best_result['k'] if best_result else None,
            'img_pca_dim': best_result['img_pca_dim'] if best_result else None,
        },
        'best_performance': {
            'ari': best_result['ari'] if best_result else None,
            'nmi': best_result['nmi'] if best_result else None,
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
            'num_neigh': r['num_neigh'],
            'k': r['k'],
            'img_pca_dim': r['img_pca_dim'],
            'ari': r['ari'],
            'nmi': r['nmi'],
        })

    df = pd.DataFrame(summary_data)
    df = df.sort_values('ari', ascending=False)
    csv_path = os.path.join(args.output_dir, 'hyperparameter_search_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to: {csv_path}")

    print("\nTop 10 configurations by ARI:")
    print(df.head(10).to_string(index=False))

    print("\nDone!")


if __name__ == '__main__':
    main()
