import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add baselines path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'staig_fusion'))

import random
import json
import gc
import torch
import numpy as np
import pandas as pd
import scanpy as sc

from staig.adata_processing import LoadSingle10xAdata
from staig.staig import STAIG
from eval_utils import clustering, calculate_metrics, draw_spatial, draw_umap, draw_paga, draw_density


# =====================
# Configuration
# =====================
INPUT_DIR = './ft_dataset/spatial_clustering/'

# Fixed seed for reproducibility
SEED = 42

SAMPLES_7_LAYERS = ['151507', '151508', '151509', '151510', '151673', '151674', '151675', '151676']
SAMPLES_5_LAYERS = ['151669', '151670', '151671', '151672']
ALL_DLPFC_SAMPLES = SAMPLES_7_LAYERS + SAMPLES_5_LAYERS

# Dataset configurations
DATASET_CONFIGS = {
    'dlpfc': {
        'output_dir': './results/spatial_clustering/st_conma/dlpfc',
        'data_dir': os.path.join(INPUT_DIR, 'DLPFC'),
        'emb_dir': './results/spatial_clustering/st_conma/dlpfc/fusion_embeddings',
        'samples': ALL_DLPFC_SAMPLES,
    },
    'human_breast_cancer': {
        'output_dir': './results/spatial_clustering/st_conma/human_breast_cancer',
        'data_dir': os.path.join(INPUT_DIR, 'human_breast_cancer'),
        'emb_dir': './results/spatial_clustering/st_conma/human_breast_cancer/fusion_embeddings',
        'samples': ['human_breast_cancer'],
        'n_clusters': 20,
    }
}

# Config for 5-layer samples
CONFIG_5LAYERS = {
    'learning_rate': 0.0005,
    'num_hidden': 64,
    'num_proj_hidden': 64,
    'activation': 'prelu',
    'base_model': 'GCNConv',
    'num_layers': 1,
    'drop_feature_rate_1': 0.1,
    'drop_feature_rate_2': 0.2,
    'tau': 30,
    'num_epochs': 450,
    'weight_decay': 0.00001,
    'num_clusters': 5,
    'num_gene': 3000,
    'num_neigh': 7,
    'k': 80,
    'img_pca_dim': 32
}

# Config for 7-layer samples
CONFIG_7LAYERS = {
    'learning_rate': 0.0005,
    'num_hidden': 64,
    'num_proj_hidden': 64,
    'activation': 'prelu',
    'base_model': 'GCNConv',
    'num_layers': 1,
    'drop_feature_rate_1': 0.1,
    'drop_feature_rate_2': 0.2,
    'tau': 30,
    'num_epochs': 450,
    'weight_decay': 0.00001,
    'num_clusters': 7,
    'num_gene': 3000,
    'num_neigh': 7,
    'k': 80,
    'img_pca_dim': 32
}

# Config for Human Breast Cancer (20 clusters)
CONFIG_HUMAN_BREAST_CANCER = {
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
    'tau': 40,
    'tau_decay': 0.00,
    'num_epochs': 500,
    'weight_decay': 0.00001,
    'num_clusters': 20,
    'num_gene': 3000,
    'num_neigh': 5,
    'num_im_neigh': 2,
    'k': 20,
    'img_pca_dim': 16
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


def get_config(sample_id, dataset='dlpfc'):
    """Get config based on sample ID and dataset."""
    if dataset == 'human_breast_cancer':
        config = CONFIG_HUMAN_BREAST_CANCER.copy()
    elif sample_id in SAMPLES_7_LAYERS:
        config = CONFIG_7LAYERS.copy()
    elif sample_id in SAMPLES_5_LAYERS:
        config = CONFIG_5LAYERS.copy()
    else:
        raise ValueError(f"Unknown sample ID: {sample_id}")
    config['seed'] = SEED
    return config


def run_single_sample(sample_id, output_dir, dataset='dlpfc'):
    print(f'\n{"="*60}')
    print(f'Processing Sample: {sample_id}')
    print(f'{"="*60}')

    # Set seed
    set_seed(SEED)

    # Get config
    config = get_config(sample_id, dataset=dataset)
    num_clusters = config['num_clusters']
    print(f"Using config: num_clusters={num_clusters}")

    sc.settings.verbosity = 0

    # Get dataset-specific paths
    dataset_config = DATASET_CONFIGS[dataset]
    emb_dir = dataset_config['emb_dir']

    try:
        # Set paths based on dataset
        if dataset == 'human_breast_cancer':
            slide_path = dataset_config['data_dir']
            dataset_name = 'human_breast_cancer'
        else:
            slide_path = os.path.join(INPUT_DIR, 'DLPFC', sample_id)
            dataset_name = 'dlpfc'

        # Create args for STAIG
        args = argparse.Namespace(
            dataset=dataset_name,
            slide=sample_id,
            config=None,
            label=True,
        )

        # Load data with text embeddings
        data = LoadSingle10xAdata(
            path=slide_path,
            n_top_genes=config['num_gene'],
            n_neighbors=config['num_neigh'],
            image_emb=True,
            label=True,
            npy_path=emb_dir,
            sample_name=sample_id,
            img_pca_dim=config.get('img_pca_dim', 32),
        ).run()

        # Train STAIG model
        staig = STAIG(args=args, config=config, single=False, refine=False)
        staig.adata = data
        staig.train()
        staig.eva()

        # Use eval_utils for clustering and visualization
        clustering(staig.adata, n_clusters=num_clusters, method='mclust')

        # Filter NA for metric calculation
        adata_filtered = staig.adata[~pd.isnull(staig.adata.obs['ground_truth'])].copy()

        # Calculate metrics using eval_utils
        metrics_dict = calculate_metrics(adata_filtered, pred_key='domain', label_key='ground_truth')

        # Save visualizations
        draw_spatial(adata_filtered, output_dir, sample_id, pred_key='domain')
        draw_umap(adata_filtered, output_dir, sample_id, pred_key='domain')
        draw_paga(adata_filtered, output_dir, sample_id, pred_key='domain')
        draw_density(adata_filtered, output_dir, sample_id)

        # Save adata
        adata_path = os.path.join(output_dir, f'{sample_id}_adata.h5ad')
        adata_filtered.write(adata_path)
        print(f"Saved: {adata_path}")

        # Get metrics
        ari = metrics_dict['ari']
        nmi = metrics_dict['nmi']

        # Save individual results
        results = {
            'sample': sample_id,
            'seed': SEED,
            'ari': ari,
            'nmi': nmi,
            'num_clusters': num_clusters
        }

        metrics_path = os.path.join(output_dir, f'{sample_id}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}")
        print(f"Saved: {metrics_path}")

        return results

    except Exception as e:
        print(f"    Error in {sample_id}: {e}")
        import traceback
        traceback.print_exc()
        return {'sample': sample_id, 'seed': SEED, 'ari': None, 'nmi': None,
                'num_clusters': num_clusters, 'error': str(e)}

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def calculate_summary(output_dir, dataset='dlpfc'):
    """Calculate summary statistics from all JSON files."""
    all_results = []

    # Get samples list based on dataset
    if dataset == 'human_breast_cancer':
        samples = ['human_breast_cancer']
    else:
        samples = ALL_DLPFC_SAMPLES

    # Collect all results
    for sample_id in samples:
        metrics_path = os.path.join(output_dir, f'{sample_id}_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                results = json.load(f)
                all_results.append(results)

    if not all_results:
        print("No results found!")
        return

    # Calculate per-sample statistics
    sample_stats = {}
    for r in all_results:
        if r['ari'] is not None:
            sample_stats[r['sample']] = {
                'ari': r['ari'],
                'nmi': r['nmi']
            }

    # Calculate overall statistics
    all_ari = [r['ari'] for r in all_results if r['ari'] is not None]
    all_nmi = [r['nmi'] for r in all_results if r['nmi'] is not None]

    summary = {
        'seed': SEED,
        'dataset': dataset,
        'total_samples': len(all_results),
        'overall': {
            'mean_ari': np.mean(all_ari) if all_ari else None,
            'mean_nmi': np.mean(all_nmi) if all_nmi else None,
        },
        'per_sample': sample_stats,
        'individual_results': all_results
    }

    # Add layer-specific stats for DLPFC
    if dataset == 'dlpfc':
        ari_7layers = [sample_stats[s]['ari'] for s in SAMPLES_7_LAYERS if s in sample_stats]
        nmi_7layers = [sample_stats[s]['nmi'] for s in SAMPLES_7_LAYERS if s in sample_stats]
        ari_5layers = [sample_stats[s]['ari'] for s in SAMPLES_5_LAYERS if s in sample_stats]
        nmi_5layers = [sample_stats[s]['nmi'] for s in SAMPLES_5_LAYERS if s in sample_stats]

        summary['7_layers'] = {
            'samples': len(ari_7layers),
            'mean_ari': np.mean(ari_7layers) if ari_7layers else None,
            'mean_nmi': np.mean(nmi_7layers) if nmi_7layers else None,
        }
        summary['5_layers'] = {
            'samples': len(ari_5layers),
            'mean_ari': np.mean(ari_5layers) if ari_5layers else None,
            'mean_nmi': np.mean(nmi_5layers) if nmi_5layers else None,
        }

    # Save summary
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nSummary saved: {summary_path}")
    if summary['overall']['mean_ari'] is not None:
        print(f"Overall ARI: {summary['overall']['mean_ari']:.4f}")
        print(f"Overall NMI: {summary['overall']['mean_nmi']:.4f}")

    # Print per-sample results
    print("\nPer-sample results:")
    for sample_id in samples:
        if sample_id in sample_stats:
            s = sample_stats[sample_id]
            print(f"  {sample_id}: ARI={s['ari']:.4f}, NMI={s['nmi']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Run ST-ConMa on spatial transcriptomics samples')
    parser.add_argument('--dataset', type=str, default='dlpfc',
                        choices=['dlpfc', 'human_breast_cancer'],
                        help='Dataset to use (dlpfc or human_breast_cancer)')
    parser.add_argument('--sample_id', type=str, default=None,
                        help='Sample ID to process')
    parser.add_argument('--all', action='store_true',
                        help='Process all samples in the dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-detected if not specified)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device')

    args = parser.parse_args()

    # Get dataset config
    config = DATASET_CONFIGS[args.dataset]
    output_dir = args.output_dir or config['output_dir']
    samples = config['samples']
    emb_dir = config['emb_dir']

    # Set CUDA device
    device_id = args.device.replace('cuda:', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("ST-ConMa Training (Text Embeddings)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {SEED}")
    print(f"Embedding: {emb_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)

    if args.all or args.dataset == 'human_breast_cancer':
        # Process all samples in the dataset
        for sample_id in samples:
            try:
                run_single_sample(sample_id, output_dir, dataset=args.dataset)
            except Exception as e:
                print(f"Error processing {sample_id}: {e}")
                continue

        # Calculate and save summary
        calculate_summary(output_dir, dataset=args.dataset)
    else:
        # Process single sample
        sample_id = args.sample_id or samples[0]
        try:
            run_single_sample(sample_id, output_dir, dataset=args.dataset)
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
