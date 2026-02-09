#!/usr/bin/env python
"""
Visualize Embeddings on Spatial Plots

This script loads embeddings and visualizes them using:
1. KMeans clustering on spatial plots
2. UMAP dimensionality reduction
3. PCA visualization

Usage:
    python embedding_viz.py --sample_id 151673 --n_clusters 7
    python embedding_viz.py --all --n_clusters 7
    python embedding_viz.py --dataset human_breast_cancer --n_clusters 20
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

warnings.filterwarnings('ignore')

# Add STAIG to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'staig_fusion'))

from staig.adata_processing import LoadSingle10xAdata

# DLPFC sample IDs
DLPFC_SAMPLES = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]

# DLPFC samples by layer count (for metric calculation)
SAMPLES_7_LAYERS = ['151507', '151508', '151509', '151510', '151673', '151674', '151675', '151676']
SAMPLES_5_LAYERS = ['151669', '151670', '151671', '151672']


def get_dlpfc_n_clusters(sample_id: str) -> int:
    """Get the number of clusters for DLPFC sample based on layer count."""
    if sample_id in SAMPLES_7_LAYERS:
        return 7
    elif sample_id in SAMPLES_5_LAYERS:
        return 5
    else:
        return 7  # default

# Dataset configurations
DATASET_CONFIGS = {
    'dlpfc': {
        'data_dir': './ft_dataset/spatial_clustering/DLPFC',
        'samples': DLPFC_SAMPLES,
        'default_n_clusters': 7,
    },
    'human_breast_cancer': {
        'data_dir': './ft_dataset/spatial_clustering/human_breast_cancer',
        'samples': ['human_breast_cancer'],
        'default_n_clusters': 20,
    }
}

# Color palette (same as Fig-3b)
COLOR_MAP = {
    '0': '#E06D71', '1': '#fdfba4', '2': '#686da7', '3': '#95c14d',
    '4': '#1B69C3', '5': '#f9dfaf', '6': '#ae68b0', '7': '#f68402',
    '8': '#75e8fd', '9': '#162e62', '10': '#A0ad30', '11': '#ba9fca',
    '12': '#C00000', '13': '#f0f040', '14': '#619870', '15': '#fbdb24',
    '16': '#74A6E2', '17': '#E19B67', '18': '#E0E0BB', '19': '#284405'
}


def plot_pca_variance(pca: PCA, output_dir: str, sample_id: str, threshold: float = 0.95):
    """Plot PCA variance plot and mark the point where cumulative variance reaches threshold (e.g., 95%)."""
    n_components = len(pca.explained_variance_ratio_)
    x = np.arange(1, n_components + 1)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the point where cumulative variance reaches threshold
    threshold_point = np.argmax(cumulative_variance >= threshold) + 1
    if cumulative_variance[-1] < threshold:
        threshold_point = None  # Threshold not reached

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Individual explained variance
    ax1 = axes[0]
    ax1.bar(x, pca.explained_variance_ratio_, alpha=0.7, color='steelblue', label='Individual')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title(f'{sample_id} - PCA Explained Variance (Individual)')
    ax1.set_xticks(x[::5])  # Show every 5th tick

    # Plot 2: Cumulative explained variance with threshold point
    ax2 = axes[1]
    ax2.plot(x, cumulative_variance, 'b-o', markersize=3, label='Cumulative')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title(f'{sample_id} - PCA Cumulative Variance')
    ax2.set_xticks(x[::5])
    ax2.grid(True, alpha=0.3)

    # Draw threshold line
    ax2.axhline(y=threshold, color='green', linestyle=':', linewidth=1.5, label=f'{threshold:.0%} threshold')

    # Mark threshold point
    if threshold_point is not None:
        threshold_variance = cumulative_variance[threshold_point - 1]
        ax2.axvline(x=threshold_point, color='red', linestyle='--', linewidth=2, label=f'{threshold:.0%} at PC {threshold_point}')
        ax2.scatter([threshold_point], [threshold_variance], color='red', s=100, zorder=5)
        ax2.annotate(
            f'PC {threshold_point}\n({threshold_variance:.2%} var)',
            xy=(threshold_point, threshold_variance),
            xytext=(threshold_point + 3, threshold_variance - 0.08),
            fontsize=10, color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
        )
    ax2.legend(loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{sample_id}_pca_variance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')

    # Save variance info to JSON
    variance_info = {
        'sample_id': sample_id,
        'threshold': threshold,
        'threshold_point': int(threshold_point) if threshold_point is not None else None,
        'threshold_cumulative_variance': float(cumulative_variance[threshold_point - 1]) if threshold_point is not None else None,
        'total_components': n_components,
        'variance_at_50_components': float(cumulative_variance[min(49, n_components - 1)]),
    }

    variance_json_path = os.path.join(output_dir, f'{sample_id}_pca_variance_info.json')
    with open(variance_json_path, 'w') as f:
        json.dump(variance_info, f, indent=4)
    print(f'Saved: {variance_json_path}')

    return threshold_point


def load_embeddings(emb_dir: str, sample_id: str) -> np.ndarray:
    # Try different naming conventions
    patterns = [
        os.path.join(emb_dir, f'{sample_id}_embeddings.npy'),
        os.path.join(emb_dir, f'{sample_id}_fusion_embeddings.npy'),
        os.path.join(emb_dir, sample_id, 'embeddings.npy'),
        os.path.join(emb_dir, 'embeddings.npy'),  # For single-sample datasets
    ]

    for emb_path in patterns:
        if os.path.exists(emb_path):
            return np.load(emb_path)

    raise FileNotFoundError(f'Embeddings not found for {sample_id} in {emb_dir}')


def load_human_breast_cancer_adata(data_dir: str) -> sc.AnnData:
    """Load Human Breast Cancer dataset in 10x Genomics format."""
    # Read 10x h5 file
    h5_path = os.path.join(data_dir, 'filtered_feature_bc_matrix.h5')
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    # Load spatial coordinates
    spatial_dir = os.path.join(data_dir, 'spatial')
    positions_path = os.path.join(spatial_dir, 'tissue_positions_list.csv')

    positions = pd.read_csv(positions_path, header=None,
                            names=['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col'])
    positions = positions.set_index('barcode')

    # Filter to match adata barcodes
    common_barcodes = adata.obs_names.intersection(positions.index)
    adata = adata[common_barcodes].copy()
    positions = positions.loc[common_barcodes]

    # Add spatial coordinates
    adata.obsm['spatial'] = positions[['pxl_col', 'pxl_row']].values

    # Load ground truth
    truth_path = os.path.join(data_dir, 'truth.txt')
    if os.path.exists(truth_path):
        truth = pd.read_csv(truth_path, sep='\t', header=None, names=['barcode', 'label'])
        truth = truth.set_index('barcode')
        common = adata.obs_names.intersection(truth.index)
        adata = adata[common].copy()
        adata.obs['ground_truth'] = truth.loc[common, 'label'].values

    return adata


def visualize_embeddings(
    data_dir: str,
    emb_dir: str,
    sample_id: str,
    n_clusters: int = 7,
    output_dir: str = None,
    show_ground_truth: bool = True,
    dataset: str = 'dlpfc'
):
    print(f'\n{"="*60}')
    print(f'Visualizing Embeddings: {sample_id}')
    print(f'{"="*60}')

    # Load data based on dataset type
    if dataset == 'human_breast_cancer':
        adata = load_human_breast_cancer_adata(data_dir)
    else:
        # DLPFC dataset
        sample_path = os.path.join(data_dir, sample_id)
        loader = LoadSingle10xAdata(
            path=sample_path,
            image_emb=False,
            label=True,
            filter_na=True
        )
        loader.load_data()
        loader.load_label()
        adata = loader.adata

    print(f'adata shape: {adata.shape}')

    # Load embeddings
    emb = load_embeddings(emb_dir, sample_id)
    emb = emb.reshape(emb.shape[0], -1)  # Flatten to 2D
    print(f'Embeddings shape: {emb.shape}')

    # Check alignment
    if emb.shape[0] != adata.shape[0]:
        print(f'Warning: Mismatch in number of spots ({emb.shape[0]} vs {adata.shape[0]})')
        n_spots = min(emb.shape[0], adata.shape[0])
        emb = emb[:n_spots]
        adata = adata[:n_spots]

    # Standardize embeddings
    scaler = StandardScaler()
    scaled = scaler.fit_transform(emb)

    # Determine number of clusters for metrics (based on ground truth layers)
    if dataset == 'dlpfc':
        n_clusters_for_metrics = get_dlpfc_n_clusters(sample_id)
    else:
        n_clusters_for_metrics = n_clusters

    # Number of clusters for visualization (always 20 for DLPFC)
    n_clusters_for_viz = 20 if dataset == 'dlpfc' else n_clusters

    # KMeans clustering for metrics (matching ground truth layers)
    print(f'Running KMeans with {n_clusters_for_metrics} clusters for metrics...')
    kmeans_metrics = KMeans(n_clusters=n_clusters_for_metrics, random_state=42, n_init=10)
    labels_metrics = kmeans_metrics.fit_predict(scaled)
    adata.obs['cluster_metrics'] = pd.Categorical([str(l) for l in labels_metrics],
                                                   categories=[str(i) for i in range(n_clusters_for_metrics)])

    # KMeans clustering for visualization (k=20 for DLPFC)
    print(f'Running KMeans with {n_clusters_for_viz} clusters for visualization...')
    kmeans_viz = KMeans(n_clusters=n_clusters_for_viz, random_state=42, n_init=10)
    labels_viz = kmeans_viz.fit_predict(scaled)
    adata.obs['cluster'] = pd.Categorical([str(l) for l in labels_viz],
                                           categories=[str(i) for i in range(n_clusters_for_viz)])

    pca = PCA(n_components=min(50, emb.shape[1]))
    adata.obsm['X_pca'] = pca.fit_transform(scaled)

    # Create output directory early for elbow plot
    os.makedirs(output_dir, exist_ok=True)

    # Plot PCA variance
    print('Plotting PCA variance...')
    threshold_point = plot_pca_variance(pca, output_dir, sample_id, threshold=0.95)
    if threshold_point is not None:
        print(f'PCA 95% variance point: {threshold_point} components')

    # UMAP
    print('Computing UMAP...')
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=15)
    sc.tl.umap(adata)

    # Calculate metrics if ground truth available (using cluster_metrics)
    ari = None
    nmi = None
    if 'ground_truth' in adata.obs.columns:
        from sklearn import metrics
        ari = metrics.adjusted_rand_score(adata.obs['ground_truth'], adata.obs['cluster_metrics'])
        nmi = metrics.normalized_mutual_info_score(adata.obs['ground_truth'], adata.obs['cluster_metrics'])
        print(f'ARI (k={n_clusters_for_metrics}): {ari:.4f}')
        print(f'NMI (k={n_clusters_for_metrics}): {nmi:.4f}')

    result = {
        'sample_id': sample_id,
        'ARI': ari,
        'NMI': nmi,
        'n_clusters_for_metrics': n_clusters_for_metrics,
        'n_clusters_for_viz': n_clusters_for_viz
    }

    save_path = os.path.join(
        output_dir,
        f'{sample_id}_ari_nmi_result.json'
        )

    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4)


    # Plot 1: Spatial clustering (using cluster for visualization with k=20)
    fig, axes = plt.subplots(1, 2 if show_ground_truth else 1, figsize=(20 if show_ground_truth else 10, 8))

    # Determine spot_size based on dataset
    spot_size = 200 if dataset == 'human_breast_cancer' else None

    # Build palette list for clusters (using n_clusters_for_viz)
    cluster_palette = [COLOR_MAP.get(str(i), '#808080') for i in range(n_clusters_for_viz)]

    if show_ground_truth and 'ground_truth' in adata.obs.columns:
        ax1 = axes[0] if show_ground_truth else axes
        sc.pl.spatial(adata, img_key=None, size=1.5, color=['ground_truth'],
                      spot_size=spot_size, ax=ax1, show=False, title=f'{sample_id} - Ground Truth')
        sc.pl.spatial(adata, img_key=None, size=1.5, color=['cluster'],
                      spot_size=spot_size, palette=cluster_palette, ax=axes[1], show=False,
                      title=f'{sample_id} - Clusters (k={n_clusters_for_viz})')
    else:
        sc.pl.spatial(adata, img_key=None, size=1.5, color=['cluster'],
                      spot_size=spot_size, palette=cluster_palette, ax=axes if not show_ground_truth else axes[0],
                      show=False, title=f'{sample_id} - Clusters (k={n_clusters_for_viz})')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{sample_id}_spatial.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')

    # Plot 2: UMAP
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    if 'ground_truth' in adata.obs.columns:
        sc.pl.umap(adata, color='ground_truth', ax=axes[0], show=False,
                   title=f'{sample_id} - Ground Truth')
    sc.pl.umap(adata, color='cluster', palette=cluster_palette, ax=axes[1],
               show=False, title=f'{sample_id} - Clusters')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{sample_id}_umap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')

    return adata


def main():
    parser = argparse.ArgumentParser(description='Visualize Embeddings')

    parser.add_argument('--dataset', type=str, default='dlpfc',
                        choices=['dlpfc', 'human_breast_cancer'],
                        help='Dataset to use (dlpfc or human_breast_cancer)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing samples (auto-detected if not specified)')
    parser.add_argument('--emb_dir', type=str,
                        default='../../results/spatial_clustering/st_conma/human_breast_cancer/fusion_embeddings',
                        help='Directory containing embeddings')
    parser.add_argument('--sample_id', type=str, default=None,
                        help='Sample ID to visualize')
    parser.add_argument('--all', action='store_true',
                        help='Visualize all samples in the dataset')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of clusters for KMeans (auto-detected if not specified)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots (auto-generated if not specified)')

    args = parser.parse_args()

    # Get dataset config
    config = DATASET_CONFIGS.get(args.dataset, DATASET_CONFIGS['dlpfc'])

    # Set defaults based on dataset
    data_dir = args.data_dir or config['data_dir']
    n_clusters = args.n_clusters or config['default_n_clusters']
    samples = config['samples']
    output_dir = args.output_dir or f"{args.emb_dir}_viz"

    if args.all:
        for sample_id in samples:
            try:
                visualize_embeddings(
                    data_dir=data_dir,
                    emb_dir=args.emb_dir,
                    sample_id=sample_id,
                    n_clusters=n_clusters,
                    output_dir=output_dir,
                    dataset=args.dataset
                )
            except Exception as e:
                print(f'Error processing {sample_id}: {e}')
                import traceback
                traceback.print_exc()
    else:
        sample_id = args.sample_id or samples[0]
        visualize_embeddings(
            data_dir=data_dir,
            emb_dir=args.emb_dir,
            sample_id=sample_id,
            n_clusters=n_clusters,
            output_dir=output_dir,
            dataset=args.dataset
        )

    print('\nDone!')


if __name__ == '__main__':
    main()
