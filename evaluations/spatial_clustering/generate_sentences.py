"""
Generate gene sentences for DLPFC and Human_breast_cancer samples.

Uses the same approach as pretraining:
1. Filter genes to common_overlap_genes (7816 genes)
2. Normalize counts (target_sum=1e4) and log1p transform
3. For each spot, sort genes by expression (descending) and take top-K as sentence

Output:
- Each sample's CSV saved to: /ft_dataset/{dataset}/{sample_id}/top100_sentences.csv
- Stats saved to: /ft_dataset/{dataset}/sentence_stats.csv

Usage:
    python generate_sentences.py --dataset dlpfc
    python generate_sentences.py --dataset breast_cancer
    python generate_sentences.py --dataset all
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
import scanpy as sc
import argparse

# Add STAIG path for adata_processing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'staig_fusion'))
from staig.adata_processing import LoadSingle10xAdata

# Load common overlap genes from pretraining
COMMON_GENES_PATH = './pt_dataset/common_overlap_genes.txt'

# DLPFC data directory
DLPFC_BASE_DIR = './ft_dataset/spatial_clustering/DLPFC'
BREAST_CANCER_DIR = './ft_dataset/spatial_clustering/human_breast_cancer'

# DLPFC sample IDs
DLPFC_SAMPLES = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]

def load_common_genes():
    """Load the common overlap genes used in pretraining."""
    with open(COMMON_GENES_PATH, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return set(genes)


def process_sample(sample_id, data_dir, common_genes, top_k=100, has_label=True):
    """Process a single sample and generate gene sentences."""

    # Load data using LoadSingle10xAdata (same as patching.py)
    loader = LoadSingle10xAdata(
        path=data_dir,
        image_emb=False,
        label=has_label,
        filter_na=True
    )

    loader.load_data()
    if has_label:
        loader.load_label()
    adata = loader.adata

    # Get gene names
    gene_names = adata.var_names.tolist()

    # Get intersection with common genes
    available_genes = set(gene_names)
    overlap_genes = sorted(available_genes.intersection(common_genes))

    if len(overlap_genes) == 0:
        print(f"  Warning: No overlapping genes found for {sample_id}")
        return []

    print(f"  {sample_id}: {len(overlap_genes)} overlapping genes (from {len(gene_names)} total)")

    # Filter to common genes
    adata_filtered = adata[:, overlap_genes].copy()

    # Normalize using scanpy (same as pretraining)
    sc.pp.normalize_total(adata_filtered, target_sum=1e4)
    sc.pp.log1p(adata_filtered)


    X = csr_matrix(adata_filtered.X)

    genes = adata_filtered.var.index.to_numpy()
    barcodes = adata_filtered.obs.index.to_numpy()

    # Generate sentences (exactly like pretraining)
    rows = []
    zero_rows = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0: # nnz = number of non-zero entries
            zero_rows.append(row)
            continue

        # Get non-zero indices and values only
        nz_idx = row.indices
        nz_val = row.data

        # Sort by expression value (descending) and take top-k
        order = np.argsort(-nz_val)
        top_idx = nz_idx[order][:top_k]
        top_genes = genes[top_idx]
        sentence = " ".join(top_genes)

        rows.append({
            "id": f"{sample_id}_{barcodes[i]}",
            "sentence": sentence
        })
    
    print(f"{sample_id} | zero inflated spots : {len(zero_rows)}")

    return rows


def generate_dlpfc_sentences(top_k=100, samples=None):
    """
    Generate sentences for DLPFC dataset.

    Args:
        top_k: Number of top genes per sentence
        samples: List of sample IDs to process (default: all DLPFC samples)

    Output:
        - Each sample's CSV: /ft_dataset/spatial_clustering/DLPFC/{sample_id}/top100_sentences.csv
        - Stats: /ft_dataset/spatial_clustering/DLPFC/sentence_stats.csv
    """
    print("=" * 60)
    print("Generating DLPFC gene sentences")
    print("=" * 60)

    common_genes = load_common_genes()
    print(f"Loaded {len(common_genes)} common overlap genes")

    if samples is None:
        samples = DLPFC_SAMPLES

    all_stats = []

    for sample_id in tqdm(samples, desc="DLPFC samples"):
        data_dir = os.path.join(DLPFC_BASE_DIR, sample_id)
        rows = process_sample(sample_id, data_dir, common_genes, top_k=top_k, has_label=True)

        if rows:
            # Save to individual sample folder
            sample_dir = os.path.join(DLPFC_BASE_DIR, sample_id)
            out_path = os.path.join(sample_dir, 'top100_sentences.csv')

            df_out = pd.DataFrame(rows)
            df_out.to_csv(out_path, index=False)

            # Calculate stats
            n_genes_per_spot = [len(r['sentence'].split()) for r in rows]
            all_stats.append({
                'sample': sample_id,
                'n_spots': len(rows),
                'mean_genes': np.mean(n_genes_per_spot),
                'min_genes': np.min(n_genes_per_spot),
                'max_genes': np.max(n_genes_per_spot)
            })

            print(f"  {sample_id}: {len(rows)} spots, avg {np.mean(n_genes_per_spot):.1f} genes/spot -> {out_path}")

    # Save stats to DLPFC base folder
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_path = os.path.join(DLPFC_BASE_DIR, 'sentence_stats.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"\nSaved stats to {stats_path}")

    total_spots = sum(s['n_spots'] for s in all_stats)
    print(f"\nTotal: {len(samples)} samples processed, {total_spots} spots")

    return all_stats


def generate_breast_cancer_sentences(top_k=100):
    """
    Generate sentences for Human breast cancer dataset.

    Args:
        top_k: Number of top genes per sentence

    Output:
        - CSV: /ft_dataset/spatial_clustering/human_breast_cancer/top100_sentences.csv
    """
    print("=" * 60)
    print("Generating Human Breast Cancer gene sentences")
    print("=" * 60)

    common_genes = load_common_genes()
    print(f"Loaded {len(common_genes)} common overlap genes")

    sample_id = "Human_breast_cancer"
    rows = process_sample(sample_id, BREAST_CANCER_DIR, common_genes, top_k=top_k, has_label=True)

    all_stats = []

    if rows:
        # Save to breast cancer folder
        out_path = os.path.join(BREAST_CANCER_DIR, 'top100_sentences.csv')

        df_out = pd.DataFrame(rows)
        df_out.to_csv(out_path, index=False)

        # Calculate stats
        n_genes_per_spot = [len(r['sentence'].split()) for r in rows]
        all_stats.append({
            'sample': sample_id,
            'n_spots': len(rows),
            'mean_genes': np.mean(n_genes_per_spot),
            'min_genes': np.min(n_genes_per_spot),
            'max_genes': np.max(n_genes_per_spot)
        })

        print(f"  {sample_id}: {len(rows)} spots, avg {np.mean(n_genes_per_spot):.1f} genes/spot -> {out_path}")

        # Save stats
        stats_df = pd.DataFrame(all_stats)
        stats_path = os.path.join(BREAST_CANCER_DIR, 'sentence_stats.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"\nSaved stats to {stats_path}")

    return all_stats


def main():
    parser = argparse.ArgumentParser(description='Generate gene sentences for spatial transcriptomics samples')
    parser.add_argument('--dataset', type=str, default='dlpfc',
                        choices=['dlpfc', 'breast_cancer', 'all'],
                        help='Dataset to process: dlpfc, breast_cancer, or all')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Number of top genes per sentence')
    parser.add_argument('--samples', type=str, default=None,
                        help='Comma-separated sample IDs to process (DLPFC only, default: all)')

    args = parser.parse_args()

    samples = None
    if args.samples:
        samples = [s.strip() for s in args.samples.split(',')]

    if args.dataset == 'dlpfc':
        generate_dlpfc_sentences(top_k=args.top_k, samples=samples)
    elif args.dataset == 'breast_cancer':
        generate_breast_cancer_sentences(top_k=args.top_k)
    elif args.dataset == 'all':
        generate_dlpfc_sentences(top_k=args.top_k, samples=samples)
        generate_breast_cancer_sentences(top_k=args.top_k)

    print("\nDone!")


if __name__ == '__main__':
    main()
