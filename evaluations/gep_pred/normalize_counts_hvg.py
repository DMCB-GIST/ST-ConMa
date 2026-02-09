"""
Normalize counts for HER2ST, CSCC, and HLT datasets using HVG (Highly Variable Genes).

1. Find common genes across all samples
2. Select top 250 HVG using batch-aware method (each sample as a batch)
3. Normalize to 1e6 (CPM) and log1p transform
4. Save normalized counts and gene lists

The batch-aware HVG selection:
- Computes HVG within each sample (batch) independently
- Combines results to find genes that are consistently variable across batches
- Avoids selecting genes that are only variable due to batch effects

Usage:
    python normalize_counts_hvg.py --dataset all --top_k 250
    python normalize_counts_hvg.py --dataset her2st --top_k 250
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import scanpy as sc
import anndata as ad
import warnings
warnings.filterwarnings("ignore")

# Base data directory
BASE_DIR = './ft_dataset/gep_pred'


def get_her2st_samples():
    """Get HER2ST sample names."""
    cnt_dir = os.path.join(BASE_DIR, 'her2st/ST-cnts')
    names = [f.replace('.tsv', '') for f in os.listdir(cnt_dir) if f.endswith('.tsv')]
    names.sort()
    return names


def get_cscc_samples():
    """Get CSCC sample names."""
    patients = ['P2', 'P5', 'P9', 'P10']
    reps = ['rep1', 'rep2', 'rep3']
    return [f'{p}_ST_{r}' for p in patients for r in reps]


def get_hlt_samples():
    """Get HLT sample names."""
    cnt_dir = os.path.join(BASE_DIR, 'hlt/ST-cnts')
    names = [f.replace('.tsv', '') for f in os.listdir(cnt_dir) if f.endswith('.tsv')]
    names.sort()
    return names


def select_hvg_batch_aware(sample_data_dict, common_genes, top_k=250):
    """
    Select top-k HVG using batch-aware method.

    Each sample is treated as a separate batch.
    scanpy computes HVG within each batch and then combines results,
    selecting genes that are consistently variable across batches.

    Args:
        sample_data_dict: dict mapping sample_name -> DataFrame
        common_genes: list of common gene names
        top_k: number of top HVG to select

    Returns:
        hvg_genes: list of top-k HVG gene names
    """
    print("  Creating AnnData objects with batch information...")
    adata_list = []

    for sample_name, df in tqdm(sample_data_dict.items(), desc="  Processing samples"):
        # Create AnnData for this sample
        adata = sc.AnnData(X=df[common_genes].values.astype(np.float32))
        adata.var_names = common_genes
        adata.obs_names = [f"{sample_name}_{idx}" for idx in df.index]
        adata.obs['batch'] = sample_name  # Batch = sample
        adata_list.append(adata)

    # Concatenate all samples
    print("  Concatenating samples...")
    adata = ad.concat(adata_list, join='outer')
    print(f"  Combined shape: {adata.shape} (spots x genes)")
    print(f"  Number of batches: {len(adata.obs['batch'].unique())}")

    # Normalize (for HVG selection only)
    print("  Normalizing for HVG selection...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Batch-aware HVG selection
    print(f"  Selecting top {top_k} HVG (batch-aware)...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=top_k,
        batch_key='batch',  # Key: compute HVG per batch, then combine
        flavor='seurat'
    )

    # Get HVG genes
    hvg_df = adata.var[adata.var['highly_variable']].copy()

    # Sort by number of batches where the gene is highly variable (more consistent = better)
    if 'highly_variable_nbatches' in hvg_df.columns:
        hvg_df = hvg_df.sort_values('highly_variable_nbatches', ascending=False)
        print(f"  HVG selected in >= {hvg_df['highly_variable_nbatches'].min()} batches")

    hvg_genes = hvg_df.index.tolist()[:top_k]

    # Print statistics
    if 'highly_variable_nbatches' in adata.var.columns:
        nbatches = adata.var.loc[hvg_genes, 'highly_variable_nbatches']
        print(f"  Batch coverage: min={nbatches.min()}, max={nbatches.max()}, mean={nbatches.mean():.1f}")

    return hvg_genes


def process_dataset_hvg(dataset_name, samples, cnt_dir, output_dir, top_k=250):
    """
    Process a dataset for HVG:
    1. Find common genes across all samples
    2. Select top-k HVG using batch-aware method
    3. Normalize (CPM 1e6) and log1p
    4. Save
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} (HVG - Batch-Aware)")
    print(f"{'='*60}")

    # Step 1: Find common genes
    print("\nStep 1: Finding common genes across all samples...")
    gene_sets = []
    for sample_name in tqdm(samples, desc="Reading samples"):
        cnt_path = os.path.join(cnt_dir, f'{sample_name}.tsv')
        if not os.path.exists(cnt_path):
            print(f"  Warning: {cnt_path} not found, skipping")
            continue
        df = pd.read_csv(cnt_path, sep='\t', index_col=0, nrows=1)  # Just read header
        gene_sets.append(set(df.columns))

    common_genes = set.intersection(*gene_sets)
    common_genes = sorted(common_genes)
    print(f"  Found {len(common_genes)} common genes across {len(gene_sets)} samples")

    # Step 2: Load all data with common genes
    print("\nStep 2: Loading all data...")
    sample_data = {}

    for sample_name in tqdm(samples, desc="Loading data"):
        cnt_path = os.path.join(cnt_dir, f'{sample_name}.tsv')
        if not os.path.exists(cnt_path):
            continue
        df = pd.read_csv(cnt_path, sep='\t', index_col=0)
        sample_data[sample_name] = df

    total_spots = sum(len(df) for df in sample_data.values())
    print(f"  Total spots: {total_spots}")
    print(f"  Samples loaded: {len(sample_data)}")

    # Step 3: Select top-k HVG using batch-aware method
    print(f"\nStep 3: Selecting top {top_k} HVG (batch-aware)...")
    hvg_genes = select_hvg_batch_aware(sample_data, common_genes, top_k=top_k)
    print(f"  Top 10 HVG: {hvg_genes[:10]}")

    # Step 4: Normalize and save
    print(f"\nStep 4: Normalizing (CPM 1e6 + log1p) and saving...")
    os.makedirs(output_dir, exist_ok=True)

    for sample_name in tqdm(samples, desc="Saving normalized data"):
        if sample_name not in sample_data:
            continue

        df = sample_data[sample_name]

        # Filter to HVG
        df_hvg = df[hvg_genes]

        # CPM normalization (per spot, target_sum=1e6)
        row_sums = df_hvg.values.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        normalized = df_hvg.values / row_sums * 1e6

        # Log1p transform
        normalized = np.log1p(normalized)

        # Save
        df_out = pd.DataFrame(normalized, index=df_hvg.index, columns=hvg_genes)
        out_path = os.path.join(output_dir, f'{sample_name}.tsv')
        df_out.to_csv(out_path, sep='\t')

    # Save gene list
    gene_list_path = os.path.join(BASE_DIR, f'{dataset_name}_hvg_cut_{top_k}.txt')
    with open(gene_list_path, 'w') as f:
        for gene in hvg_genes:
            f.write(f'{gene}\n')
    print(f"  Saved gene list to: {gene_list_path}")

    print(f"\nDone! Saved {len(sample_data)} normalized files to: {output_dir}")

    return hvg_genes


def main():
    parser = argparse.ArgumentParser(description='Normalize ST counts using HVG (batch-aware)')
    parser.add_argument('--dataset', type=str, choices=['her2st', 'cscc', 'hlt', 'all'], default='all',
                        help='Dataset to process')
    parser.add_argument('--top_k', type=int, default=250,
                        help='Number of top HVG to select')

    args = parser.parse_args()

    if args.dataset in ['cscc', 'all']:
        cscc_cnt_dir = os.path.join(BASE_DIR, 'cscc/ST-cnts')
        cscc_output_dir = os.path.join(BASE_DIR, 'cscc/ST-cnts-normalized-hvg')
        cscc_samples = get_cscc_samples()
        process_dataset_hvg('cscc', cscc_samples, cscc_cnt_dir, cscc_output_dir, top_k=args.top_k)

    if args.dataset in ['her2st', 'all']:
        her2st_cnt_dir = os.path.join(BASE_DIR, 'her2st/ST-cnts')
        her2st_output_dir = os.path.join(BASE_DIR, 'her2st/ST-cnts-normalized-hvg')
        her2st_samples = get_her2st_samples()
        process_dataset_hvg('her2st', her2st_samples, her2st_cnt_dir, her2st_output_dir, top_k=args.top_k)

    if args.dataset in ['hlt', 'all']:
        hlt_cnt_dir = os.path.join(BASE_DIR, 'hlt/ST-cnts')
        hlt_output_dir = os.path.join(BASE_DIR, 'hlt/ST-cnts-normalized-hvg')
        hlt_samples = get_hlt_samples()
        process_dataset_hvg('hlt', hlt_samples, hlt_cnt_dir, hlt_output_dir, top_k=args.top_k)

    print("\nAll done!")


if __name__ == '__main__':
    main()
