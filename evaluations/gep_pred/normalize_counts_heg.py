"""
Normalize counts for HER2ST and CSCC datasets.

1. Find common genes across all samples
2. Select top 250 HEG (highly expressed genes) based on mean expression
3. Normalize to 1e6 (CPM) and log1p transform
4. Save normalized counts and gene lists
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

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


def process_dataset(dataset_name, samples, cnt_dir, output_dir, top_k=250):
    """
    Process a dataset:
    1. Find common genes across all samples
    2. Select top-k HEG based on mean expression
    3. Normalize (CPM 1e6) and log1p
    4. Save
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
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

    # Step 2: Load all data with common genes and compute mean expression
    print("\nStep 2: Computing mean expression across all samples...")
    all_data = []
    sample_data = {}

    for sample_name in tqdm(samples, desc="Loading data"):
        cnt_path = os.path.join(cnt_dir, f'{sample_name}.tsv')
        if not os.path.exists(cnt_path):
            continue
        df = pd.read_csv(cnt_path, sep='\t', index_col=0)
        df = df[common_genes]  # Filter to common genes
        sample_data[sample_name] = df
        all_data.append(df.values)

    # Concatenate all spots
    all_data = np.concatenate(all_data, axis=0)
    print(f"  Total spots: {all_data.shape[0]}")

    # Compute mean expression per gene (across all spots)
    mean_expression = np.mean(all_data, axis=0)

    # Step 3: Select top-k HEG
    print(f"\nStep 3: Selecting top {top_k} highly expressed genes...")
    top_indices = np.argsort(mean_expression)[::-1][:top_k]
    heg_genes = [common_genes[i] for i in top_indices]
    print(f"  Top 10 HEG: {heg_genes[:10]}")
    print(f"  Mean expression range: {mean_expression[top_indices[0]]:.2f} - {mean_expression[top_indices[-1]]:.2f}")

    # Step 4: Normalize and save
    print(f"\nStep 4: Normalizing (CPM 1e6 + log1p) and saving...")
    os.makedirs(output_dir, exist_ok=True)

    for sample_name in tqdm(samples, desc="Saving normalized data"):
        if sample_name not in sample_data:
            continue

        df = sample_data[sample_name]

        # Filter to HEG
        df_heg = df[heg_genes]

        # CPM normalization (per spot, target_sum=1e6)
        row_sums = df_heg.values.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        normalized = df_heg.values / row_sums * 1e6

        # Log1p transform
        normalized = np.log1p(normalized)

        # Save
        df_out = pd.DataFrame(normalized, index=df_heg.index, columns=heg_genes)
        out_path = os.path.join(output_dir, f'{sample_name}.tsv')
        df_out.to_csv(out_path, sep='\t')

    # Save gene list
    gene_list_path = os.path.join(BASE_DIR, f'{dataset_name}_heg_cut_{top_k}.txt')
    with open(gene_list_path, 'w') as f:
        for gene in heg_genes:
            f.write(f'{gene}\n')
    print(f"  Saved gene list to: {gene_list_path}")

    print(f"\nDone! Saved {len(samples)} normalized files to: {output_dir}")

    return heg_genes


def main():
    parser = argparse.ArgumentParser(description='Normalize ST counts')
    parser.add_argument('--dataset', type=str, choices=['her2st', 'cscc', 'hlt', 'all'], default='all',
                        help='Dataset to process')
    parser.add_argument('--top_k', type=int, default=250,
                        help='Number of top HEG to select')

    args = parser.parse_args()

    if args.dataset in ['cscc', 'all']:
        cscc_cnt_dir = os.path.join(BASE_DIR, 'cscc/ST-cnts')
        cscc_output_dir = os.path.join(BASE_DIR, 'cscc/ST-cnts-normalized')
        cscc_samples = get_cscc_samples()
        process_dataset('cscc', cscc_samples, cscc_cnt_dir, cscc_output_dir, top_k=args.top_k)

    if args.dataset in ['her2st', 'all']:
        her2st_cnt_dir = os.path.join(BASE_DIR, 'her2st/ST-cnts')
        her2st_output_dir = os.path.join(BASE_DIR, 'her2st/ST-cnts-normalized')
        her2st_samples = get_her2st_samples()
        process_dataset('her2st', her2st_samples, her2st_cnt_dir, her2st_output_dir, top_k=args.top_k)

    if args.dataset in ['hlt', 'all']:
        hlt_cnt_dir = os.path.join(BASE_DIR, 'hlt/ST-cnts')
        hlt_output_dir = os.path.join(BASE_DIR, 'hlt/ST-cnts-normalized')
        hlt_samples = get_hlt_samples()
        process_dataset('hlt', hlt_samples, hlt_cnt_dir, hlt_output_dir, top_k=args.top_k)

    print("\nAll done!")


if __name__ == '__main__':
    main()
