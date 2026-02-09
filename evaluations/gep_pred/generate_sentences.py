import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
import scanpy as sc
import argparse

# Base data directory
BASE_DIR = './ft_dataset/gep_pred'

# Load common overlap genes from pretraining
COMMON_GENES_PATH = './pt_dataset/common_overlap_genes.txt'


def load_common_genes():
    """Load the common overlap genes used in pretraining."""
    with open(COMMON_GENES_PATH, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return set(genes)


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


def process_sample(cnt_path, sample_name, common_genes, top_k=100):
    """
    Process a single sample and generate gene sentences.

    Uses the same approach as pretraining:
    1. Filter to common_overlap_genes
    2. sc.pp.normalize_total(target_sum=1e4)
    3. sc.pp.log1p
    4. For each spot, get non-zero genes only, sort by expression, take top-k
    """
    import anndata

    # Load count matrix (genes as columns, spots as rows)
    df = pd.read_csv(cnt_path, sep='\t', index_col=0)

    # Get intersection with common genes (sorted like pretraining)
    available_genes = set(df.columns)
    overlap_genes = sorted(available_genes.intersection(common_genes))

    if len(overlap_genes) == 0:
        print(f"  Warning: No overlapping genes found for {sample_name}")
        return []

    # Filter to common genes
    df = df[overlap_genes]

    # Create AnnData object
    adata = anndata.AnnData(X=df.values.astype(np.float32))
    adata.obs_names = df.index.astype(str)
    adata.var_names = df.columns

    # Normalize using scanpy (same as pretraining)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Convert to sparse matrix (same as pretraining)
    X = csr_matrix(adata.X)
    genes = adata.var.index.to_numpy()
    barcodes = adata.obs.index.to_numpy()

    # Generate sentences (exactly like pretraining)
    rows = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
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
            "id": f"{sample_name}_{barcodes[i]}",
            "sentence": sentence
        })

    return rows


def generate_her2st_sentences(output_dir, top_k=100):
    """Generate sentences for HER2ST dataset."""
    print("=" * 60)
    print("Generating HER2ST gene sentences")
    print("=" * 60)

    common_genes = load_common_genes()
    print(f"Loaded {len(common_genes)} common overlap genes")

    cnt_dir = os.path.join(BASE_DIR, 'her2st/ST-cnts')
    samples = get_her2st_samples()

    os.makedirs(output_dir, exist_ok=True)

    all_stats = []

    for sample_name in tqdm(samples, desc="HER2ST samples"):
        cnt_path = os.path.join(cnt_dir, f'{sample_name}.tsv')

        rows = process_sample(cnt_path, sample_name, common_genes, top_k=top_k)

        if rows:
            # Save individual sample CSV
            df_out = pd.DataFrame(rows)
            out_path = os.path.join(output_dir, f'{sample_name}.csv')
            df_out.to_csv(out_path, index=False)

            # Calculate stats
            n_genes_per_spot = [len(r['sentence'].split()) for r in rows]
            all_stats.append({
                'sample': sample_name,
                'n_spots': len(rows),
                'mean_genes': np.mean(n_genes_per_spot),
                'min_genes': np.min(n_genes_per_spot),
                'max_genes': np.max(n_genes_per_spot)
            })

            print(f"  {sample_name}: {len(rows)} spots, avg {np.mean(n_genes_per_spot):.1f} genes/spot")

    # Save stats
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(os.path.join(output_dir, 'stats.csv'), index=False)
    print(f"\nTotal: {len(samples)} samples processed")


def generate_cscc_sentences(output_dir, top_k=100):
    """Generate sentences for CSCC dataset."""
    print("=" * 60)
    print("Generating CSCC gene sentences")
    print("=" * 60)

    common_genes = load_common_genes()
    print(f"Loaded {len(common_genes)} common overlap genes")

    cnt_dir = os.path.join(BASE_DIR, 'cscc/ST-cnts')
    samples = get_cscc_samples()

    os.makedirs(output_dir, exist_ok=True)

    all_stats = []

    for sample_name in tqdm(samples, desc="CSCC samples"):
        cnt_path = os.path.join(cnt_dir, f'{sample_name}.tsv')

        if not os.path.exists(cnt_path):
            print(f"  Warning: {cnt_path} not found, skipping")
            continue

        rows = process_sample(cnt_path, sample_name, common_genes, top_k=top_k)

        if rows:
            # Save individual sample CSV
            df_out = pd.DataFrame(rows)
            out_path = os.path.join(output_dir, f'{sample_name}.csv')
            df_out.to_csv(out_path, index=False)

            # Calculate stats
            n_genes_per_spot = [len(r['sentence'].split()) for r in rows]
            all_stats.append({
                'sample': sample_name,
                'n_spots': len(rows),
                'mean_genes': np.mean(n_genes_per_spot),
                'min_genes': np.min(n_genes_per_spot),
                'max_genes': np.max(n_genes_per_spot)
            })

            print(f"  {sample_name}: {len(rows)} spots, avg {np.mean(n_genes_per_spot):.1f} genes/spot")

    # Save stats
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(os.path.join(output_dir, 'stats.csv'), index=False)
    print(f"\nTotal: {len(samples)} samples processed")


def generate_hlt_sentences(output_dir, top_k=100):
    """Generate sentences for HLT dataset."""
    print("=" * 60)
    print("Generating HLT gene sentences")
    print("=" * 60)

    common_genes = load_common_genes()
    print(f"Loaded {len(common_genes)} common overlap genes")

    cnt_dir = os.path.join(BASE_DIR, 'hlt/ST-cnts')
    samples = get_hlt_samples()

    os.makedirs(output_dir, exist_ok=True)

    all_stats = []

    for sample_name in tqdm(samples, desc="HLT samples"):
        cnt_path = os.path.join(cnt_dir, f'{sample_name}.tsv')

        if not os.path.exists(cnt_path):
            print(f"  Warning: {cnt_path} not found, skipping")
            continue

        rows = process_sample(cnt_path, sample_name, common_genes, top_k=top_k)

        if rows:
            # Save individual sample CSV
            df_out = pd.DataFrame(rows)
            out_path = os.path.join(output_dir, f'{sample_name}.csv')
            df_out.to_csv(out_path, index=False)

            # Calculate stats
            n_genes_per_spot = [len(r['sentence'].split()) for r in rows]
            all_stats.append({
                'sample': sample_name,
                'n_spots': len(rows),
                'mean_genes': np.mean(n_genes_per_spot),
                'min_genes': np.min(n_genes_per_spot),
                'max_genes': np.max(n_genes_per_spot)
            })

            print(f"  {sample_name}: {len(rows)} spots, avg {np.mean(n_genes_per_spot):.1f} genes/spot")

    # Save stats
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(os.path.join(output_dir, 'stats.csv'), index=False)
    print(f"\nTotal: {len(samples)} samples processed")


def main():
    parser = argparse.ArgumentParser(description='Generate gene sentences for ST datasets')
    parser.add_argument('--dataset', type=str, choices=['her2st', 'cscc', 'hlt', 'all'], default='all',
                        help='Dataset to process')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Number of top genes per sentence')

    args = parser.parse_args()

    her2st_output = os.path.join(BASE_DIR, 'her2st/ST-sentences')
    cscc_output = os.path.join(BASE_DIR, 'cscc/ST-sentences')
    hlt_output = os.path.join(BASE_DIR, 'hlt/ST-sentences')

    if args.dataset in ['her2st', 'all']:
        generate_her2st_sentences(her2st_output, top_k=args.top_k)

    if args.dataset in ['cscc', 'all']:
        generate_cscc_sentences(cscc_output, top_k=args.top_k)

    if args.dataset in ['hlt', 'all']:
        generate_hlt_sentences(hlt_output, top_k=args.top_k)

    print("\nDone")


if __name__ == '__main__':
    main()
