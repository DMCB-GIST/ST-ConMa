"""
HLT (Human Lymphoid Tissue) Data Preparation Script

Step 1: Creates ST-cnts (raw count matrices) from 10x Visium format
        This script handles Step 1 only.

After running this script, run:
    python extract_patches.py --dataset hlt      (ST-patches)
    python normalize_counts.py --dataset hlt     (ST-cnts-normalized)
    python generate_sentences.py --dataset hlt   (ST-sentences)

Directory structure:
hlt/
├── ST-cnts/          
├── ST-patches/        
├── ST-cnts-normalized/ 
├── ST-sentences/      
├── ST-imgs/
├── ST-spotfiles/
└── filtered_expression_matrices/

Usage:
    python prepare_hlt_data.py
"""

import os
import numpy as np
import pandas as pd
from scipy.io import mmread
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = './ft_dataset/gep_pred/hlt'
OUTPUT_BASE_DIR = BASE_DIR

# Mapping: sample index -> (expression_matrix_dir, spotfile, sample_name)
# 4 samples (1,2,3,4) corresponding to (A1,B1,C1,D1)
SAMPLE_MAPPING = {
    '1': {
        'expr_dir': f'{BASE_DIR}/filtered_expression_matrices/1',
        'spotfile': f'{BASE_DIR}/ST-spotfiles/tissue_positions_list_1.csv',
        'name': 'A1'
    },
    '2': {
        'expr_dir': f'{BASE_DIR}/filtered_expression_matrices/2',
        'spotfile': f'{BASE_DIR}/ST-spotfiles/tissue_positions_list_2.csv',
        'name': 'B1'
    },
    '3': {
        'expr_dir': f'{BASE_DIR}/filtered_expression_matrices/3',
        'spotfile': f'{BASE_DIR}/ST-spotfiles/tissue_positions_list_3.csv',
        'name': 'C1'
    },
    '4': {
        'expr_dir': f'{BASE_DIR}/filtered_expression_matrices/4',
        'spotfile': f'{BASE_DIR}/ST-spotfiles/tissue_positions_list_4.csv',
        'name': 'D1'
    }
}


# ============================================================================
# Helper Functions
# ============================================================================

def load_10x_data(expr_dir):
    """
    Load 10x Genomics expression data (matrix.mtx, barcodes.tsv, features.tsv).

    Returns:
        matrix: sparse matrix (genes x spots)
        barcodes: list of barcode strings
        gene_names: list of gene names
    """
    # Load matrix
    matrix_path = os.path.join(expr_dir, 'matrix.mtx')
    matrix = mmread(matrix_path).tocsr()  # genes x spots

    # Load barcodes
    barcodes_path = os.path.join(expr_dir, 'barcodes.tsv')
    with open(barcodes_path, 'r') as f:
        barcodes = [line.strip() for line in f]

    # Load features (gene_id, gene_name, feature_type)
    features_path = os.path.join(expr_dir, 'features.tsv')
    gene_names = []
    with open(features_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            gene_names.append(parts[1])  # gene name is second column

    return matrix, barcodes, gene_names


def load_tissue_positions(spotfile_path):
    """
    Load tissue positions from 10x Visium format.

    Format: barcode,in_tissue,array_row,array_col,pixel_row,pixel_col

    Returns:
        DataFrame with columns: barcode, in_tissue, array_row, array_col, pixel_y, pixel_x
    """
    df = pd.read_csv(spotfile_path, header=None,
                     names=['barcode', 'in_tissue', 'array_row', 'array_col', 'pixel_y', 'pixel_x'])
    return df


# ============================================================================
# Main Processing Functions
# ============================================================================

def process_sample(sample_idx, sample_info):

    sample_name = sample_info['name']
    print(f"\n{'='*60}")
    print(f"Processing sample {sample_idx}: {sample_name}")
    print(f"{'='*60}")

    # Load expression data
    print("Loading expression data...")
    matrix, barcodes, gene_names = load_10x_data(sample_info['expr_dir'])
    print(f"  Matrix shape: {matrix.shape} (genes x spots)")
    print(f"  Barcodes: {len(barcodes)}")
    print(f"  Genes: {len(gene_names)}")

    # Load tissue positions
    print("Loading tissue positions...")
    positions = load_tissue_positions(sample_info['spotfile'])
    print(f"  Total positions: {len(positions)}")

    # Filter to only in-tissue spots with valid barcodes
    in_tissue = positions[positions['in_tissue'] == 1].copy()
    print(f"  In-tissue spots: {len(in_tissue)}")

    # Match barcodes between expression and positions
    barcode_set = set(barcodes)
    in_tissue = in_tissue[in_tissue['barcode'].isin(barcode_set)]
    print(f"  Matched spots: {len(in_tissue)}")

    # Create barcode to index mapping
    barcode_to_idx = {b: i for i, b in enumerate(barcodes)}

    # Create spot IDs (array_row x array_col format, same as HER2ST)
    in_tissue['spot_id'] = in_tissue.apply(
        lambda row: f"{int(row['array_row'])}x{int(row['array_col'])}", axis=1
    )

    # Extract expression for matched spots (transpose: genes x spots -> spots x genes)
    spot_indices = [barcode_to_idx[b] for b in in_tissue['barcode']]
    expr_matrix = matrix[:, spot_indices].T.toarray()  # spots x genes
    print(f"  Expression matrix: {expr_matrix.shape} (spots x genes)")

    # Save raw count matrix (ST-cnts)
    print("\nSaving raw count matrix...")
    cnts_dir = os.path.join(OUTPUT_BASE_DIR, 'ST-cnts')
    os.makedirs(cnts_dir, exist_ok=True)

    # Create DataFrame with spot_id as index and gene_names as columns
    count_df = pd.DataFrame(
        expr_matrix,
        index=in_tissue['spot_id'].values,
        columns=gene_names
    )

    cnts_path = os.path.join(cnts_dir, f'{sample_name}.tsv')
    count_df.to_csv(cnts_path, sep='\t')
    print(f"  Saved: {cnts_path}")
    print(f"  Shape: {count_df.shape}")

    # Return statistics
    stats = {
        'sample_name': sample_name,
        'n_spots': len(in_tissue),
        'n_genes': len(gene_names),
    }

    return stats


def main():
    """Main function to process all HLT samples."""
    print("="*60)
    print("HLT Data Preparation - ST-cnts")
    print("="*60)
    print("\nThis script creates ST-cnts/ (raw count matrices)")

    # Process each sample
    all_stats = []
    for sample_idx, sample_info in SAMPLE_MAPPING.items():
        stats = process_sample(sample_idx, sample_info)
        all_stats.append(stats)

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for stats in all_stats:
        print(f"  {stats['sample_name']}: {stats['n_spots']} spots, {stats['n_genes']} genes")

    total_spots = sum(s['n_spots'] for s in all_stats)
    print(f"\n  Total spots: {total_spots}")

    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print("1. Run: python extract_patches.py --dataset hlt")
    print("2. Run: python normalize_counts.py --dataset hlt")
    print("3. Run: python generate_sentences.py --dataset hlt")
    print("4. Update dataset.py to add HLT support")
    print("5. Update train_patient_cv.py to add 'hlt' dataset option")


if __name__ == '__main__':
    main()
