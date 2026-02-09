"""
Saves 224x224 patches centered on each spot location.
Patches are saved as: ST-patches/{sample_name}/{spot_id}.png
"""

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
import argparse

# Handle large images
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

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


def extract_her2st_patches(output_dir, patch_size=224):
    """Extract patches from HER2ST dataset."""
    print("=" * 60)
    print("Extracting HER2ST patches")
    print("=" * 60)

    img_dir = os.path.join(BASE_DIR, 'her2st/ST-imgs')
    pos_dir = os.path.join(BASE_DIR, 'her2st/ST-spotfiles')

    r = patch_size // 2
    samples = get_her2st_samples()

    for sample_name in tqdm(samples, desc="HER2ST samples", leave=False):
        # Create output directory
        sample_output_dir = os.path.join(output_dir, sample_name)
        os.makedirs(sample_output_dir, exist_ok=True)

        img_subdir = os.path.join(img_dir, sample_name[0], sample_name)
        fig_name = os.listdir(img_subdir)[0]
        img_path = os.path.join(img_subdir, fig_name)
        img = Image.open(img_path)

        # Load spot positions
        pos_path = os.path.join(pos_dir, f'{sample_name}_selection.tsv')
        pos_df = pd.read_csv(pos_path, sep='\t')

        # Get pixel coordinates
        pixel_x = np.floor(pos_df['pixel_x'].values).astype(int)
        pixel_y = np.floor(pos_df['pixel_y'].values).astype(int)

        # Create spot IDs
        x = np.around(pos_df['x'].values).astype(int)
        y = np.around(pos_df['y'].values).astype(int)
        spot_ids = [f"{x[i]}x{y[i]}" for i in range(len(x))]

        # Extract patches
        for i, spot_id in enumerate(spot_ids):
            cx, cy = pixel_x[i], pixel_y[i]
            patch = img.crop((cx - r, cy - r, cx + r, cy + r))

            patch_path = os.path.join(sample_output_dir, f"{spot_id}.png")
            patch.save(patch_path)

        print(f"  {sample_name}: {len(spot_ids)} patches saved")


def extract_cscc_patches(output_dir, patch_size=224):
    """Extract patches from CSCC dataset."""
    print("=" * 60)
    print("Extracting CSCC patches")
    print("=" * 60)

    img_dir = os.path.join(BASE_DIR, 'cscc/ST-imgs')
    pos_dir = os.path.join(BASE_DIR, 'cscc/ST-spotfiles')

    r = patch_size // 2
    samples = get_cscc_samples()

    for sample_name in tqdm(samples, desc="CSCC samples", leave=False):
        # Create output directory
        sample_output_dir = os.path.join(output_dir, sample_name)
        os.makedirs(sample_output_dir, exist_ok=True)

        # Load image (structure: ST-imgs/{sample_name}/image.jpg)
        img_subdir = os.path.join(img_dir, sample_name)
        fig_name = [f for f in os.listdir(img_subdir) if f.endswith('.jpg') or f.endswith('.tif')][0]
        img_path = os.path.join(img_subdir, fig_name)
        img = Image.open(img_path)

        # Load spot positions
        pos_path = os.path.join(pos_dir, f'{sample_name}.tsv')
        pos_df = pd.read_csv(pos_path, sep='\t')

        # Get pixel coordinates
        pixel_x = np.floor(pos_df['pixel_x'].values).astype(int)
        pixel_y = np.floor(pos_df['pixel_y'].values).astype(int)

        # Create spot IDs
        x = np.around(pos_df['x'].values).astype(int)
        y = np.around(pos_df['y'].values).astype(int)
        spot_ids = [f"{x[i]}x{y[i]}" for i in range(len(x))]

        # Extract patches
        for i, spot_id in enumerate(spot_ids):
            cx, cy = pixel_x[i], pixel_y[i]
            patch = img.crop((cx - r, cy - r, cx + r, cy + r))

            patch_path = os.path.join(sample_output_dir, f"{spot_id}.png")
            patch.save(patch_path)

        print(f"  {sample_name}: {len(spot_ids)} patches saved")


def extract_hlt_patches(output_dir, patch_size=224):
    """Extract patches from HLT dataset."""
    print("=" * 60)
    print("Extracting HLT patches")
    print("=" * 60)

    img_dir = os.path.join(BASE_DIR, 'hlt/ST-imgs')
    spotfile_dir = os.path.join(BASE_DIR, 'hlt/ST-spotfiles')

    # Mapping: sample name -> (image file, spotfile)
    sample_mapping = {
        'A1': ('GSM7697868_GEX_C73_A1_Merged.tiff', 'tissue_positions_list_1.csv'),
        'B1': ('GSM7697869_GEX_C73_B1_Merged.tiff', 'tissue_positions_list_2.csv'),
        'C1': ('GSM7697870_GEX_C73_C1_Merged.tiff', 'tissue_positions_list_3.csv'),
        'D1': ('GSM7697871_GEX_C73_D1_Merged.tiff', 'tissue_positions_list_4.csv'),
    }

    r = patch_size // 2
    samples = get_hlt_samples()

    for sample_name in tqdm(samples, desc="HLT samples", leave=False):
        if sample_name not in sample_mapping:
            print(f"  Warning: {sample_name} not in mapping, skipping")
            continue

        img_file, spotfile = sample_mapping[sample_name]

        # Create output directory
        sample_output_dir = os.path.join(output_dir, sample_name)
        os.makedirs(sample_output_dir, exist_ok=True)

        # Load image
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).convert('RGB')

        pos_path = os.path.join(spotfile_dir, spotfile)
        pos_df = pd.read_csv(pos_path, header=None,
                             names=['barcode', 'in_tissue', 'array_row', 'array_col', 'pixel_y', 'pixel_x'])

        # Filter to in-tissue spots only
        pos_df = pos_df[pos_df['in_tissue'] == 1]

        # Load corresponding count matrix to get valid spot IDs
        cnt_path = os.path.join(BASE_DIR, 'hlt/ST-cnts', f'{sample_name}.tsv')
        cnt_df_full = pd.read_csv(cnt_path, sep='\t', index_col=0, usecols=[0])  # Get just the index column
        valid_spot_ids = set(cnt_df_full.index.astype(str))

        # Create spot IDs and filter to valid ones
        pos_df['spot_id'] = pos_df.apply(
            lambda row: f"{int(row['array_row'])}x{int(row['array_col'])}", axis=1
        )
        pos_df = pos_df[pos_df['spot_id'].isin(valid_spot_ids)]

        # Get pixel coordinates
        pixel_x = pos_df['pixel_x'].values.astype(int)
        pixel_y = pos_df['pixel_y'].values.astype(int)
        spot_ids = pos_df['spot_id'].values

        # Extract patches
        for i, spot_id in enumerate(spot_ids):
            cx, cy = pixel_x[i], pixel_y[i]
            patch = img.crop((cx - r, cy - r, cx + r, cy + r))

            patch_path = os.path.join(sample_output_dir, f"{spot_id}.png")
            patch.save(patch_path)

        print(f"  {sample_name}: {len(spot_ids)} patches saved")


def main():
    parser = argparse.ArgumentParser(description='Extract ST patches')
    parser.add_argument('--dataset', type=str, choices=['her2st', 'cscc', 'hlt', 'all'], default='all',
                        help='Dataset to extract patches from')

    args = parser.parse_args()

    her2st_output = os.path.join(BASE_DIR, 'her2st/ST-patches')
    cscc_output = os.path.join(BASE_DIR, 'cscc/ST-patches')
    hlt_output = os.path.join(BASE_DIR, 'hlt/ST-patches')

    if args.dataset in ['her2st', 'all']:
        extract_her2st_patches(her2st_output)

    if args.dataset in ['cscc', 'all']:
        extract_cscc_patches(cscc_output)

    if args.dataset in ['hlt', 'all']:
        extract_hlt_patches(hlt_output)

    print("\nDone")


if __name__ == '__main__':
    main()
