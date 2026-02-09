import argparse
import cv2
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# Add STAIG path for adata_processing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines/STAIG/staig'))
from adata_processing import LoadSingle10xAdata


DLPFC_BASE_DIR = './ft_dataset/spatial_clustering/DLPFC'
BREAST_CANCER_DIR = './ft_dataset/spatial_clustering/human_breast_cancer'

DLPFC_SAMPLES = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]

# Patch size: 224x224 (112 pixels on each side from center)
PATCH_SIZE = 224
HALF_SIZE = PATCH_SIZE // 2  # 112


def extract_patches_for_sample(path, sample_id, has_label=True):
    """Extract patches for a single sample."""
    print(f"\n{'='*60}")
    print(f"Processing sample: {sample_id}")
    print(f"{'='*60}")

    # Initialize data loader
    loader = LoadSingle10xAdata(path=path, image_emb=False, label=has_label, filter_na=True)
    loader.load_data()
    if has_label:
        loader.load_label()
    adata = loader.adata

    # Get barcodes
    barcodes = adata.obs.index.to_numpy()

    tif_path = os.path.join(path, "spatial", "tissue_full_image.tif")
    if os.path.exists(tif_path):
        print("File exists.")
    else:
        print("File does not exist.")
        return

    # Read tiff image
    im = cv2.imread(tif_path, cv2.IMREAD_COLOR)
    if im is None:
        print(f"Error: Could not read image at {tif_path}")
        return

    img_height, img_width = im.shape[:2]
    print(f"Image size: {img_width} x {img_height}")

    # Create directory for st_images
    st_images_path = os.path.join(path, 'st_images')
    try:
        os.makedirs(st_images_path)
        print("Folder 'st_images' created successfully")
    except FileExistsError:
        print("Folder 'st_images' already exist")

    # Process and save 224x224 patches with {sample_id}_{barcode}.png naming
    skipped = 0
    saved = 0

    for i, coord in tqdm(enumerate(adata.obsm['spatial']), total=len(adata.obsm['spatial']), desc=f"{sample_id}"):
        # Calculate patch coordinates (112 pixels on each side from center)
        center_x = int(coord[0])
        center_y = int(coord[1])

        left = center_x - HALF_SIZE
        top = center_y - HALF_SIZE
        right = center_x + HALF_SIZE
        bottom = center_y + HALF_SIZE

        # Check bounds
        if left < 0 or top < 0 or right > img_width or bottom > img_height:
            skipped += 1
            continue

        # Extract 224x224 patch directly (no resize needed)
        patch = im[top:bottom, left:right]

        # Verify patch size
        if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
            skipped += 1
            continue

        # Save patch with {sample_id}_{barcode}.png naming
        barcode = barcodes[i]
        filename = f'{sample_id}_{barcode}.png'
        cv2.imwrite(os.path.join(st_images_path, filename), patch)
        saved += 1

    print(f"  {sample_id}: {saved} patches saved, {skipped} skipped (out of bounds)")
    print(f"  Output: {st_images_path}")


def process_dlpfc():
    """Process all DLPFC samples."""
    print("\n" + "="*60)
    print("Processing DLPFC Dataset")
    print("="*60)
    for sample_id in DLPFC_SAMPLES:
        path = os.path.join(DLPFC_BASE_DIR, sample_id)
        extract_patches_for_sample(path, sample_id, has_label=True)


def process_breast_cancer():
    """Process Human breast cancer sample."""
    print("\n" + "="*60)
    print("Processing Human Breast Cancer Dataset")
    print("="*60)
    sample_id = "Human_breast_cancer"
    extract_patches_for_sample(BREAST_CANCER_DIR, sample_id, has_label=True)


def main():
    parser = argparse.ArgumentParser(description='Extract 224x224 patches for spatial transcriptomics samples')
    parser.add_argument('--dataset', type=str, default='dlpfc',
                        choices=['dlpfc', 'breast_cancer', 'all'],
                        help='Dataset to process: dlpfc, breast_cancer, or all')
    args = parser.parse_args()

    if args.dataset == 'dlpfc':
        process_dlpfc()
    elif args.dataset == 'breast_cancer':
        process_breast_cancer()
    elif args.dataset == 'all':
        process_dlpfc()
        process_breast_cancer()

    print("\nDone!")


if __name__ == '__main__':
    main()
