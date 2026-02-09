"""
Extract images from h5 files and save as individual PNG files
"""

import h5py
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from pathlib import Path
import cv2
from tqdm import tqdm


def extract_and_save_images(
    h5_file_path,
    output_dir,
    target_size=(224, 224),
    format='png'
    ):
    """
    Extract images from h5 file and save as individual files.

    Args:
        h5_file_path: path to h5 file
        output_dir: directory to save images
        target_size: resize images to this size (height, width)
        format: image format ('png', 'jpg', etc.)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get filename without extension
    filename = Path(h5_file_path).stem

    print(f"Processing {h5_file_path}")
    print(f"Output directory: {output_dir}")

    # Open h5 file
    with h5py.File(h5_file_path, 'r') as f:
        # Load images and barcodes
        imgs = f['img'][:]
        barcodes = f['barcode'][:]

        # Decode barcodes properly
        decoded_barcodes = []
        for b in barcodes:
            if isinstance(b, bytes):
                # Decode bytes to string
                decoded_barcodes.append(b.decode('utf-8')[3:-2])
            elif isinstance(b, str):
                # Already a string
                decoded_barcodes.append(b[3:-2])
            else:
                # Convert to string (fallback)
                decoded_barcodes.append(str(b)[3:-2])


        barcodes = decoded_barcodes

        # Save each image
        for idx, (img, barcode) in enumerate(tqdm(zip(imgs, barcodes), total=len(imgs))):
            # Create filename: filename_barcode.png
            image_filename = f"{filename}_{barcode}.{format}"
            image_path = os.path.join(output_dir, image_filename)

            # Handle different image formats
            if img.dtype == np.float32 or img.dtype == np.float64:
                # Normalize to 0-255
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)

            # Convert to PIL Image
            if len(img.shape) == 2:
                # Grayscale
                pil_img = Image.fromarray(img, mode='L')
            elif img.shape[2] == 3:
                # RGB
                pil_img = Image.fromarray(img, mode='RGB')
            elif img.shape[2] == 4:
                # RGBA
                pil_img = Image.fromarray(img, mode='RGBA')
            else:
                print(f"Warning: Unexpected image shape {img.shape}, skipping")
                continue

            # Resize to target size
            if target_size:
                pil_img = pil_img.resize(target_size, Image.BILINEAR)

            # Save image
            pil_img.save(image_path)

        print(f"\nSaved {len(imgs)} images to {output_dir}")


def process_multiple_h5_files(
    h5_dir,
    output_dir,
    target_size=(224, 224),
    format='png'
    ):
    """
    Process multiple h5 files in a directory.

    Args:
        h5_dir: directory containing h5 files
        output_dir: directory to save images
        target_size: resize images to this size
        format: image format
    """
    # Find all h5 files
    h5_files = list(Path(h5_dir).glob('*.h5'))

    print(f"Found {len(h5_files)} h5 files")

    for h5_file in tqdm(h5_files):
        try:
            extract_and_save_images(
                str(h5_file),
                output_dir,
                target_size,
                format
            )
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract images from h5 files')
    parser.add_argument('--input', type=str, default='./pt_dataset/hest_data/patches',
                       help='Input h5 file or directory containing h5 files')
    parser.add_argument('--output', type=str, default='./pt_dataset/st_images',
                       help='Output directory for images')
    parser.add_argument('--size', type=int, nargs=2, default=[224, 224],
                       help='Target image size (height width)')
    parser.add_argument('--format', type=str, default='png',
                       choices=['png', 'jpg', 'jpeg'],
                       help='Image format')

    args = parser.parse_args()

    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single file
        extract_and_save_images(
            args.input,
            args.output,
            target_size=tuple(args.size),
            format=args.format
        )
    elif os.path.isdir(args.input):
        # Directory
        process_multiple_h5_files(
            args.input,
            args.output,
            target_size=tuple(args.size),
            format=args.format
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")
