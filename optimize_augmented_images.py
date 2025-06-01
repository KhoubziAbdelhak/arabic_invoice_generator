#!/usr/bin/env python3
"""
Optimize augmented images to reduce file size.
This script takes images from the output/augmented_images directory,
optimizes them and saves them to output/optimized_images directory.
"""

import os
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ImageOptimizer")

class ImageOptimizer:
    def __init__(self,
                 input_dir="output/augmented_images",
                 output_dir="output/optimized_augmented_images",
                 max_size=1500,
                 jpeg_quality=85,
                 png_compression=9,
                 convert_to_jpg=True):
        """
        Initialize the image optimizer.

        Args:
            input_dir: Directory containing images to optimize
            output_dir: Directory to save optimized images
            max_size: Maximum dimension (width or height) for the optimized images
            jpeg_quality: Quality for JPEG compression (0-100)
            png_compression: Compression level for PNG (0-9)
            convert_to_jpg: Whether to convert PNG to JPEG to save space
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_size = max_size
        self.jpeg_quality = jpeg_quality
        self.png_compression = png_compression
        self.convert_to_jpg = convert_to_jpg

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Check if input directory exists
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory {self.input_dir} not found.")

    def get_image_files(self):
        """Get all image files from input directory"""
        extensions = ['.jpg', '.jpeg', '.png']
        image_files = []

        for ext in extensions:
            image_files.extend(list(self.input_dir.glob(f"*{ext}")))
            image_files.extend(list(self.input_dir.glob(f"*{ext.upper()}")))

        return image_files

    def optimize_image(self, image_path):
        """
        Optimize a single image

        Args:
            image_path: Path to the image file

        Returns:
            tuple: (success, original_size, new_size, output_path)
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return False, 0, 0, None

            # Get original size
            original_size = os.path.getsize(image_path)
            height, width = image.shape[:2]

            # Resize if necessary
            if max(width, height) > self.max_size:
                scale = self.max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Determine output path and format
            stem = image_path.stem
            if self.convert_to_jpg and image_path.suffix.lower() in ['.png']:
                output_path = self.output_dir / f"{stem}.jpg"
                # Save as JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                cv2.imwrite(str(output_path), image, encode_param)
            else:
                # Keep original format
                output_path = self.output_dir / image_path.name

                if image_path.suffix.lower() in ['.jpg', '.jpeg']:
                    # Save as JPEG with specified quality
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                    cv2.imwrite(str(output_path), image, encode_param)
                elif image_path.suffix.lower() == '.png':
                    # Save as PNG with specified compression
                    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), self.png_compression]
                    cv2.imwrite(str(output_path), image, encode_param)
                else:
                    # For other formats, just save
                    cv2.imwrite(str(output_path), image)

            # Get new size
            new_size = os.path.getsize(output_path)

            return True, original_size, new_size, output_path

        except Exception as e:
            logger.error(f"Error optimizing {image_path}: {e}")
            return False, 0, 0, None

    def process_all_images(self, num_workers=4):
        """Process all images in the input directory using multiple workers"""
        image_files = self.get_image_files()
        total_files = len(image_files)

        if total_files == 0:
            logger.warning(f"No image files found in {self.input_dir}")
            return

        logger.info(f"Found {total_files} images to optimize")

        total_original_size = 0
        total_new_size = 0
        successful_files = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=total_files, desc="Optimizing images") as pbar:
                futures = {executor.submit(self.optimize_image, img_path): img_path for img_path in image_files}

                for future in futures:
                    success, original_size, new_size, output_path = future.result()
                    if success:
                        successful_files += 1
                        total_original_size += original_size
                        total_new_size += new_size

                        # Calculate percentage reduction
                        reduction = (1 - (new_size / original_size)) * 100
                        pbar.set_postfix({"reduction": f"{reduction:.1f}%"})

                    pbar.update(1)

        # Print summary
        logger.info(f"Optimization complete: {successful_files}/{total_files} files processed successfully")

        if successful_files > 0:
            total_reduction = (1 - (total_new_size / total_original_size)) * 100
            original_mb = total_original_size / (1024 * 1024)
            new_mb = total_new_size / (1024 * 1024)

            logger.info(f"Total size reduction: {original_mb:.2f} MB → {new_mb:.2f} MB ({total_reduction:.1f}% smaller)")


def main():
    parser = argparse.ArgumentParser(description="Optimize images to reduce file size")
    parser.add_argument("--input", "-i", default="output/augmented_images",
                        help="Directory containing images to optimize")
    parser.add_argument("--output", "-o", default="output/optimized_images",
                        help="Directory to save optimized images")
    parser.add_argument("--max-size", "-s", type=int, default=1500,
                        help="Maximum dimension (width or height) for the optimized images")
    parser.add_argument("--jpeg-quality", "-j", type=int, default=85,
                        help="Quality for JPEG compression (0-100)")
    parser.add_argument("--png-compression", "-p", type=int, default=9,
                        help="Compression level for PNG (0-9)")
    parser.add_argument("--keep-png", action="store_false", dest="convert_to_jpg",
                        help="Keep PNG format instead of converting to JPEG")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of worker processes to use for parallel processing")

    args = parser.parse_args()

    optimizer = ImageOptimizer(
        input_dir=args.input,
        output_dir=args.output,
        max_size=args.max_size,
        jpeg_quality=args.jpeg_quality,
        png_compression=args.png_compression,
        convert_to_jpg=args.convert_to_jpg
    )

    optimizer.process_all_images(num_workers=args.workers)


if __name__ == "__main__":
    main()
