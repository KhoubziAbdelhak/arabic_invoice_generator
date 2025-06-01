import os
import logging
from pathlib import Path
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ImageOptimizer")

class ImageOptimizer:
    """
    Utility class for optimizing images to reduce file size.
    This can be used both as a standalone utility or integrated into
    the document augmentation pipeline.
    """
    
    def __init__(self, 
                 max_size=1500,
                 jpeg_quality=85,
                 png_compression=9,
                 convert_to_jpg=True):
        """
        Initialize the image optimizer.
        
        Args:
            max_size: Maximum dimension (width or height) for the optimized images
            jpeg_quality: Quality for JPEG compression (0-100)
            png_compression: Compression level for PNG (0-9)
            convert_to_jpg: Whether to convert PNG to JPEG to save space
        """
        self.max_size = max_size
        self.jpeg_quality = jpeg_quality
        self.png_compression = png_compression
        self.convert_to_jpg = convert_to_jpg
    
    def optimize_image(self, input_path=None, output_path=None, in_place=False, image=None):
        """
        Optimize a single image
        
        Args:
            input_path: Path to the input image file (optional if image data is provided)
            output_path: Path to save the optimized image (required if image data is provided)
            in_place: Whether to overwrite the input file
            image: Image data as numpy array (optional, if provided input_path is not read)
            
        Returns:
            tuple: (success, original_size, new_size, output_path)
        """
        try:
            # Check if we have image data or need to read from file
            if image is None and input_path is None:
                logger.error("Either input_path or image must be provided")
                return False, 0, 0, None
                
            # If image data is provided directly
            if image is not None:
                if output_path is None:
                    logger.error("output_path is required when providing image data directly")
                    return False, 0, 0, None
                    
                # Estimate original size from image dimensions
                height, width = image.shape[:2]
                channels = 3 if len(image.shape) == 3 else 1
                original_size = width * height * channels
                
                # Ensure output directory exists
                output_path = Path(output_path)
                os.makedirs(output_path.parent, exist_ok=True)
            else:
                # Process from file path
                input_path = Path(input_path)
                
                # Handle output path
                if in_place:
                    output_path = input_path
                elif output_path is None:
                    # Create output path with '_optimized' suffix if not provided
                    if self.convert_to_jpg and input_path.suffix.lower() in ['.png']:
                        output_path = input_path.with_stem(f"{input_path.stem}_optimized").with_suffix(".jpg")
                    else:
                        output_path = input_path.with_stem(f"{input_path.stem}_optimized")
                else:
                    output_path = Path(output_path)
                
                # Ensure output directory exists
                os.makedirs(output_path.parent, exist_ok=True)
                
                # Read image
                image = cv2.imread(str(input_path))
                if image is None:
                    logger.error(f"Failed to read image: {input_path}")
                    return False, 0, 0, None
                
                # Get original size
                original_size = os.path.getsize(input_path)
                height, width = image.shape[:2]
            
            # Resize if necessary
            if max(width, height) > self.max_size:
                scale = self.max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Save with appropriate settings
            if self.convert_to_jpg and input_path.suffix.lower() in ['.png']:
                # If output path still has png extension, change it to jpg
                if output_path.suffix.lower() == '.png':
                    output_path = output_path.with_suffix('.jpg')
                
                # Save as JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                cv2.imwrite(str(output_path), image, encode_param)
            else:
                # Save with original format but optimized
                if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                    cv2.imwrite(str(output_path), image, encode_param)
                elif output_path.suffix.lower() == '.png':
                    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), self.png_compression]
                    cv2.imwrite(str(output_path), image, encode_param)
                else:
                    # For other formats, just save
                    cv2.imwrite(str(output_path), image)
            
            # Get new size
            new_size = os.path.getsize(output_path)
            
            # Calculate percentage reduction
            reduction = (1 - (new_size / original_size)) * 100
            if input_path:
                logger.debug(f"Optimized {input_path.name}: {original_size/1024:.1f}KB → {new_size/1024:.1f}KB ({reduction:.1f}% smaller)")
            else:
                logger.debug(f"Optimized image: {original_size/1024:.1f}KB → {new_size/1024:.1f}KB ({reduction:.1f}% smaller)")
            
            return True, original_size, new_size, output_path
            
        except Exception as e:
            if input_path:
                logger.error(f"Error optimizing {input_path}: {e}")
            else:
                logger.error(f"Error optimizing image: {e}")
            return False, 0, 0, None
    
    def batch_optimize(self, input_dir, output_dir=None, extensions=None):
        """
        Optimize all images in a directory
        
        Args:
            input_dir: Directory containing images to optimize
            output_dir: Directory to save optimized images (if None, will create 'optimized' subdirectory)
            extensions: List of file extensions to process (default: ['.jpg', '.jpeg', '.png'])
            
        Returns:
            tuple: (success_count, total_count, original_size_bytes, new_size_bytes)
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            logger.error(f"Input directory {input_dir} does not exist")
            return 0, 0, 0, 0
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = input_dir / "optimized"
        else:
            output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Default extensions if not provided
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png']
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(list(input_dir.glob(f"*{ext}")))
            image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))
        
        total_files = len(image_files)
        if total_files == 0:
            logger.warning(f"No image files found in {input_dir}")
            return 0, 0, 0, 0
        
        logger.info(f"Found {total_files} images to optimize")
        
        # Process each image
        success_count = 0
        total_original_size = 0
        total_new_size = 0
        
        for i, image_path in enumerate(image_files):
            logger.info(f"Processing image {i+1}/{total_files}: {image_path.name}")
            
            # Determine output path
            output_path = output_dir / image_path.name
            if self.convert_to_jpg and image_path.suffix.lower() == '.png':
                output_path = output_path.with_suffix('.jpg')
            
            # Optimize the image
            success, original_size, new_size, _ = self.optimize_image(image_path, output_path)
            
            if success:
                success_count += 1
                total_original_size += original_size
                total_new_size += new_size
        
        # Print summary
        if success_count > 0:
            total_reduction = (1 - (total_new_size / total_original_size)) * 100
            original_mb = total_original_size / (1024 * 1024)
            new_mb = total_new_size / (1024 * 1024)
            
            logger.info(f"Optimization complete: {success_count}/{total_files} files processed successfully")
            logger.info(f"Total size reduction: {original_mb:.2f} MB → {new_mb:.2f} MB ({total_reduction:.1f}% smaller)")
        
        return success_count, total_files, total_original_size, total_new_size


# Standalone usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize images to reduce file size")
    parser.add_argument("--input", "-i", required=True, 
                        help="Input image file or directory")
    parser.add_argument("--output", "-o", 
                        help="Output path (file or directory depending on input)")
    parser.add_argument("--max-size", "-s", type=int, default=1500, 
                        help="Maximum dimension (width or height) for the optimized images")
    parser.add_argument("--jpeg-quality", "-j", type=int, default=85, 
                        help="Quality for JPEG compression (0-100)")
    parser.add_argument("--png-compression", "-p", type=int, default=9, 
                        help="Compression level for PNG (0-9)")
    parser.add_argument("--keep-png", action="store_false", dest="convert_to_jpg",
                        help="Keep PNG format instead of converting to JPEG")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite input files (only applies to single file mode)")
    
    args = parser.parse_args()
    
    optimizer = ImageOptimizer(
        max_size=args.max_size,
        jpeg_quality=args.jpeg_quality,
        png_compression=args.png_compression,
        convert_to_jpg=args.convert_to_jpg
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file mode
        success, original_size, new_size, output_path = optimizer.optimize_image(
            input_path, 
            args.output, 
            in_place=args.in_place
        )
        
        if success:
            reduction = (1 - (new_size / original_size)) * 100
            print(f"Optimized {input_path.name}: {original_size/1024:.1f}KB → {new_size/1024:.1f}KB ({reduction:.1f}% smaller)")
            print(f"Saved to: {output_path}")
        else:
            print(f"Failed to optimize {input_path}")
    
    elif input_path.is_dir():
        # Directory mode
        optimizer.batch_optimize(input_path, args.output)
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")