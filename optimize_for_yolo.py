#!/usr/bin/env python3
"""
Image Optimizer for YOLO8-OBB Training

This script optimizes a dataset of images and annotations for efficient
YOLO8-OBB training by:
1. Resizing images to the target dimensions
2. Converting to JPG with quality optimization
3. Organizing files in YOLO-compatible format
4. Optionally converting to grayscale

Usage:
    python optimize_for_yolo.py --input-dir images/ --annotation-dir annotations/ --output-dir yolo_dataset/
"""

import os
import cv2
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import random
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("YOLO-Optimizer")

class YOLODatasetOptimizer:
    def __init__(self, 
                 input_dir,
                 annotation_dir,
                 output_dir,
                 target_size=(1024, 1024),
                 jpeg_quality=85,
                 use_grayscale=False,
                 max_file_size_kb=300,
                 split_ratio=(0.8, 0.1, 0.1)):
        """
        Initialize the YOLO dataset optimizer
        
        Args:
            input_dir: Directory containing input images
            annotation_dir: Directory containing annotations
            output_dir: Directory to save optimized dataset
            target_size: Target image dimensions (width, height)
            jpeg_quality: JPEG quality (0-100)
            use_grayscale: Whether to convert images to grayscale
            max_file_size_kb: Maximum file size in KB
            split_ratio: Train/val/test split ratio (e.g., 0.8, 0.1, 0.1)
        """
        self.input_dir = Path(input_dir)
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.jpeg_quality = jpeg_quality
        self.use_grayscale = use_grayscale
        self.max_file_size_kb = max_file_size_kb
        self.split_ratio = split_ratio
        
        # Create output directories
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.images_train_dir = self.images_dir / 'train'
        self.images_val_dir = self.images_dir / 'val'
        self.images_test_dir = self.images_dir / 'test'
        self.labels_train_dir = self.labels_dir / 'train'
        self.labels_val_dir = self.labels_dir / 'val'
        self.labels_test_dir = self.labels_dir / 'test'
        
        for dir_path in [self.images_train_dir, self.images_val_dir, self.images_test_dir,
                         self.labels_train_dir, self.labels_val_dir, self.labels_test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Statistics
        self.stats = {
            'processed': 0,
            'skipped': 0,
            'train_count': 0,
            'val_count': 0,
            'test_count': 0,
            'avg_file_size_kb': 0,
            'total_size_mb': 0,
            'total_objects': 0,
            'avg_objects_per_image': 0
        }
    
    def find_image_annotation_pairs(self):
        """Find all image and annotation file pairs"""
        image_files = list(self.input_dir.glob('**/*.jpg')) + list(self.input_dir.glob('**/*.jpeg')) + list(self.input_dir.glob('**/*.png'))
        
        pairs = []
        for img_path in image_files:
            base_name = img_path.stem
            json_path = self.annotation_dir / f"{base_name}.json"
            
            if json_path.exists():
                pairs.append((img_path, json_path))
            else:
                logger.warning(f"No annotation found for image: {img_path}")
                
        logger.info(f"Found {len(pairs)} image-annotation pairs")
        return pairs
    
    def split_dataset(self, pairs):
        """Split dataset into train/val/test sets"""
        random.shuffle(pairs)
        
        total = len(pairs)
        train_count = int(total * self.split_ratio[0])
        val_count = int(total * self.split_ratio[1])
        
        train_pairs = pairs[:train_count]
        val_pairs = pairs[train_count:train_count + val_count]
        test_pairs = pairs[train_count + val_count:]
        
        self.stats['train_count'] = len(train_pairs)
        self.stats['val_count'] = len(val_pairs)
        self.stats['test_count'] = len(test_pairs)
        
        logger.info(f"Split dataset: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
        return train_pairs, val_pairs, test_pairs
    
    def convert_to_yolo_format(self, annotation_data, img_width, img_height):
        """
        Convert JSON annotation to YOLO format
        
        YOLO format:
        <class_id> <x_center> <y_center> <width> <height> [<rotation_angle>]
        
        All values are normalized to [0, 1]
        """
        yolo_annotations = []
        object_count = 0
        
        if 'entities' in annotation_data and 'ocr_text' in annotation_data['entities']:
            for text_entry in annotation_data['entities']['ocr_text']:
                # For simplicity, class 0 = all text
                class_id = 0
                
                # Get bounding box information
                if 'bbox' in text_entry:
                    # Regular bbox format
                    bbox = text_entry['bbox']
                    x, y = bbox['x'], bbox['y']
                    w, h = bbox['width'], bbox['height']
                    
                    # Calculate center coordinates
                    x_center = (x + w/2) / img_width
                    y_center = (y + h/2) / img_height
                    
                    # Normalize width and height
                    norm_width = w / img_width
                    norm_height = h / img_height
                    
                    # Basic validation
                    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and norm_width > 0 and norm_height > 0:
                        # Standard YOLO format (no rotation)
                        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
                        object_count += 1
                
                elif 'bounding_poly' in text_entry and 'vertices' in text_entry['bounding_poly']:
                    # Polygon format for oriented bounding boxes
                    vertices = text_entry['bounding_poly']['vertices']
                    if len(vertices) == 4:
                        # Extract coordinates
                        points = [(v['x'], v['y']) for v in vertices]
                        
                        # Calculate center, width, height, and angle
                        # This is an approximation - proper OBB calculation is more complex
                        min_x = min(p[0] for p in points)
                        max_x = max(p[0] for p in points)
                        min_y = min(p[1] for p in points)
                        max_y = max(p[1] for p in points)
                        
                        # Center coordinates
                        x_center = (min_x + max_x) / 2 / img_width
                        y_center = (min_y + max_y) / 2 / img_height
                        
                        # Width and height
                        width = (max_x - min_x) / img_width
                        height = (max_y - min_y) / img_height
                        
                        # Calculate rotation angle
                        # This is an approximation - proper OBB angle calculation would be more complex
                        dx1 = points[1][0] - points[0][0]
                        dy1 = points[1][1] - points[0][1]
                        angle = np.arctan2(dy1, dx1)
                        angle_degrees = np.degrees(angle) % 180
                        norm_angle = angle_degrees / 180.0  # Normalize to [0, 1]
                        
                        # Basic validation
                        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and width > 0 and height > 0:
                            # YOLO OBB format with rotation
                            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {norm_angle:.6f}")
                            object_count += 1
        
        return yolo_annotations, object_count
    
    def optimize_image(self, image_path, output_path):
        """Optimize an image for YOLO training"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return False, 0
            
            # Convert to grayscale if needed
            if self.use_grayscale and len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels for YOLO
            
            # Resize to target size
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Encode with quality parameter
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            success = cv2.imwrite(str(output_path), image, encode_param)
            
            if success:
                # Check file size
                file_size_kb = os.path.getsize(output_path) / 1024
                
                # Reduce quality if file is too large
                if file_size_kb > self.max_file_size_kb and self.jpeg_quality > 30:
                    current_quality = self.jpeg_quality
                    while file_size_kb > self.max_file_size_kb and current_quality > 30:
                        current_quality -= 10
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), current_quality]
                        cv2.imwrite(str(output_path), image, encode_param)
                        file_size_kb = os.path.getsize(output_path) / 1024
                
                return True, file_size_kb
            
            return False, 0
            
        except Exception as e:
            logger.error(f"Error optimizing image {image_path}: {e}")
            return False, 0
    
    def process_dataset(self):
        """Process the entire dataset"""
        start_time = datetime.now()
        logger.info(f"Starting dataset optimization at {start_time}")
        
        # Find all image-annotation pairs
        pairs = self.find_image_annotation_pairs()
        
        # Split dataset
        train_pairs, val_pairs, test_pairs = self.split_dataset(pairs)
        
        total_file_size = 0
        total_objects = 0
        
        # Process each split
        for split_name, split_pairs, images_dir, labels_dir in [
            ('train', train_pairs, self.images_train_dir, self.labels_train_dir),
            ('val', val_pairs, self.images_val_dir, self.labels_val_dir),
            ('test', test_pairs, self.images_test_dir, self.labels_test_dir)
        ]:
            logger.info(f"Processing {split_name} split: {len(split_pairs)} images")
            
            for img_path, json_path in tqdm(split_pairs, desc=f"Processing {split_name}"):
                # Create output paths
                output_img_path = images_dir / f"{img_path.stem}.jpg"
                output_label_path = labels_dir / f"{img_path.stem}.txt"
                
                # Optimize image
                success, file_size_kb = self.optimize_image(img_path, output_img_path)
                
                if success:
                    # Load annotation
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            annotation_data = json.load(f)
                        
                        # Get image dimensions
                        if 'image_size' in annotation_data:
                            orig_width = annotation_data['image_size'].get('width', self.target_size[0])
                            orig_height = annotation_data['image_size'].get('height', self.target_size[1])
                        else:
                            # Use original image dimensions
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                orig_height, orig_width = img.shape[:2]
                            else:
                                orig_width, orig_height = self.target_size
                        
                        # Convert to YOLO format
                        yolo_annotations, object_count = self.convert_to_yolo_format(
                            annotation_data, orig_width, orig_height
                        )
                        
                        # Write YOLO format labels
                        with open(output_label_path, 'w') as f:
                            for ann in yolo_annotations:
                                f.write(f"{ann}\n")
                        
                        # Update stats
                        self.stats['processed'] += 1
                        total_file_size += file_size_kb
                        total_objects += object_count
                        
                    except Exception as e:
                        logger.error(f"Error processing annotation {json_path}: {e}")
                        self.stats['skipped'] += 1
                else:
                    self.stats['skipped'] += 1
        
        # Calculate final statistics
        if self.stats['processed'] > 0:
            self.stats['avg_file_size_kb'] = total_file_size / self.stats['processed']
            self.stats['total_size_mb'] = total_file_size / 1024
            self.stats['total_objects'] = total_objects
            self.stats['avg_objects_per_image'] = total_objects / self.stats['processed']
        
        # Generate dataset.yaml file for YOLO
        self.generate_yaml()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.info(f"Dataset optimization completed in {processing_time:.1f} seconds")
        
        return self.stats
    
    def generate_yaml(self):
        """Generate dataset.yaml file for YOLO training"""
        yaml_content = f"""# YOLOv8 dataset configuration
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

path: {os.path.abspath(self.output_dir)}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (optional)

# Classes
names:
  0: text  # text regions
"""
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"Generated YOLO dataset config: {yaml_path}")
    
    def print_summary(self):
        """Print summary of the optimization process"""
        print("\n" + "="*60)
        print(f"YOLO DATASET OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Total images processed: {self.stats['processed']}")
        print(f"Images skipped: {self.stats['skipped']}")
        print(f"Dataset split:")
        print(f"  - Training:   {self.stats['train_count']} images")
        print(f"  - Validation: {self.stats['val_count']} images")
        print(f"  - Testing:    {self.stats['test_count']} images")
        print(f"Image properties:")
        print(f"  - Dimensions: {self.target_size[0]}x{self.target_size[1]} pixels")
        print(f"  - Format:     {'Grayscale JPG' if self.use_grayscale else 'RGB JPG'}")
        print(f"  - Quality:    {self.jpeg_quality}")
        print(f"File sizes:")
        print(f"  - Average size:  {self.stats['avg_file_size_kb']:.1f} KB")
        print(f"  - Total size:    {self.stats['total_size_mb']:.1f} MB")
        print(f"Objects:")
        print(f"  - Total objects: {self.stats['total_objects']}")
        print(f"  - Avg per image: {self.stats['avg_objects_per_image']:.1f}")
        print("\nDataset ready for YOLOv8-OBB training at:")
        print(f"  {os.path.abspath(self.output_dir)}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Optimize images and annotations for YOLO8-OBB training")
    
    parser.add_argument("--input-dir", "-i", required=True, 
                      help="Directory containing input images")
    parser.add_argument("--annotation-dir", "-a", required=True,
                      help="Directory containing input annotations")
    parser.add_argument("--output-dir", "-o", required=True,
                      help="Directory to save optimized dataset")
    
    parser.add_argument("--width", type=int, default=1024,
                      help="Target image width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024,
                      help="Target image height (default: 1024)")
    parser.add_argument("--quality", "-q", type=int, default=85,
                      help="JPEG quality (0-100, default: 85)")
    parser.add_argument("--grayscale", "-g", action="store_true",
                      help="Convert images to grayscale")
    parser.add_argument("--max-size", type=int, default=300,
                      help="Maximum file size in KB (default: 300)")
    
    parser.add_argument("--split", type=str, default="0.8,0.1,0.1",
                      help="Train/val/test split ratio (default: 0.8,0.1,0.1)")
                      
    args = parser.parse_args()
    
    # Parse split ratio
    try:
        split_ratio = [float(x) for x in args.split.split(',')]
        if len(split_ratio) != 3 or sum(split_ratio) != 1.0:
            logger.warning("Invalid split ratio. Must be three values that sum to 1.0. Using default 0.8,0.1,0.1")
            split_ratio = (0.8, 0.1, 0.1)
    except:
        logger.warning("Failed to parse split ratio. Using default 0.8,0.1,0.1")
        split_ratio = (0.8, 0.1, 0.1)
    
    # Initialize optimizer
    optimizer = YOLODatasetOptimizer(
        input_dir=args.input_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        target_size=(args.width, args.height),
        jpeg_quality=args.quality,
        use_grayscale=args.grayscale,
        max_file_size_kb=args.max_size,
        split_ratio=split_ratio
    )
    
    # Process dataset
    optimizer.process_dataset()
    
    # Print summary
    optimizer.print_summary()

if __name__ == "__main__":
    main()