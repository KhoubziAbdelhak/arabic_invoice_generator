import os
import json
import cv2
import numpy as np
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

class CorrectedOCRDatasetPreparer:
    def __init__(self, images_dir, annotations_dir, output_dir, padding=0, min_height=10, min_width=10):
        """
        Initialize the OCR dataset preparer for corrected bounding boxes.
        
        Args:
            images_dir: Directory containing the original images
            annotations_dir: Directory containing the corrected annotations
            output_dir: Base directory to save the prepared dataset
            padding: Additional padding to apply when cropping (can be negative)
            min_height: Minimum height for valid text regions
            min_width: Minimum width for valid text regions
        """
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)
        self.padding = padding
        self.min_height = min_height
        self.min_width = min_width
        
        # Create output directories
        self.images_output_dir = self.output_dir / 'images'
        self.labels_output_dir = self.output_dir / 'labels'
        
        self.images_output_dir.mkdir(exist_ok=True, parents=True)
        self.labels_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Statistics
        self.total_images = 0
        self.total_text_regions = 0
        self.valid_text_regions = 0
        self.skipped_images = 0
        
    def prepare_dataset(self):
        """Prepare the OCR dataset using corrected annotations"""
        # Find all annotation files
        annotation_files = list(self.annotations_dir.glob('*.json'))
        print(f"Found {len(annotation_files)} annotation files")
        
        # Process each annotation file
        for ann_file in tqdm(annotation_files):
            self.process_annotation(ann_file)
            
        # Print statistics
        print("\nDataset preparation complete:")
        print(f"Total images processed: {self.total_images}")
        print(f"Total text regions found: {self.total_text_regions}")
        print(f"Valid text regions extracted: {self.valid_text_regions}")
        print(f"Skipped images: {self.skipped_images}")
        print(f"Dataset saved to: {self.output_dir}")
            
    def process_annotation(self, annotation_file):
        """Process a single annotation file and extract text regions"""
        try:
            # Find corresponding image file
            base_name = annotation_file.stem
            image_file = None
            
            for ext in ['.png', '.jpg', '.jpeg']:
                img_path = self.images_dir / f"{base_name}{ext}"
                if img_path.exists():
                    image_file = img_path
                    break
                    
            if image_file is None:
                print(f"Could not find image for {annotation_file}")
                self.skipped_images += 1
                return
                
            # Read the image
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"Failed to read image: {image_file}")
                self.skipped_images += 1
                return
                
            image_height, image_width = img.shape[:2]
            
            # Read the annotation
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.total_images += 1
            
            # Create ground truth file
            gt_file_path = self.labels_output_dir / f"{base_name}.txt"
            
            with open(gt_file_path, 'w', encoding='utf-8') as gt_file:
                # Extract text and bounding boxes
                if "entities" in data and "ocr_text" in data["entities"]:
                    for idx, item in enumerate(data["entities"]["ocr_text"]):
                        self.total_text_regions += 1
                        
                        if 'text' not in item or not item['text'].strip():
                            continue
                            
                        if 'bbox' not in item:
                            continue
                            
                        # Extract bbox
                        bbox = item['bbox']
                        if not isinstance(bbox, dict) or not all(k in bbox for k in ['x', 'y', 'width', 'height']):
                            continue
                            
                        # Get coordinates with padding
                        x = max(0, int(bbox['x']) - self.padding)
                        y = max(0, int(bbox['y']) - self.padding)
                        w = min(image_width - x, int(bbox['width']) + 2 * self.padding)
                        h = min(image_height - y, int(bbox['height']) + 2 * self.padding)
                        
                        # Skip if too small
                        if w < self.min_width or h < self.min_height:
                            continue
                            
                        # Crop the region
                        text_img = img[y:y+h, x:x+w]
                        
                        # Skip if empty
                        if text_img.size == 0:
                            continue
                            
                        # Save the cropped image
                        text_img_path = f"{base_name}_{idx}.png"
                        cv2.imwrite(str(self.images_output_dir / text_img_path), text_img)
                        
                        # Write to ground truth file
                        gt_file.write(f"{text_img_path}\t{item['text']}\n")
                        
                        self.valid_text_regions += 1
                
        except Exception as e:
            print(f"Error processing {annotation_file}: {e}")
            import traceback
            print(traceback.format_exc())
            self.skipped_images += 1
            
    def create_train_val_split(self, val_ratio=0.2):
        """Create train/val split for the dataset"""
        # Create train/val directories
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'
        
        for split_dir in [train_dir, val_dir]:
            (split_dir / 'images').mkdir(exist_ok=True, parents=True)
            (split_dir / 'labels').mkdir(exist_ok=True, parents=True)
            
        # Get all ground truth files
        gt_files = list(self.labels_output_dir.glob('*.txt'))
        
        # Shuffle the files
        np.random.shuffle(gt_files)
        
        # Split into train and val
        split_idx = int(len(gt_files) * (1 - val_ratio))
        train_files = gt_files[:split_idx]
        val_files = gt_files[split_idx:]
        
        print(f"\nCreating train/val split:")
        print(f"Training samples: {len(train_files)}")
        print(f"Validation samples: {len(val_files)}")
        
        # Process train files
        for gt_file in tqdm(train_files, desc="Processing training files"):
            self._copy_sample_to_split(gt_file, train_dir)
            
        # Process val files
        for gt_file in tqdm(val_files, desc="Processing validation files"):
            self._copy_sample_to_split(gt_file, val_dir)
            
    def _copy_sample_to_split(self, gt_file, split_dir):
        """Copy a sample (ground truth file and its images) to the split directory"""
        try:
            # Copy ground truth file
            shutil.copy(gt_file, split_dir / 'labels' / gt_file.name)
            
            # Read ground truth file to get image filenames
            with open(gt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) < 1:
                        continue
                        
                    img_file = parts[0]
                    
                    # Copy image file
                    src_img_path = self.images_output_dir / img_file
                    if src_img_path.exists():
                        shutil.copy(src_img_path, split_dir / 'images' / img_file)
                        
        except Exception as e:
            print(f"Error copying sample {gt_file}: {e}")

def create_easyocr_compatible_dataset(dataset_dir):
    """
    Create an EasyOCR-compatible dataset structure from the prepared OCR dataset.
    
    Args:
        dataset_dir: Base directory of the prepared dataset
    """
    dataset_dir = Path(dataset_dir)
    
    for split in ['train', 'val']:
        split_dir = dataset_dir / split
        
        if not split_dir.exists():
            print(f"Split directory {split_dir} not found. Skipping.")
            continue
            
        # Create EasyOCR directory structure
        easyocr_dir = dataset_dir / f"easyocr_{split}"
        (easyocr_dir / 'images').mkdir(exist_ok=True, parents=True)
        
        # Create label file
        label_file = easyocr_dir / f"{split}_label.txt"
        
        with open(label_file, 'w', encoding='utf-8') as out_f:
            # Read all ground truth files
            gt_files = list((split_dir / 'labels').glob('*.txt'))
            
            for gt_file in tqdm(gt_files, desc=f"Creating EasyOCR {split} dataset"):
                with open(gt_file, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        if not line.strip():
                            continue
                            
                        parts = line.strip().split('\t')
                        if len(parts) < 2:
                            continue
                            
                        img_file, text = parts[0], parts[1]
                        
                        # Copy image
                        src_img_path = split_dir / 'images' / img_file
                        if src_img_path.exists():
                            shutil.copy(src_img_path, easyocr_dir / 'images' / img_file)
                            
                            # Write to label file
                            out_f.write(f"images/{img_file}\t{text}\n")
                            
    print(f"EasyOCR compatible dataset created in {dataset_dir}")

def main():
    parser = argparse.ArgumentParser(description='Prepare OCR dataset with corrected bounding boxes')
    
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Directory containing original images')
    parser.add_argument('--annotations-dir', type=str, required=True,
                        help='Directory containing corrected annotations')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save the prepared dataset')
    parser.add_argument('--padding', type=int, default=0,
                        help='Additional padding to apply when cropping (can be negative to tighten)')
    parser.add_argument('--min-height', type=int, default=10,
                        help='Minimum height for valid text regions')
    parser.add_argument('--min-width', type=int, default=10,
                        help='Minimum width for valid text regions')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Ratio of validation samples (default: 0.2)')
    parser.add_argument('--easyocr-format', action='store_true',
                        help='Create EasyOCR-compatible dataset')
    
    args = parser.parse_args()
    
    # Create dataset preparer
    preparer = CorrectedOCRDatasetPreparer(
        args.images_dir,
        args.annotations_dir,
        args.output_dir,
        args.padding,
        args.min_height,
        args.min_width
    )
    
    # Prepare the dataset
    preparer.prepare_dataset()
    
    # Create train/val split
    preparer.create_train_val_split(args.val_ratio)
    
    # Create EasyOCR compatible dataset if requested
    if args.easyocr_format:
        create_easyocr_compatible_dataset(args.output_dir)

if __name__ == "__main__":
    main()