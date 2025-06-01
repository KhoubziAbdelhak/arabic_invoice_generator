import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from config.config import *

class OCRDatasetPreparer:
    def __init__(self, 
                 images_dir=IMAGES_DIR,
                 annotations_dir=ANNOTATIONS_DIR,
                 augmented_images_dir=AUGMENTED_IMAGES_DIR,
                 augmented_annotations_dir=AUGMENTED_ANNOTATIONS_DIR,
                 output_dir=os.path.join(BASE_DIR, 'ocr_dataset')):
        """
        Initialize the OCR dataset preparer with source and destination directories.
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.augmented_images_dir = augmented_images_dir
        self.augmented_annotations_dir = augmented_annotations_dir
        self.output_dir = output_dir
        
        # Create output directories
        self.train_images_dir = os.path.join(output_dir, 'train', 'images')
        self.train_labels_dir = os.path.join(output_dir, 'train', 'labels')
        self.val_images_dir = os.path.join(output_dir, 'val', 'images')
        self.val_labels_dir = os.path.join(output_dir, 'val', 'labels')
        self.test_images_dir = os.path.join(output_dir, 'test', 'images')
        self.test_labels_dir = os.path.join(output_dir, 'test', 'labels')
        
        for directory in [self.train_images_dir, self.train_labels_dir,
                         self.val_images_dir, self.val_labels_dir,
                         self.test_images_dir, self.test_labels_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Set split ratios
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
    
    def collect_all_data(self):
        """
        Collect all image paths and their corresponding annotation paths.
        """
        data_pairs = []
        
        # Regular images
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in image_files:
            img_path = os.path.join(self.images_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            
            # Find corresponding annotation file
            ann_file = f"{base_name}.json"
            ann_path = os.path.join(self.annotations_dir, ann_file)
            
            if os.path.exists(ann_path):
                data_pairs.append((img_path, ann_path))
        
        # Augmented images
        if os.path.exists(self.augmented_images_dir):
            aug_image_files = [f for f in os.listdir(self.augmented_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in aug_image_files:
                img_path = os.path.join(self.augmented_images_dir, img_file)
                base_name = os.path.splitext(img_file)[0]
                
                # Find corresponding annotation file
                ann_file = f"{base_name}.json"
                ann_path = os.path.join(self.augmented_annotations_dir, ann_file)
                
                if os.path.exists(ann_path):
                    data_pairs.append((img_path, ann_path))
        
        return data_pairs
    
    def split_dataset(self, data_pairs):
        """
        Split the dataset into train, validation, and test sets.
        """
        random.shuffle(data_pairs)
        total = len(data_pairs)
        
        train_size = int(total * self.train_ratio)
        val_size = int(total * self.val_ratio)
        
        train_pairs = data_pairs[:train_size]
        val_pairs = data_pairs[train_size:train_size + val_size]
        test_pairs = data_pairs[train_size + val_size:]
        
        return train_pairs, val_pairs, test_pairs
    
    def convert_to_ocr_format(self, annotation_path):
        """
        Convert the annotation JSON to a format suitable for OCR training.
        Returns a list of (text, coordinates) tuples.
        """
        ocr_annotations = []
        
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle the specific structure of your JSON annotations
            if "entities" in data and "ocr_text" in data["entities"]:
                for item in data["entities"]["ocr_text"]:
                    if 'text' in item and 'bbox' in item:
                        text = item['text']
                        
                        # Skip empty text
                        if not text or not text.strip():
                            continue
                        
                        # Extract bbox coordinates
                        if isinstance(item['bbox'], dict) and all(k in item['bbox'] for k in ['x', 'y', 'width', 'height']):
                            bbox = [
                                item['bbox']['x'],
                                item['bbox']['y'],
                                item['bbox']['width'],
                                item['bbox']['height']
                            ]
                        else:
                            # If bbox is already in the expected format
                            bbox = item['bbox']
                        
                        # For standard OCR models, we need text and its bounding coordinates
                        ocr_annotations.append((text, bbox))
            elif isinstance(data, list):
                # Try the original format as a fallback
                for item in data:
                    if 'text' in item and 'bbox' in item:
                        text = item['text']
                        coords = item['bbox']
                        
                        # Skip empty text
                        if not text.strip():
                            continue
                        
                        # For standard OCR models, we need text and its bounding coordinates
                        ocr_annotations.append((text, coords))
                        
            print(f"Extracted {len(ocr_annotations)} text regions from {annotation_path}")
        except Exception as e:
            print(f"Error processing annotation {annotation_path}: {e}")
            import traceback
            print(traceback.format_exc())
        
        return ocr_annotations
    
    def create_line_images(self, image_path, annotations, output_dir, image_name):
        """
        Extract line images from the document based on bounding boxes.
        Creates individual images for each text line and a corresponding ground truth file.
        """
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {image_path}")
                return []
            
            # Create ground truth file path
            base_name = os.path.splitext(image_name)[0]
            gt_filename = os.path.join(output_dir.replace('images', 'labels'), f"{base_name}.txt")
            
            created_files = []
            
            with open(gt_filename, 'w', encoding='utf-8') as gt_file:
                for idx, (text, bbox) in enumerate(annotations):
                    try:
                        # Extract coordinates
                        if len(bbox) == 4:  # [x, y, width, height]
                            # Handle both float and int values
                            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                            x1, y1 = int(x), int(y)
                            x2, y2 = int(x + w), int(y + h)
                        elif isinstance(bbox, dict) and all(k in bbox for k in ['x', 'y', 'width', 'height']):
                            # Handle dictionary format
                            x, y = float(bbox['x']), float(bbox['y'])
                            w, h = float(bbox['width']), float(bbox['height'])
                            x1, y1 = int(x), int(y)
                            x2, y2 = int(x + w), int(y + h)
                        elif len(bbox) == 8:  # [x1, y1, x2, y2, x3, y3, x4, y4]
                            # Convert polygon to rectangle (simplified approach)
                            x_coords = [float(bbox[i]) for i in range(0, len(bbox), 2)]
                            y_coords = [float(bbox[i]) for i in range(1, len(bbox), 2)]
                            x1, y1 = int(min(x_coords)), int(min(y_coords))
                            x2, y2 = int(max(x_coords)), int(max(y_coords))
                        else:
                            print(f"Unsupported bbox format: {bbox}")
                            continue
                        
                        # Ensure coordinates are within image bounds
                        h, w = img.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        # Skip invalid boxes
                        if x2 <= x1 or y2 <= y1 or x2-x1 < 5 or y2-y1 < 5:
                            print(f"Skipping invalid box: {x1},{y1},{x2},{y2}")
                            continue
                        
                        # Extract the line image
                        line_img = img[y1:y2, x1:x2]
                        
                        # Skip empty or invalid images
                        if line_img.size == 0:
                            continue
                        
                        # Create a unique filename for this line
                        line_filename = f"{base_name}_{idx}.png"
                        line_path = os.path.join(output_dir, line_filename)
                        
                        # Save the line image
                        cv2.imwrite(line_path, line_img)
                        created_files.append(line_path)
                        
                        # Write to ground truth file: filename and text
                        gt_file.write(f"{line_filename}\t{text}\n")
                    except Exception as e:
                        print(f"Error processing annotation {idx}: {e}")
                        continue
            
            created_files.append(gt_filename)
            return created_files
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            import traceback
            print(traceback.format_exc())
            return []
    
    def process_data_split(self, data_pairs, images_dir, labels_dir):
        """
        Process a data split (train/val/test) and create the OCR dataset structure.
        """
        processed_files = []
        
        for img_path, ann_path in tqdm(data_pairs, desc=f"Processing {Path(images_dir).name} set"):
            # Extract file name
            img_name = os.path.basename(img_path)
            
            # Convert annotations to OCR format
            ocr_annotations = self.convert_to_ocr_format(ann_path)
            
            # Create line images and ground truth file
            created_files = self.create_line_images(img_path, ocr_annotations, images_dir, img_name)
            processed_files.extend(created_files)
            
            # Copy full page image (optional)
            # full_img_dest = os.path.join(images_dir, img_name)
            # shutil.copy2(img_path, full_img_dest)
            # processed_files.append(full_img_dest)
        
        return processed_files
    
    def prepare_dataset(self):
        """
        Main method to prepare the OCR dataset.
        """
        print("Collecting data...")
        data_pairs = self.collect_all_data()
        print(f"Found {len(data_pairs)} image-annotation pairs")
        
        print("Splitting dataset...")
        train_pairs, val_pairs, test_pairs = self.split_dataset(data_pairs)
        
        print(f"Processing training set ({len(train_pairs)} samples)...")
        train_files = self.process_data_split(train_pairs, self.train_images_dir, self.train_labels_dir)
        
        print(f"Processing validation set ({len(val_pairs)} samples)...")
        val_files = self.process_data_split(val_pairs, self.val_images_dir, self.val_labels_dir)
        
        print(f"Processing test set ({len(test_pairs)} samples)...")
        test_files = self.process_data_split(test_pairs, self.test_images_dir, self.test_labels_dir)
        
        print(f"Created {len(train_files) + len(val_files) + len(test_files)} files")
        print(f"Dataset prepared in {self.output_dir}")
        
        # Create dataset statistics
        stats = {
            "train_samples": len(train_pairs),
            "val_samples": len(val_pairs),
            "test_samples": len(test_pairs),
            "train_files": len(train_files),
            "val_files": len(val_files),
            "test_files": len(test_files)
        }
        
        with open(os.path.join(self.output_dir, "dataset_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        return stats

def create_tesseract_training_files():
    """
    Creates additional files needed for Tesseract OCR training.
    This is a simplified version - actual Tesseract training requires more setup.
    """
    ocr_dataset_dir = os.path.join(BASE_DIR, 'ocr_dataset')
    
    # Create box files and lstmf files directories
    box_dir = os.path.join(ocr_dataset_dir, 'box_files')
    lstmf_dir = os.path.join(ocr_dataset_dir, 'lstmf_files')
    
    os.makedirs(box_dir, exist_ok=True)
    os.makedirs(lstmf_dir, exist_ok=True)
    
    print("For Tesseract training, you'll need to:")
    print("1. Convert images to specific formats (tif recommended)")
    print("2. Generate box files for each image")
    print("3. Run text2image to generate training data")
    print("4. Create lstmf files")
    print("5. Create a training list file")
    print("6. Run training with lstmtraining")
    
    # This would require additional processing specific to Tesseract

def create_easyocr_compatible_dataset():
    """
    Prepare dataset in a format compatible with EasyOCR fine-tuning
    """
    ocr_dataset_dir = os.path.join(BASE_DIR, 'ocr_dataset')
    easyocr_dir = os.path.join(ocr_dataset_dir, 'easyocr')
    os.makedirs(easyocr_dir, exist_ok=True)
    
    # For EasyOCR, we need a simple annotation format:
    # image_path\ttext
    total_samples = 0
    for split in ['train', 'val']:
        images_dir = os.path.join(ocr_dataset_dir, split, 'images')
        labels_dir = os.path.join(ocr_dataset_dir, split, 'labels')
        
        annotation_file = os.path.join(easyocr_dir, f"{split}_annotations.txt")
        samples_count = 0
        
        with open(annotation_file, 'w', encoding='utf-8') as f:
            for label_file in os.listdir(labels_dir):
                if label_file.endswith('.txt'):
                    label_path = os.path.join(labels_dir, label_file)
                    
                    with open(label_path, 'r', encoding='utf-8') as lf:
                        for line in lf:
                            parts = line.strip().split('\t')
                            if len(parts) == 2:
                                img_name, text = parts
                                img_path = os.path.join(images_dir, img_name)
                                if os.path.exists(img_path):
                                    f.write(f"{img_path}\t{text}\n")
                                    samples_count += 1
        
        total_samples += samples_count
        print(f"Created {samples_count} samples in {split} set")
    
    print(f"Created EasyOCR-compatible dataset with {total_samples} total samples in {easyocr_dir}")

if __name__ == "__main__":
    print("Starting OCR dataset preparation...")
    
    # Prepare the main dataset
    preparer = OCRDatasetPreparer()
    stats = preparer.prepare_dataset()
    
    # Create format-specific files
    create_easyocr_compatible_dataset()
    
    print("OCR dataset preparation complete!")
    print(f"Dataset statistics: {stats}")
    
    # Check if we have any data
    if stats["train_files"] > 0:
        print("\nRecommendations for fine-tuning:")
        print("1. For EasyOCR: Use the files in ocr_dataset/easyocr")
        print("2. For Tesseract: Additional processing required, see documentation")
        print("3. For Transformer-based OCR: Use the train/val/test splits directly")
    else:
        print("\nWARNING: No data was processed. Check that your annotation files and images exist and match.")
        print("Run the debug_annotations.py script to diagnose problems with your annotation format.")