import os
import json
import cv2
import numpy as np
from pathlib import Path
from config.config import *

def inspect_annotation_structure(annotation_path):
    """
    Examine the structure of an annotation file and print its key elements
    """
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\nInspecting annotation: {annotation_path}")
        
        # Print the top-level keys
        print(f"Top-level keys: {list(data.keys())}")
        
        # Check for entities structure
        if "entities" in data:
            print(f"Entity types: {list(data['entities'].keys())}")
            
            # Check for OCR text
            if "ocr_text" in data["entities"]:
                ocr_items = data["entities"]["ocr_text"]
                print(f"OCR text items count: {len(ocr_items)}")
                
                if ocr_items:
                    # Print structure of first item
                    first_item = ocr_items[0]
                    print(f"First OCR item keys: {list(first_item.keys())}")
                    
                    # Print text sample
                    if "text" in first_item:
                        print(f"Sample text: {first_item['text']}")
                    
                    # Print bbox structure
                    if "bbox" in first_item:
                        bbox = first_item["bbox"]
                        print(f"Bbox type: {type(bbox)}")
                        print(f"Bbox structure: {bbox}")
        
        # Check for list structure as fallback
        elif isinstance(data, list):
            print(f"List structure with {len(data)} items")
            if data:
                first_item = data[0]
                print(f"First item keys: {list(first_item.keys())}")
        
        return data
    except Exception as e:
        print(f"Error reading annotation {annotation_path}: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def visualize_annotation(image_path, annotation_path, output_path=None):
    """
    Visualize the annotations on the image for debugging
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")
            return
        
        # Make a copy for visualization
        vis_img = img.copy()
        
        # Read the annotation
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text and bounding boxes
        text_items = []
        
        if "entities" in data and "ocr_text" in data["entities"]:
            for item in data["entities"]["ocr_text"]:
                if 'text' in item and 'bbox' in item:
                    text = item['text']
                    
                    # Skip empty text
                    if not text or not text.strip():
                        continue
                    
                    # Extract bbox coordinates
                    if isinstance(item['bbox'], dict) and all(k in item['bbox'] for k in ['x', 'y', 'width', 'height']):
                        x = int(item['bbox']['x'])
                        y = int(item['bbox']['y'])
                        w = int(item['bbox']['width'])
                        h = int(item['bbox']['height'])
                        
                        text_items.append((text, (x, y, w, h)))
        
        # Draw bounding boxes and text
        for text, (x, y, w, h) in text_items:
            # Draw rectangle
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(vis_img, text[:10], (x, y - 5), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Save or display the result
        if output_path:
            cv2.imwrite(output_path, vis_img)
            print(f"Visualization saved to: {output_path}")
        else:
            # Create visualization directory if it doesn't exist
            vis_dir = os.path.join(BASE_DIR, 'bbox_visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Save with a standard name
            base_name = os.path.basename(image_path)
            output_path = os.path.join(vis_dir, f"debug_{base_name}")
            cv2.imwrite(output_path, vis_img)
            print(f"Visualization saved to: {output_path}")
        
        return text_items
    except Exception as e:
        print(f"Error visualizing {annotation_path}: {e}")
        import traceback
        print(traceback.format_exc())
        return []

def process_sample_files():
    """
    Process a few sample files to debug the annotation structure
    """
    # Find a few sample image-annotation pairs
    sample_files = []
    
    # Get first 3 files for testing
    for i in range(3):
        img_path = os.path.join(IMAGES_DIR, f"invoice_{i:03d}.png")
        ann_path = os.path.join(ANNOTATIONS_DIR, f"invoice_{i:03d}.json")
        
        if os.path.exists(img_path) and os.path.exists(ann_path):
            sample_files.append((img_path, ann_path))
    
    # Process each sample
    for img_path, ann_path in sample_files:
        # Inspect the annotation structure
        data = inspect_annotation_structure(ann_path)
        
        # Visualize the annotations
        text_items = visualize_annotation(img_path, ann_path)
        
        print(f"\nFound {len(text_items)} text items in {os.path.basename(img_path)}")
        print("First few text items:")
        for i, (text, bbox) in enumerate(text_items[:5]):
            print(f"{i+1}. '{text}' at {bbox}")
        print("\n" + "-"*50)

def test_extraction_methods():
    """
    Test different extraction methods for annotations
    """
    print("\nTesting different extraction methods...")
    
    # Choose a sample annotation file
    sample_ann_path = os.path.join(ANNOTATIONS_DIR, "invoice_000.json")
    
    if not os.path.exists(sample_ann_path):
        print(f"Sample annotation file not found: {sample_ann_path}")
        return
    
    # Read the annotation
    with open(sample_ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Method 1: Original method
    method1_results = []
    if isinstance(data, list):
        for item in data:
            if 'text' in item and 'bbox' in item:
                method1_results.append((item['text'], item['bbox']))
    
    # Method 2: Nested structure method
    method2_results = []
    if "entities" in data and "ocr_text" in data["entities"]:
        for item in data["entities"]["ocr_text"]:
            if 'text' in item and 'bbox' in item:
                method2_results.append((item['text'], item['bbox']))
    
    print(f"Method 1 (original): Found {len(method1_results)} items")
    print(f"Method 2 (nested): Found {len(method2_results)} items")
    
    # Print first few results from most successful method
    results = method2_results if len(method2_results) > len(method1_results) else method1_results
    print("\nSample extracted items:")
    for i, (text, bbox) in enumerate(results[:5]):
        print(f"{i+1}. '{text}' at {bbox}")

if __name__ == "__main__":
    print("=== Annotation Debug Tool ===")
    process_sample_files()
    test_extraction_methods()
    print("\nDebug complete!")