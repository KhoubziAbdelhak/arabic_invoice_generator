import os
import json
import argparse
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from difflib import SequenceMatcher
from collections import Counter
from pathlib import Path
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import editdistance
import Levenshtein

# Import OCR engines
try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import paddleocr
except ImportError:
    paddleocr = None

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
except ImportError:
    TrOCRProcessor = None


class OCREvaluator:
    def __init__(self, model_type, model_path=None, device="cuda"):
        """
        Initialize OCR evaluator with the selected model type
        
        Args:
            model_type (str): Type of OCR model to use ('easyocr', 'tesseract', 'paddleocr', 'trocr')
            model_path (str): Path to the model or model name
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the specified OCR model"""
        print(f"Loading {self.model_type} model...")
        
        if self.model_type == "easyocr":
            if easyocr is None:
                raise ImportError("EasyOCR is not installed. Please install with 'pip install easyocr'")
            
            if self.model_path:
                self.model = easyocr.Reader(
                    ['ar'], 
                    gpu=(self.device == "cuda"),
                    model_storage_directory=self.model_path,
                    user_network_directory=self.model_path,
                    recog_network='arabic_invoice_recognizer'
                )
            else:
                self.model = easyocr.Reader(['ar'], gpu=(self.device == "cuda"))
        
        elif self.model_type == "tesseract":
            if pytesseract is None:
                raise ImportError("Pytesseract is not installed. Please install with 'pip install pytesseract'")
            
            # No specific loading required for Tesseract
            # But we can configure the tesseract command
            if self.model_path:
                pytesseract.pytesseract.tesseract_cmd = self.model_path
        
        elif self.model_type == "paddleocr":
            if paddleocr is None:
                raise ImportError("PaddleOCR is not installed. Please install with 'pip install paddleocr'")
            
            self.model = paddleocr.PaddleOCR(use_angle_cls=True, lang="ar", use_gpu=(self.device == "cuda"))
        
        elif self.model_type == "trocr":
            if TrOCRProcessor is None:
                raise ImportError("Transformers is not installed. Please install with 'pip install transformers'")
            
            if self.model_path:
                self.processor = TrOCRProcessor.from_pretrained(self.model_path)
                self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
            else:
                # Default to Microsoft's TrOCR model
                self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base")
                self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base")
            
            self.model.to(self.device)
        
        else:
            raise ValueError(f"Unsupported OCR model type: {self.model_type}")
        
        print(f"Model loaded successfully")
    
    def recognize_text(self, image_path):
        """
        Recognize text in an image using the loaded OCR model
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            list: List of (text, confidence, bbox) tuples
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                return []
            
            if self.model_type == "easyocr":
                results = self.model.readtext(img)
                # Format: [(bbox, text, confidence), ...]
                return [(text, conf, bbox) for bbox, text, conf in results]
            
            elif self.model_type == "tesseract":
                # Get text with bounding boxes
                data = pytesseract.image_to_data(img, lang='ara', output_type=pytesseract.Output.DICT)
                results = []
                
                n_boxes = len(data['text'])
                for i in range(n_boxes):
                    if int(data['conf'][i]) > 0:  # Filter out low confidence
                        text = data['text'][i]
                        if not text.strip():
                            continue
                            
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        bbox = (x, y, x + w, y + h)
                        conf = float(data['conf'][i]) / 100.0
                        results.append((text, conf, bbox))
                
                return results
            
            elif self.model_type == "paddleocr":
                results = self.model.ocr(img, cls=True)
                if not results or not results[0]:
                    return []
                    
                # Format: [[[bbox], (text, confidence)], ...]
                return [(text, conf, bbox) for (bbox, (text, conf)) in results[0]]
            
            elif self.model_type == "trocr":
                # TrOCR doesn't return bounding boxes, so we treat the whole image as one text block
                pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values)
                    
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                h, w = img.shape[:2]
                
                # Return the text with a dummy confidence score and full-image bbox
                return [(text, 1.0, (0, 0, w, h))]
            
        except Exception as e:
            print(f"Error recognizing text in {image_path}: {e}")
            import traceback
            print(traceback.format_exc())
            return []
    
    def evaluate_dataset(self, test_data_dir):
        """
        Evaluate the OCR model on a test dataset
        
        Args:
            test_data_dir (str): Directory containing test images and labels
            
        Returns:
            dict: Evaluation metrics
        """
        test_images_dir = os.path.join(test_data_dir, "images")
        test_labels_dir = os.path.join(test_data_dir, "labels")
        
        if not os.path.exists(test_images_dir) or not os.path.exists(test_labels_dir):
            raise ValueError(f"Test data directory structure invalid. Expected 'images' and 'labels' subdirectories in {test_data_dir}")
        
        image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        results = []
        char_errors = 0
        total_chars = 0
        word_errors = 0
        total_words = 0
        total_time = 0
        
        confusion_matrix = Counter()
        
        for img_file in tqdm(image_files, desc="Evaluating"):
            img_path = os.path.join(test_images_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            
            # Find corresponding label file
            label_file = f"{base_name}.txt"
            label_path = os.path.join(test_labels_dir, label_file)
            
            if not os.path.exists(label_path):
                print(f"Warning: No label file found for {img_file}")
                continue
            
            # Read ground truth
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            ground_truth_map = {}
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    gt_img, gt_text = parts
                    ground_truth_map[gt_img] = gt_text
            
            # Check if this image has ground truth
            if img_file not in ground_truth_map:
                # This is a full page image, skip it
                continue
                
            ground_truth = ground_truth_map[img_file]
            
            # Recognize text in image
            start_time = time.time()
            detected_results = self.recognize_text(img_path)
            end_time = time.time()
            
            if not detected_results:
                print(f"Warning: No text detected in {img_file}")
                results.append({
                    'image': img_file,
                    'ground_truth': ground_truth,
                    'detected_text': '',
                    'cer': 1.0,
                    'wer': 1.0,
                    'time': 0
                })
                continue
            
            # For simplicity, concatenate all detected text
            detected_text = " ".join([text for text, _, _ in detected_results])
            
            # Calculate Character Error Rate (CER)
            edit_distance = editdistance.eval(detected_text, ground_truth)
            cer = edit_distance / max(len(ground_truth), 1)
            
            # Calculate Word Error Rate (WER)
            gt_words = ground_truth.split()
            detected_words = detected_text.split()
            
            wer_distance = Levenshtein.distance(detected_text, ground_truth)
            wer = wer_distance / max(len(gt_words), 1)
            
            # Update counters
            char_errors += edit_distance
            total_chars += len(ground_truth)
            word_errors += wer_distance
            total_words += len(gt_words)
            
            # Processing time
            processing_time = end_time - start_time
            total_time += processing_time
            
            # Build confusion matrix (character level)
            for i in range(min(len(detected_text), len(ground_truth))):
                if detected_text[i] != ground_truth[i]:
                    confusion_pair = (ground_truth[i], detected_text[i])
                    confusion_matrix[confusion_pair] += 1
            
            results.append({
                'image': img_file,
                'ground_truth': ground_truth,
                'detected_text': detected_text,
                'cer': cer,
                'wer': wer,
                'time': processing_time
            })
        
        # Calculate overall metrics
        overall_cer = char_errors / max(total_chars, 1)
        overall_wer = word_errors / max(total_words, 1)
        average_time = total_time / max(len(results), 1)
        
        # Create final results dictionary
        evaluation = {
            'model_type': self.model_type,
            'overall_cer': overall_cer,
            'overall_wer': overall_wer,
            'average_processing_time': average_time,
            'num_samples': len(results),
            'confusion_matrix': dict(confusion_matrix),
            'sample_results': results
        }
        
        return evaluation
    
    def visualize_results(self, evaluation, output_dir):
        """
        Visualize evaluation results and save plots
        
        Args:
            evaluation (dict): Evaluation results from evaluate_dataset
            output_dir (str): Directory to save visualization outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame from results
        df = pd.DataFrame(evaluation['sample_results'])
        
        # Plot CER and WER distributions
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df['cer'], bins=20, kde=True)
        plt.axvline(evaluation['overall_cer'], color='r', linestyle='--', 
                    label=f'Overall CER: {evaluation["overall_cer"]:.4f}')
        plt.title('Character Error Rate Distribution')
        plt.xlabel('CER')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        sns.histplot(df['wer'], bins=20, kde=True)
        plt.axvline(evaluation['overall_wer'], color='r', linestyle='--', 
                    label=f'Overall WER: {evaluation["overall_wer"]:.4f}')
        plt.title('Word Error Rate Distribution')
        plt.xlabel('WER')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_distributions.png'))
        
        # Plot confusion matrix (top N most common errors)
        top_n = 20
        cm = evaluation['confusion_matrix']
        top_confusions = sorted(cm.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if top_confusions:
            chars_gt, chars_pred = zip(*[x[0] for x in top_confusions])
            counts = [x[1] for x in top_confusions]
            
            plt.figure(figsize=(15, 8))
            confusion_df = pd.DataFrame({
                'Ground Truth': [get_display(reshape(c)) if c else 'ε' for c in chars_gt],
                'Predicted': [get_display(reshape(c)) if c else 'ε' for c in chars_pred],
                'Count': counts
            })
            
            # Create a pivot table for the heatmap
            pivot_df = confusion_df.pivot_table(
                index='Ground Truth', 
                columns='Predicted', 
                values='Count', 
                fill_value=0
            )
            
            sns.heatmap(pivot_df, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Top {top_n} Character Confusions')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        
        # Plot processing time distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['time'], bins=20, kde=True)
        plt.axvline(evaluation['average_processing_time'], color='r', linestyle='--',
                    label=f'Average: {evaluation["average_processing_time"]:.4f}s')
        plt.title('Processing Time Distribution')
        plt.xlabel('Time (seconds)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'processing_time.png'))
        
        # Save detailed results
        df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
        
        # Save evaluation metrics
        metrics = {
            'model_type': evaluation['model_type'],
            'overall_cer': evaluation['overall_cer'],
            'overall_wer': evaluation['overall_wer'],
            'average_processing_time': evaluation['average_processing_time'],
            'num_samples': evaluation['num_samples']
        }
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate OCR models on Arabic invoice dataset')
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['easyocr', 'tesseract', 'paddleocr', 'trocr'],
                        help='Type of OCR model to evaluate')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model or model name')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data directory with images and labels subdirectories')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')
                        
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = OCREvaluator(
        model_type=args.model_type,
        model_path=args.model_path,
        device=args.device
    )
    
    # Run evaluation
    evaluation = evaluator.evaluate_dataset(args.test_data)
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize and save results
    evaluator.visualize_results(evaluation, output_dir)
    
    # Print summary
    print("\n===== Evaluation Summary =====")
    print(f"Model: {args.model_type}")
    print(f"Samples evaluated: {evaluation['num_samples']}")
    print(f"Character Error Rate (CER): {evaluation['overall_cer']:.4f}")
    print(f"Word Error Rate (WER): {evaluation['overall_wer']:.4f}")
    print(f"Average processing time: {evaluation['average_processing_time']:.4f} seconds")
    print(f"Detailed results saved to: {output_dir}")


if __name__ == "__main__":
    main()