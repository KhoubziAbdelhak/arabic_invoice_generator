import os
import json
from PIL import Image
import shutil

# Attempt to import from config.py. If it's in a different location,
# you might need to adjust Python's path or pass these as arguments.
try:
    from config import AUGMENTED_IMAGES_DIR, AUGMENTED_ANNOTATIONS_DIR, OUTPUT_DIR
except ImportError:
    print("Warning: Could not import from config.py. Using default relative paths for 'output' directory.")
    # Define default paths if config.py is not found or not in PYTHONPATH
    # This assumes the script is run from the project root where 'output' is a subdir.
    BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Adjust if script is nested
    if os.path.basename(os.path.dirname(__file__)) == "arabic_invoice_generator": # If script is in root
         BASE_PROJECT_DIR = os.path.dirname(__file__)

    OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, 'output')
    AUGMENTED_IMAGES_DIR = os.path.join(OUTPUT_DIR, 'augmented_images')
    AUGMENTED_ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, 'augmented_annotations')


class OCRDatasetPreparer:
    def __init__(self, augmented_images_dir, augmented_annotations_dir, output_base_dir, padding=5):
        self.augmented_images_dir = augmented_images_dir
        self.augmented_annotations_dir = augmented_annotations_dir

        self.ocr_dataset_root_dir = os.path.join(output_base_dir, 'ocr_finetune_dataset')
        self.cropped_images_output_dir = os.path.join(self.ocr_dataset_root_dir, 'images')
        self.labels_file_path = os.path.join(self.ocr_dataset_root_dir, 'labels.txt')

        self.padding = padding
        self.labels_data = []

        # Ensure base output directory exists if not using global OUTPUT_DIR
        os.makedirs(output_base_dir, exist_ok=True)
        # Create specific dataset directories
        os.makedirs(self.cropped_images_output_dir, exist_ok=True)

    def _get_bounding_box_from_poly(self, vertices):
        """Calculates the axis-aligned bounding box from polygon vertices."""
        if not vertices:
            return None
        all_x = [v['x'] for v in vertices]
        all_y = [v['y'] for v in vertices]
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        return int(min_x), int(min_y), int(max_x), int(max_y)

    def process_single_document(self, image_filename, annotation_filename):
        """Processes one image and its annotation to extract Arabic text crops."""
        image_path = os.path.join(self.augmented_images_dir, image_filename)
        annotation_path = os.path.join(self.augmented_annotations_dir, annotation_filename)

        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return
        if not os.path.exists(annotation_path):
            print(f"Annotation file not found: {annotation_path}")
            return

        try:
            img = Image.open(image_path).convert("RGB") # Ensure RGB for consistency
            img_width, img_height = img.size

            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)

            if "entities" not in annotation_data or "ocr_text" not in annotation_data["entities"]:
                print(f"No 'ocr_text' entities found in {annotation_filename}")
                return

            crop_index = 0
            for text_entry in annotation_data["entities"]["ocr_text"]:
                if text_entry.get("direction") == "rtl":
                    text_label = text_entry.get("text", "")
                    if not text_label.strip():  # Skip empty or whitespace-only labels
                        continue

                    if "bounding_poly" in text_entry and "vertices" in text_entry["bounding_poly"]:
                        vertices = text_entry["bounding_poly"]["vertices"]
                        bbox = self._get_bounding_box_from_poly(vertices)

                        if bbox is None:
                            print(f"Skipping entry with no vertices: '{text_label}' in {annotation_filename}")
                            continue

                        min_x, min_y, max_x, max_y = bbox

                        # Add padding
                        padded_min_x = max(0, min_x - self.padding)
                        padded_min_y = max(0, min_y - self.padding)
                        padded_max_x = min(img_width, max_x + self.padding)
                        padded_max_y = min(img_height, max_y + self.padding)

                        # Ensure the box has a positive area after padding
                        if padded_max_x <= padded_min_x or padded_max_y <= padded_min_y:
                            # print(f"Skipping zero-area crop for '{text_label}' in {image_filename} after padding.")
                            continue

                        try:
                            cropped_img = img.crop((padded_min_x, padded_min_y, padded_max_x, padded_max_y))
                        except Exception as crop_e:
                            print(f"Error cropping '{text_label}' in {image_filename} with box {bbox}: {crop_e}")
                            continue


                        # Save cropped image (using PNG for potentially better OCR quality)
                        base_image_name_no_ext = os.path.splitext(image_filename)[0]
                        cropped_filename = f"{base_image_name_no_ext}_crop_{crop_index}.png"
                        cropped_image_save_path = os.path.join(self.cropped_images_output_dir, cropped_filename)

                        try:
                            cropped_img.save(cropped_image_save_path)
                             # Store label data: relative path from labels.txt location which is ocr_dataset_root_dir
                            relative_cropped_path = os.path.join('images', cropped_filename)
                            self.labels_data.append(f"{relative_cropped_path}\t{text_label}")
                            crop_index += 1
                        except Exception as save_e:
                            print(f"Error saving cropped image {cropped_image_save_path}: {save_e}")
                            continue
                    else:
                        # This case might occur if some RTL entries don't use bounding_poly
                        # print(f"Skipping RTL text entry without 'bounding_poly': '{text_label}' in {annotation_filename}")
                        pass


        except Exception as e:
            print(f"Error processing document {image_filename}: {e}")

    def create_dataset(self, clean_existing_dataset=True):
        """
        Main method to iterate through all documents and create the dataset.
        :param clean_existing_dataset: If True, removes the old dataset directory before creating a new one.
        """
        if clean_existing_dataset and os.path.exists(self.ocr_dataset_root_dir):
            print(f"Cleaning up existing dataset directory: {self.ocr_dataset_root_dir}")
            shutil.rmtree(self.ocr_dataset_root_dir)

        # Recreate directories after cleaning or if they didn't exist
        os.makedirs(self.cropped_images_output_dir, exist_ok=True)

        self.labels_data = [] # Reset labels data if called multiple times or after cleaning

        print(f"Looking for annotations in: {self.augmented_annotations_dir}")
        print(f"Looking for images in: {self.augmented_images_dir}")

        annotation_files = [f for f in os.listdir(self.augmented_annotations_dir) if f.endswith('.json')]
        if not annotation_files:
            print(f"No annotation files found in {self.augmented_annotations_dir}. Please check the path and content.")
            return

        processed_count = 0
        for annotation_filename in annotation_files:
            base_name_no_ext = os.path.splitext(annotation_filename)[0]

            # Try to find the corresponding image file (jpg, jpeg, png)
            image_filename = None
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_image_path = os.path.join(self.augmented_images_dir, base_name_no_ext + ext)
                if os.path.exists(potential_image_path):
                    image_filename = base_name_no_ext + ext
                    break

            if image_filename:
                # print(f"Processing: Image: {image_filename}, Annotation: {annotation_filename}")
                self.process_single_document(image_filename, annotation_filename)
                processed_count +=1
            else:
                print(f"Warning: Corresponding image not found for annotation: {annotation_filename}")

        if processed_count == 0:
            print("No documents were processed. Check if images and annotations are correctly paired.")
            return

        # Write the labels file
        if self.labels_data:
            with open(self.labels_file_path, 'w', encoding='utf-8') as f:
                for line in self.labels_data:
                    f.write(line + '\n')
            print(f"\nDataset preparation complete.")
            print(f"Cropped images saved in: {self.cropped_images_output_dir}")
            print(f"Labels file saved as: {self.labels_file_path} (with {len(self.labels_data)} entries)")
        else:
            print("\nNo Arabic text segments found or processed. The labels file will be empty.")


if __name__ == "__main__":
    print("Starting OCR Dataset Preparation...")

    # These paths are usually defined in your config.py
    # Ensure they point to your 'augmented_images' and 'augmented_annotations' directories
    # and the general 'output' directory for placing the new dataset.

    # If config.py import failed, these are fallback (might need adjustment)
    # Check if the global vars from try-except block are populated
    if 'AUGMENTED_IMAGES_DIR' not in globals():
        print("CRITICAL: AUGMENTED_IMAGES_DIR is not defined. Exiting. Check config.py import or manual paths.")
        exit(1)
    if 'AUGMENTED_ANNOTATIONS_DIR' not in globals():
        print("CRITICAL: AUGMENTED_ANNOTATIONS_DIR is not defined. Exiting. Check config.py import or manual paths.")
        exit(1)
    if 'OUTPUT_DIR' not in globals():
        print("CRITICAL: OUTPUT_DIR is not defined. Exiting. Check config.py import or manual paths.")
        exit(1)

    preparer = OCRDatasetPreparer(
        augmented_images_dir=AUGMENTED_IMAGES_DIR,
        augmented_annotations_dir=AUGMENTED_ANNOTATIONS_DIR,
        output_base_dir=OUTPUT_DIR, # The 'ocr_finetune_dataset' will be created inside this
        padding=5  # Adjust padding pixels as needed
    )

    # Set to True to remove any previous dataset for a clean build
    preparer.create_dataset(clean_existing_dataset=True)
