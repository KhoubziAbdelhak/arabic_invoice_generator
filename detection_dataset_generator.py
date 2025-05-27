import json
import os
import math
import shutil
import random

# Import configuration
try:
    from config.config import *
except ImportError:
    print("Error: config.py not found or cannot be imported.")
    print("Please ensure config/config.py is correctly set up and accessible.")
    print("Script cannot proceed without config.py.")
    exit(1)

# Configuration settings
USE_AUGMENTED_DATA = True  # Set to False to use original data instead

# Data splitting configuration
TRAIN_RATIO = 0.7  # 70% for training
VAL_RATIO = 0.2    # 20% for validation
TEST_RATIO = 0.1   # 10% for testing
RANDOM_SEED = 42   # For reproducible splits

# Select data paths based on configuration
if USE_AUGMENTED_DATA:
    SOURCE_ANNOTATIONS_DIR = AUGMENTED_ANNOTATIONS_DIR
    SOURCE_IMAGES_DIR = AUGMENTED_IMAGES_DIR
    DATASET_SUFFIX = "_augmented"
    print("Using AUGMENTED dataset paths")
else:
    SOURCE_ANNOTATIONS_DIR = ANNOTATIONS_DIR
    SOURCE_IMAGES_DIR = IMAGES_DIR
    DATASET_SUFFIX = "_original"
    print("Using ORIGINAL dataset paths")


def calculate_normalized_obb_from_vertices(vertices, img_width, img_height):
    """
    Calculates normalized oriented bounding box coordinates from vertex points.

    Args:
        vertices: List of vertex dictionaries with 'x' and 'y' keys
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        List of 8 normalized coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    if len(vertices) != 4:
        raise ValueError(f"Expected 4 vertices, got {len(vertices)}")

    if img_width == 0 or img_height == 0:
        raise ValueError("Image width or height cannot be zero.")

    # Extract and validate coordinates
    coords = []
    for vertex in vertices:
        if 'x' not in vertex or 'y' not in vertex:
            raise ValueError(f"Vertex missing 'x' or 'y' coordinate: {vertex}")

        x = vertex['x']
        y = vertex['y']

        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"Vertex coordinates must be numbers: x={x}, y={y}")

        # Normalize coordinates
        norm_x = x / img_width
        norm_y = y / img_height
        coords.extend([norm_x, norm_y])

    return coords


def calculate_normalized_obb_from_bbox(bbox_dict, img_width, img_height):
    """
    Calculates normalized oriented bounding box coordinates from axis-aligned bbox.
    Converts to 4 corner points format.

    Args:
        bbox_dict: Dictionary with 'x', 'y', 'width', 'height' keys
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        List of 8 normalized coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    required_keys = ['x', 'y', 'width', 'height']
    for key in required_keys:
        if key not in bbox_dict:
            raise ValueError(f"Bounding box dictionary is missing key: {key}")
        if not isinstance(bbox_dict[key], (int, float)):
            raise ValueError(f"Bounding box value for '{key}' is not a number: {bbox_dict[key]}")

    x = bbox_dict['x']
    y = bbox_dict['y']
    w = bbox_dict['width']
    h = bbox_dict['height']

    if img_width == 0 or img_height == 0:
        raise ValueError("Image width or height cannot be zero.")
    if w < 0 or h < 0:
        raise ValueError("Bounding box width or height cannot be negative.")

    # Create 4 corner points (top-left, top-right, bottom-right, bottom-left)
    vertices = [
        {'x': x, 'y': y},                    # top-left
        {'x': x + w, 'y': y},                # top-right
        {'x': x + w, 'y': y + h},            # bottom-right
        {'x': x, 'y': y + h}                 # bottom-left
    ]

    return calculate_normalized_obb_from_vertices(vertices, img_width, img_height)


def collect_all_files(source_annotations_dir, source_images_dir):
    """
    Collects all image and annotation file pairs from the source directories.
    Returns a list of tuples (image_path, json_path, base_name).
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    json_extensions = ['.json']

    # Collect all image files
    image_files = []
    for root, dirs, files in os.walk(source_images_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                image_files.append((image_path, base_name))

    # Collect all JSON files
    json_files = {}
    for root, dirs, files in os.walk(source_annotations_dir):
        for file in files:
            if file.lower().endswith('.json'):
                json_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                json_files[base_name] = json_path

    # Match image files with their corresponding JSON files
    matched_files = []
    for image_path, base_name in image_files:
        if base_name in json_files:
            matched_files.append((image_path, json_files[base_name], base_name))
        else:
            print(f"Warning: No annotation found for image {base_name}")

    return matched_files


def split_files(matched_files, train_ratio, val_ratio, test_ratio, random_seed=42):
    """
    Splits the matched files into train, validation, and test sets.
    Returns three lists: train_files, val_files, test_files.
    """
    # Ensure ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Ratios don't sum to 1.0 (sum={total_ratio}). Normalizing...")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Shuffle the files
    shuffled_files = matched_files.copy()
    random.shuffle(shuffled_files)

    total_files = len(shuffled_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count  # Remainder goes to test

    train_files = shuffled_files[:train_count]
    val_files = shuffled_files[train_count:train_count + val_count]
    test_files = shuffled_files[train_count + val_count:]

    print(f"Data split summary:")
    print(f"  Total files: {total_files}")
    print(f"  Train: {len(train_files)} ({len(train_files)/total_files*100:.1f}%)")
    print(f"  Validation: {len(val_files)} ({len(val_files)/total_files*100:.1f}%)")
    print(f"  Test: {len(test_files)} ({len(test_files)/total_files*100:.1f}%)")

    return train_files, val_files, test_files


def copy_files_to_split(files, split_name, yolo_images_dir, yolo_labels_dir):
    """
    Copies image files to the appropriate split directory and processes their annotations.
    Returns the number of annotations processed.
    """
    images_split_dir = os.path.join(yolo_images_dir, split_name)
    labels_split_dir = os.path.join(yolo_labels_dir, split_name)

    os.makedirs(images_split_dir, exist_ok=True)
    os.makedirs(labels_split_dir, exist_ok=True)

    total_annotations = 0
    successful_files = 0

    for image_path, json_path, base_name in files:
        try:
            # Copy image file
            dest_image_path = os.path.join(images_split_dir, os.path.basename(image_path))
            shutil.copy2(image_path, dest_image_path)

            # Process JSON annotation
            success, annotation_count = process_single_json_for_split(
                json_path, labels_split_dir, base_name
            )

            if success:
                successful_files += 1
                total_annotations += annotation_count
            else:
                print(f"Warning: Failed to process annotation for {base_name}")

        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    print(f"Split '{split_name}': {successful_files}/{len(files)} files processed successfully, {total_annotations} annotations")
    return total_annotations


def process_single_json_for_split(json_file_path, output_labels_dir, base_name):
    """
    Processes a single JSON file and creates a corresponding YOLO OBB label file.
    Modified version that doesn't need to determine split from path.
    Returns tuple (success, annotation_count)
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return False, 0
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return False, 0

    image_size = data.get("image_size")
    if not image_size or 'width' not in image_size or 'height' not in image_size:
        print(f"Warning: 'image_size' with 'width' and 'height' not found in {json_file_path}. Skipping file.")
        return True, 0

    img_width = image_size['width']
    img_height = image_size['height']

    if not isinstance(img_width, (int, float)) or not isinstance(img_height, (int, float)):
        print(f"Warning: Image width or height is not a number in {json_file_path}. Skipping file.")
        return True, 0
    if img_width <= 0 or img_height <= 0:
        print(f"Warning: Image width or height is non-positive in {json_file_path}. Skipping file.")
        return True, 0

    labels = []
    entities = data.get("entities", {})

    # Class 0: text (OCR text elements)
    ocr_text_entities = entities.get("ocr_text", [])
    for text_entity in ocr_text_entities:
        bounding_poly = text_entity.get("bounding_poly")
        if bounding_poly and "vertices" in bounding_poly:
            vertices = bounding_poly["vertices"]
            try:
                obb_coords = calculate_normalized_obb_from_vertices(vertices, img_width, img_height)
                coords_str = " ".join(f"{coord:.6f}" for coord in obb_coords)
                labels.append(f"0 {coords_str}")
            except Exception as e:
                print(f"Warning: Could not process a 'text' bounding_poly in {base_name}: {e}")

    # Class 1: table
    table_entities = entities.get("table", [])
    for table_entity in table_entities:
        bbox = table_entity.get("bbox")
        if bbox:
            try:
                obb_coords = calculate_normalized_obb_from_bbox(bbox, img_width, img_height)
                coords_str = " ".join(f"{coord:.6f}" for coord in obb_coords)
                labels.append(f"1 {coords_str}")
            except Exception as e:
                print(f"Warning: Could not process a 'table' bbox in {base_name}: {e}")

    if not labels:
        return True, 0

    # Write label file
    output_label_file = os.path.join(output_labels_dir, f"{base_name}.txt")
    try:
        with open(output_label_file, 'w', encoding='utf-8') as f:
            for label_line in labels:
                f.write(label_line + "\n")
        return True, len(labels)
    except IOError as e:
        print(f"Error writing label file {output_label_file}: {e}")
        return False, 0


def generate_yaml_config(yolo_dataset_root_dir):
    """
    Generates the obb.yaml configuration file for OBB detection.
    Uses relative paths within the dataset structure.
    """
    yaml_content_dict = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,
        "names": {
            0: "text",
            1: "table"
        }
    }

    yaml_file_path = os.path.join(yolo_dataset_root_dir, "obb.yaml")

    try:
        with open(yaml_file_path, 'w', encoding='utf-8') as f:
            f.write(f"# YOLO OBB (Oriented Bounding Box) Dataset Configuration\n")
            f.write(f"# Generated for invoice text and table detection\n\n")
            f.write(f"# Path to training images (relative to this YAML file's directory)\n")
            f.write(f"train: {yaml_content_dict['train']}\n\n")
            f.write(f"# Path to validation images (relative to this YAML file's directory)\n")
            f.write(f"val: {yaml_content_dict['val']}\n\n")
            f.write(f"# Path to test images (optional, relative)\n")
            f.write(f"test: {yaml_content_dict['test']}\n\n")
            f.write(f"# Number of classes\n")
            f.write(f"nc: {yaml_content_dict['nc']}\n\n")
            f.write(f"# Class names\n")
            f.write("names:\n")
            for key, value in yaml_content_dict['names'].items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n# Task type for OBB detection\n")
            f.write(f"task: obb\n")
        print(f"Successfully created YAML configuration file: {yaml_file_path}")
        return True
    except IOError as e:
        print(f"Error writing YAML file {yaml_file_path}: {e}")
        return False


def create_empty_split_directories(base_dir, subdirs=['train', 'val', 'test']):
    """
    Creates empty train/val/test directories if they don't exist.
    """
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        os.makedirs(dir_path, exist_ok=True)


def main():
    """
    Main function to orchestrate the YOLO OBB data generation with automatic data splitting.
    """
    print("Starting YOLO OBB data generation script with automatic data splitting...")
    print(f"Split ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    print(f"Random seed: {RANDOM_SEED}")

    # Define YOLO dataset directory based on OUTPUT_DIR
    YOLO_DATASET_ROOT_DIR = os.path.join(OUTPUT_DIR, f'obb-dataset{DATASET_SUFFIX}')

    # Create the required directory structure
    yolo_images_dir = os.path.join(YOLO_DATASET_ROOT_DIR, 'images')
    yolo_labels_dir = os.path.join(YOLO_DATASET_ROOT_DIR, 'labels')

    # Ensure the main YOLO dataset directory exists
    try:
        os.makedirs(YOLO_DATASET_ROOT_DIR, exist_ok=True)
        print(f"Created dataset root directory: {YOLO_DATASET_ROOT_DIR}")
    except OSError as e:
        print(f"Error creating output directory {YOLO_DATASET_ROOT_DIR}: {e}")
        return

    # Create empty split directories for both images and labels
    create_empty_split_directories(yolo_images_dir)
    create_empty_split_directories(yolo_labels_dir)
    print("Created train/val/test subdirectories for images and labels")

    # Check if source directories exist
    if not os.path.isdir(SOURCE_ANNOTATIONS_DIR):
        print(f"Error: Source annotations directory not found: {SOURCE_ANNOTATIONS_DIR}")
        return

    if not os.path.isdir(SOURCE_IMAGES_DIR):
        print(f"Error: Source images directory not found: {SOURCE_IMAGES_DIR}")
        return

    # Collect all image-annotation file pairs
    print("Collecting image and annotation files...")
    matched_files = collect_all_files(SOURCE_ANNOTATIONS_DIR, SOURCE_IMAGES_DIR)

    if not matched_files:
        print("No matching image-annotation pairs found!")
        return

    print(f"Found {len(matched_files)} matching image-annotation pairs")

    # Split files into train/val/test
    print("Splitting files into train/validation/test sets...")
    train_files, val_files, test_files = split_files(
        matched_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    # Process each split
    total_annotations = 0
    split_stats = {}

    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        if files:  # Only process if there are files in this split
            print(f"\nProcessing {split_name} split...")
            annotation_count = copy_files_to_split(files, split_name, yolo_images_dir, yolo_labels_dir)
            split_stats[split_name] = annotation_count
            total_annotations += annotation_count
        else:
            split_stats[split_name] = 0

    # Generate obb.yaml
    generate_yaml_config(YOLO_DATASET_ROOT_DIR)

    # Print final verification
    print(f"\n--- Final Dataset Structure Verification ---")
    for split in ['train', 'val', 'test']:
        images_split_dir = os.path.join(yolo_images_dir, split)
        labels_split_dir = os.path.join(yolo_labels_dir, split)

        image_count = len([f for f in os.listdir(images_split_dir) if os.path.isfile(os.path.join(images_split_dir, f))]) if os.path.exists(images_split_dir) else 0
        label_count = len([f for f in os.listdir(labels_split_dir) if f.endswith('.txt')]) if os.path.exists(labels_split_dir) else 0

        print(f"{split}: {image_count} images, {label_count} labels")

    print(f"\n--- YOLO OBB Dataset Structure Created ---")
    print(f"Dataset root directory: {YOLO_DATASET_ROOT_DIR}")
    print(f"├── images/")
    print(f"│   ├── train/")
    print(f"│   ├── val/")
    print(f"│   └── test/")
    print(f"├── labels/")
    print(f"│   ├── train/")
    print(f"│   ├── val/")
    print(f"│   └── test/")
    print(f"└── obb.yaml")

    print(f"\n--- Processing Summary ---")
    print(f"Total Annotations: {total_annotations}")
    for split, count in split_stats.items():
        if count > 0:
            print(f"  - {split}: {count} annotations")

    print(f"\nClasses:")
    print(f"  - Class 0 (text): OCR text elements with oriented bounding boxes")
    print(f"  - Class 1 (table): Table regions with axis-aligned bounding boxes")
    print(f"\nData Source: {'Augmented' if USE_AUGMENTED_DATA else 'Original'} dataset")
    print(f"Random Seed: {RANDOM_SEED} (for reproducible splits)")

    print(f"\n--- Ready for Training ---")
    print(f"Dataset has been automatically split into train/val/test sets.")
    print(f"To train with Ultralytics:")
    print(f"  pip install ultralytics")
    print(f"  yolo obb train data=obb.yaml model=yolov8n-obb.pt epochs=100 imgsz=640")
    print("--- End of Summary ---")


if __name__ == "__main__":
    # Check if critical config paths are defined
    critical_paths_defined = True
    required_paths = [
        ('ANNOTATIONS_DIR', ANNOTATIONS_DIR),
        ('AUGMENTED_ANNOTATIONS_DIR', AUGMENTED_ANNOTATIONS_DIR),
        ('OUTPUT_DIR', OUTPUT_DIR),
        ('IMAGES_DIR', IMAGES_DIR),
        ('AUGMENTED_IMAGES_DIR', AUGMENTED_IMAGES_DIR)
    ]

    for path_name, path_value in required_paths:
        if path_value is None:
            print(f"Critical configuration variable '{path_name}' is not defined.")
            critical_paths_defined = False

    if critical_paths_defined:
        main()
    else:
        print("Script cannot proceed due to missing critical path configurations in config.py.")
        print("Please ensure all required paths are defined in your config/config.py file.")
