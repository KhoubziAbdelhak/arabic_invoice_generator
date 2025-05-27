import cv2
import json
import os

def visualize_annotations(image_path, annotation_path, output_dir="visualized_output"):
    """
    Loads an image and its JSON annotation file, draws the bounding boxes,
    and displays it. Optionally saves the visualized image.

    Args:
        image_path (str): Path to the image file.
        annotation_path (str): Path to the JSON annotation file.
        output_dir (str, optional): Directory to save the visualized image.
                                     If None, image is only displayed.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Load the annotations
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {annotation_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {annotation_path}")
        return

    # Draw bounding boxes
    if 'entities' in annotations and 'ocr_text' in annotations['entities']:
        for i, text_entry in enumerate(annotations['entities']['ocr_text']):
            if 'bbox' in text_entry:
                bbox = text_entry['bbox']
                x = int(bbox['x'])
                y = int(bbox['y'])
                width = int(bbox['width'])
                height = int(bbox['height'])

                # Top-left corner
                pt1 = (x, y)
                # Bottom-right corner
                pt2 = (x + width, y + height)

                # Draw the rectangle (BGR color, thickness)
                # Cycle through a few colors for better visibility if many boxes
                color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)][i % 6]
                cv2.rectangle(image, pt1, pt2, color, 2)

                # Optionally, put text label (e.g., the OCR'd text or confidence)
                label = text_entry.get('text', '') # Get text if available
                # conf = text_entry.get('confidence')
                # if conf is not None:
                #     label = f"{label} ({conf:.2f})"

                if label:
                    # Put text slightly above the top-left corner of the bbox
                    cv2.putText(image, label, (x, y - 10 if y > 20 else y + height + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display the image
    window_name = f"Annotations: {os.path.basename(image_path)}"
    cv2.imshow(window_name, image)
    print(f"Displaying {window_name}. Press any key to close this window.")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    # Optionally save the image
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"visualized_{os.path.basename(image_path)}")
        cv2.imwrite(output_filename, image)
        print(f"Saved visualized image to {output_filename}")

if __name__ == "__main__":
    # --- Configuration: Set these paths to your files ---
    # Example: Use one of the augmented images and its annotation
    # Make sure your AUGMENTED_IMAGES_DIR and AUGMENTED_ANNOTATIONS_DIR from the
    # main augmentation script are correctly referenced here or provide absolute paths.

    # Assuming the augmentation script created these directories in the same location
    # and used a consistent naming convention (e.g., augmented_0_doc_for_geom.jpg)
    base_augmented_dir = "augmented_images"
    base_augmented_annotations_dir = "augmented_annotations"
    output_visualization_dir = "visualized_annotations_output"

    # Try to find an example augmented image and its annotation
    example_image_name = None
    example_annotation_name = None

    if os.path.exists(base_augmented_dir) and os.path.exists(base_augmented_annotations_dir):
        augmented_images = sorted([f for f in os.listdir(base_augmented_dir) if f.startswith("augmented_") and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if augmented_images:
            example_image_name = augmented_images[0] # Pick the first one
            base_name_parts = os.path.splitext(example_image_name)[0].split('_', 1) # "augmented_0_doc_for_geom" -> ["augmented", "0_doc_for_geom"]
            if len(base_name_parts) > 1:
                original_base_name = base_name_parts[1] # "0_doc_for_geom"
                example_annotation_name = f"augmented_{original_base_name}.json"
            else: # Fallback if parsing fails, less robust
                 example_annotation_name = os.path.splitext(example_image_name)[0] + ".json"


    if example_image_name and example_annotation_name:
        test_image_path = os.path.join(base_augmented_dir, example_image_name)
        test_annotation_path = os.path.join(base_augmented_annotations_dir, example_annotation_name)

        print(f"Attempting to visualize:\nImage: {test_image_path}\nAnnotation: {test_annotation_path}")

        if os.path.exists(test_image_path) and os.path.exists(test_annotation_path):
            visualize_annotations(test_image_path, test_annotation_path, output_visualization_dir)
        else:
            if not os.path.exists(test_image_path):
                print(f"ERROR: Example image not found at {test_image_path}")
            if not os.path.exists(test_annotation_path):
                print(f"ERROR: Example annotation not found at {test_annotation_path}")
            print("\nPlease ensure your augmentation script has run and produced output,")
            print("or manually set 'test_image_path' and 'test_annotation_path' to valid files.")
    else:
        print("No augmented images found to visualize. Please run the augmentation script first,")
        print("or manually set 'test_image_path' and 'test_annotation_path' in this script.")

    # --- OR: Manually specify paths to any image and its annotation ---
    # manual_image_path = "path/to/your/image.jpg"
    # manual_annotation_path = "path/to/your/annotation.json"
    # if os.path.exists(manual_image_path) and os.path.exists(manual_annotation_path):
    #     visualize_annotations(manual_image_path, manual_annotation_path, output_visualization_dir)
    # else:
    #     print(f"Manual paths not valid: {manual_image_path}, {manual_annotation_path}")
