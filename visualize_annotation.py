import os
import json
from PIL import Image, ImageDraw, ImageFont
import sys

class AnnotationVisualizer:
    def __init__(self, base_dir):
        self.base_dir = base_dir

        # Normal Images
        # self.images_dir = os.path.join(base_dir, 'output', 'images')
        # self.annotations_dir = os.path.join(base_dir, 'output', 'annotations')

        # Augmented Images
        self.images_dir = os.path.join(base_dir, 'output', 'augmented_images')
        self.annotations_dir = os.path.join(base_dir, 'output', 'augmented_annotations')

        self.output_dir = os.path.join(base_dir, 'visualization_output')
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize_all_documents(self):
        """Process all images and their annotations"""
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            annotation_file = f"{base_name}.json"

            print(f"Processing {image_file}...")
            self.visualize_single_document(image_file, annotation_file)

    def visualize_single_document(self, image_file, annotation_file):
        """Create visualization for a single document"""
        image_path = os.path.join(self.images_dir, image_file)
        annotation_path = os.path.join(self.annotations_dir, annotation_file)

        # Check if files exist
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_file}")
            return
        if not os.path.exists(annotation_path):
            print(f"Annotation file not found: {annotation_file}")
            return

        # Load image and annotation
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        # Visualize all text elements
        self.visualize_text_elements(image, draw, annotation)

        # Save the output
        output_path = os.path.join(self.output_dir, f'{os.path.splitext(image_file)[0]}_annotated.jpg')
        image.save(output_path, "JPEG", quality=85)

    def get_bounding_info(self, text_entry):
        """Extract bounding information from either bbox or bounding_poly format"""
        if "bbox" in text_entry:
            # Original rectangular bbox format
            bbox = text_entry["bbox"]
            # Convert all values to integers for drawing
            x = int(bbox["x"])
            y = int(bbox["y"])
            w = int(bbox["width"])
            h = int(bbox["height"])
            return {
                'type': 'rectangle',
                'coords': [
                    (x, y),
                    (x + w, y),
                    (x + w, y + h),
                    (x, y + h)
                ],
                'top_left': (x, y)
            }
        elif "bounding_poly" in text_entry and "vertices" in text_entry["bounding_poly"]:
            # Rotated polygon format
            vertices = text_entry["bounding_poly"]["vertices"]
            coords = [(int(vertex["x"]), int(vertex["y"])) for vertex in vertices]
            # Find the topmost-leftmost point for text label placement
            top_left = min(coords, key=lambda p: (p[1], p[0]))
            return {
                'type': 'polygon',
                'coords': coords,
                'top_left': top_left
            }
        else:
            # Fallback - return None if no valid bounding info found
            return None

    def visualize_text_elements(self, image, draw, annotation):
        """Draw text elements with direction-based colors"""
        if "entities" not in annotation or "ocr_text" not in annotation["entities"]:
            print("No OCR text entities found in annotation")
            return

        for text_entry in annotation["entities"]["ocr_text"]:
            bounding_info = self.get_bounding_info(text_entry)

            if bounding_info is None:
                print(f"No valid bounding information found for text: {text_entry.get('text', 'Unknown')[:20]}")
                continue

            direction = text_entry.get("direction", "ltr")
            text = text_entry.get("text", "")

            # Red for RTL (Arabic), Blue for LTR (English)
            color = "red" if direction == "rtl" else "blue"

            # Draw bounding box/polygon
            if bounding_info['type'] == 'rectangle':
                # Draw rectangle
                x1, y1 = bounding_info['coords'][0]
                x2, y2 = bounding_info['coords'][2]
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            else:
                # Draw polygon - fix by connecting points in sequence
                points = bounding_info['coords']
                for i in range(len(points)):
                    start_point = points[i]
                    end_point = points[(i + 1) % len(points)]  # Wrap around to close the polygon
                    draw.line([start_point, end_point], fill=color, width=2)

            # Draw text label above/near the bounding area
            label_x, label_y = bounding_info['top_left']
            draw.text(
                (label_x, max(0, label_y - 15)),
                text[:20],  # Truncate long text
                fill=color,
                font=self.get_font(12)
            )

    @staticmethod
    def get_font(size):
        """Get a font that supports both Arabic and English"""
        try:
            # Try to load a font that supports Arabic
            # First try project relative path, then absolute path, then default
            font_paths = [
                "data/Amiri-Regular.ttf",
                "arabic_invoice_generator/data/Amiri-Regular.ttf",
                "/var/home/abdelhak/programming/pfe/arabic_invoice_generator/data/Amiri-Regular.ttf"
            ]
            
            for path in font_paths:
                if os.path.exists(path):
                    return ImageFont.truetype(path, size)
            
            # If none of the paths work, fall back to default
            return ImageFont.load_default()
        except:
            # Fallback to default font
            return ImageFont.load_default()

def main():
    # Try to determine the project base directory
    try:
        # Current working directory
        cwd = os.getcwd()
        if cwd.endswith('arabic_invoice_generator'):
            base_dir = cwd
        elif os.path.exists(os.path.join(cwd, 'arabic_invoice_generator')):
            base_dir = cwd
        else:
            # Fallback to original path
            base_dir = "/var/home/abdelhak/programming/pfe/arabic_invoice_generator/"
            if not os.path.exists(base_dir):
                base_dir = "."
    except:
        base_dir = "."

    print(f"Using base directory: {base_dir}")

    # Initialize visualizer
    visualizer = AnnotationVisualizer(base_dir)

    # Process all documents
    # visualizer.visualize_all_documents()

    # Process only x number of documents
    number_of_documents = 30
    try:
        image_files = [f for f in os.listdir(visualizer.images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][:number_of_documents]
        if not image_files:
            print(f"No image files found in {visualizer.images_dir}")
        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            annotation_file = f"{base_name}.json"
            visualizer.visualize_single_document(image_file, annotation_file)

        print("\nVisualization complete! Check the 'visualization_output' directory for results.\n")
    except Exception as e:
        print(f"Error processing images: {str(e)}")

if __name__ == "__main__":
    main()
