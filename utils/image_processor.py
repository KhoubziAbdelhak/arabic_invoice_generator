from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import os

class ImageProcessor:
    @staticmethod
    def pdf_to_images(pdf_path, output_dir, dpi=300, format="jpg", quality=85):
        images = convert_from_path(pdf_path, dpi=dpi)
        image_paths = []

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        for i, image in enumerate(images):
            if i>0:
                pass
            # Use JPG format instead of PNG to save space
            image_path = os.path.join(output_dir, f'{pdf_name}.{format.lower()}')
            # PIL expects 'JPEG' not 'JPG' for the format parameter
            save_format = "JPEG" if format.upper() == "JPG" else format.upper()
            image.save(image_path, save_format, quality=quality)
            image_paths.append(image_path)
            print(f"Saved image: {image_path}")
            
        return image_paths

    @staticmethod
    def draw_annotations(image_path, annotations, output_path, format="jpg", quality=85):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        for ann in annotations:
            x, y, w, h = ann['coordinates'].values()
            draw.rectangle(((x, y), (x + w, y + h)), outline='red', width=2)
        
        # Save with format and quality specified
        if format.lower() == "jpg" or format.lower() == "jpeg":
            # PIL expects 'JPEG' not 'JPG' for the format parameter
            save_format = "JPEG" if format.upper() == "JPG" else format.upper()
            image.save(output_path, save_format, quality=quality)
        else:
            image.save(output_path, format.upper())
    