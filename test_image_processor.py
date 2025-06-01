import os
from utils.image_processor import ImageProcessor
from PIL import Image
import numpy as np

def test_jpg_format():
    # Create test output directory
    test_dir = os.path.join(os.path.dirname(__file__), "test_output")
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a simple test image
    img = Image.new('RGB', (500, 300), color=(255, 255, 255))
    img_array = np.array(img)
    
    # Draw some text on the image
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Test JPG Format", fill=(0, 0, 0))
    
    # Save as both PNG and JPG for comparison
    png_path = os.path.join(test_dir, "test_image.png")
    jpg_path = os.path.join(test_dir, "test_image.jpg")
    
    # Save directly using PIL
    img.save(png_path, "PNG")
    img.save(jpg_path, "JPEG", quality=85)
    
    print(f"Direct PIL save:")
    print(f"PNG size: {os.path.getsize(png_path)/1024:.2f} KB")
    print(f"JPG size: {os.path.getsize(jpg_path)/1024:.2f} KB")
    
    # Test the JPG format in our ImageProcessor class
    processor_jpg_path = os.path.join(test_dir, "processor_test_image.jpg")
    
    # Simulate drawing annotations
    annotations = [
        {"coordinates": {"x": 50, "y": 50, "width": 200, "height": 100}}
    ]
    
    # Test the draw_annotations method with JPG format
    ImageProcessor.draw_annotations(
        png_path, 
        annotations, 
        processor_jpg_path,
        format="jpg",
        quality=85
    )
    
    print(f"\nImageProcessor save:")
    print(f"JPG size: {os.path.getsize(processor_jpg_path)/1024:.2f} KB")
    print(f"Image saved successfully: {os.path.exists(processor_jpg_path)}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_jpg_format()