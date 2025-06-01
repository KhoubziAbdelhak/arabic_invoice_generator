import os
import sys
import subprocess
from utils.image_processor import ImageProcessor
from config.config import *

def test_jpg_conversion():
    # Make sure the output directory exists
    os.makedirs("test_output", exist_ok=True)
    
    # Find a sample PDF to convert
    pdf_files = []
    if os.path.exists(PDF_DIR):
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in PDF_DIR.")
        print("Looking for any PDF in the system...")
        
        # Try to find any PDF file in the current directory and subdirectories
        found_pdfs = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.pdf'):
                    found_pdfs.append(os.path.join(root, file))
                    if len(found_pdfs) >= 1:  # Find just one PDF
                        break
            if found_pdfs:
                break
                
        if found_pdfs:
            pdf_path = found_pdfs[0]
            print(f"Found PDF: {pdf_path}")
        else:
            print("No PDF files found. Test cannot continue.")
            sys.exit(1)
    else:
        # Use the first PDF found
        pdf_path = os.path.join(PDF_DIR, pdf_files[0])
        print(f"Using existing PDF: {pdf_path}")
    
    # Test PNG conversion
    print("\nTesting PNG conversion...")
    png_paths = ImageProcessor.pdf_to_images(
        pdf_path=pdf_path,
        output_dir="test_output",
        format="png"
    )
    if png_paths:
        png_size = os.path.getsize(png_paths[0])
        print(f"PNG file created: {png_paths[0]}")
        print(f"PNG file size: {png_size/1024:.2f} KB")
    
    # Test JPG conversion
    print("\nTesting JPG conversion...")
    jpg_paths = ImageProcessor.pdf_to_images(
        pdf_path=pdf_path,
        output_dir="test_output",
        format="jpg",
        quality=85
    )
    if jpg_paths:
        jpg_size = os.path.getsize(jpg_paths[0])
        print(f"JPG file created: {jpg_paths[0]}")
        print(f"JPG file size: {jpg_size/1024:.2f} KB")
        
        if 'png_size' in locals():
            reduction = (1 - (jpg_size / png_size)) * 100
            print(f"\nFile size reduction: {reduction:.2f}%")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_jpg_conversion()