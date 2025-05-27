#!/usr/bin/env python3
"""
Document Scanner Script - CamScanner-like image processing
Processes images to make them look like clean, scanned documents
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path


def order_points(pts):
    """Order points in top-left, top-right, bottom-right, bottom-left order"""
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def four_point_transform(image, pts):
    """Apply perspective transformation to get bird's eye view"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width and height of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute and apply the perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def find_document_contour(image):
    """Find the document contour in the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    h, w = image.shape[:2]
    min_area = (w * h) * 0.1  # Minimum 10% of image area

    # Find contour with 4 corners (document) that's large enough
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)  # More lenient approximation

        if len(approx) == 4:
            # Ensure the contour is reasonably sized and not too small
            rect = cv2.boundingRect(approx)
            if rect[2] > w * 0.5 and rect[3] > h * 0.5:  # At least 50% of image dimensions
                return approx.reshape(4, 2)

    # If no suitable 4-corner contour found, add small margin to image corners
    margin = min(w, h) * 0.02  # 2% margin
    return np.array([
        [margin, margin],
        [w - margin, margin],
        [w - margin, h - margin],
        [margin, h - margin]
    ], dtype="float32")


def enhance_document(image):
    """Apply document enhancement (contrast, brightness, noise reduction)"""
    # Convert to LAB color space for better processing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge channels and convert back to BGR
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Apply bilateral filter for noise reduction while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # Slight sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced


def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """Adjust brightness and contrast"""
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def process_document(image_path, output_path=None, auto_crop=True, enhance=True,
                    brightness=10, contrast=20, debug=False):
    """Main document processing function"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Resize if too large (for faster processing)
    height, width = image.shape[:2]
    scale_factor = 1.0
    if width > 1500:
        scale_factor = 1500 / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = cv2.resize(image, (new_width, new_height))

    original = image.copy()

    # Auto-crop document if enabled
    if auto_crop:
        try:
            doc_contour = find_document_contour(image)

            # Check if the detected contour is reasonable (not too different from original)
            h, w = image.shape[:2]
            contour_area = cv2.contourArea(doc_contour)
            image_area = h * w
            area_ratio = contour_area / image_area

            if debug:
                print(f"Debug: Contour area ratio: {area_ratio:.2f}")
                # Save debug image showing detected contour
                debug_img = original.copy()
                cv2.drawContours(debug_img, [doc_contour.astype(int)], -1, (0, 255, 0), 3)
                debug_path = str(Path(image_path).parent / f"debug_{Path(image_path).stem}.jpg")
                cv2.imwrite(debug_path, debug_img)
                print(f"Debug: Contour visualization saved to {debug_path}")

            # Only apply transformation if contour seems reasonable
            if area_ratio > 0.3:  # At least 30% of image area
                transformed = four_point_transform(image, doc_contour)
                # Additional check: make sure transformed image isn't too small
                th, tw = transformed.shape[:2]
                if tw > w * 0.3 and th > h * 0.3:  # At least 30% of original dimensions
                    image = transformed
                    print(f"✓ Applied perspective correction (area ratio: {area_ratio:.2f})")
                else:
                    print(f"! Perspective correction resulted in too small image, using original")
                    image = original
            else:
                print(f"! Detected contour too small ({area_ratio:.2f}), using original")
                image = original

        except Exception as e:
            print(f"! Perspective correction failed, using original: {e}")
            image = original

    # Enhance document
    if enhance:
        image = enhance_document(image)
        print(f"✓ Applied enhancement filters")

    # Adjust brightness and contrast
    if brightness != 0 or contrast != 0:
        image = adjust_brightness_contrast(image, brightness, contrast)
        print(f"✓ Adjusted brightness (+{brightness}) and contrast (+{contrast})")

    # Save processed image
    if output_path is None:
        base = Path(image_path)
        output_path = base.parent / f"{base.stem}_scanned{base.suffix}"

    cv2.imwrite(str(output_path), image)
    print(f"✓ Saved processed document: {output_path}")

    return str(output_path)


def batch_process(input_dir, output_dir=None, **kwargs):
    """Process multiple images in a directory"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    else:
        output_path = input_path / "scanned"
        output_path.mkdir(exist_ok=True)

    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    image_files = [f for f in input_path.iterdir()
                  if f.suffix.lower() in extensions]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Processing {len(image_files)} images...")

    for img_file in image_files:
        try:
            output_file = output_path / f"{img_file.stem}_scanned{img_file.suffix}"
            process_document(str(img_file), str(output_file), **kwargs)
        except Exception as e:
            print(f"✗ Failed to process {img_file}: {e}")

    print(f"\nBatch processing complete! Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Document Scanner - CamScanner-like processing")

    # Input/output arguments
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("-b", "--batch", action="store_true",
                       help="Process all images in input directory")

    # Processing options
    parser.add_argument("--no-crop", action="store_true",
                       help="Disable automatic document cropping")
    parser.add_argument("--no-enhance", action="store_true",
                       help="Disable image enhancement")
    parser.add_argument("--brightness", type=int, default=10,
                       help="Brightness adjustment (-100 to 100, default: 10)")
    parser.add_argument("--contrast", type=int, default=20,
                       help="Contrast adjustment (-100 to 100, default: 20)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (saves contour visualization)")

    args = parser.parse_args()

    # Validate brightness and contrast values
    if not (-100 <= args.brightness <= 100):
        print("Brightness must be between -100 and 100")
        return
    if not (-100 <= args.contrast <= 100):
        print("Contrast must be between -100 and 100")
        return

    try:
        if args.batch:
            batch_process(
                args.input,
                args.output,
                auto_crop=not args.no_crop,
                enhance=not args.no_enhance,
                brightness=args.brightness,
                contrast=args.contrast
            )
        else:
            process_document(
                args.input,
                args.output,
                auto_crop=not args.no_crop,
                enhance=not args.no_enhance,
                brightness=args.brightness,
                contrast=args.contrast,
                debug=args.debug
            )

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())


# Example usage:
"""
# Process single image with default settings
python doc_scanner.py image1.jpg

# Process single image with custom output
python doc_scanner.py image1.jpg -o scanned_image1.jpg

# Process all images in a directory
python doc_scanner.py /path/to/images -b -o /path/to/output

# Process with custom brightness/contrast
python doc_scanner.py image1.jpg --brightness 15 --contrast 25

# Process without auto-cropping
python doc_scanner.py image1.jpg --no-crop

# Process without enhancement filters
python doc_scanner.py image1.jpg --no-enhance
"""
