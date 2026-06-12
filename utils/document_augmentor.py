# document_augmentor.py
import argparse
import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
import json
from pathlib import Path

# Ensure project root is on sys.path so `from config.config import *` works
# when this script is executed directly (python utils/document_augmentor.py)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.config import *
import os
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_augmentor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentAugmentor:
    def __init__(self):
        self.noise_textures = []
        self.paper_textures = []
        self.load_textures()
        self.transformation_matrix = None
        self.inverse_transformation_matrix = None
        self.current_image_shape = None
        self.perspective_matrix = None
        self.ink_bleed_kernel = (3, 3)
        self.distortion_params = {
            # Mild lens distortion from office scanners and phone captures.
            "k1": random.uniform(-0.003, 0.003),
            "k2": random.uniform(-0.0015, 0.0015),
            "p1": random.uniform(-0.0015, 0.0015),
            "p2": random.uniform(-0.0015, 0.0015)
        }

    def load_textures(self):
        # Fixed: Using absolute path might cause issues, using relative paths
        texture_dir = Path("data/textures")
        if texture_dir.exists():
            for texture_path in texture_dir.glob("*.jpg"):
                try:
                    texture = cv2.imread(str(texture_path))
                    if texture is not None:
                        self.noise_textures.append(texture)
                    else:
                        logger.warning(f"Failed to load texture: {texture_path}")
                except Exception as e:
                    logger.error(f"Error loading texture {texture_path}: {e}")
        else:
            logger.warning("Texture directory not found. Augmentation using textures will be skipped.")

    def rotate_image(self, image, angle=None):
        if angle is None:
            # Phone/scanner skew, still small enough to keep text readable.
            angle = random.uniform(-8, 8)

        # Safe handling of image dimensions
        if image is None:
            logger.error("Cannot rotate None image")
            return image, 0

        # Get dimensions safely
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Calculate rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # Calculate new dimensions
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        self.transformation_matrix = M
        self.current_image_shape = (new_h, new_w)

        # Use safer border mode and handle potential errors
        try:
            rotated_image = cv2.warpAffine(
                image, M, (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,  # Changed to CONSTANT
                borderValue=(255, 255, 255)      # White border
            )

            # Calculate inverse matrix
            inverse_matrix = np.vstack([M, [0, 0, 1]])
            inverse_matrix = np.linalg.inv(inverse_matrix)[:2]
            self.inverse_transformation_matrix = inverse_matrix

            return rotated_image, angle
        except Exception as e:
            logger.error(f"Error in rotate_image: {e}")
            return image, 0

    def apply_random_perspective(self, image):
        if image is None:
            logger.error("Cannot apply perspective to None image")
            return image

        h, w = image.shape[:2]

        # Fixed: Define source points properly
        src_points = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ])

        # Real phone captures often have visible trapezoid distortion.
        perspective_strength = random.uniform(0.035, 0.085)
        max_offset_w = w * perspective_strength
        max_offset_h = h * perspective_strength

        dst_points = np.float32([
            [random.uniform(0, max_offset_w), random.uniform(0, max_offset_h)],
            [w - random.uniform(0, max_offset_w), random.uniform(0, max_offset_h)],
            [w - random.uniform(0, max_offset_w), h - random.uniform(0, max_offset_h)],
            [random.uniform(0, max_offset_w), h - random.uniform(0, max_offset_h)]
        ])

        try:
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            warped = cv2.warpPerspective(
                image, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
            self.perspective_matrix = M
            return warped
        except Exception as e:
            logger.error(f"Error in apply_random_perspective: {e}")
            return image

    def transform_bbox_to_rotated_polygon(self, bbox):
        if self.transformation_matrix is None:
            x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            return [
                {'x': int(x), 'y': int(y)},
                {'x': int(x + width), 'y': int(y)},
                {'x': int(x + width), 'y': int(y + height)},
                {'x': int(x), 'y': int(y + height)}
            ]

        required_keys = ['x', 'y', 'width', 'height']
        for key in required_keys:
            if key not in bbox:
                logger.error("Missing key in bbox: %s", key)
                return None

        x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']

        corners = np.array([
            [x, y, 1],
            [x + width, y, 1],
            [x + width, y + height, 1],
            [x, y + height, 1]
        ], dtype=np.float32).T

        try:
            transformed_corners = self.transformation_matrix @ corners
            transformed_corners = transformed_corners.T[:, :2]

            h, w = self.current_image_shape[:2]

            polygon_vertices = []
            for corner in transformed_corners:
                x_coord = max(0, min(w - 1, corner[0]))
                y_coord = max(0, min(h - 1, corner[1]))
                polygon_vertices.append({
                    'x': int(x_coord),
                    'y': int(y_coord)
                })

            return polygon_vertices
        except Exception as e:
            logger.error(f"Error in transform_bbox_to_rotated_polygon: {e}")
            return None

    def transform_bbox_to_axis_aligned(self, bbox):
        if self.transformation_matrix is None:
            return bbox

        required_keys = ['x', 'y', 'width', 'height']
        for key in required_keys:
            if key not in bbox:
                logger.error("Missing key in bbox: %s", key)
                return None

        x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']

        corners = np.array([
            [x, y, 1],
            [x + width, y, 1],
            [x + width, y + height, 1],
            [x, y + height, 1]
        ], dtype=np.float32).T

        try:
            transformed_corners = self.transformation_matrix @ corners
            transformed_corners = transformed_corners.T[:, :2]

            h, w = self.current_image_shape[:2]

            min_x = max(0, np.min(transformed_corners[:, 0]))
            max_x = min(w - 1, np.max(transformed_corners[:, 0]))
            min_y = max(0, np.min(transformed_corners[:, 1]))
            max_y = min(h - 1, np.max(transformed_corners[:, 1]))

            new_width = max(1, max_x - min_x)
            new_height = max(1, max_y - min_y)

            return {
                'x': float(min_x),
                'y': float(min_y),
                'width': float(new_width),
                'height': float(new_height)
            }
        except Exception as e:
            logger.error(f"Error in transform_bbox_to_axis_aligned: {e}")
            return bbox

    def transform_polygon_vertices(self, vertices):
        if self.transformation_matrix is None:
            return vertices

        if not isinstance(vertices, list):
            logger.error("Vertices must be a list, got %s", type(vertices))
            return vertices

        transformed_vertices = []
        for vertex in vertices:
            if 'x' not in vertex or 'y' not in vertex:
                logger.error("Vertex missing x or y coordinate: %s", vertex)
                continue

            point = np.array([vertex['x'], vertex['y'], 1], dtype=np.float32)
            try:
                transformed_point = self.transformation_matrix @ point
                transformed_vertices.append({
                    'x': int(max(0, min(self.current_image_shape[1] - 1, transformed_point[0]))),
                    'y': int(max(0, min(self.current_image_shape[0] - 1, transformed_point[1])))
                })
            except Exception as e:
                logger.error(f"Error transforming vertex: {e}")
                transformed_vertices.append(vertex)  # Keep original in case of error

        return transformed_vertices

    def transform_perspective_polygon(self, vertices):
        """Transform polygon vertices using perspective transformation matrix"""
        if self.perspective_matrix is None:
            return vertices

        transformed_vertices = []

        try:
            # Convert the vertices dictionary to array format that cv2 can use
            points_array = np.array([[vertex['x'], vertex['y']] for vertex in vertices], dtype=np.float32)

            # Need to reshape for perspectiveTransform
            points_array = points_array.reshape(-1, 1, 2)

            # Apply perspective transformation
            transformed_points = cv2.perspectiveTransform(points_array, self.perspective_matrix)

            # Convert back to our format
            for point in transformed_points.reshape(-1, 2):
                transformed_vertices.append({
                    'x': int(max(0, min(self.current_image_shape[1]-1, point[0]))),
                    'y': int(max(0, min(self.current_image_shape[0]-1, point[1])))
                })

            return transformed_vertices
        except Exception as e:
            logger.error(f"Error in transform_perspective_polygon: {e}")
            return vertices  # Return original on error

    def apply_ink_bleed(self, image):
        """Applies stronger ink bleed effect to simulate poor quality printing/handling."""
        if image is None or image.size == 0:
            logger.error("Cannot apply ink bleed to invalid image")
            return image

        try:
            # Stronger blur for more visible ink bleed
            bleed_kernel = random.choice([(3, 3), (5, 5), (5, 5), (7, 7)])
            blurred = cv2.GaussianBlur(image, bleed_kernel, 0)
            # More aggressive ink spread
            return cv2.addWeighted(image, 0.75, blurred, 0.25, 0)
        except Exception as e:
            logger.error(f"Error in apply_ink_bleed: {e}")
            return image

    def apply_distortion(self, image):
        """Distorts the image using lens distortion parameters (simulates scanner lens effects)."""
        if image is None:
            logger.error("Cannot apply distortion to None image")
            return image

        h, w = image.shape[:2]

        # Fixed: Define proper camera matrix
        camera_matrix = np.array([
            [w * 1.5, 0, w/2],
            [0, h * 1.5, h/2],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.float32([
            self.distortion_params["k1"],
            self.distortion_params["k2"],
            self.distortion_params["p1"],
            self.distortion_params["p2"],
            0
        ])

        try:
            return cv2.undistort(image, camera_matrix, dist_coeffs)
        except Exception as e:
            logger.error(f"Error in apply_distortion: {e}")
            return image

    def apply_shadow(self, image):
        """Adds realistic shadow effects from scanner lid or uneven lighting"""
        if image is None:
            logger.error("Cannot apply shadow to None image")
            return image

        try:
            h, w = image.shape[:2]
            shadow_image = image.copy().astype(np.float32)

            # Aggressive shadow effects (scanner lid shadows, uneven lighting)
            shadow_type = random.choice(['edge', 'gradient', 'corner', 'strong_gradient'])

            if shadow_type == 'edge':
                # Strong edge shadow (common in scanner margins)
                shadow_intensity = random.uniform(0.25, 0.48)
                gradient = np.linspace(shadow_intensity, 0, w // 2)
                gradient = np.tile(gradient, (h, 1))
                
                for i in range(3):
                    shadow_image[:, :w//2, i] *= (1 - gradient * 0.5)
                    
            elif shadow_type == 'gradient':
                # Uneven lighting gradient (strong)
                shadow_intensity = random.uniform(0.18, 0.36)
                X_m, Y_m = np.mgrid[0:h, 0:w]
                gradient = (Y_m / w + X_m / h) / 2
                
                for i in range(3):
                    shadow_image[:, :, i] *= (1 - gradient * shadow_intensity)
                    
            elif shadow_type == 'corner':
                # Strong corner shadow
                shadow_intensity = random.uniform(0.24, 0.42)
                X_m, Y_m = np.mgrid[0:h, 0:w]
                distance = np.sqrt((X_m/h)**2 + (Y_m/w)**2)
                gradient = np.clip(distance, 0, 1)
                
                for i in range(3):
                    shadow_image[:, :, i] *= (1 - gradient * shadow_intensity)
            
            elif shadow_type == 'strong_gradient':
                # Very strong lighting gradient (poor scanner)
                shadow_intensity = random.uniform(0.30, 0.52)
                X_m, Y_m = np.mgrid[0:h, 0:w]
                gradient = np.sin(X_m / h * np.pi) * np.cos(Y_m / w * np.pi)
                gradient = (gradient + 1) / 2  # Normalize to 0-1
                
                for i in range(3):
                    shadow_image[:, :, i] *= (1 - gradient * shadow_intensity)

            return np.clip(shadow_image, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in apply_shadow: {e}")
            return image

    def apply_random_texture(self, image):
        if not self.noise_textures:  # Check if textures loaded correctly
            return image

        if image is None:
            logger.error("Cannot apply texture to None image")
            return image

        try:
            # Choose a random texture
            texture = random.choice(self.noise_textures)

            # Handle potential size mismatch safely
            if texture.shape[0] < image.shape[0] or texture.shape[1] < image.shape[1]:
                # Use a tiling approach instead of resize for large images
                rows_needed = (image.shape[0] // texture.shape[0]) + 1
                cols_needed = (image.shape[1] // texture.shape[1]) + 1
                texture_tiled = np.tile(texture, (rows_needed, cols_needed, 1))
                texture = texture_tiled[:image.shape[0], :image.shape[1]]
            else:
                # Resize normally
                texture = cv2.resize(texture, (image.shape[1], image.shape[0]))

            # Higher opacity for more visible paper texture
            alpha = random.uniform(0.16, 0.36)
            return cv2.addWeighted(image, 1 - alpha, texture, alpha, 0)
        except Exception as e:
            logger.error(f"Error in apply_random_texture: {e}")
            return image

    def apply_jpeg_artifacts(self, image):
        """Simulates strong JPEG compression artifacts from low-quality scans."""
        if image is None:
            logger.error("Cannot apply JPEG artifacts to None image")
            return image

        try:
            # Lower JPEG quality for visible compression artifacts.
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(22, 62)]
            _, encoded_image = cv2.imencode('.jpg', image, encode_param)
            decoded_image = cv2.imdecode(encoded_image, 1)

            if decoded_image is None:
                logger.error("Failed to decode JPEG image")
                return image

            return decoded_image
        except Exception as e:
            logger.error(f"Error in apply_jpeg_artifacts: {e}")
            return image

    def add_scanner_lines(self, image):
        """Add prominent scanner lines/artifacts common in low-quality scanned documents"""
        if image is None:
            logger.error("Cannot add scanner lines to None image")
            return image

        try:
            h, w = image.shape[:2]
            img_copy = image.copy()

            # Add more frequent horizontal lines (scanner artifacts)
            num_lines = random.randint(6, 16)
            for _ in range(num_lines):
                y = random.randint(0, h-1)
                line_color = (random.randint(180, 230), random.randint(180, 230), random.randint(180, 230))
                thickness = random.choice([1, 1, 2])  # Some thicker lines
                cv2.line(img_copy, (0, y), (w, y), line_color, thickness)

            # Add occasional vertical streaks (dust on scanner)
            if random.random() < 0.55:
                num_streaks = random.randint(1, 6)
                for _ in range(num_streaks):
                    x = random.randint(0, w-1)
                    streak_color = (random.randint(190, 240), random.randint(190, 240), random.randint(190, 240))
                    cv2.line(img_copy, (x, 0), (x, h), streak_color, 1)

            return img_copy
        except Exception as e:
            logger.error(f"Error in add_scanner_lines: {e}")
            return image

    def add_uneven_illumination(self, image):
        """Simulate severe uneven scanner illumination (very common in office scanners)"""
        if image is None:
            logger.error("Cannot add uneven illumination to None image")
            return image

        try:
            h, w = image.shape[:2]
            img_copy = image.copy().astype(np.float32)

            # Create strong illumination gradient
            X_m, Y_m = np.mgrid[0:h, 0:w]
            center_x, center_y = w // 2, h // 2
            distance = np.sqrt(((X_m - center_x) / (w/2))**2 + ((Y_m - center_y) / (h/2))**2)
            
            # Normalize to 0-1 range
            distance = distance / distance.max()
            
            # Create stronger illumination effect.
            intensity = random.uniform(0.20, 0.42)
            illumination = 1 - (distance * intensity)

            # Apply to all channels
            for i in range(3):
                img_copy[:, :, i] = img_copy[:, :, i] * illumination

            return np.clip(img_copy, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in add_uneven_illumination: {e}")
            return image

    def apply_color_jitter(self, image):
        if image is None:
            logger.error("Cannot apply color jitter to None image")
            return image

        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            enhancer_brightness = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer_brightness.enhance(random.uniform(0.82, 1.18))

            enhancer_contrast = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer_contrast.enhance(random.uniform(0.78, 1.28))

            enhancer_color = ImageEnhance.Color(pil_image)
            pil_image = enhancer_color.enhance(random.uniform(0.70, 1.18))

            enhancer_sharpness = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer_sharpness.enhance(random.uniform(0.75, 1.38))

            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error in apply_color_jitter: {e}")
            return image

    def apply_thermal_print_effect(self, image):
        if image is None:
            logger.error("Cannot apply thermal print effect to None image")
            return image

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            noise = np.random.normal(0, random.uniform(5, 12), gray.shape).astype(np.float32)
            gray = np.clip(gray + noise, 0, 255).astype(np.uint8)
            gray = cv2.GaussianBlur(gray, (3, 3), 0.5)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.error(f"Error in apply_thermal_print_effect: {e}")
            return image

    def apply_low_resolution_rescan(self, image):
        """Simulate WhatsApp/email re-save or low-resolution office scanner output."""
        if image is None:
            logger.error("Cannot apply low-resolution rescan to None image")
            return image

        try:
            h, w = image.shape[:2]
            scale = random.uniform(0.45, 0.82)
            small_w = max(32, int(w * scale))
            small_h = max(32, int(h * scale))
            small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
            interpolation = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])
            return cv2.resize(small, (w, h), interpolation=interpolation)
        except Exception as e:
            logger.error(f"Error in apply_low_resolution_rescan: {e}")
            return image

    def add_motion_blur(self, image):
        """Add light directional blur from hand-held phone capture."""
        if image is None:
            logger.error("Cannot add motion blur to None image")
            return image

        try:
            kernel_size = random.choice([5, 7, 9])
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            kernel[kernel_size // 2, :] = 1.0
            angle = random.uniform(-35, 35)
            center = (kernel_size // 2, kernel_size // 2)
            rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
            kernel_sum = kernel.sum()
            if kernel_sum > 0:
                kernel /= kernel_sum
            blurred = cv2.filter2D(image, -1, kernel)
            return cv2.addWeighted(image, 0.35, blurred, 0.65, 0)
        except Exception as e:
            logger.error(f"Error in add_motion_blur: {e}")
            return image

    def add_vignette_shadow(self, image):
        """Darken edges like phone camera shadow or curled paper."""
        if image is None:
            logger.error("Cannot add vignette shadow to None image")
            return image

        try:
            h, w = image.shape[:2]
            x_kernel = cv2.getGaussianKernel(w, w * random.uniform(0.38, 0.58))
            y_kernel = cv2.getGaussianKernel(h, h * random.uniform(0.38, 0.58))
            mask = y_kernel @ x_kernel.T
            mask = mask / mask.max()
            strength = random.uniform(0.22, 0.45)
            vignette = (1 - strength) + (mask * strength)
            out = image.astype(np.float32)
            for channel in range(3):
                out[:, :, channel] *= vignette
            return np.clip(out, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in add_vignette_shadow: {e}")
            return image

    def add_stamp_or_pen_marks(self, image):
        """Add faint real-world marks without covering too much invoice text."""
        if image is None:
            logger.error("Cannot add stamp or pen marks to None image")
            return image

        try:
            h, w = image.shape[:2]
            overlay = image.copy()
            mark_color = random.choice([(35, 35, 180), (150, 55, 35), (40, 90, 160)])

            if random.random() < 0.6:
                center = (random.randint(w // 5, 4 * w // 5), random.randint(h // 6, 5 * h // 6))
                axes = (random.randint(w // 18, w // 10), random.randint(h // 45, h // 25))
                angle = random.uniform(-15, 15)
                cv2.ellipse(overlay, center, axes, angle, 0, 360, mark_color, random.choice([2, 3]))
                cv2.line(
                    overlay,
                    (center[0] - axes[0], center[1]),
                    (center[0] + axes[0], center[1]),
                    mark_color,
                    random.choice([1, 2]),
                )
            else:
                for _ in range(random.randint(1, 3)):
                    x1 = random.randint(w // 8, 7 * w // 8)
                    y1 = random.randint(h // 8, 7 * h // 8)
                    x2 = min(w - 1, max(0, x1 + random.randint(-w // 6, w // 6)))
                    y2 = min(h - 1, max(0, y1 + random.randint(-h // 18, h // 18)))
                    cv2.line(overlay, (x1, y1), (x2, y2), mark_color, random.choice([1, 2]))

            return cv2.addWeighted(image, 0.82, overlay, 0.18, 0)
        except Exception as e:
            logger.error(f"Error in add_stamp_or_pen_marks: {e}")
            return image

    def add_paper_texture(self, image):
        """Add paper texture overlay"""
        # This is just an alias for apply_random_texture for clarity in the code
        return self.apply_random_texture(image)

    def add_noise(self, image):
        """Add aggressive scanner noise (high noise from low-quality sensors)"""
        if image is None:
            logger.error("Cannot add noise to None image")
            return image

        try:
            # High noise levels for low-quality scanners
            noise_level = random.uniform(8, 22)
            
            # Strong Gaussian noise
            gaussian_noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
            
            # Add noise
            noisy_image = image.astype(np.float32) + gaussian_noise
            
            return np.clip(noisy_image, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in add_noise: {e}")
            return image

    def add_paper_aging(self, image):
        """Simulate strong paper aging/yellowing effect for old documents"""
        if image is None:
            logger.error("Cannot add paper aging to None image")
            return image

        try:
            img_copy = image.copy().astype(np.float32)
            
            # Strong yellowing effect
            aging_intensity = random.uniform(0.10, 0.22)
            
            # Increase warm tones (red/yellow channels)
            img_copy[:, :, 2] = np.clip(img_copy[:, :, 2] * (1 + aging_intensity), 0, 255)  # Red channel
            img_copy[:, :, 1] = np.clip(img_copy[:, :, 1] * (1 + aging_intensity * 0.7), 0, 255)  # Green channel
            # Slightly reduce blue (makes it more yellow/brown)
            img_copy[:, :, 0] = np.clip(img_copy[:, :, 0] * (1 - aging_intensity * 0.3), 0, 255)
            
            return img_copy.astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in add_paper_aging: {e}")
            return image

    def add_blur(self, image):
        """Add stronger blur (motion blur, out-of-focus scanning)"""
        if image is None:
            logger.error("Cannot add blur to None image")
            return image

        try:
            # Larger kernel and sigma for more visible blur
            kernel_size = random.choice([3, 5, 5, 7])
            sigma = random.uniform(0.9, 2.6)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        except Exception as e:
            logger.error(f"Error in add_blur: {e}")
            return image

    def adjust_brightness_contrast(self, image):
        """Adjust brightness and contrast"""
        if image is None:
            logger.error("Cannot adjust brightness/contrast of None image")
            return image

        try:
            alpha = random.uniform(0.68, 1.42)  # Contrast
            beta = random.uniform(-42, 38)      # Brightness
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        except Exception as e:
            logger.error(f"Error in adjust_brightness_contrast: {e}")
            return image

    def add_fold_effect(self, image):
        """Add more prominent fold/crease effect"""
        if image is None:
            logger.error("Cannot add fold effect to None image")
            return image

        try:
            h, w = image.shape[:2]
            img_copy = image.copy()
            
            # Add 1-4 fold lines
            num_folds = random.randint(1, 5)
            for _ in range(num_folds):
                fold_x = random.randint(w // 5, 4 * w // 5)
                fold_y = random.randint(h // 5, 4 * h // 5)
                
                # Horizontal or vertical fold
                if random.random() < 0.5:
                    # Horizontal fold
                    cv2.line(img_copy, (0, fold_y), (w, fold_y), (185, 185, 185), random.choice([1, 2, 2]))
                else:
                    # Vertical fold
                    cv2.line(img_copy, (fold_x, 0), (fold_x, h), (185, 185, 185), random.choice([1, 2, 2]))
            
            return img_copy
        except Exception as e:
            logger.error(f"Error in add_fold_effect: {e}")
            return image

    def add_coffee_stain(self, image):
        """Add more prominent coffee stain effect"""
        if image is None:
            logger.error("Cannot add coffee stain to None image")
            return image

        try:
            h, w = image.shape[:2]

            # Check for reasonable image dimensions
            if h <= 100 or w <= 100:
                logger.warning("Image too small for coffee stain effect")
                return image

            stain_x = random.randint(50, w - 50)
            stain_y = random.randint(50, h - 50)
            radius = random.randint(35, min(85, min(w, h) // 3))

            # Create circular stain with stronger effect
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (stain_x, stain_y), radius, 255, -1)
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            mask = mask.astype(np.float32) / 255.0

            stain_color = np.array([120, 100, 80], dtype=np.uint8)

            # Apply the stain with stronger intensity
            img_copy = image.copy()
            for i in range(3):
                img_copy[:, :, i] = np.clip(
                    image[:, :, i] * (1 - mask * 0.32) + stain_color[i] * mask * 0.32,
                    0, 255
                ).astype(np.uint8)

            return img_copy
        except Exception as e:
            logger.error(f"Error in add_coffee_stain: {e}")
            return image

    def add_wrinkles(self, image):
        """Add more prominent wrinkle effects"""
        if image is None:
            logger.error("Cannot add wrinkles to None image")
            return image

        try:
            h, w = image.shape[:2]
            img_copy = image.copy()

            # Add more random lines to simulate wrinkles
            for _ in range(random.randint(4, 10)):
                pt1 = (random.randint(0, w-1), random.randint(0, h-1))
                pt2 = (random.randint(0, w-1), random.randint(0, h-1))
                thickness = random.choice([1, 1, 2])
                cv2.line(img_copy, pt1, pt2, (210, 210, 210), thickness)

            return img_copy
        except Exception as e:
            logger.error(f"Error in add_wrinkles: {e}")
            return image

    def add_dust_spots(self, image):
        """Add dust spots and specks (common on scanner glass)"""
        if image is None:
            logger.error("Cannot add dust spots to None image")
            return image

        try:
            h, w = image.shape[:2]
            img_copy = image.copy()

            # Add multiple dust spots
            num_spots = random.randint(18, 55)
            for _ in range(num_spots):
                x = random.randint(0, w-1)
                y = random.randint(0, h-1)
                radius = random.randint(1, 4)
                
                # Random brightness (white or dark spots)
                if random.random() < 0.6:
                    # White/light dust
                    spot_color = (random.randint(220, 255), random.randint(220, 255), random.randint(220, 255))
                else:
                    # Dark dust
                    spot_color = (random.randint(0, 40), random.randint(0, 40), random.randint(0, 40))
                
                cv2.circle(img_copy, (x, y), radius, spot_color, -1)

            return img_copy
        except Exception as e:
            logger.error(f"Error in add_dust_spots: {e}")
            return image

    def validate_bounding_box(self, bbox, image_shape):
        """Validate bounding box coordinates"""
        if image_shape is None or len(image_shape) < 2:
            logger.error(f"Invalid image shape: {image_shape}")
            return False

        h, w = image_shape[:2]

        # Check if bbox has required keys
        if not all(k in bbox for k in ['x', 'y', 'width', 'height']):
            logger.error(f"Bounding box missing required keys: {bbox}")
            return False

        # Check if coordinates are within image bounds
        if bbox['x'] < 0 or bbox['y'] < 0 or bbox['x'] + bbox['width'] > w or bbox['y'] + bbox['height'] > h:
            logger.warning(f"Bounding box exceeds image bounds: {bbox}, image: {w}x{h}")

            # Fix the bounding box
            bbox['x'] = max(0, min(bbox['x'], w))
            bbox['y'] = max(0, min(bbox['y'], h))
            bbox['width'] = min(bbox['width'], w - bbox['x'])
            bbox['height'] = min(bbox['height'], h - bbox['y'])

            logger.info(f"Fixed bounding box: {bbox}")

        # Check for valid dimensions
        if bbox['width'] <= 0 or bbox['height'] <= 0:
            logger.error(f"Invalid bounding box dimensions: {bbox}")
            return False

        return True

    def process_image_and_annotations(self, image_path, annotation_path, output_image_path, output_annotation_path):
        # Check if files exist first
        if not os.path.isfile(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return False

        if not os.path.isfile(annotation_path):
            logger.error(f"Annotation file does not exist: {annotation_path}")
            return False

        # Load image with proper error handling
        try:
            # Set maximum image size to prevent memory issues
            MAX_IMAGE_SIZE = 4000  # pixels

            # Read image using PIL first to check dimensions
            pil_img = Image.open(image_path)
            w, h = pil_img.size

            # If image is too large, resize it before processing
            if w > MAX_IMAGE_SIZE or h > MAX_IMAGE_SIZE:
                logger.warning(f"Image too large ({w}x{h}), resizing for processing")

                # Calculate new dimensions while maintaining aspect ratio
                if w > h:
                    new_w = MAX_IMAGE_SIZE
                    new_h = int(h * MAX_IMAGE_SIZE / w)
                else:
                    new_h = MAX_IMAGE_SIZE
                    new_w = int(w * MAX_IMAGE_SIZE / h)

                # Convert to numpy array and use cv2 for resize
                image = np.array(pil_img)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Also adjust annotations proportionally
                scale_x = new_w / w
                scale_y = new_h / h
            else:
                # Load directly with OpenCV if size is okay
                image = cv2.imread(image_path)
                scale_x = 1.0
                scale_y = 1.0

            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return False

        # Load annotations
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            # If we resized the image, update annotations
            if scale_x != 1.0 or scale_y != 1.0:
                if 'image_size' in annotations:
                    annotations['image_size']['width'] *= scale_x
                    annotations['image_size']['height'] *= scale_y

                # Scale all coordinates
                if 'entities' in annotations:
                    if 'ocr_text' in annotations['entities']:
                        for text_entry in annotations['entities']['ocr_text']:
                            if 'bbox' in text_entry:
                                text_entry['bbox']['x'] *= scale_x
                                text_entry['bbox']['y'] *= scale_y
                                text_entry['bbox']['width'] *= scale_x
                                text_entry['bbox']['height'] *= scale_y

                            if 'bounding_poly' in text_entry and 'vertices' in text_entry['bounding_poly']:
                                for vertex in text_entry['bounding_poly']['vertices']:
                                    if 'x' in vertex:
                                        vertex['x'] *= scale_x
                                    if 'y' in vertex:
                                        vertex['y'] *= scale_y

                    if 'table' in annotations['entities']:
                        for table_entry in annotations['entities']['table']:
                            if 'bbox' in table_entry:
                                table_entry['bbox']['x'] *= scale_x
                                table_entry['bbox']['y'] *= scale_y
                                table_entry['bbox']['width'] *= scale_x
                                table_entry['bbox']['height'] *= scale_y

                            if 'bounding_poly' in table_entry and 'vertices' in table_entry['bounding_poly']:
                                for vertex in table_entry['bounding_poly']['vertices']:
                                    if 'x' in vertex:
                                        vertex['x'] *= scale_x
                                    if 'y' in vertex:
                                        vertex['y'] *= scale_y

        except Exception as e:
            logger.error(f"Failed to read annotations: {annotation_path} ({str(e)})")
            return False

        # Initialize transformation matrices
        self.current_image_shape = image.shape
        self.transformation_matrix = None
        self.perspective_matrix = None

        # Track confidence reduction due to transformations
        confidence_multiplier = 1.0
        rotation_applied = False
        perspective_applied = False

        # Aggressive augmentation pipeline for real-world scanning conditions
        # Check image integrity at each step
        try:
            # Start with uneven illumination (very common in real scans)
            if image is not None and random.random() < 0.85:
                image = self.add_uneven_illumination(image)

            # Paper aging effect (strong yellowing)
            if image is not None and random.random() < 0.60:
                image = self.add_paper_aging(image)

            # Paper texture (more aggressive)
            if image is not None and random.random() < 0.75:
                image = self.apply_random_texture(image)

            # Thermal print effect
            if image is not None and random.random() < 0.35:
                image = self.apply_thermal_print_effect(image)

            # Re-scan / forwarded image degradation
            if image is not None and random.random() < 0.38:
                image = self.apply_low_resolution_rescan(image)

            # Add folds (more likely)
            if image is not None and random.random() < 0.35:
                image = self.add_fold_effect(image)

            # Coffee stains (more frequent)
            if image is not None and random.random() < 0.18:
                image = self.add_coffee_stain(image)

            # Wrinkles (more prominent)
            if image is not None and random.random() < 0.45:
                image = self.add_wrinkles(image)

            # Dust spots (very common)
            if image is not None and random.random() < 0.75:
                image = self.add_dust_spots(image)

            # Faint stamp/pen marks seen on handled invoices
            if image is not None and random.random() < 0.20:
                image = self.add_stamp_or_pen_marks(image)

            # Apply ink bleed (stronger)
            if image is not None and random.random() < 0.40:
                image = self.apply_ink_bleed(image)

            # Add scanner lines (very common in real scans)
            if image is not None and random.random() < 0.78:
                image = self.add_scanner_lines(image)

            # Blur (more aggressive)
            if image is not None and random.random() < 0.58:
                image = self.add_blur(image)

            # Motion blur from phone capture
            if image is not None and random.random() < 0.34:
                image = self.add_motion_blur(image)

            # JPEG artifacts (more aggressive)
            if image is not None and random.random() < 0.85:
                image = self.apply_jpeg_artifacts(image)

            # Noise (higher levels)
            if image is not None and random.random() < 0.75:
                image = self.add_noise(image)

            # Distortion (subtle but more likely)
            if image is not None and random.random() < 0.32:
                image = self.apply_distortion(image)

            # Global exposure/contrast shift
            if image is not None and random.random() < 0.55:
                image = self.adjust_brightness_contrast(image)

            # Color jitter
            if image is not None and random.random() < 0.70:
                image = self.apply_color_jitter(image)

            # Shadow effects (very common)
            if image is not None and random.random() < 0.80:
                image = self.apply_shadow(image)

            # Phone-camera edge darkening
            if image is not None and random.random() < 0.50:
                image = self.add_vignette_shadow(image)

            # Geometric transformations last (more aggressive)
            if image is not None:
                if random.random() < 0.45:  # perspective with 45% probability
                    image = self.apply_random_perspective(image)
                    perspective_applied = True
                    confidence_multiplier *= 0.88  # Reduce confidence more
                    logger.info("Applied perspective transformation")
                elif random.random() < 0.45:  # Rotation with 45% probability only if perspective is not applied
                    image, rotation_angle = self.rotate_image(image)
                    rotation_applied = True
                    confidence_multiplier *= 0.82  # Reduce confidence more for rotation
                    logger.info(f"Applied rotation: {rotation_angle:.2f} degrees")

            # Update current image shape
            if image is not None:
                self.current_image_shape = image.shape

        except Exception as e:
            logger.error(f"Error during augmentation pipeline: {e}")
            logger.error(traceback.format_exc())
            # Continue with original image if augmentation fails
            image = cv2.imread(image_path)

        # Transform annotations if geometric transformations were applied
        if image is not None and ((rotation_applied and self.transformation_matrix is not None) or
                                 (perspective_applied and self.perspective_matrix is not None)):
            try:
                if 'entities' in annotations:
                    # Keep token-level and line-level labels aligned after geometry.
                    for text_key in ('ocr_text', 'text_lines'):
                        if text_key not in annotations['entities']:
                            continue

                        for text_entry in annotations['entities'][text_key]:
                            if 'confidence' in text_entry:
                                text_entry['confidence'] *= confidence_multiplier

                            if 'bounding_poly' in text_entry and 'vertices' in text_entry['bounding_poly']:
                                vertices = text_entry['bounding_poly']['vertices']
                                if rotation_applied:
                                    text_entry['bounding_poly']['vertices'] = self.transform_polygon_vertices(vertices)
                                elif perspective_applied:
                                    text_entry['bounding_poly']['vertices'] = self.transform_perspective_polygon(vertices)

                            elif 'bbox' in text_entry:
                                if rotation_applied:
                                    rotated_polygon = self.transform_bbox_to_rotated_polygon(text_entry['bbox'])
                                    if rotated_polygon:  # Only update if transformation was successful
                                        del text_entry['bbox']
                                        text_entry['bounding_poly'] = {'vertices': rotated_polygon}
                                elif perspective_applied:
                                    x, y, w, h = text_entry['bbox']['x'], text_entry['bbox']['y'], text_entry['bbox']['width'], text_entry['bbox']['height']
                                    polygon = [
                                        {'x': int(x), 'y': int(y)},
                                        {'x': int(x + w), 'y': int(y)},
                                        {'x': int(x + w), 'y': int(y + h)},
                                        {'x': int(x), 'y': int(y + h)}
                                    ]
                                    transformed_polygon = self.transform_perspective_polygon(polygon)
                                    del text_entry['bbox']  # Remove bbox and add bounding_poly
                                    text_entry['bounding_poly'] = {'vertices': transformed_polygon}
                if 'kie_fields' in annotations['entities']:
                    for field in annotations['entities']['kie_fields']:
                        if 'bbox' in field:
                            if rotation_applied:
                                new_bbox = self.transform_bbox_to_axis_aligned(field['bbox'])
                                if new_bbox:
                                    field['bbox'] = new_bbox
                            elif perspective_applied:
                                x, y, w, h = field['bbox']['x'], field['bbox']['y'], \
                                    field['bbox']['width'], field['bbox']['height']
                                polygon = [{'x': int(x), 'y': int(y)}, {'x': int(x + w), 'y': int(y)},
                                           {'x': int(x + w), 'y': int(y + h)}, {'x': int(x), 'y': int(y + h)}]
                                tp = self.transform_perspective_polygon(polygon)
                                xs = [v['x'] for v in tp];
                                ys = [v['y'] for v in tp]
                                field['bbox'] = {'x': min(xs), 'y': min(ys),
                                                 'width': max(xs) - min(xs), 'height': max(ys) - min(ys)}

                if 'entities' in annotations and 'table' in annotations['entities']:
                    # Rotate tables also
                    for table_entry in annotations['entities']['table']:
                        if 'bbox' in table_entry:
                            if rotation_applied:
                                rotated_polygon = self.transform_bbox_to_rotated_polygon(table_entry['bbox'])
                                if rotated_polygon:
                                    del table_entry['bbox']
                                    table_entry['bounding_poly'] = {'vertices': rotated_polygon}
                            elif perspective_applied:
                                x, y, w, h = table_entry['bbox']['x'], table_entry['bbox']['y'], table_entry['bbox']['width'], table_entry['bbox']['height']
                                polygon = [
                                    {'x': int(x), 'y': int(y)},
                                    {'x': int(x + w), 'y': int(y)},
                                    {'x': int(x + w), 'y': int(y + h)},
                                    {'x': int(x), 'y': int(y + h)}
                                ]
                                transformed_polygon = self.transform_perspective_polygon(polygon)
                                del table_entry['bbox']
                                table_entry['bounding_poly'] = {'vertices': transformed_polygon}
            except Exception as e:
                logger.error(f"Error transforming annotations: {e}")
                logger.error(traceback.format_exc())

        # Update image size in annotations
        if image is not None and 'image_size' in annotations:
            annotations['image_size']['width'] = self.current_image_shape[1]
            annotations['image_size']['height'] = self.current_image_shape[0]

        # Save augmented image and annotations
        try:
            if image is not None:
                # Create output directories if they don't exist
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(output_annotation_path), exist_ok=True)

                # Save as JPG with quality setting for better file size
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                cv2.imwrite(output_image_path, image, encode_param)

                with open(output_annotation_path, 'w', encoding='utf-8') as f:
                    json.dump(annotations, f, ensure_ascii=False, indent=2)

                logger.info(f"Successfully saved to {output_image_path} with annotations")
                return True
            else:
                logger.error("Failed to generate valid augmented image")
                return False
        except Exception as e:
            logger.error(f"Failed to save output: {str(e)}")
            logger.error(traceback.format_exc())
            return False

def _output_paths_for(input_filename, output_dir, output_annotation_dir):
    base_name = os.path.splitext(input_filename)[0]
    ext = os.path.splitext(input_filename)[1].lower()
    if ext in {'.jpg', '.jpeg'}:
        output_filename = f"augmented_{input_filename}"
    else:
        output_filename = f"augmented_{base_name}.jpg"
    return (
        os.path.join(output_dir, output_filename),
        os.path.join(output_annotation_dir, f"augmented_{base_name}.json"),
    )


def _is_complete_output(output_image_path, output_annotation_path):
    return (
        os.path.isfile(output_image_path)
        and os.path.getsize(output_image_path) > 0
        and os.path.isfile(output_annotation_path)
        and os.path.getsize(output_annotation_path) > 0
    )


def process_directory(input_dir, annotation_dir, output_dir, output_annotation_dir, max_images=None, force=False):
    """Process images and annotations, resuming by skipping complete outputs."""
    augmentor = DocumentAugmentor()

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_annotation_dir, exist_ok=True)

    processed_count = 0
    failed_count = 0
    skipped_count = 0
    missing_annotation_count = 0

    # Get list of all image files first, then process
    image_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(filename)

    # Sort files for consistent processing order
    image_files.sort()

    total_source_images = len(image_files)
    if max_images is not None:
        image_files = image_files[:max_images]

    logger.info(
        "Found %s source image(s); selected %s for augmentation%s",
        total_source_images,
        len(image_files),
        " (resume mode)" if not force else " (force mode)",
    )

    # Process each file
    for index, filename in enumerate(image_files, start=1):
        base_name = os.path.splitext(filename)[0]

        input_image_path = os.path.join(input_dir, filename)
        input_annotation_path = os.path.join(annotation_dir, f"{base_name}.json")

        output_image_path, output_annotation_path = _output_paths_for(filename, output_dir, output_annotation_dir)

        if not force and _is_complete_output(output_image_path, output_annotation_path):
            skipped_count += 1
            if skipped_count <= 5 or skipped_count % 500 == 0:
                logger.info("Skipping existing output %s (%s/%s)", filename, index, len(image_files))
            continue

        if os.path.exists(input_annotation_path):
            logger.info(f"Processing {filename} ({index}/{len(image_files)})...")
            try:
                success = augmentor.process_image_and_annotations(
                    input_image_path,
                    input_annotation_path,
                    output_image_path,
                    output_annotation_path
                )

                if success:
                    processed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                logger.error(traceback.format_exc())
                failed_count += 1
        else:
            logger.warning(f"No annotation file found for {filename}")
            missing_annotation_count += 1

    logger.info(
        "Processing complete: %s processed, %s skipped, %s failed, %s missing annotations",
        processed_count,
        skipped_count,
        failed_count,
        missing_annotation_count,
    )
    return processed_count, failed_count, skipped_count, missing_annotation_count

def visualize_bounding_boxes(image_path, annotation_path, output_path=None):
    """
    Visualize bounding boxes on an image to verify alignment.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
    import matplotlib.pyplot as plt
    import numpy as np

    # Check if files exist
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return False

    if not os.path.exists(annotation_path):
        logger.error(f"Annotation file not found: {annotation_path}")
        return False

    # Load image with error handling
    try:
        # Use PIL for safer image loading
        from PIL import Image as PILImage
        pil_img = PILImage.open(image_path)
        image = np.array(pil_img)

        # Convert if needed
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image, image, image], axis=2)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
    except Exception as e:
        logger.error(f"Failed to read image: {image_path} - {str(e)}")
        return False

    # Load annotations
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read annotations: {annotation_path} ({str(e)})")
        return False

    # Create figure
    plt.figure(figsize=(20, 20))
    plt.imshow(image)

    # Draw bounding boxes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Color palette

    # Process OCR text entities
    if 'entities' in annotations and 'ocr_text' in annotations['entities']:
        for i, text_entity in enumerate(annotations['entities']['ocr_text']):
            color = colors[i % len(colors)][:3]  # RGB

            if 'bounding_poly' in text_entity and 'vertices' in text_entity['bounding_poly']:
                vertices = text_entity['bounding_poly']['vertices']
                points = np.array([vertex for vertex in vertices])

                # Draw polygon
                plt.plot(points[:, 0], points[:, 1], '-', color=color, linewidth=2)
                plt.plot([points[-1, 0], points[0, 0]], [points[-1, 1], points[0, 1]],
                         '-', color=color, linewidth=2)

                # Add text for debugging
                if 'text' in text_entity:
                    centroid = points.mean(axis=0)
                    plt.text(centroid[0], centroid[1], text_entity['text'],
                             color='white', fontsize=8,
                             bbox=dict(facecolor=color, alpha=0.5))
            elif 'bbox' in text_entity:
                x, y = text_entity['bbox']['x'], text_entity['bbox']['y']
                w, h = text_entity['bbox']['width'], text_entity['bbox']['height']

                # Draw rectangle
                rect = plt.Rectangle((x, y), w, h, linewidth=2,
                                    edgecolor=color, facecolor='none')
                plt.gca().add_patch(rect)

                # Add text
                if 'text' in text_entity:
                    plt.text(x, y-5, text_entity['text'],
                             color='white', fontsize=8,
                             bbox=dict(facecolor=color, alpha=0.5))

    # Process table entities
    if 'entities' in annotations and 'table' in annotations['entities']:
        for i, table in enumerate(annotations['entities']['table']):
            color = 'lime'  # Distinct color for tables

            if 'bounding_poly' in table and 'vertices' in table['bounding_poly']:
                vertices = table['bounding_poly']['vertices']
                points = np.array([vertex for vertex in vertices])

                # Draw polygon
                plt.plot(points[:, 0], points[:, 1], '-', color=color, linewidth=3)
                plt.plot([points[-1, 0], points[0, 0]], [points[-1, 1], points[0, 1]],
                         '-', color=color, linewidth=3)
                plt.text(points[0, 0], points[0, 1]-10, f"Table {i+1}",
                         color='white', fontsize=12,
                         bbox=dict(facecolor=color, alpha=0.7))
            elif 'bbox' in table:
                x, y = table['bbox']['x'], table['bbox']['y']
                w, h = table['bbox']['width'], table['bbox']['height']

                # Draw rectangle
                rect = plt.Rectangle((x, y), w, h, linewidth=3,
                                    edgecolor=color, facecolor='none')
                plt.gca().add_patch(rect)
                plt.text(x, y-10, f"Table {i+1}", color='white', fontsize=12,
                         bbox=dict(facecolor=color, alpha=0.7))

    plt.axis('off')

    # Save or show the visualization
    try:
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()  # Close to free memory
            logger.info(f"Visualization saved to {output_path}")
            return True
        else:
            plt.show()
            return True
    except Exception as e:
        logger.error(f"Error saving visualization: {str(e)}")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Augment invoice images and annotations with resume support.")
    parser.add_argument("--input-dir", default=None, help="Source image directory. Defaults to IMAGES_DIR from config.")
    parser.add_argument("--annotation-dir", default=None, help="Source annotation directory. Defaults to ANNOTATIONS_DIR from config.")
    parser.add_argument("--output-dir", default=None, help="Augmented image directory. Defaults to AUGMENTED_IMAGES_DIR from config.")
    parser.add_argument("--output-annotation-dir", default=None, help="Augmented annotation directory. Defaults to AUGMENTED_ANNOTATIONS_DIR from config.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for smoke runs. Omit to process all source images.")
    parser.add_argument("--force", action="store_true", help="Recreate outputs even when image and annotation already exist.")
    parser.add_argument("--no-visualize", action="store_true", help="Skip sample bounding-box visualizations.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()

        # Define directories with safer error handling
        try:
            # Try to get from config
            input_dir = args.input_dir or IMAGES_DIR
            annotation_dir = args.annotation_dir or ANNOTATIONS_DIR
            output_dir = args.output_dir or AUGMENTED_IMAGES_DIR
            annotation_output_dir = args.output_annotation_dir or AUGMENTED_ANNOTATIONS_DIR
        except NameError:
            # Fallback to defaults
            logger.warning("Configuration variables not found, using default directories")
            input_dir = args.input_dir or "data/images"
            annotation_dir = args.annotation_dir or "data/annotations"
            output_dir = args.output_dir or "data/augmented_images"
            annotation_output_dir = args.output_annotation_dir or "data/augmented_annotations"

        # Process images with proper error handling
        try:
            max_images = args.max_images
            try:
                max_images = max_images if max_images is not None else NUM_AUGMENTED_IMAGES
            except NameError:
                pass

            processed, failed, skipped, missing_annotations = process_directory(
                input_dir,
                annotation_dir,
                output_dir,
                annotation_output_dir,
                max_images=max_images,
                force=args.force,
            )

            # Only visualize if we have successful augmentations
            if processed > 0 and not args.no_visualize:
                # Create visualization directory
                vis_dir = "bbox_visualizations"
                os.makedirs(vis_dir, exist_ok=True)

                # Get list of augmented images (prioritizing JPG format)
                augmented_images = [
                    f for f in os.listdir(output_dir)
                    if f.endswith(('.jpg', '.jpeg', '.png'))
                ]

                # Limit to 5 samples max
                num_samples = min(5, len(augmented_images))
                if num_samples > 0:
                    import random
                    samples = random.sample(augmented_images, num_samples)

                    for sample in samples:
                        base_name = os.path.splitext(sample)[0]
                        image_path = os.path.join(output_dir, sample)
                        annotation_path = os.path.join(annotation_output_dir, f"{base_name}.json")
                        output_path = os.path.join(vis_dir, f"viz_{base_name}.png")

                        if os.path.exists(annotation_path):
                            visualize_bounding_boxes(image_path, annotation_path, output_path)

        except Exception as e:
            logger.error(f"Error in main execution: {e}")
            logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"Critical error: {e}")
        logger.error(traceback.format_exc())
