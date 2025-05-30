# document_augmentor.py
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
import json
from pathlib import Path
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
            "k1": random.uniform(-0.005, 0.005),
            "k2": random.uniform(-0.002, 0.002),
            "p1": random.uniform(-0.002, 0.002),
            "p2": random.uniform(-0.002, 0.002)
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
            angle = random.uniform(-7, 7)

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

        max_offset_w = w * 0.03
        max_offset_h = h * 0.03

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
        """Applies a subtle ink bleed effect."""
        if image is None or image.size == 0:
            logger.error("Cannot apply ink bleed to invalid image")
            return image

        try:
            blurred = cv2.GaussianBlur(image, self.ink_bleed_kernel, 0)
            # Slightly darken the blurred image to simulate ink spread
            return cv2.addWeighted(image, 0.9, blurred, 0.1, 0)
        except Exception as e:
            logger.error(f"Error in apply_ink_bleed: {e}")
            return image

    def apply_distortion(self, image):
        """Distorts the image using lens distortion parameters."""
        if image is None:
            logger.error("Cannot apply distortion to None image")
            return image

        h, w = image.shape[:2]

        # Fixed: Define proper camera matrix
        camera_matrix = np.array([
            [w, 0, w/2],
            [0, h, h/2],
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
        """Adds a subtle shadow effect to simulate depth"""
        if image is None:
            logger.error("Cannot apply shadow to None image")
            return image

        try:
            h, w = image.shape[:2]
            mask = np.zeros_like(image[:, :, 0]).astype(np.float32)

            shadow_x_offset = 3
            shadow_y_offset = 3
            shadow_intensity = 0.3

            X_m, Y_m = np.mgrid[0:h, 0:w]
            shadow_mask = (Y_m / w + X_m / h) - shadow_x_offset / shadow_y_offset

            shadow_image = image.copy().astype(np.float32)

            for i in range(3):
                channel = shadow_image[:, :, i]
                channel[shadow_mask > 0] *= (1 - shadow_intensity)
                shadow_image[:, :, i] = channel

            return shadow_image.astype(np.uint8)
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

            alpha = random.uniform(0.05, 0.2)  # Lower opacity for more subtle texture
            return cv2.addWeighted(image, 1 - alpha, texture, alpha, 0)
        except Exception as e:
            logger.error(f"Error in apply_random_texture: {e}")
            return image

    def apply_jpeg_artifacts(self, image):
        """Simulates JPEG compression artifacts."""
        if image is None:
            logger.error("Cannot apply JPEG artifacts to None image")
            return image

        try:
            # Fixed: Use proper parameter format for imencode
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(70, 95)]
            _, encoded_image = cv2.imencode('.jpg', image, encode_param)
            decoded_image = cv2.imdecode(encoded_image, 1)

            if decoded_image is None:
                logger.error("Failed to decode JPEG image")
                return image

            return decoded_image
        except Exception as e:
            logger.error(f"Error in apply_jpeg_artifacts: {e}")
            return image

    def apply_color_jitter(self, image):
        if image is None:
            logger.error("Cannot apply color jitter to None image")
            return image

        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            enhancer_brightness = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer_brightness.enhance(random.uniform(0.9, 1.1))

            enhancer_contrast = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer_contrast.enhance(random.uniform(0.9, 1.1))

            enhancer_color = ImageEnhance.Color(pil_image)
            pil_image = enhancer_color.enhance(random.uniform(0.9, 1.1))

            enhancer_sharpness = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer_sharpness.enhance(random.uniform(0.9, 1.1))

            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error in apply_color_jitter: {e}")
            return image

    def apply_thermal_print_effect(self, image):
        if image is None:
            logger.error("Cannot apply thermal print effect to None image")
            return image

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            noise = np.random.normal(0, 3, gray.shape).astype(np.uint8)
            gray = cv2.add(gray, noise)
            gray = cv2.GaussianBlur(gray, (3, 3), 0.5)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.error(f"Error in apply_thermal_print_effect: {e}")
            return image

    def add_paper_texture(self, image):
        """Add paper texture overlay"""
        # This is just an alias for apply_random_texture for clarity in the code
        return self.apply_random_texture(image)

    def add_noise(self, image):
        """Add random noise"""
        if image is None:
            logger.error("Cannot add noise to None image")
            return image

        try:
            noise_level = random.uniform(5, 15)
            noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
            return cv2.add(image, noise)
        except Exception as e:
            logger.error(f"Error in add_noise: {e}")
            return image

    def add_blur(self, image):
        """Add slight blur"""
        if image is None:
            logger.error("Cannot add blur to None image")
            return image

        try:
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.5, 1.5)
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
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-20, 20)    # Brightness
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        except Exception as e:
            logger.error(f"Error in adjust_brightness_contrast: {e}")
            return image

    def add_fold_effect(self, image):
        """Add fold/crease effect"""
        if image is None:
            logger.error("Cannot add fold effect to None image")
            return image

        try:
            h, w = image.shape[:2]
            # Create a simple fold line
            fold_x = random.randint(w // 4, 3 * w // 4)
            cv2.line(image, (fold_x, 0), (fold_x, h), (200, 200, 200), 2)
            return image
        except Exception as e:
            logger.error(f"Error in add_fold_effect: {e}")
            return image

    def add_coffee_stain(self, image):
        """Add coffee stain effect"""
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
            radius = random.randint(20, min(50, min(w, h) // 4))

            # Create circular stain
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (stain_x, stain_y), radius, 255, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            mask = mask.astype(np.float32) / 255.0

            stain_color = np.array([139, 119, 101], dtype=np.uint8)

            # Apply the stain safely
            img_copy = image.copy()
            for i in range(3):
                img_copy[:, :, i] = np.clip(
                    image[:, :, i] * (1 - mask * 0.3) + stain_color[i] * mask * 0.3,
                    0, 255
                ).astype(np.uint8)

            return img_copy
        except Exception as e:
            logger.error(f"Error in add_coffee_stain: {e}")
            return image

    def add_wrinkles(self, image):
        """Add wrinkle effects"""
        if image is None:
            logger.error("Cannot add wrinkles to None image")
            return image

        try:
            h, w = image.shape[:2]
            img_copy = image.copy()

            # Add some random lines to simulate wrinkles
            for _ in range(random.randint(2, 5)):
                pt1 = (random.randint(0, w-1), random.randint(0, h-1))
                pt2 = (random.randint(0, w-1), random.randint(0, h-1))
                cv2.line(img_copy, pt1, pt2, (220, 220, 220), 1)

            return img_copy
        except Exception as e:
            logger.error(f"Error in add_wrinkles: {e}")
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

        # Augmentation pipeline with varied probabilities and order
        # Check image integrity at each step
        try:
            # Start with thermal effect and paper texture
            if image is not None and random.random() < 0.85:
                image = self.apply_thermal_print_effect(image)

            if image is not None and random.random() < 0.9:
                image = self.apply_random_texture(image)

            # Add fold, coffee stain and wrinkles. Applying folds before coffee stains
            if image is not None and random.random() < 0.3:
                image = self.add_fold_effect(image)

            if image is not None and random.random() < 0.15:
                image = self.add_coffee_stain(image)

            if image is not None and random.random() < 0.4:
                image = self.add_wrinkles(image)

            # Apply ink bleed before blur, noise, and color jitter
            if image is not None and random.random() < 0.3:
                image = self.apply_ink_bleed(image)

            if image is not None and random.random() < 0.85:
                image = self.add_blur(image)

            # Apply noise and jpeg artifacts at end
            if image is not None and random.random() < 0.9:
                image = self.apply_jpeg_artifacts(image)

            if image is not None and random.random() < 0.8:
                image = self.add_noise(image)

            # Apply distortion after noise but before color jitter and shadow
            if image is not None and random.random() < 0.2:
                image = self.apply_distortion(image)

            if image is not None and random.random() < 0.9:
                image = self.apply_color_jitter(image)

            # Apply Shadow with higher probability
            if image is not None and random.random() < 0.8:
                image = self.apply_shadow(image)

            # Geometric transformations last, to ensure coordinate mapping works correctly
            if image is not None:
                if random.random() < 0.3:  # perspective with 30% probability
                    image = self.apply_random_perspective(image)
                    perspective_applied = True
                    confidence_multiplier *= 0.95  # Reduce confidence slightly
                    logger.info("Applied perspective transformation")
                elif random.random() < 0.6:  # Rotation with 60% probability only if perspective is not applied
                    image, rotation_angle = self.rotate_image(image)
                    rotation_applied = True
                    confidence_multiplier *= 0.9
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
                if 'entities' in annotations and 'ocr_text' in annotations['entities']:
                    for text_entry in annotations['entities']['ocr_text']:
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

                cv2.imwrite(output_image_path, image)

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

def process_directory(input_dir, annotation_dir, output_dir, output_annotation_dir, max_images=None):
    """Process all images and their annotations in a directory"""
    augmentor = DocumentAugmentor()

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_annotation_dir, exist_ok=True)

    # Get defined constant or use default
    if max_images is None:
        try:
            num_images = NUM_AUGMENTED_IMAGES  # From config.config
        except NameError:
            num_images = 100  # Default if not defined
    else:
        num_images = max_images

    # Process images
    processed_count = 0
    failed_count = 0

    # Get list of all image files first, then process
    image_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(filename)

    # Sort files for consistent processing order
    image_files.sort()

    # Limit to requested number
    image_files = image_files[:num_images]

    # Process each file
    for filename in image_files:
        base_name = os.path.splitext(filename)[0]

        input_image_path = os.path.join(input_dir, filename)
        input_annotation_path = os.path.join(annotation_dir, f"{base_name}.json")

        output_image_path = os.path.join(output_dir, f"augmented_{filename}")
        output_annotation_path = os.path.join(output_annotation_dir, f"augmented_{base_name}.json")

        if os.path.exists(input_annotation_path):
            logger.info(f"Processing {filename}...")
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

    logger.info(f"Processing complete: {processed_count} successful, {failed_count} failed")
    return processed_count, failed_count

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

if __name__ == "__main__":
    try:
        # Get number of images to process from config or use default
        try:
            num_images = NUM_AUGMENTED_IMAGES
        except NameError:
            num_images = 10  # Default to a lower number for testing
            logger.info(f"NUM_AUGMENTED_IMAGES not defined, using {num_images}")

        # Define directories with safer error handling
        try:
            # Try to get from config
            input_dir = IMAGES_DIR
            annotation_dir = ANNOTATIONS_DIR
            output_dir = AUGMENTED_IMAGES_DIR
            annotation_output_dir = AUGMENTED_ANNOTATIONS_DIR
        except NameError:
            # Fallback to defaults
            logger.warning("Configuration variables not found, using default directories")
            input_dir = "data/images"
            annotation_dir = "data/annotations"
            output_dir = "data/augmented_images"
            annotation_output_dir = "data/augmented_annotations"

        # Process images with proper error handling
        try:
            processed, failed = process_directory(
                input_dir,
                annotation_dir,
                output_dir,
                annotation_output_dir,
                num_images
            )

            # Only visualize if we have successful augmentations
            if processed > 0:
                # Create visualization directory
                vis_dir = "bbox_visualizations"
                os.makedirs(vis_dir, exist_ok=True)

                # Get list of augmented images
                augmented_images = [
                    f for f in os.listdir(output_dir)
                    if f.endswith(('.png', '.jpg', '.jpeg'))
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
