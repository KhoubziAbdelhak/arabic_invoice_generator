import pdfplumber
import json
from PIL import Image, ImageDraw
from arabic_reshaper import ArabicReshaper
from bidi.algorithm import get_display

class AnnotationProcessor:
    def __init__(self):
        self.annotations = {
            "version": "v1",
            "entities": {
                "ocr_text": [],
                "table": []  # New category for table annotations
            }
        }
        self.reshaper_config = {
            'delete_harakat': False,
            'support_ligatures': True,
            'language': 'Arabic',
            'use_unshaped_instead_of_isolated': True
        }
        self.reshaper = ArabicReshaper(configuration=self.reshaper_config)

    def process_pdf(self, pdf_path, image_path):
        with Image.open(image_path) as img:
            image_width = img.width
            image_height = img.height
            self.annotations["image_path"] = image_path
            self.annotations["image_size"] = {
                "width": image_width,
                "height": image_height
            }

        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            pdf_width = page.width
            pdf_height = page.height
            scale_x = image_width / pdf_width
            scale_y = image_height / pdf_height

            # Extract word-level text annotations
            for word in page.extract_words():
                text = word['text']
                text = get_display(text)  # Convert to visual display order
                x0 = word['x0'] * scale_x
                y0 = word['top'] * scale_y
                x1 = word['x1'] * scale_x
                y1 = word['bottom'] * scale_y
                self._add_text_entry(
                    text=text,
                    x=x0,
                    y=y0,
                    width=x1 - x0,
                    height=y1 - y0,
                    direction='rtl'
                )

            # Extract table annotations with content
            tables = page.find_tables()
            for table in tables:
                bbox = table.bbox  # (x0, top, x1, bottom)
                x0 = bbox[0] * scale_x
                y0 = bbox[1] * scale_y
                x1 = bbox[2] * scale_x
                y1 = bbox[3] * scale_y
                table_content = table.extract()  # Extract table content as list of lists
                self._add_table_entry(
                    x=x0,
                    y=y0,
                    width=x1 - x0,
                    height=y1 - y0,
                    content=table_content  # Include table content
                )

        return self.annotations

    def _add_text_entry(self, text, x, y, width, height, direction='rtl'):
        """Add text entry for OCR"""
        self.annotations["entities"]["ocr_text"].append({
            "text": text,
            "bbox": {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            },
            "direction": direction,
            "confidence": 1.0
        })

    def _add_table_entry(self, x, y, width, height, content):
        """Add table entry with content"""
        self.annotations["entities"]["table"].append({
            "bbox": {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            },
            "content": content,  # New field for table content
            "confidence": 1.0
        })

    def _process_line(self, chars, scale_x, scale_y):
        """Process line of characters with proper Arabic shaping"""
        chars_sorted = sorted(chars, key=lambda c: c['x0'], reverse=True)
        text = ''.join([c['text'] for c in chars_sorted])
        text = get_display(text)  # Convert to visual display order
        x0 = min(c['x0'] for c in chars_sorted) * scale_x
        y0 = min(c['top'] for c in chars_sorted) * scale_y
        x1 = max(c['x1'] for c in chars_sorted) * scale_x
        y1 = max(c['bottom'] for c in chars_sorted) * scale_y
        self._add_text_entry(
            text=text,
            x=x0,
            y=y0,
            width=x1 - x0,
            height=y1 - y0,
            direction='rtl'
        )

    def save_annotations(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)