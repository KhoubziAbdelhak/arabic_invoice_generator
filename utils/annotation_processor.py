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
                "table": [],  # New category for table annotations
                "kie_fields": [] # KIE annotations
            }
        }

    def process_pdf(self, pdf_path, image_path, invoice_data=None):
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
                text = get_display(word['text'])
                # Re-added get_display(text) to fix backwards Arabic text from pdfplumber
                x0 = word['x0'] * scale_x
                y0 = word['top'] * scale_y
                x1 = word['x1'] * scale_x
                y1 = word['bottom'] * scale_y

                if word['text'].replace('.', '').replace(',', '').replace('%', '').replace('-', '').isdigit():
                    y0 -= 5

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

            # Match KIE fields if invoice_data is provided
            if invoice_data:
                kie_keys = ['invoice_ref', 'company_name', 'issue_datetime', 'subtotal', 'tax', 'total', 'recipient_name']
                words = page.extract_words()
                for key in kie_keys:
                    val = str(invoice_data.get(key, ""))
                    if not val:
                        continue

                    matched_words = []
                    # Simple heuristic: find words whose text is in the target valid value
                    # and join them to form bounding box
                    for w in words:
                        if w['text'] in val:
                            matched_words.append(w)

                    if matched_words:
                        mx0 = min(w['x0'] for w in matched_words) * scale_x
                        my0 = min(w['top'] for w in matched_words) * scale_y
                        mx1 = max(w['x1'] for w in matched_words) * scale_x
                        my1 = max(w['bottom'] for w in matched_words) * scale_y
                        self.annotations["entities"]["kie_fields"].append({
                            "label": key,
                            "text": val,
                            "bbox": {
                                "x": round(mx0),
                                "y": round(my0),
                                "width": round(mx1 - mx0),
                                "height": round(my1 - my0)
                            }
                        })

        return self.annotations

    def _add_text_entry(self, text, x, y, width, height, direction='rtl'):
        """Add text entry for OCR"""
        self.annotations["entities"]["ocr_text"].append({
            "text": text,
            "bbox": {
                "x": round(x),
                "y": round(y),
                "width": round(width),
                "height": round(height)
            },
            "direction": direction,
            "confidence": 1.0
        })

    def _add_table_entry(self, x, y, width, height, content):
        """Add table entry with content"""
        self.annotations["entities"]["table"].append({
            "bbox": {
                "x": round(x),
                "y": round(y),
                "width": round(width),
                "height": round(height)
            },
            "content": content,  # New field for table content
            "confidence": 1.0
        })

    def _process_line(self, chars, scale_x, scale_y):
        """Process line of characters with proper Arabic shaping"""
        chars_sorted = sorted(chars, key=lambda c: c['x0'], reverse=True)
        text = ''.join([c['text'] for c in chars_sorted])
        # Removed get_display(text) for OCR ground truth
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
