"""
Improved annotation processor for synthetic Arabic invoice dataset generation.
Produces precise word-level bounding boxes compatible with LiLT/LayoutLM architectures.

Features:
- Word-level PDF extraction using pdfplumber
- Coordinate normalization to 0-1000 scale (LiLT standard)
- BIO tagging with precise entity matching
- Hugging Face datasets format output
- Arabic RTL reading order handling
"""

import pdfplumber
import json
import os
import re
from collections import defaultdict
from PIL import Image
from typing import List, Dict, Tuple, Optional


class LiLTAnnotationProcessor:
    """
    Annotation processor optimized for LiLT/LayoutLM KIE models.
    Extracts word-level bounding boxes from PDFs and generates BIO-tagged annotations.
    """
    
    # Entity label mapping from invoice_data keys to NER tags
    ENTITY_LABEL_MAP = {
        'invoice_number': 'INVOICE_NUMBER',
        'invoice_ref': 'INVOICE_NUMBER',
        'customer_name': 'CUSTOMER_NAME',
        'customer_vat_number': 'CUSTOMER_VAT',
        'customer_address': 'CUSTOMER_ADDRESS',
        'customer_number': 'CUSTOMER_NUMBER',
        'seller_name': 'SELLER_NAME',
        'seller_vat_number': 'SELLER_VAT',
        'seller_address': 'SELLER_ADDRESS',
        'company_name': 'COMPANY_NAME',
        'seller_trade_name_ar': 'SELLER_NAME',
        'seller_trade_name_en': 'SELLER_NAME',
        'seller_location_line': 'SELLER_ADDRESS',
        'issue_datetime': 'INVOICE_DATE',
        'issue_time': 'INVOICE_TIME',
        'branch_name': 'BRANCH',
        'invoice_type': 'INVOICE_TYPE',
        'subtotal': 'SUBTOTAL',
        'discount': 'DISCOUNT',
        'net_amount': 'NET_AMOUNT',
        'tax': 'TAX',
        'total': 'TOTAL',
        'recipient_name': 'CUSTOMER_NAME',
        'recipient_phone': 'CUSTOMER_PHONE',
        'phone_number': 'SELLER_PHONE',
        'pos_phone_number': 'SELLER_PHONE',
        'pos_phone_number_2': 'SELLER_PHONE',
        'pos_phone_number_3': 'SELLER_PHONE',
        'email': 'EMAIL',
        'website': 'WEBSITE',
        'special_instructions': 'INSTRUCTIONS',
        'page_number': 'PAGE_NUMBER',
        'cashier_id': 'CASHIER_ID',
    }
    
    # Keys that should be treated as numeric values
    NUMERIC_KEYS = {
        'subtotal', 'tax', 'total', 'discount', 'net_amount',
        'unit_price', 'line_subtotal', 'quantity'
    }
    
    def __init__(self):
        self.annotations = None
    
    def _normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text for comparison."""
        # Normalize Arabic-Indic digits to Western digits
        arabic_digits = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
        text = str(text).translate(arabic_digits)
        
        # Remove tashkeel (diacritics)
        text = re.sub(r"[\u0617-\u061A\u064B-\u065F\u0670]", "", text)
        
        # Normalize Alef forms
        text = re.sub(r"[إأآٱ]", "ا", text)
        
        # Normalize other characters
        text = text.replace("ى", "ي")
        text = text.replace("ة", "ه")
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def _normalize_number(self, text: str) -> str:
        """Extract numeric value from text."""
        arabic_digits = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
        text = str(text).translate(arabic_digits)
        text = text.replace(",", "")
        return re.sub(r"[^\d.]", "", text)
    
    def _is_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters."""
        for char in text:
            if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F' or '\u08A0' <= char <= '\u08FF':
                return True
        return False
    
    def _determine_direction(self, text: str) -> str:
        """Determine text direction (RTL or LTR)."""
        return 'rtl' if self._is_arabic(text) else 'ltr'
    
    def _normalize_coordinates_to_lilt(
        self, 
        x0: float, 
        top: float, 
        x1: float, 
        bottom: float,
        page_width: float,
        page_height: float
    ) -> List[int]:
        """
        Normalize PDF coordinates to 0-1000 scale for LiLT.
        
        Args:
            x0, top, x1, bottom: PDF coordinates
            page_width: PDF page width in points
            page_height: PDF page height in points
            
        Returns:
            [x_min, y_min, x_max, y_max] normalized to 0-1000
        """
        x_min = int(1000 * (x0 / page_width))
        y_min = int(1000 * (top / page_height))
        x_max = int(1000 * (x1 / page_width))
        y_max = int(1000 * (bottom / page_height))
        
        # Clip to valid range
        x_min = max(0, min(1000, x_min))
        y_min = max(0, min(1000, y_min))
        x_max = max(0, min(1000, x_max))
        y_max = max(0, min(1000, y_max))
        
        # Prevent zero-width/height bboxes
        if x_min == x_max:
            x_max = min(1000, x_min + 1)
        if y_min == y_max:
            y_max = min(1000, y_min + 1)
        
        return [x_min, y_min, x_max, y_max]
    
    def _extract_words_with_pdfplumber(
        self, 
        page
    ) -> List[Dict]:
        """
        Extract words with precise bounding boxes using pdfplumber.
        
        Args:
            page: pdfplumber page object
            
        Returns:
            List of word dictionaries with text and coordinates
        """
        words = page.extract_words(
            x_tolerance=1,
            y_tolerance=1,
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=['fontname', 'size']
        )
        
        if not words:
            return []
        
        extracted = []
        for word in words:
            text = word['text'].strip()
            if not text:
                continue
            
            extracted.append({
                'text': text,
                'x0': word['x0'],
                'top': word['top'],
                'x1': word['x1'],
                'bottom': word['bottom'],
                'direction': self._determine_direction(text),
                'fontname': word.get('fontname', ''),
                'size': word.get('size', 0)
            })
        
        return extracted
    
    def _sort_words_rtl(self, words: List[Dict]) -> List[Dict]:
        """
        Sort words in Arabic RTL reading order.
        Primary: top to bottom (y-coordinate)
        Secondary: right to left (x-coordinate) within each line
        """
        if not words:
            return []
        
        # Group words into lines based on y-coordinate proximity
        line_groups = defaultdict(list)
        y_tolerance = 5  # pixels tolerance for line grouping
        
        # Sort by y first
        sorted_by_y = sorted(words, key=lambda w: w['top'])
        
        current_line_y = sorted_by_y[0]['top']
        current_line_idx = 0
        line_groups[current_line_idx].append(sorted_by_y[0])
        
        for word in sorted_by_y[1:]:
            if abs(word['top'] - current_line_y) < y_tolerance:
                line_groups[current_line_idx].append(word)
            else:
                current_line_idx += 1
                current_line_y = word['top']
                line_groups[current_line_idx].append(word)
        
        # Sort each line by x (RTL: right to left means descending x)
        sorted_words = []
        for line_idx in sorted(line_groups.keys()):
            line_words = line_groups[line_idx]
            # Check if line is predominantly Arabic
            is_rtl = sum(1 for w in line_words if w['direction'] == 'rtl') > len(line_words) / 2
            
            if is_rtl:
                # RTL: sort right to left (descending x0)
                line_words.sort(key=lambda w: -w['x0'])
            else:
                # LTR: sort left to right (ascending x0)
                line_words.sort(key=lambda w: w['x0'])
            
            sorted_words.extend(line_words)
        
        return sorted_words
    
    def _match_entity_to_words(
        self,
        entity_value: str,
        entity_key: str,
        words: List[Dict],
        used_word_indices: set
    ) -> List[int]:
        """
        Match entity value to specific words in the extracted word list.
        
        Args:
            entity_value: The value to match (e.g., customer name)
            entity_key: The entity key (e.g., 'customer_name')
            words: List of extracted words
            used_word_indices: Set of already-used word indices
            
        Returns:
            List of word indices that match this entity
        """
        if not entity_value:
            return []
        
        # Normalize entity value for matching
        normalized_value = self._normalize_arabic(str(entity_value))
        
        # For numeric fields, extract just the number
        if entity_key in self.NUMERIC_KEYS:
            normalized_value = self._normalize_number(entity_value)
            if not normalized_value:
                return []
            
            # Find word with matching numeric value
            for idx, word in enumerate(words):
                if idx in used_word_indices:
                    continue
                word_num = self._normalize_number(word['text'])
                if word_num == normalized_value:
                    return [idx]
            
            return []
        
        # For text fields, try to match contiguous sequence of words
        value_words = normalized_value.split()
        if not value_words:
            return []
        
        # Try to find exact contiguous match
        best_match = []
        best_score = 0
        
        for start_idx in range(len(words)):
            if start_idx in used_word_indices:
                continue
            
            matched_indices = []
            matched_words = []
            
            for offset, target_word in enumerate(value_words):
                current_idx = start_idx + offset
                if current_idx >= len(words):
                    break
                if current_idx in used_word_indices:
                    break
                
                word_text = self._normalize_arabic(words[current_idx]['text'])
                
                # Check for exact or partial match
                if word_text == target_word or target_word in word_text or word_text in target_word:
                    matched_indices.append(current_idx)
                    matched_words.append(word_text)
                else:
                    break
            
            # Score this match
            score = len(matched_indices)
            if score > best_score and score >= min(len(value_words), 2):
                best_score = score
                best_match = matched_indices
        
        return best_match
    
    def _assign_bio_tags(
        self,
        words: List[Dict],
        entity_matches: Dict[str, List[int]]
    ) -> List[str]:
        """
        Assign BIO (Begin-Inside-Outside) NER tags to words.
        
        Args:
            words: List of extracted words
            entity_matches: Dict mapping entity labels to word indices
            
        Returns:
            List of BIO tags
        """
        num_words = len(words)
        bio_tags = ['O'] * num_words
        word_to_entity = {}
        
        # Map word indices to entity labels
        for entity_label, word_indices in entity_matches.items():
            for idx in word_indices:
                if idx not in word_to_entity:
                    word_to_entity[idx] = entity_label
        
        # Assign BIO tags
        prev_entity = None
        for idx in range(num_words):
            if idx in word_to_entity:
                entity = word_to_entity[idx]
                if entity != prev_entity:
                    # Begin of new entity
                    bio_tags[idx] = f'B-{entity}'
                else:
                    # Inside existing entity
                    bio_tags[idx] = f'I-{entity}'
                prev_entity = entity
            else:
                bio_tags[idx] = 'O'
                prev_entity = None
        
        return bio_tags
    
    def process_pdf_for_lilt(
        self,
        pdf_path: str,
        image_path: str,
        invoice_data: Dict,
        page_index: int = 0
    ) -> Dict:
        """
        Process PDF to generate LiLT-compatible annotations.
        
        Args:
            pdf_path: Path to PDF file
            image_path: Path to rendered image
            invoice_data: Dictionary of synthetic invoice data
            page_index: Page index to process (default: 0)
            
        Returns:
            Dictionary with LiLT format: {id, tokens, bboxes, ner_tags, image_path}
        """
        # Get image dimensions
        with Image.open(image_path) as img:
            image_width = img.width
            image_height = img.height
        
        doc_id = os.path.splitext(os.path.basename(image_path))[0]
        
        # Open PDF with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if page_index < 0 or page_index >= len(pdf.pages):
                raise IndexError(f"page_index {page_index} out of range")
            
            page = pdf.pages[page_index]
            page_width = page.width
            page_height = page.height
            
            # Extract words with precise bounding boxes
            words = self._extract_words_with_pdfplumber(page)
            
            if not words:
                return None
            
            # Sort words in RTL reading order
            words = self._sort_words_rtl(words)
            
            # Normalize coordinates to 0-1000 scale
            tokens = []
            bboxes = []
            
            for word in words:
                bbox_lilt = self._normalize_coordinates_to_lilt(
                    word['x0'], word['top'], word['x1'], word['bottom'],
                    page_width, page_height
                )
                tokens.append(word['text'])
                bboxes.append(bbox_lilt)
            
            # Match entities to words
            entity_matches = {}
            used_word_indices = set()
            
            for key, value in invoice_data.items():
                if key not in self.ENTITY_LABEL_MAP:
                    continue
                if not value:
                    continue
                
                entity_label = self.ENTITY_LABEL_MAP[key]
                matched_indices = self._match_entity_to_words(
                    value, key, words, used_word_indices
                )
                
                if matched_indices:
                    entity_matches[entity_label] = matched_indices
                    used_word_indices.update(matched_indices)
            
            # Assign BIO tags
            ner_tags = self._assign_bio_tags(words, entity_matches)
            
            # Build LiLT-compatible output
            annotation = {
                'id': doc_id,
                'tokens': tokens,
                'bboxes': bboxes,
                'ner_tags': ner_tags,
                'image_path': image_path,
                # Additional metadata
                'doc_id': doc_id,
                'template_name': invoice_data.get('template_name', ''),
                'template_style': invoice_data.get('template_style', ''),
                'page_index': page_index,
                'language': 'ar',
            }
            
            return annotation
    
    def save_lilt_annotation(
        self,
        annotation: Dict,
        output_path: str
    ):
        """
        Save LiLT annotation as JSON.
        
        Args:
            annotation: LiLT annotation dictionary
            output_path: Output JSON file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)
    
    def save_lilt_annotations_jsonl(
        self,
        annotations: List[Dict],
        output_path: str
    ):
        """
        Save multiple LiLT annotations as JSONL (one per line).
        
        Args:
            annotations: List of LiLT annotation dictionaries
            output_path: Output JSONL file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for annotation in annotations:
                f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
