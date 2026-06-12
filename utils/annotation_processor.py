import pymupdf as fitz  # PyMuPDF
import json
import os
import re
from collections import defaultdict
from PIL import Image

class AnnotationProcessor:
    def __init__(self):
        self.annotations = None

    _ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

    def _normalize_arabic(self, text):
        text = str(text).translate(self._ARABIC_DIGITS)
        text = re.sub(r"[\u0617-\u061A\u064B-\u065F\u0670]", "", text)
        text = re.sub(r"[إأآٱ]", "ا", text)
        text = text.replace("ى", "ي")
        text = text.replace("ة", "ه")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_text(self, text):
        text = self._normalize_arabic(text)
        if text == "لاير":
            text = "ريال"
        text = text.replace("لاير", "ريال")
        text = text.replace("،", ",")
        text = re.sub(r"[^\w\u0600-\u06FF@./:+#%-]+", " ", text)
        return re.sub(r"\s+", " ", text).strip().lower()

    def _normalize_number(self, text):
        text = str(text).translate(self._ARABIC_DIGITS)
        text = text.replace(",", "")
        return re.sub(r"[^\d.]", "", text)

    def _clean_extracted_text(self, text):
        text = str(text or "").strip()
        replacements = {
            "لاير": "ريال",
            "باللاير": "بالريال",
            "االسم": "الاسم",
            "االحتفاظ": "الاحتفاظ",
            "األمير": "الأمير",
            "اإللكتروني": "الإلكتروني",
            "األسعار": "الأسعار",
            "اإلجمالي": "الإجمالي",
            "مالحظات": "ملاحظات",
            "خالل": "خلال",
            "وفً ا": "وفقًا",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _contains_arabic(self, text):
        return any(
            '\u0600' <= c <= '\u06FF'
            or '\u0750' <= c <= '\u077F'
            or '\u08A0' <= c <= '\u08FF'
            for c in str(text)
        )

    def _contains_latin(self, text):
        return bool(re.search(r"[A-Za-z]", str(text)))

    def _tokens_are_rtl(self, tokens):
        if not tokens:
            return False
        arabic_tokens = sum(
            1 for token in tokens
            if token.get("direction") == "rtl" or self._contains_arabic(token.get("text", ""))
        )
        latin_tokens = sum(
            1 for token in tokens
            if token.get("direction") == "ltr" and self._contains_latin(token.get("text", ""))
        )
        return arabic_tokens > 0 and arabic_tokens >= latin_tokens

    def _sort_tokens_reading_order(self, tokens):
        is_rtl = self._tokens_are_rtl(tokens)
        return sorted(tokens, key=lambda t: t["x"], reverse=is_rtl)

    def _group_words_into_lines(self, raw_tokens, line_y_threshold=0.4, word_gap_threshold=1.2):
        if not raw_tokens:
            return []

        raw_tokens.sort(key=lambda t: t["y"])

        lines = []
        current_line = [raw_tokens[0]]

        for token in raw_tokens[1:]:
            line_y0 = min(t["y"] for t in current_line)
            line_y1 = max(t["y"] + t["height"] for t in current_line)
            
            token_mid_y = token["y"] + token["height"] / 2
            
            if line_y0 <= token_mid_y <= line_y1:
                current_line.append(token)
            else:
                overlap = max(0, min(line_y1, token["y"] + token["height"]) - max(line_y0, token["y"]))
                overlap_ratio = overlap / max(1, min(line_y1 - line_y0, token["height"]))
                if overlap_ratio >= line_y_threshold:
                    current_line.append(token)
                else:
                    lines.append(current_line)
                    current_line = [token]
        if current_line:
            lines.append(current_line)

        refined_lines = []
        for line in lines:
            line.sort(key=lambda t: t["x"])

            phrases = []
            curr_phrase = [line[0]]

            for token in line[1:]:
                prev_token = curr_phrase[-1]
                gap = token["x"] - (prev_token["x"] + prev_token["width"])

                avg_char_w = token["width"] / max(1, len(token["text"]))
                if gap > (avg_char_w * word_gap_threshold):
                    phrases.append(curr_phrase)
                    curr_phrase = [token]
                else:
                    curr_phrase.append(token)
            if curr_phrase:
                phrases.append(curr_phrase)
            refined_lines.extend(phrases)

        refined_lines.sort(key=lambda p: (sum(t["y"] for t in p) / len(p), min(t["x"] for t in p)))
        return refined_lines

    def _merge_tokens_to_field(self, tokens, line_id):
        if not tokens:
            return None

        x0 = min(t['x'] for t in tokens)
        y0 = min(t['y'] for t in tokens)
        x1 = max(t['x'] + t['width'] for t in tokens)
        y1 = max(t['y'] + t['height'] for t in tokens)

        is_rtl = self._tokens_are_rtl(tokens)
        sorted_tokens = self._sort_tokens_reading_order(tokens)

        return {
            "text": " ".join([t["text"] for t in sorted_tokens]),
            "bbox": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
            "words": [t["id"] for t in sorted_tokens],
            "line_id": line_id,
            "reading_order": "rtl" if is_rtl else "ltr"
        }

    def _sort_line_tokens(self, line):
        is_rtl = self._tokens_are_rtl(line)
        line.sort(key=lambda t: t["x"], reverse=is_rtl)
        return is_rtl

    def _determine_direction(self, text):
        # Check if the token has any Arabic characters
        for c in text:
            if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF':
                return 'rtl'
        return 'ltr'

    def _normalize_value(self, key, text):
        if key == 'issue_datetime':
            # handled separately in the kie loop
            pass
        if key in ['subtotal', 'tax', 'total'] or 'price' in key or 'quantity' in key or 'total' in key:
            return re.sub(r'[^\d.]', '', text)
        return text

    def process_pdf(self, pdf_path, image_path, invoice_data=None, page_index=0):
        doc_id = os.path.splitext(os.path.basename(image_path))[0]
        with Image.open(image_path) as img:
            image_width = img.width
            image_height = img.height

        self.annotations = {
            "version": "v2",
            "doc_id": doc_id,
            "language": "ar",
            "image_path": image_path,
            "image_size": {
                "width": image_width,
                "height": image_height
            },
            "entities": {
                "ocr_text": [],
                "text_lines": [],
                "kie_fields": [],
                "line_items": []
            },
            "metadata": {
                "template_name": invoice_data.get("template_name") if invoice_data else None,
                "template_style": invoice_data.get("template_style") if invoice_data else None,
                "page_index": page_index
            },
            "quality": {
                "warnings": []
            }
        }

        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(f"page_index {page_index} out of range for PDF with {len(doc)} pages")
        page = doc[page_index]

        # Compute scale factors: image pixels per PDF point (1 point = 1/72 inch)
        # The image size is in pixels; we need the page dimensions in points.
        page_rect = page.rect  # (x0, y0, x1, y1) in points
        pdf_width_pt = page_rect.width
        pdf_height_pt = page_rect.height
        scale_x = image_width / pdf_width_pt
        scale_y = image_height / pdf_height_pt

        # Extract words with positions using PyMuPDF
        # page.get_text("words") returns list of (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        # Coordinates are in PDF points, top-left origin.
        raw_words = page.get_text("words")

        tokens = []
        token_idx = 1
        raw_tokens = []

        for word in raw_words:
            x0_pt, y0_pt, x1_pt, y1_pt, raw_text, block_no, line_no, word_no = word
            raw_text = raw_text.strip()
            if not raw_text:
                continue

            text = self._clean_extracted_text(raw_text)
            direction = self._determine_direction(text)

            # Convert coordinates from points to image pixels
            x = x0_pt * scale_x
            y = y0_pt * scale_y
            width = (x1_pt - x0_pt) * scale_x
            height = (y1_pt - y0_pt) * scale_y

            raw_token = {
                "raw_text": raw_text,
                "text": text,
                "x": round(x),
                "y": round(y),
                "width": round(width),
                "height": round(height),
                "direction": direction,
                "confidence": 1.0,
                # store original PyMuPDF data for debugging
                "_block_no": block_no,
                "_line_no": line_no,
                "_word_no": word_no
            }
            raw_tokens.append(raw_token)

        # =========================================================
        # DEDUPLICATION - we'll remove EXACT duplicate tokens
        # (same text and nearly the same position) that arise from
        # PyMuPDF sometimes splitting a single glyph into two tokens.
        # We do not remove tokens that share the same text but are in
        # different positions. A simple tolerance-based dedup.
        # =========================================================
        def dedup_tokens(tokens_list, tolerance=2):
            """Remove tokens with identical text and almost identical bounding boxes."""
            unique = []
            for t in tokens_list:
                duplicate = False
                for u in unique:
                    if (t['raw_text'] == u['raw_text'] and
                        abs(t['x'] - u['x']) <= tolerance and
                        abs(t['y'] - u['y']) <= tolerance and
                        abs(t['width'] - u['width']) <= tolerance and
                        abs(t['height'] - u['height']) <= tolerance):
                        duplicate = True
                        break
                if not duplicate:
                    unique.append(t)
            return unique

        raw_tokens = dedup_tokens(raw_tokens, tolerance=2)

        lines = self._group_words_into_lines(raw_tokens)

        # Assign line IDs and token IDs, and correct reading order
        line_idx = 1
        phrase_idx = 1
        word_tokens_for_matching = []
        
        for phrase in lines:
            is_rtl = self._sort_line_tokens(phrase)

            line_id = f"L_{line_idx:02d}"
            
            phrase_text = " ".join([t['text'] for t in phrase])
            
            phrase_x = min(t['x'] for t in phrase)
            phrase_y = min(t['y'] for t in phrase)
            phrase_width = max(t['x'] + t['width'] for t in phrase) - phrase_x
            phrase_height = max(t['y'] + t['height'] for t in phrase) - phrase_y
            
            self.annotations["entities"]["text_lines"].append({
                "id": line_id,
                "text": phrase_text,
                "bbox": {
                    "x": phrase_x,
                    "y": phrase_y,
                    "width": phrase_width,
                    "height": phrase_height
                },
                "direction": "rtl" if is_rtl else "ltr"
            })

            t_id = f"t_{phrase_idx:03d}"
            
            self.annotations["entities"]["ocr_text"].append({
                "id": t_id,
                "text": phrase_text,
                "bbox": {
                    "x": phrase_x,
                    "y": phrase_y,
                    "width": phrase_width,
                    "height": phrase_height
                },
                "direction": "rtl" if is_rtl else "ltr",
                "confidence": min(t.get("confidence", 1.0) for t in phrase),
                "line_id": line_id
            })
            
            for word in phrase:
                word['phrase_id'] = t_id
                word['line_id'] = line_id
                word['id'] = f"w_{len(word_tokens_for_matching):04d}"
                word_tokens_for_matching.append(word)
                
            phrase_idx += 1
            line_idx += 1

        raw_tokens = word_tokens_for_matching

        # =========================================================
        # KIE field mapping (unchanged logic, but now token texts are correct)
        # =========================================================
        if invoice_data:
            from datetime import datetime
            def parse_date(d_str):
                if not d_str:
                    return ""
                formats = [
                    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
                    "%Y/%m/%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y",
                    "%B %d, %Y"
                ]
                for fmt in formats:
                    try:
                        return datetime.strptime(d_str.strip(), fmt).strftime("%Y-%m-%d")
                    except ValueError:
                        pass
                return d_str

            kie_keys = [
                'invoice_ref', 'company_name', 'seller_name', 'seller_address',
                'seller_vat_number', 'issue_datetime', 'email', 'website',
                'phone_number', 'fax_number', 'subtotal', 'tax', 'total',
                'recipient_name', 'recipient_company', 'street_address',
                'city_postcode', 'recipient_phone', 'shipping_recipient_name',
                'shipping_recipient_company', 'shipping_street_address',
                'shipping_city_postcode', 'shipping_recipient_phone',
                'special_instructions', 'issue_time', 'branch_name',
                'invoice_type', 'invoice_number', 'page_number', 'cashier_id',
                'customer_number', 'customer_vat_number', 'customer_name',
                'customer_address', 'pos_phone_number', 'pos_phone_number_2',
                'pos_phone_number_3', 'side_serial_left', 'side_serial_right',
                'seller_trade_name_ar', 'seller_trade_name_en',
                'seller_location_line', 'discount', 'net_amount'
            ]

            numeric_keys = {
                'subtotal', 'tax', 'total', 'discount', 'net_amount',
                'unit_price', 'line_subtotal'
            }
            line_item_numeric_keys = {
                'line_item_quantity', 'line_item_unit_price', 'line_item_total',
                'line_item_discount_percent', 'line_item_line_subtotal'
            }
            numeric_context_keywords = {
                'subtotal': ['الاجمالي', 'قبل الخصم', 'subtotal'],
                'tax': ['ضريبه', 'القيمه المضافه', 'vat', 'tax'],
                'total': ['شامل', 'شاملا', 'اجمالي الصافي', 'total'],
                'discount': ['الخصم', 'discount'],
                'net_amount': ['الصافي', 'net'],
            }
            used_token_ids = set()
            matched_value_norms = set()

            def token_available(token, ignore_used):
                return not (ignore_used and token.get('id') in used_token_ids)

            def tokens_by_line():
                line_map = defaultdict(list)
                for token in raw_tokens:
                    line_map[token.get("line_id")].append(token)
                return line_map

            line_map = tokens_by_line()
            token_order = {token["id"]: i for i, token in enumerate(raw_tokens)}
            def sort_by_reading(tokens):
                return self._sort_tokens_reading_order(tokens)

            line_text_by_id = {
                line_id: self._normalize_text(" ".join(t["text"] for t in sort_by_reading(tokens)))
                for line_id, tokens in line_map.items()
            }

            def same_line_currency_tokens(token, target_text, ignore_used):
                target_norm = self._normalize_text(target_text)
                if not any(currency in target_norm for currency in ("ريال", "ر.س", "رس", "sar")):
                    return []
                line_tokens = line_map.get(token.get("line_id"), [])
                currency = []
                for candidate in line_tokens:
                    if candidate is token or not token_available(candidate, ignore_used):
                        continue
                    norm = self._normalize_text(candidate["text"])
                    if norm in {"ريال", "ر.س", "رس", "sar"}:
                        currency.append(candidate)
                return currency[:1]

            def find_numeric_tokens(target_text, key=None, ignore_used=True):
                target_num = self._normalize_number(target_text)
                if not target_num:
                    return []
                exact = [
                    t for t in raw_tokens
                    if token_available(t, ignore_used)
                    and self._normalize_number(t["text"]) == target_num
                ]
                if not exact:
                    return []

                context_words = numeric_context_keywords.get(key, [])
                if context_words:
                    def context_score(token):
                        line_text = line_text_by_id.get(token.get("line_id"), "")
                        hits = sum(1 for word in context_words if word in line_text)
                        return (hits, token["y"], -token["x"])
                    best = max(exact, key=context_score)
                else:
                    best = min(exact, key=lambda t: (t["y"], -t["x"]))
                return sort_by_reading([best] + same_line_currency_tokens(best, target_text, ignore_used))

            def span_score(tokens, target_words):
                token_words = [self._normalize_text(t["text"]) for t in tokens]
                covered = 0
                for word in target_words:
                    if word and word in token_words:
                        covered += 1
                extra = max(0, len(token_words) - covered)
                return (covered, -extra, len(tokens))

            def find_tokens(target_text, key=None, ignore_used=True):
                target_str = str(target_text).strip()
                if not target_str:
                    return []

                if key in numeric_keys or key in line_item_numeric_keys:
                    return find_numeric_tokens(target_str, key=key, ignore_used=ignore_used)

                target_norm = self._normalize_text(target_str)
                target_words = target_norm.split()
                if not target_words:
                    return []

                # First pass: exact contiguous span on one detected line.
                span_candidates = []
                for line_tokens in line_map.values():
                    line_tokens = [t for t in line_tokens if token_available(t, ignore_used)]
                    ordered = sort_by_reading(line_tokens)
                    norm_words = [self._normalize_text(t["text"]) for t in ordered]
                    for start in range(len(ordered)):
                        for end in range(start + 1, min(len(ordered), start + len(target_words) + 2) + 1):
                            span = ordered[start:end]
                            span_text = " ".join(norm_words[start:end])
                            if span_text == target_norm:
                                return span
                            score = span_score(span, target_words)
                            if score[0] > 0:
                                span_candidates.append((score, span))
                if span_candidates:
                    best_score, best_span = max(span_candidates, key=lambda item: item[0])
                    min_covered = len(target_words) if len(target_words) <= 3 else max(2, len(target_words) // 2)
                    if best_score[0] >= min_covered and (
                        key != "line_item_description" or best_score[0] == len(target_words)
                    ):
                        return best_span

                # Second pass: bag-of-words cluster on same line. Handles mixed RTL/LTR names.
                candidate_tokens = []
                for t in raw_tokens:
                    if not token_available(t, ignore_used):
                        continue
                    tok_norm = self._normalize_text(t['text'])
                    raw_norm = self._normalize_text(t['raw_text'])
                    if tok_norm in target_words or raw_norm in target_words:
                        candidate_tokens.append(t)
                    elif any(ch.isdigit() for ch in t['raw_text']) and any(ch.isdigit() for ch in target_str):
                        num_t = self._normalize_number(t['raw_text'])
                        num_target = self._normalize_number(target_str)
                        if num_t and num_t == num_target:
                            candidate_tokens.append(t)
                    elif len(tok_norm) > 2 and tok_norm in target_norm:
                        candidate_tokens.append(t)

                if not candidate_tokens:
                    return []

                if key == "line_item_description":
                    ordered_candidates = sorted(candidate_tokens, key=lambda t: token_order.get(t["id"], 0))
                    covered = len({self._normalize_text(t["text"]) for t in ordered_candidates} & set(target_words))
                    if covered >= max(2, len(target_words) // 2):
                        return ordered_candidates

                clusters = defaultdict(list)
                for token in candidate_tokens:
                    clusters[token.get("line_id")].append(token)

                def cluster_score(cluster):
                    cluster = sort_by_reading(cluster)
                    score = span_score(cluster, target_words)
                    rtl_count = sum(1 for tc in cluster if tc['direction'] == 'rtl')
                    avg_x = sum(tc['x'] for tc in cluster) / len(cluster)
                    return (score[0], score[1], rtl_count, avg_x)

                best_cluster = max(clusters.values(), key=cluster_score)
                best_cluster = sort_by_reading(best_cluster)
                min_covered = len(target_words) if len(target_words) <= 3 else max(2, len(target_words) // 2)
                if cluster_score(best_cluster)[0] < min_covered:
                    return []
                return best_cluster

            def match_is_valid(key, val_str, matched_tokens):
                if not matched_tokens:
                    return False
                if key in numeric_keys:
                    target_num = self._normalize_number(val_str)
                    matched_nums = [self._normalize_number(t["text"]) for t in matched_tokens]
                    return target_num in matched_nums
                target_words = self._normalize_text(val_str).split()
                matched_words = [self._normalize_text(t["text"]) for t in matched_tokens]
                if len(target_words) <= 3:
                    return all(word in matched_words for word in target_words)
                covered = sum(1 for word in target_words if word in matched_words)
                return covered >= max(2, len(target_words) // 2)

            def target_is_visible(val_str, key=None):
                if key in numeric_keys:
                    target_num = self._normalize_number(val_str)
                    if not target_num:
                        return False
                    return any(self._normalize_number(token["text"]) == target_num for token in raw_tokens)

                target_words = self._normalize_text(val_str).split()
                if not target_words:
                    return False
                visible_words = {self._normalize_text(token["text"]) for token in raw_tokens}
                if len(target_words) <= 3:
                    return all(word in visible_words for word in target_words)
                covered = sum(1 for word in target_words if word in visible_words)
                return covered >= max(2, len(target_words) // 2)

            for key in kie_keys:
                val = invoice_data.get(key, "")
                if not val:
                    continue
                val_str = str(val)
                value_norm = self._normalize_text(val_str)
                matched_tokens = find_tokens(val_str, key=key, ignore_used=True)
                if matched_tokens and match_is_valid(key, val_str, matched_tokens):
                    for t in matched_tokens:
                        used_token_ids.add(t['id'])
                    matched_value_norms.add(value_norm)

                    token_ids = []
                    for t in matched_tokens:
                        if t['phrase_id'] not in token_ids:
                            token_ids.append(t['phrase_id'])

                    x0 = min(t['x'] for t in matched_tokens)
                    y0 = min(t['y'] for t in matched_tokens)
                    x1 = max(t['x'] + t['width'] for t in matched_tokens)
                    y1 = max(t['y'] + t['height'] for t in matched_tokens)
                    norm_val = val_str
                    if key == 'issue_datetime':
                        norm_val = parse_date(val_str)
                    elif key in ['subtotal', 'tax', 'total', 'discount', 'net_amount']:
                        norm_val = re.sub(r'[^\d.]', '', val_str)

                    self.annotations["entities"]["kie_fields"].append({
                        "label": key,
                        "text": val_str,
                        "normalized_value": norm_val,
                        "bbox": {
                            "x": x0,
                            "y": y0,
                            "width": x1 - x0,
                            "height": y1 - y0
                        },
                        "token_ids": token_ids
                    })
                elif value_norm not in matched_value_norms and target_is_visible(val_str, key=key):
                    self.annotations["quality"]["warnings"].append({
                        "type": "kie_field_not_matched",
                        "label": key,
                        "text": val_str
                    })

            # Line items
            products = invoice_data.get('products', [])
            for i, prod in enumerate(products):
                fields_data = {}
                for f in [
                    'description', 'quantity', 'unit_price', 'total',
                    'item_code', 'location', 'discount_percent', 'line_subtotal'
                ]:
                    pv = str(prod.get(f, ''))
                    if not pv:
                        continue
                    m_tokens = find_tokens(pv, key=f"line_item_{f}")
                    
                    token_ids = []
                    for t in m_tokens:
                        if t['phrase_id'] not in token_ids:
                            token_ids.append(t['phrase_id'])
                            
                    fields_data[f] = {
                        "text": pv,
                        "token_ids": token_ids
                    }
                self.annotations["entities"]["line_items"].append({
                    "item_index": i,
                    "fields": fields_data
                })

        doc.close()
        return self.annotations

    def save_annotations(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
