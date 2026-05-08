import os
import json
import glob
import re
def to_logical(text):
    # Split text into tokens. If token contains Arabic, reverse it (because pdfplumber output was made visual LTR)
    words = []
    # simple split might break spaces, let's just reverse the whole string if it's pure text, 
    # but strings like "10 لاير" -> "ريال 10" is tricky.
    for w in text.split():
        if any('\u0600' <= c <= '\u06FF' for c in w):
            words.append(w[::-1])
        else:
            words.append(w)
    return " ".join(words[::-1] if len(words) > 1 else words)
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    ocr_text = data.get('entities', {}).get('ocr_text', [])
    # Sort by y then x (right to left for Arabic)
    ocr_text.sort(key=lambda item: (item['bbox']['y'], -item['bbox']['x']))
    key_information = {}
    looking_for_vendor = False
    for i, item in enumerate(ocr_text):
        original_text = item['text']
        logical_text = to_logical(original_text)
        # update the item to have logically correct text for KIE training
        item['text_logical'] = logical_text
        # Heuristics on logical text
        if "البائع" in logical_text:
            looking_for_vendor = True
        elif looking_for_vendor and "اسم" in logical_text:
            if i + 1 < len(ocr_text):
                vendor_val = to_logical(ocr_text[i+1]['text'])
                key_information['vendor_name'] = vendor_val
                ocr_text[i+1]['label'] = 'vendor_name'
                looking_for_vendor = False
        if "ضريبة" in logical_text or "فاتورة" in logical_text:
             if i + 1 < len(ocr_text) and any(c.isdigit() for c in ocr_text[i+1]['text']):
                 val = to_logical(ocr_text[i+1]['text'])
                 key_information['invoice_id'] = val
                 ocr_text[i+1]['label'] = 'invoice_id'
        if "تاريخ" in logical_text:
             if i + 1 < len(ocr_text) and "-" in ocr_text[i+1]['text']:
                 val = to_logical(ocr_text[i+1]['text'])
                 key_information['date'] = val
                 ocr_text[i+1]['label'] = 'date'
        if "اإلجمالي" in logical_text:
             if i + 1 < len(ocr_text) and any(c.isdigit() for c in ocr_text[i+1]['text']):
                 val = to_logical(ocr_text[i+1]['text'])
                 key_information['total_amount'] = val
                 ocr_text[i+1]['label'] = 'total_amount'
    data['key_information'] = key_information
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
if __name__ == '__main__':
    annotations_dir = 'output/annotations'
    for file_path in glob.glob(os.path.join(annotations_dir, '*.json')):
         process_file(file_path)
    print("Done v2.")
