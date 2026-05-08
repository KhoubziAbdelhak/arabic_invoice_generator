import os
import json
import glob
from bidi.algorithm import get_display
def reverse_visual_to_logical(text):
    # If text is primarily Arabic visual, reverse it back. 
    # Since get_display was used, it reversed Arabic. 
    # For training, reverse back. Simple cheat: text[::-1] if mostly Arabic characters.
    # More robust: just use get_display again, it often reverts visual to logical symmetrically for plain characters.
    # Actually, best is text[::-1] for words that are pure Arabic because get_display just reversed them.
    # But mixed strings like "10 لاير" might get messy. Let's do simple token reversal if it contains arabic.
    # Actually, simplest is to just reverse if the dataset currently has it fully reversed by get_display.
    # wait, "6509.60 لاير" -> "ريال 6509.60" if reversed completely.
    return text[::-1]
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    ocr_text = data.get('entities', {}).get('ocr_text', [])
    # Sort by y then x (right to left)
    ocr_text.sort(key=lambda item: (item['bbox']['y'], -item['bbox']['x']))
    key_information = {}
    # state vars
    looking_for_vendor = False
    looking_for_tax = False
    looking_for_date = False
    looking_for_total = False
    for i, item in enumerate(ocr_text):
        text = item['text']
        # fix direction. If get_display was used on generation, we might need to fix it.
        # But wait, in Python, if it was get_display(logical), then to get logical, we reverse it.
        # Let's just store logical in key_information. 
        # Actually it's safer to reverse just the Arabic words or do full string reverse.
        logical_text = " ".join([w[::-1] if any("\u0600" <= c <= "\u06FF" for c in w) else w for w in text.split()])
        # Also clean up
        # Vendor heuristics
        if "البائع" in logical_text or "البائع" in text[::-1]:
            looking_for_vendor = True
        elif looking_for_vendor and ("اسم" in logical_text or "اسم" in text[::-1]):
            # Next item is probably vendor
            if i + 1 < len(ocr_text):
                vendor_item = ocr_text[i+1]['text']
                # reverse parts
                vendor_logical = " ".join([w[::-1] if any("\u0600" <= c <= "\u06FF" for c in w) else w for w in vendor_item.split()])
                key_information['vendor_name'] = vendor_logical
                ocr_text[i+1]['label'] = 'vendor_name'
                looking_for_vendor = False
        # Tax/Invoice number
        if "ضريبة" in text[::-1] or "فاتورة" in text[::-1] or "رقم" in text[::-1]:
             if i + 1 < len(ocr_text) and any(c.isdigit() for c in ocr_text[i+1]['text']):
                 key_information['invoice_id'] = ocr_text[i+1]['text']
                 ocr_text[i+1]['label'] = 'invoice_id'
        # Date
        if "تاريخ" in text[::-1]:
             if i + 1 < len(ocr_text) and "-" in ocr_text[i+1]['text']:
                 key_information['date'] = ocr_text[i+1]['text']
                 ocr_text[i+1]['label'] = 'date'
        # Total
        if "اإلجمالي" in text[::-1]:
             if i + 1 < len(ocr_text) and any(c.isdigit() for c in ocr_text[i+1]['text']):
                 key_information['total_amount'] = ocr_text[i+1]['text']
                 ocr_text[i+1]['label'] = 'total_amount'
    data['key_information'] = key_information
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
if __name__ == '__main__':
    annotations_dir = 'output/annotations' # Assuming run from project root
    for file_path in glob.glob(os.path.join(annotations_dir, '*.json')):
        process_file(file_path)
    print("Done generating KIE dataset tags.")
