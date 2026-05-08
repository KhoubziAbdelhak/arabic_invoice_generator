# Arabic Invoice Generator

## Project Overview

This is a Python-based synthetic data generation and OCR fine-tuning pipeline for **Arabic invoices**. It generates realistic Arabic invoices (DOCX → PDF → images), annotates text elements and tables with bounding boxes, augments the dataset, and prepares it for fine-tuning OCR models (EasyOCR, Tesseract, PaddleOCR, TrOCR) and YOLO OBB (Oriented Bounding Box) detection models.

### Key Capabilities

1. **Invoice Generation** — Generates synthetic Arabic invoices from DOCX templates using Faker (ar_SA locale) and product data from a CSV.
2. **Annotation** — Extracts text and table bounding boxes from PDFs using `pdfplumber`, producing JSON annotations.
3. **Document Augmentation** — Applies realistic document effects (noise, blur, distortion) to increase dataset diversity.
4. **OCR Dataset Preparation** — Crops RTL text regions and creates a labeled dataset for OCR recognition fine-tuning.
5. **YOLO OBB Dataset Generation** — Converts annotations to YOLO Oriented Bounding Box format with train/val/test splits for text and table detection.
6. **OCR Evaluation** — Evaluates trained OCR models with metrics like CER, WER, confusion matrices, and processing time.

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3 |
| Document Processing | `python-docx`, `pdfplumber`, `pdf2image`, LibreOffice (headless PDF conversion) |
| Image Processing | `Pillow`, `OpenCV`, `scikit-image` |
| OCR Engines | EasyOCR, Tesseract, PaddleOCR, TrOCR (transformers) |
| Data / ML | `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn` |
| Text Rendering | `arabic-reshaper`, `python-bidi` |
| Fake Data | `Faker` (ar_SA locale) |

## Directory Structure

```
arabic_invoice_generator/
├── config/
│   ├── config.py            # Paths, constants, template validation
│   └── template_config.py   # Template-to-placeholder mappings
├── data/
│   ├── products.csv          # ~38K product catalog (Arabic + English)
│   ├── products.py           # ProductGenerator class
│   ├── Amiri-Regular.ttf     # Arabic font
│   ├── arabic.ttf            # Arabic font
│   ├── templates/            # DOCX invoice templates (.docx)
│   └── textures/             # Document texture assets
├── utils/
│   ├── document_processor.py # DOCX filling, PDF conversion
│   ├── image_processor.py    # PDF → image conversion
│   ├── annotation_processor.py # PDF text/table → JSON annotations
│   ├── document_augmentor.py # Document augmentation effects
│   └── document_scanner.py   # Document scanning simulation
├── generate_dataset.py        # Main pipeline: generate N invoices
├── prepare_ocr_dataset.py     # OCR recognition dataset preparation
├── detection_dataset_generator.py  # YOLO OBB detection dataset
├── evaluate_ocr.py            # OCR model evaluation (CER/WER)
├── visualize_annotation.py    # Annotation visualization
├── OCR_FINE_TUNING.md         # Guide for fine-tuning OCR models
├── requirements.txt           # Python dependencies
└── QWEN.md                    # This file
```

## Building and Running

### Prerequisites

- Python 3.10+
- LibreOffice (headless) for DOCX → PDF conversion
- Tesseract OCR (optional, for evaluation)
- A CUDA-capable GPU is recommended for OCR training/evaluation

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Invoices

```bash
python generate_dataset.py
```

This generates `NUM_INVOICES_TO_GENERATE` (default: 1300) invoices. For each invoice:
1. Random template selected from `data/templates/`
2. Products sampled from `data/products.csv`
3. DOCX filled with placeholder values → converted to PDF → rendered as images
4. JSON annotations with text and table bounding boxes saved

Output directories:
- `output/docx/` — Generated DOCX files (cleaned up after conversion)
- `output/pdf/` — Generated PDFs (cleaned up after image conversion)
- `output/images/` — Invoice page images (JPG)
- `output/annotations/` — JSON annotation files

### Prepare OCR Recognition Dataset

```bash
python prepare_ocr_dataset.py
```

Crops RTL text regions from augmented images and creates `output/ocr_finetune_dataset/` with:
- `images/` — Cropped text region images (PNG)
- `labels.txt` — Tab-separated file: `image_path\tarabic_text`

### Generate YOLO OBB Detection Dataset

```bash
python detection_dataset_generator.py
```

Creates `output/obb-dataset_original/` or `output/obb-dataset_augmented/` with:
- `images/train|val|test/` — Split images
- `labels/train|val|test/` — YOLO OBB label files (class_id x1 y1 x2 y2 x3 y3 x4 y4)
- `obb.yaml` — Dataset configuration for Ultralytics YOLO

Classes: `0` = text, `1` = table  
Split: 70/20/10 train/val/test (seeded with `RANDOM_SEED=42`)

### Evaluate OCR Models

```bash
python evaluate_ocr.py --model_type easyocr --test_data path/to/test_data --output_dir evaluation_results
```

Supported model types: `easyocr`, `tesseract`, `paddleocr`, `trocr`

Outputs:
- `evaluation_results/<model_type>_<timestamp>/metrics.json`
- Error distribution plots, confusion matrix, processing time histograms
- `detailed_results.csv`

## Configuration

Key settings in `config/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `VAT_RATE` | 0.15 | VAT rate applied to invoices |
| `NUM_PRODUCTS_PER_INVOICE` | 5 | Average products per invoice (±40% variation) |
| `NUM_INVOICES_TO_GENERATE` | 1300 | Total invoices to generate |
| `NUM_AUGMENTED_IMAGES` | = NUM_INVOICES | Number of augmented images |

In `detection_dataset_generator.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `USE_AUGMENTED_DATA` | True | Use augmented vs. original data |
| `TRAIN_RATIO` | 0.7 | Training set ratio |
| `VAL_RATIO` | 0.2 | Validation set ratio |
| `TEST_RATIO` | 0.1 | Test set ratio |
| `RANDOM_SEED` | 42 | Reproducibility seed |

## Invoice Template Requirements

DOCX templates in `data/templates/` must contain these placeholders:

```
{{invoice_ref}}, {{seller_name}}, {{seller_address}}, {{seller_vat_number}},
{{issue_datetime}}, {{products_table}}, {{subtotal}}, {{tax}}, {{total}},
{{email}}, {{website}}, {{phone_number}}, {{fax_number}}
```

Templates are validated on startup via `validate_template()` in `config.py`.

## Annotation Format (JSON)

```json
{
  "version": "v1",
  "image_path": "...",
  "image_size": { "width": 2550, "height": 3300 },
  "entities": {
    "ocr_text": [
      {
        "text": "...",
        "bbox": { "x": 100, "y": 200, "width": 300, "height": 40 },
        "bounding_poly": { "vertices": [...] },
        "direction": "rtl",
        "confidence": 1.0
      }
    ],
    "table": [
      {
        "bbox": { "x": 100, "y": 300, "width": 500, "height": 800 },
        "content": [["header1", "header2"], ["row1col1", "row1col2"]],
        "confidence": 1.0
      }
    ]
  }
}
```

## Fine-Tuning OCR Models

See `OCR_FINE_TUNING.md` for detailed guides on fine-tuning:
- **EasyOCR** — Custom recognition network
- **Tesseract** — LSTM training with tesstrain
- **TrOCR** — Hugging Face VisionEncoderDecoder fine-tuning
- **PaddleOCR** — Arabic language model training

## Development Conventions

- **No hardcoded paths in scripts** — All paths go through `config/config.py`
- **Cleanup intermediate files** — DOCX and PDF files are removed after image extraction to save disk space
- **JPG format for images** — Used instead of PNG to reduce storage
- **Arabic-only products** — CSV products are filtered to only include Arabic text items
- **RTL text handling** — Uses `arabic-reshaper` + `python-bidi` for proper Arabic text shaping and display
- **Retry mechanisms** — PDF conversion includes timeout + retry logic
