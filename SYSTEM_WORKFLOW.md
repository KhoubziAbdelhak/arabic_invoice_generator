# 2. System Workflow (End-to-End Pipeline)

This section documents the complete pipeline used by the project to generate synthetic Arabic invoice images and their annotations. The description below is **implementation-faithful** and derived directly from the repository source code.

## 2.1 Data generation (invoice content synthesis)

**Entry point:** `generate_dataset.py`

For each invoice (looping over `range(NUM_INVOICES_TO_GENERATE)`), the generator performs the following steps:

1. **Initialize Arabic faker and product generator**
   - Arabic locale faker:
     - `fake = Faker('ar_SA')`
   - Product generator is initialized with a CSV product catalog:
     - `product_generator = ProductGenerator(csv_path=PRODUCTS_CSV_PATH)`

2. **Select a DOCX template (randomly)**
   - The generator scans `TEMPLATE_DIR` and keeps only `.docx` templates:
     - `template_files = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith('.docx')]`
   - For each invoice iteration:
     - `selected_template = random.choice(template_files)`

3. **Create a zero-padded invoice index for filenames**
   - The number of digits depends on the total invoices requested:
     - `padding_digits = len(str(NUM_INVOICES_TO_GENERATE))`
   - Each invoice uses:
     - `invoice_num = str(i).zfill(padding_digits)`
   - Output files are named using this padded index (e.g., `invoice_0001.docx`, `invoice_0001.json`).

4. **Sample the number of products with ±40% variation**
   - The code applies a uniform multiplicative jitter around `NUM_PRODUCTS_PER_INVOICE`:
     - `num_products = int(NUM_PRODUCTS_PER_INVOICE * (1 + random.uniform(-0.4, 0.4)))`

5. **Generate product line items from CSV (preferred path)**
   - Products are created via:
     - `products = product_generator.generate_products_from_csv(num_products)`

   In `data/products.py`, `ProductGenerator`:
   - Loads `data/products.csv` using **Pandas**.
   - Filters to **Arabic-only** product names by checking that `Item_Name` contains at least one character in the Arabic Unicode range (`\u0600`–`\u06FF`). Non-Arabic products are skipped.
   - Generates per-line-item fields including:
     - `description`: Arabic product name (sometimes prefixed with an Arabic brand if present)
     - `quantity`: random integer (1–10)
     - `unit_price`: formatted string with currency suffix (e.g., `"12.34 ريال"`)
     - `unit_price_numeric`: numeric price
     - `total`: formatted string (price × quantity, with `" ريال"`)
     - `total_numeric`: numeric total
     - Additional metadata: `row_index`, and in the CSV-based path also `pack` and `unit`

6. **Compute invoice totals**
   - Subtotal:
     - `subtotal = sum(float(p['unit_price_numeric']) * p['quantity'] for p in products)`
   - Tax:
     - `tax = round(subtotal * VAT_RATE, 2)`
   - Total:
     - `total = subtotal + tax`

7. **Generate invoice fields (Arabic locale)**
   - The generator builds an `invoice_data` dictionary that includes:
     - Identifiers: `invoice_ref` (e.g., `INV-####-2024` via `fake.numerify`)
     - Seller/company fields: `company_name`, `seller_name`, `seller_address`, `seller_vat_number`
     - Date: `issue_datetime` (via `fake.date()`)
     - Contact fields: `email`, `website`, `phone_number`, `fax_number`
     - Monetary fields: `subtotal`, `tax`, `total` (strings formatted as `"X.XX ريال"`)
     - `products`: the full list of product dictionaries

   It also includes additional “recipient” and “shipping” fields such as:
   - `recipient_name`, `recipient_company`, `street_address`, `city_postcode`, `recipient_phone`
   - `shipping_recipient_name`, `shipping_recipient_company`, `shipping_street_address`, `shipping_city_postcode`, `shipping_recipient_phone`

   And a free-text field:
   - `special_instructions = fake.paragraph(nb_sentences=2)`

8. **Add uppercase duplicate keys for template compatibility**
   - The generator builds a second mapping where keys are uppercased:
     - `uppercase_keys = {key.upper(): value for key, value in invoice_data.items()}`
   - Then merges them back:
     - `invoice_data.update(uppercase_keys)`

   This allows DOCX templates to use either `{{invoice_ref}}` or `{{INVOICE_REF}}` (and similarly for other fields).

**Data flow summary:**

`Faker('ar_SA')` + `ProductGenerator` → `invoice_data` dict → DOCX placeholder replacement and table synthesis.

---

## 2.2 Invoice synthesis method (DOCX placeholder filling + table construction)

**Module:** `utils/document_processor.py`

### Placeholder replacement

`DocumentProcessor.create_invoice(template_path, invoice_data, output_path)`:

1. Loads the DOCX template:
   - `doc = Document(template_path)`
2. Replaces placeholders using **plain string replacement**:
   - Placeholders are built as `{{KEY}}` for each key in `invoice_data`.
3. Searches and replaces in two places:
   - `doc.paragraphs`
   - all table cells (row → cell → paragraph)

### Products table synthesis

The product table is identified as the first table that contains the literal marker `{{products_table}}` in any cell.

If found, `_format_products_table(table, invoice_data['products'])`:

- Removes all rows except the first (header) row.
- Ensures a 4-column layout and applies fixed column widths.
- Sets Arabic headers:
  - `"الوصف"`, `"الكمية"`, `"سعر الوحدة"`, `"الإجمالي"`
- Adds one row per product, mapping:
  - `description`, `quantity`, `unit_price`, `total`
- Right-aligns text in header and body cells (`alignment = 2`).
- Adds cell borders using Word XML elements (`OxmlElement`, `qn`).

**Key point:** invoice appearance (fonts, layout, styling) is primarily governed by the **DOCX templates** in `data/templates/`. The code injects content and formats the products table but does not generate a full layout from scratch.

---

## 2.3 Rendering process (DOCX → PDF → image)

### DOCX → PDF

**Module:** `utils/document_processor.py`

`DocumentProcessor.convert_to_pdf(input_path, output_dir)` runs LibreOffice in headless mode:

```bash
libreoffice --headless --convert-to pdf --outdir <output_dir> <input_docx>
```

Implementation characteristics:
- Retries up to `max_retries = 3`.
- Uses a per-attempt timeout of 30 seconds.

**External dependency:** the `libreoffice` CLI must be installed and available in the system path.

### PDF → raster image(s)

**Module:** `utils/image_processor.py`

`ImageProcessor.pdf_to_images(pdf_path, output_dir, dpi=300, format="jpg", quality=...)`:

- Converts the PDF to PIL images with:
  - `convert_from_path(pdf_path, dpi=300)` (from `pdf2image`)
- Saves images using PIL; the generator uses JPG to reduce file size:
  - in `generate_dataset.py`: `format="jpg"`, `quality=75`

**Implementation detail:** the output filename is `f"{pdf_name}.{format.lower()}"` (no page index). Therefore, if a PDF contains multiple pages, later pages would overwrite earlier ones. The pipeline is effectively designed for single-page invoices.

---

## 2.4 Annotation generation (PDF text/table → JSON aligned to image pixels)

**Module:** `utils/annotation_processor.py`

`AnnotationProcessor.process_pdf(pdf_path, image_path)` produces a JSON annotation object aligned to the rendered invoice image.

### Coordinate alignment strategy

1. Reads the rasterized image (`image_path`) to obtain:
   - `image_width`, `image_height`
2. Opens the PDF (`pdf_path`) with `pdfplumber` and uses the first page (`pages[0]`) to get:
   - `pdf_width`, `pdf_height`
3. Computes scaling from PDF coordinate space to image pixel space:

- `scale_x = image_width / pdf_width`
- `scale_y = image_height / pdf_height`

### Word-level extraction (text annotations)

- Iterates through words:
  - `for word in page.extract_words():`
- Converts extracted word text to visual display order using `python-bidi`:
  - `text = get_display(word['text'])`
- Converts word bounding coordinates into image pixel coordinates:
  - `x0 = word['x0'] * scale_x`
  - `y0 = word['top'] * scale_y`
  - `x1 = word['x1'] * scale_x`
  - `y1 = word['bottom'] * scale_y`
- Stores each word under `entities.ocr_text[]` with:
  - `text`
  - `bbox: {x, y, width, height}` (rounded)
  - `direction: 'rtl'`
  - `confidence: 1.0`

### Table extraction (table annotations)

- Finds tables:
  - `tables = page.find_tables()`
- For each detected table:
  - Uses `table.bbox` (PDF-space) scaled to image-space
  - Extracts content as a list-of-lists:
    - `table_content = table.extract()`
- Stores each table under `entities.table[]` with:
  - `bbox: {x, y, width, height}`
  - `content: table_content`
  - `confidence: 1.0`

### JSON schema (as produced)

The annotation object written via `AnnotationProcessor.save_annotations()` contains:

- `version: "v1"`
- `image_path: <path-to-rendered-image>`
- `image_size: { width, height }`
- `entities`:
  - `ocr_text`: list of word-level entries (`text`, `bbox`, `direction`, `confidence`)
  - `table`: list of table region entries (`bbox`, `content`, `confidence`)

---

## Implementation note: cleanup of intermediate files

At the end of each invoice iteration in `generate_dataset.py`, the script deletes:
- the generated `.docx` file (`os.remove(docx_path)`)
- the generated `.pdf` file (`os.remove(pdf_path)`)

This keeps the output focused on the final rendered images and annotations.

