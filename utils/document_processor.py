from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor, Inches
import subprocess
import os
import re
import time


class DocumentProcessor:

    # ------------------------------------------------------------------ #
    #  Paragraph-level placeholder replacement (run-aware)                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _replace_in_paragraph(paragraph, replacements):
        """
        Replace {{key}} placeholders in a paragraph without destroying formatting.

        Word frequently splits a single placeholder across multiple runs, e.g.
        run[0]="{{", run[1]="company", run[2]="_name}}".  Reading
        `paragraph.text` sees the full string, but writing `paragraph.text`
        nukes every run's formatting.

        Strategy:
          1. Concatenate all run texts → find whether any placeholder exists.
          2. Apply every replacement to the concatenated string.
          3. Write the result back into run[0] and blank out all other runs,
             preserving run[0]'s character formatting (bold, italic, font …).
        """
        if not paragraph.runs:
            return

        full_text = "".join(run.text for run in paragraph.runs)

        # Quick exit if nothing to replace
        if "{{" not in full_text:
            return

        changed = False
        for key, value in replacements.items():
            placeholder = "{{" + key + "}}"
            if placeholder in full_text:
                full_text = full_text.replace(placeholder, str(value))
                changed = True

        if changed:
            paragraph.runs[0].text = full_text
            for run in paragraph.runs[1:]:
                run.text = ""

    # ------------------------------------------------------------------ #
    #  RTL helpers                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _set_paragraph_rtl(paragraph):
        """Mark a paragraph as right-to-left (Arabic / Hebrew)."""
        pPr = paragraph._p.get_or_add_pPr()

        # Schema-compliant bidi insertion
        if pPr.find(qn("w:bidi")) is None:
            bidi = OxmlElement("w:bidi")
            bidi.set(qn("w:val"), "1")
            jc = pPr.find(qn("w:jc"))
            if jc is not None:
                jc.addprevious(bidi)
            else:
                pPr.append(bidi)

        DocumentProcessor._set_paragraph_jc(paragraph, "start")

    @staticmethod
    def _set_paragraph_jc(paragraph, value):
        """Set raw paragraph justification, including OOXML start/end values."""
        pPr = paragraph._p.get_or_add_pPr()
        jc = pPr.find(qn("w:jc"))
        if jc is None:
            jc = OxmlElement("w:jc")
            pPr.append(jc)
        jc.set(qn("w:val"), value)

    @staticmethod
    def _set_run_rtl(run):
        """Mark a run's characters as RTL."""
        run.font.rtl = True
        run.font.name = "Arial"
        rPr = run._r.get_or_add_rPr()
        if rPr.find(qn("w:rtl")) is None:
            rtl = OxmlElement("w:rtl")
            rtl.set(qn("w:val"), "1")
            rPr.append(rtl)

        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is None:
            rFonts = OxmlElement("w:rFonts")
            rPr.insert(0, rFonts)
        rFonts.set(qn("w:ascii"), "Arial")
        rFonts.set(qn("w:hAnsi"), "Arial")
        rFonts.set(qn("w:cs"), "Arial")

    @staticmethod
    def _set_table_rtl(table):
        """Render table columns in right-to-left visual order."""
        tblPr = table._tbl.tblPr
        if tblPr is None:
            tblPr = OxmlElement("w:tblPr")
            table._tbl.insert(0, tblPr)

        if tblPr.find(qn("w:bidiVisual")) is None:
            bidi_visual = OxmlElement("w:bidiVisual")
            tblPr.append(bidi_visual)

        jc = tblPr.find(qn("w:jc"))
        if jc is None:
            jc = OxmlElement("w:jc")
            tblPr.append(jc)
        jc.set(qn("w:val"), "right")

    @staticmethod
    def _contains_arabic(text):
        return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", str(text or "")))

    @staticmethod
    def _is_invoice_title(text):
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        text = text.replace("\u0640", "")
        if not text:
            return False

        title_terms = (
            "فاتورة",
            "invoice",
            "tax invoice",
            "sales invoice",
            "credit invoice",
        )
        lower_text = text.lower()
        if not any(term in lower_text for term in title_terms):
            return False

        field_label_terms = (
            "رقم",
            "تاريخ",
            "نوع",
            "total",
            "subtotal",
            "الإجمالي",
            "{{",
        )
        if any(term in lower_text for term in field_label_terms):
            return False

        return len(text) <= 80

    @staticmethod
    def _apply_paragraph_direction(paragraph):
        """Right-align Arabic paragraphs while preserving LTR-only paragraphs."""
        if not DocumentProcessor._contains_arabic(paragraph.text):
            return

        DocumentProcessor._set_paragraph_rtl(paragraph)
        if DocumentProcessor._is_invoice_title(paragraph.text):
            DocumentProcessor._set_paragraph_jc(paragraph, "center")
        for run in paragraph.runs:
            # Mixed Arabic/Latin invoice lines need the whole run marked as
            # complex RTL text, otherwise LibreOffice can anchor fragments from
            # the left side of the cell during PDF export.
            DocumentProcessor._set_run_rtl(run)

    # ------------------------------------------------------------------ #
    #  Main entry point                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_invoice(template_path, invoice_data, output_path):
        doc = Document(template_path)
        products_table = None

        # Set proper page margins for Arabic invoices (narrower margins for more content)
        for section in doc.sections:
            section.top_margin = Inches(0.5)    # 0.5 inch top margin
            section.bottom_margin = Inches(0.5) # 0.5 inch bottom margin
            section.left_margin = Inches(0.7)   # 0.7 inch left margin
            section.right_margin = Inches(0.7)  # 0.7 inch right margin

        # --- paragraphs outside tables ---
        for paragraph in doc.paragraphs:
            # Set proper spacing for Arabic text
            paragraph.paragraph_format.space_before = Pt(2)
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.line_spacing = Pt(14)
            DocumentProcessor._replace_in_paragraph(paragraph, invoice_data)
            DocumentProcessor._apply_paragraph_direction(paragraph)

        # --- tables ---
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        # Detect products table *before* replacing
                        if "{{products_table}}" in paragraph.text:
                            products_table = table

                        DocumentProcessor._replace_in_paragraph(paragraph, invoice_data)
                        DocumentProcessor._apply_paragraph_direction(paragraph)

        # --- format the products table ---
        if products_table is not None:
            if invoice_data.get("template_style") == "al_murabaha":
                DocumentProcessor._format_al_murabaha_table(
                    products_table, invoice_data["products"], invoice_data
                )
            else:
                DocumentProcessor._format_products_table(
                    products_table, invoice_data["products"], invoice_data
                )

        # Some table rows are created after placeholder replacement.
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        DocumentProcessor._apply_paragraph_direction(paragraph)

        doc.save(output_path)
        return output_path

    # ------------------------------------------------------------------ #
    #  Table borders                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def set_normal_borders(table, color="000000", size="4"):
        """Apply single-line borders to every cell in *table*."""
        for row in table.rows:
            for cell in row.cells:
                tc = cell._tc
                tcPr = tc.get_or_add_tcPr()

                # Remove any existing border element
                for existing in tcPr.findall(qn("w:tcBorders")):
                    tcPr.remove(existing)

                tcBorders = OxmlElement("w:tcBorders")
                for border_name in ["top", "left", "bottom", "right", "insideH", "insideV"]:
                    border = OxmlElement(f"w:{border_name}")
                    border.set(qn("w:val"), "single")
                    border.set(qn("w:sz"), size)
                    border.set(qn("w:space"), "0")
                    border.set(qn("w:color"), color)
                    tcBorders.append(border)

                tcPr.append(tcBorders)

    # keep the old misspelled name as an alias so nothing breaks
    set_nomal_borders = set_normal_borders

    # ------------------------------------------------------------------ #
    #  Products table                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _set_table_grid(table, widths):
        tblPr = table._tbl.tblPr
        tblW = tblPr.find(qn("w:tblW"))
        if tblW is None:
            tblW = OxmlElement("w:tblW")
            tblPr.append(tblW)
        tblW.set(qn("w:w"), str(sum(widths)))
        tblW.set(qn("w:type"), "dxa")

        tblGrid = table._tbl.tblGrid
        if tblGrid is not None:
            for gridCol in list(tblGrid):
                tblGrid.remove(gridCol)
        else:
            tblGrid = OxmlElement("w:tblGrid")
            table._tbl.insert(0, tblGrid)
        for width in widths:
            gridCol = OxmlElement("w:gridCol")
            gridCol.set(qn("w:w"), str(width))
            tblGrid.append(gridCol)

    @staticmethod
    def _write_table_cell(cell, text, bold=False, size=8, color="2C2B5F", align="right"):
        for para in cell.paragraphs:
            for run in para.runs:
                run.text = ""
        p = cell.paragraphs[0] if cell.paragraphs else cell.add_paragraph()
        p.clear()
        # Proper cell padding for Arabic invoices
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.line_spacing = Pt(12)

        # Set cell margins
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        
        # Add cell margin settings
        tcMar = OxmlElement('w:tcMar')
        for margin in ['top', 'left', 'bottom', 'right']:
            mar = OxmlElement(f'w:{margin}')
            mar.set(qn('w:w'), '50')  # 50 twentieths of a point
            mar.set(qn('w:type'), 'dxa')
            tcMar.append(mar)
        tcPr.append(tcMar)

        pPr = p._p.get_or_add_pPr()
        if pPr.find(qn("w:bidi")) is None:
            bidi = OxmlElement("w:bidi")
            bidi.set(qn("w:val"), "1")
            jc = pPr.find(qn("w:jc"))
            if jc is not None:
                jc.addprevious(bidi)
            else:
                pPr.append(bidi)
                
        if align == "right":
            DocumentProcessor._set_paragraph_jc(p, "start")
        elif align == "center":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == "left":
            DocumentProcessor._set_paragraph_jc(p, "end")

        run = p.add_run(str(text))
        run.bold = bold
        run.font.name = "Arial"
        run.font.size = Pt(size)
        run.font.color.rgb = RGBColor.from_string(color)
        DocumentProcessor._set_run_rtl(run)

    @staticmethod
    def _format_al_murabaha_table(table, products, invoice_data=None):
        invoice_data = invoice_data or {}
        widths = [794, 1077, 624, 3856, 1191, 964, 1247, 454]
        DocumentProcessor._set_table_rtl(table)
        DocumentProcessor._set_table_grid(table, widths)

        marker_row_idx = None
        for idx, row in enumerate(table.rows):
            if any("{{products_table}}" in p.text for cell in row.cells for p in cell.paragraphs):
                marker_row_idx = idx
                break
        if marker_row_idx is None:
            return

        base_row = table.rows[marker_row_idx]

        def prepare_row(row):
            while len(row.cells) < 8:
                row.cells[-1]._tc.addnext(OxmlElement("w:tc"))
            for col_idx, cell in enumerate(row.cells[:8]):
                tcPr = cell._tc.get_or_add_tcPr()
                tcW = tcPr.find(qn("w:tcW"))
                if tcW is None:
                    tcW = OxmlElement("w:tcW")
                    tcPr.insert(0, tcW)
                tcW.set(qn("w:w"), str(widths[col_idx]))
                tcW.set(qn("w:type"), "dxa")

        def fill_product_row(row, product, idx, tall=False):
            prepare_row(row)
            unit_price = float(product.get("unit_price_numeric", 0) or 0)
            quantity = int(product.get("quantity", 1) or 1)
            line_before_tax = unit_price * quantity
            line_tax = round(line_before_tax * 0.05, 2)
            line_after_tax = line_before_tax + line_tax
            values = [
                product.get("color", product.get("location", "")),
                product.get("chassis_no", product.get("item_code", "")),
                product.get("model", "2019"),
                product.get("description", ""),
                f"{line_before_tax:,.2f}",
                f"{line_tax:,.2f}",
                f"{line_after_tax:,.2f}",
                str(idx),
            ]
            for col_idx, value in enumerate(values):
                DocumentProcessor._write_table_cell(
                    row.cells[col_idx],
                    value,
                    size=8,
                    align="center" if col_idx != 3 else "right",
                )
            trPr = row._tr.get_or_add_trPr()
            height = trPr.find(qn("w:trHeight"))
            if height is None:
                height = OxmlElement("w:trHeight")
                trPr.append(height)
            height.set(qn("w:val"), "3000" if tall else "520")
            height.set(qn("w:hRule"), "atLeast")

        if products:
            fill_product_row(base_row, products[0], 1, tall=len(products) == 1)
            insert_after = base_row._tr
            for idx, product in enumerate(products[1:], start=2):
                new_row = table.add_row()
                insert_after.addnext(new_row._tr)
                fill_product_row(new_row, product, idx)
                insert_after = new_row._tr

        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        if run.font.color.rgb is None:
                            run.font.color.rgb = RGBColor.from_string("2C2B5F")
        DocumentProcessor.set_normal_borders(table, color="2C2B5F", size="6")

    @staticmethod
    def _format_products_table(table, products, invoice_data=None):
        """
        Replace the placeholder row with a proper header + one row per product.
        Columns (RTL order, displayed right→left):
            [0] الوصف  [1] الكمية  [2] سعر الوحدة  [3] الإجمالي
        """
        invoice_data = invoice_data or {}
        # ---- remove all existing rows ----
        while len(table.rows) > 0:
            table._element.remove(table.rows[-1]._element)

        # ---- column config ----
        if invoice_data.get("template_style") == "al_murabaha":
            HEADERS = [
                "اللون", "رقم الهيكل", "الموديل", "المواصفات",
                "السعر قبل الضريبة", "الضريبة %5", "السعر بعد الضريبة", "م."
            ]
            COL_WIDTHS = [794, 1077, 624, 3856, 1191, 964, 1247, 454]
        elif invoice_data.get("template_style") in {"competitive_price", "auto_parts"}:
            HEADERS = ["##", "رقم الصنف", "الوصف", "الموقع", "السعر", "كمية", "% خصم", "الإجمالي"]
            COL_WIDTHS = [500, 1500, 3300, 900, 1000, 900, 900, 1200]
        else:
            HEADERS = ["الوصف", "الكمية", "سعر الوحدة", "الإجمالي"]
            COL_WIDTHS = [int(w * 914_400) for w in [3.0, 1.0, 1.8, 1.8]]
        NUM_COLS = len(HEADERS)

        tblPr = table._tbl.tblPr
        DocumentProcessor._set_table_rtl(table)
        DocumentProcessor._set_table_grid(table, COL_WIDTHS)
        tblInd = tblPr.find(qn("w:tblInd"))
        if tblInd is None:
            tblInd = OxmlElement("w:tblInd")
            tblPr.append(tblInd)
        tblInd.set(qn("w:w"), "0")
        tblInd.set(qn("w:type"), "dxa")
        tblGrid = table._tbl.tblGrid
        if tblGrid is not None:
            for gridCol in list(tblGrid):
                tblGrid.remove(gridCol)
        else:
            tblGrid = OxmlElement("w:tblGrid")
            table._tbl.insert(0, tblGrid)
        for width in COL_WIDTHS:
            gridCol = OxmlElement("w:gridCol")
            gridCol.set(qn("w:w"), str(width))
            tblGrid.append(gridCol)

        def _add_row_with_cols(t, n):
            """Add a row, ensuring it has exactly *n* cells."""
            row = t.add_row()
            # If the table was created with fewer columns, Word may give us
            # fewer cells; pad to NUM_COLS.
            while len(row.cells) < n:
                row.cells[-1]._tc.addnext(OxmlElement("w:tc"))
            return row

        def _set_cell(cell, text, bold=False, width=None, bg_color=None, color=None):
            """Write *text* into *cell*, applying RTL + optional styling."""
            # Set background shading
            if bg_color:
                tc = cell._tc
                tcPr = tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear")
                shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"), bg_color)
                tcPr.append(shd)

            # Set cell width
            if width:
                tc = cell._tc
                tcPr = tc.get_or_add_tcPr()
                tcW = OxmlElement("w:tcW")
                tcW.set(qn("w:w"), str(width))
                tcW.set(qn("w:type"), "dxa")
                tcPr.insert(0, tcW)

            # Clear and write text
            for para in cell.paragraphs:
                for run in para.runs:
                    run.text = ""
            p = cell.paragraphs[0] if cell.paragraphs else cell.add_paragraph()
            p.clear()
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)

            # RTL paragraph direction
            pPr = p._p.get_or_add_pPr()
            if pPr.find(qn("w:bidi")) is None:
                bidi_el = OxmlElement("w:bidi")
                bidi_el.set(qn("w:val"), "1")
                jc_el = pPr.find(qn("w:jc"))
                if jc_el is not None:
                    jc_el.addprevious(bidi_el)
                else:
                    pPr.append(bidi_el)
            DocumentProcessor._set_paragraph_jc(p, "start")

            run = p.add_run(text)
            run.bold = bold
            run.font.name = "Arial"
            run.font.size = Pt(8 if invoice_data.get("template_style") in {"competitive_price", "auto_parts", "al_murabaha"} else 10)
            if color:
                run.font.color.rgb = RGBColor.from_string(color)
            DocumentProcessor._set_run_rtl(run)

        # ---- header row ----
        hrow = _add_row_with_cols(table, NUM_COLS)
        hrow.cells[0].text = ""  # clear default content

        for col_idx, header_text in enumerate(HEADERS):
            _set_cell(
                hrow.cells[col_idx],
                header_text,
                bold=True,
                width=COL_WIDTHS[col_idx],
                bg_color=None if invoice_data.get("template_style") == "al_murabaha" else "D9E1F2",
                color="2C2B5F" if invoice_data.get("template_style") == "al_murabaha" else None,
            )

        # ---- product rows ----
        for idx, product in enumerate(products, start=1):
            row = _add_row_with_cols(table, NUM_COLS)
            if invoice_data.get("template_style") == "al_murabaha":
                unit_price = float(product.get("unit_price_numeric", 0) or 0)
                quantity = int(product.get("quantity", 1) or 1)
                line_before_tax = unit_price * quantity
                line_tax = round(line_before_tax * 0.05, 2)
                line_after_tax = line_before_tax + line_tax
                values = [
                    product.get("color", ""),
                    product.get("chassis_no", product.get("item_code", "")),
                    product.get("model", "2019"),
                    product.get("description", ""),
                    f"{line_before_tax:,.2f}",
                    f"{line_tax:,.2f}",
                    f"{line_after_tax:,.2f}",
                    str(idx),
                ]
            elif invoice_data.get("template_style") in {"competitive_price", "auto_parts"}:
                values = [
                    str(idx),
                    product.get("item_code", ""),
                    product.get("description", ""),
                    product.get("location", ""),
                    str(product.get("unit_price", "")).replace(" ريال", ""),
                    str(product.get("quantity", "")),
                    product.get("discount_percent", "0.00"),
                    str(product.get("total", "")).replace(" ريال", ""),
                ]
            else:
                values = [
                    product.get("description", ""),
                    str(product.get("quantity", "")),
                    product.get("unit_price", ""),
                    product.get("total", ""),
                ]
            for col_idx, val in enumerate(values):
                _set_cell(
                    row.cells[col_idx],
                    val,
                    width=COL_WIDTHS[col_idx],
                    color="2C2B5F" if invoice_data.get("template_style") == "al_murabaha" else None,
                )

        # ---- borders ----
        if invoice_data.get("template_style") == "al_murabaha":
            DocumentProcessor.set_normal_borders(table, color="2C2B5F", size="6")
        else:
            DocumentProcessor.set_normal_borders(table)

    # ------------------------------------------------------------------ #
    #  PDF conversion                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def convert_to_pdf(input_path, output_dir):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                subprocess.run(
                    [
                        "libreoffice",
                        "--headless",
                        "--convert-to", "pdf",
                        "--outdir", output_dir,
                        input_path,
                    ],
                    check=True,
                    timeout=30,
                )
                pdf_name = os.path.splitext(os.path.basename(input_path))[0] + ".pdf"
                pdf_path = os.path.join(output_dir, pdf_name)
                if os.path.exists(pdf_path):
                    return pdf_path
            except subprocess.TimeoutExpired:
                if attempt == max_retries - 1:
                    raise
                print(f"Conversion timeout — retrying ({attempt + 1}/{max_retries})")
                time.sleep(2)
            except Exception as e:
                print(f"PDF conversion failed: {e}")
                return None
        return None
