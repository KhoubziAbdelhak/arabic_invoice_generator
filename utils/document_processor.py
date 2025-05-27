from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import subprocess
import os
import time  

class DocumentProcessor:
    @staticmethod
    def create_invoice(template_path, invoice_data, output_path):
        doc = Document(template_path)
        products_table = None

        # Process regular placeholders in paragraphs
        for paragraph in doc.paragraphs:
            for key, value in invoice_data.items():
                placeholder = '{{' + key + '}}'
                if placeholder in paragraph.text:
                    paragraph.text = paragraph.text.replace(placeholder, str(value))

        # Process tables
        for table in doc.tables:
            if products_table is None:  # Find products table
                for row in table.rows:
                    for cell in row.cells:
                        if '{{products_table}}' in cell.text:
                            products_table = table
                            break

            # Process placeholders in table cells
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        for key, value in invoice_data.items():
                            placeholder = '{{' + key + '}}'
                            if placeholder in paragraph.text:
                                paragraph.text = paragraph.text.replace(placeholder, str(value))

        # Format products table
        if products_table:
            DocumentProcessor._format_products_table(products_table, invoice_data['products'])

        doc.save(output_path)
        return output_path

    @staticmethod
    def set_nomal_borders(table):
        for row in table.rows:
            for cell in row.cells:
                tc = cell._tc
                tcPr = tc.get_or_add_tcPr()
                tcBorders = OxmlElement('w:tcBorders')

                for border_name in ['top', 'left', 'bottom', 'right']:
                    border = OxmlElement(f'w:{border_name}')
                    border.set(qn('w:val'), 'single')
                    border.set(qn('w:sz'), '4')
                    border.set(qn('w:space'), '0')
                    border.set(qn('w:space'), '0')
                    tcBorders.append(border)

                tcPr.append(tcBorders)

    @staticmethod
    def _format_products_table(table, products):
        # Clear existing rows except header
        while len(table.rows) > 1:
            table._element.remove(table.rows[-1]._element)


        # Add product rows
        headers = ["الوصف", "الكمية", "سعر الوحدة", "الإجمالي"]

        num_columns = len(headers)
        # Add columns if necessary
        for _ in range(num_columns - len(table.columns)):
            table.add_column(width=914400)
        # Set column widths
        column_widths = [2.5, 1.0, 2.0, 2.0]
        for i, width in enumerate(column_widths):
            if i < len(table.columns):
                table.columns[i].width = int(width * 914400)
        
        # Set header
        for i, header in enumerate(headers):
            table.rows[0].cells[i].text = header
            para = table.rows[0].cells[i].paragraphs[0]
            para.alignment = 2  # Right align
            para.runs[0].bold = True

        # Add products
        for product in products:
            row = table.add_row()
            row.cells[0].text = product['description']
            row.cells[1].text = str(product['quantity'])
            row.cells[2].text = product['unit_price']
            row.cells[3].text = product['total']
            
            # Right align all cells
            for cell in row.cells:
                cell.paragraphs[0].alignment = 2
        
        # Set normal borders
        DocumentProcessor.set_nomal_borders(table)



    @staticmethod
    def convert_to_pdf(input_path, output_dir):
        try:
            # Add timeout and retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = subprocess.run([
                        "libreoffice",
                        "--headless",
                        "--convert-to", "pdf",
                        "--outdir", output_dir,
                        input_path
                    ], check=True, timeout=30)  # 30 second timeout
                    
                    pdf_name = os.path.splitext(os.path.basename(input_path))[0] + '.pdf'
                    pdf_path = os.path.join(output_dir, pdf_name)
                    
                    if os.path.exists(pdf_path):
                        return pdf_path
                except subprocess.TimeoutExpired:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Conversion timeout, attempt {attempt + 1} of {max_retries}")
                    time.sleep(2)  # Wait before retry
                    
            return None
        except Exception as e:
            print(f"PDF conversion failed: {e}")
            return None
