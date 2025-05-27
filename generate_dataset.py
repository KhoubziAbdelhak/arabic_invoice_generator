import random
from config.config import *
from data.products import ProductGenerator
from utils.document_processor import DocumentProcessor
from utils.image_processor import ImageProcessor
from utils.annotation_processor import AnnotationProcessor
from faker import Faker
import os

def generate_dataset():
    fake = Faker('ar_SA')
    product_generator = ProductGenerator(csv_path=PRODUCTS_CSV_PATH)

    # Get all template files from the TEMPLATE_DIR
    template_files = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith('.docx')]

    # Calculate the number of digits needed for zero padding
    total_invoices = NUM_INVOICES_TO_GENERATE
    padding_digits = len(str(total_invoices))

    for i in range(NUM_INVOICES_TO_GENERATE):
        try:
            # Create zero-padded invoice number
            invoice_num = str(i).zfill(padding_digits)
            print(f"\nProcessing invoice {invoice_num}...")

            # * change the NUM_PRODUCTS_PER_INVOICE with 40% variation
            num_products = int(NUM_PRODUCTS_PER_INVOICE * (1 + random.uniform(-0.4, 0.4)))

            # Generate invoice data
            products = product_generator.generate_products_from_csv(num_products)

            # Calculate totals
            subtotal = sum(float(p['unit_price_numeric']) * p['quantity'] for p in products)
            tax = round(subtotal * VAT_RATE, 2)
            total = subtotal + tax

            invoice_data = {
                'invoice_ref': f"INV-{fake.numerify(text='####')}-2024",
                'seller_name': fake.company(),
                'seller_address': fake.address(),
                'seller_vat_number': fake.numerify(text='###########'),
                'issue_datetime': fake.date(),
                'email': fake.email(),
                'website': fake.domain_name(),
                'phone_number': fake.phone_number(),
                'fax_number': fake.phone_number(),
                'subtotal': f"{subtotal:.2f} ريال",
                'tax': f"{tax:.2f} ريال",
                'total': f"{total:.2f} ريال",
                'products': products
            }

            print("Generated invoice data")

            # Generate document with zero-padded naming
            docx_path = os.path.join(DOCX_DIR, f'invoice_{invoice_num}.docx')
            print(f"Creating DOCX at: {docx_path}")

            selected_template = random.choice(template_files)
            print(f"Using template: {selected_template}")

            DocumentProcessor.create_invoice(
                os.path.join(TEMPLATE_DIR, selected_template),
                invoice_data,
                docx_path
            )
            print("Created DOCX successfully")

            # Convert to PDF
            print("Converting to PDF...")
            pdf_path = DocumentProcessor.convert_to_pdf(docx_path, PDF_DIR)

            if pdf_path and os.path.exists(pdf_path):
                print(f"Created PDF at: {pdf_path}")

                # Convert to images
                print("Converting to images...")
                image_paths = ImageProcessor.pdf_to_images(pdf_path, IMAGES_DIR)

                # Create annotation processor instance
                processor = AnnotationProcessor()

                # Generate and save annotations
                print("Generating annotations...")
                for j, image_path in enumerate(image_paths):
                    if j > 0:
                        pass

                    # Process PDF and get annotations
                    annotations = processor.process_pdf(
                        pdf_path=pdf_path,
                        image_path=image_path
                    )

                    # Save annotations with zero-padded naming
                    annotation_path = os.path.join(ANNOTATIONS_DIR, f'invoice_{invoice_num}.json')
                    processor.save_annotations(annotation_path)

                print(f"Annotations saved at: {annotation_path}")
            else:
                print(f"Failed to generate PDF for invoice {invoice_num}")

            # remove the .docx
            os.remove(docx_path)
            # remove the .pdf
            os.remove(pdf_path)

        except Exception as e:
            print(f"Error processing invoice {invoice_num}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue

if __name__ == "__main__":
    generate_dataset()
