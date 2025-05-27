# template_config.py

TEMPLATES = {
    "standard_invoice": {
        "template_file": "standard_invoice_template.docx",
        "mapping": {
            "invoice_ref": "[رقم الفاتورة]",
            "seller_name": "[اسم الشركة]",
            "seller_address": "[عنوان الشارع]",
            "seller_city": "[المدينة، الرمز البريدي للشارع]",
            "seller_vat_number": "[الرقم الضريبي]",
            "issue_datetime": "[أدخل التاريخ]",
            "email": "[البريد الإلكتروني]",
            "website": "[موقع الويب]",
            "phone_number": "[أدخل رقم الهاتف]",
            "fax_number": "[أدخل الفاكس]",
            "products": {
                "description": "[أدخل الوصف {}]",  # Will be formatted with number
                "amount": "[أدخل المبلغ]"
            }
        }
    },
    "simple_invoice": {  # Your new template
        "template_file": "test_template.docx",
        "mapping": {
            "company_name": "[اسم الشركة]",
            "company_logo": "[شعار الشركة]",
            "street_address": "[عنوان الشارع]",
            "city_zip": "[المدينة، الرمز البريدي للشارع]",
            "phone": "[أدخل رقم الهاتف]",
            "fax": "[أدخل الفاكس]",
            "email": "[الإيميل]",
            "website": "[موقع الويب]",
            "invoice_number": "[رقم الفاتورة]",
            "date": "[أدخل التاريخ]",
            "client_name": "[الاسم]",
            "client_company": "[اسم الشركة]",
            "client_address": "[عنوان الشارع]",
            "client_city_zip": "[المدينة، الرمز البريدي للشارع]",
            "client_phone": "[أدخل الهاتف]",
            "client_email": "[البريد الإلكتروني]",
            "project_description": "[وصف المشروع أو الخدمة]",
            "purchase_order": "[رقم طلب الشراء]",
            "products": {
                "description": "[أدخل الوصف {}]",
                "amount": "[أدخل المبلغ]"
            },
            "total": "[أدخل القيمة الإجمالية]",
            "contact_name": "[الاسم]",
            "contact_phone": "[الهاتف]"
        }
    }
}

