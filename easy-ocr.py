import easyocr
reader = easyocr.Reader(['ar']) # this needs to run only once to load the model into memory
result = reader.readtext('/var/home/abdelhak/programming/pfe/arabic_invoice_generator/output/images/invoice_031.png')
print(result)
