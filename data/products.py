import pandas as pd
import random
from faker import Faker
import os

fake = Faker('ar_SA')


class ProductGenerator:
    def __init__(self, csv_path=None):
        # Initialize original product types, adjectives, brands, and colors
        self.product_types = [
            # Electronics
            "حاسوب محمول", "هاتف ذكي", "شاشة عرض", "طابعة ليزر", "ماوس لاسلكي",
            "لوحة مفاتيح", "سماعات رأس", "كاميرا رقمية", "قرص صلب خارجي", "جهاز راوتر",
            "مكبر صوت", "ميكروفون", "شاحن متنقل", "حقيبة لابتوب", "حامل هاتف",
            "ساعة ذكية", "جهاز لوحي", "قارئ كتب إلكتروني", "كاميرا مراقبة", "جهاز تحكم عن بعد",

            # Home Appliances
            "ثلاجة", "غسالة أطباق", "فرن كهربائي", "ميكروويف", "خلاط",
            "عصارة", "مكواة بخار", "مكنسة كهربائية", "مروحة", "مكيف هواء",
            "سخان ماء", "فرن غاز", "غسالة ملابس", "مجفف ملابس", "محمصة خبز",

            # Furniture
            "طاولة طعام", "كرسي مكتب", "أريكة", "سرير", "خزانة ملابس",
            "رف كتب", "طاولة جانبية", "مكتب دراسة", "كنبة", "مقعد",

            # Clothing
            "قميص رجالي", "بلوزة نسائية", "بنطال جينز", "حذاء رياضي", "معطف شتوي",
            "جاكيت", "فستان", "تنورة", "حقيبة يد", "حذاء رسمي",

            # Groceries
            "زيت زيتون", "أرز بسمتي", "سكر", "ملح", "قهوة عربية",
            "شاي", "معكرونة", "حليب", "بيض", "دقيق",

            # Office Supplies
            "علبة أوراق", "قلم حبر", "مجلد", "دفتر", "ملصقات",
            "مشبك ورق", "لوحة بيضاء", "علامات لونية", "آلة حاسبة", "دباسة"
        ]

        self.adjectives = [
            "متطور", "احترافي", "فاخر", "اقتصادي", "محمول", "ممتاز", "عالي الجودة",
            "سريع", "قوي", "خفيف", "متعدد الوظائف", "ذكي", "آمن", "مريح", "أنيق",
            "عملي", "مبتكر", "حديث", "كلاسيكي", "مدمج", "متين", "فعال", "سلس",
            "مشرق", "هادئ", "قوي التحمل", "مقاوم للماء", "لاسلكي", "صديق للبيئة"
        ]

        self.brands = [
            "تكنوساينس", "سمارت تك", "إلكترونيكا", "ديجيتال بلس", "تك برو",
            "أوميجا", "جولد لاين", "إيليت", "بريمو", "سوبرا",
            "ماكسيموم", "أوفيس برو", "هوم لايف", "كوزي", "ستايل",
            "رويال", "بست تشويس", "فريش", "ناتشرال", "بريميوم"
        ]

        self.colors = [
            "أسود", "أبيض", "فضي", "ذهبي", "أحمر",
            "أزرق", "أخضر", "وردي", "بنفسجي", "برتقالي",
            "رمادي", "بني", "كريمي", "ذهبي", "فضي"
        ]

        # Load CSV data if path is provided
        self.csv_products = []
        if csv_path and os.path.exists(csv_path):
            self._load_csv_products(csv_path)

    def _load_csv_products(self, csv_path):
        """Load products from a CSV file - only including Arabic products"""
        try:
            df = pd.read_csv(csv_path)

            # Process each row in the CSV
            for _, row in df.iterrows():
                # Extract item name
                item_name = row.get('Item_Name', '')

                # Skip non-Arabic products
                if not any('\u0600' <= c <= '\u06FF' for c in item_name):
                    continue

                # Extract brand (if available)
                brand = row.get('Brand', '')
                if pd.isna(brand) or brand == '':
                    brand = None

                # Extract category (class)
                category = row.get('class', '')

                # Extract weight/size information (stored but not used in name)
                weight = row.get('Weight', '')

                # Extract unit information
                unit = row.get('Unit', '')
                pack = row.get('Pack', '')

                # Create product dictionary
                product = {
                    'name': item_name,
                    'brand': brand,
                    'category': category,
                    'weight': weight,
                    'unit': unit,
                    'pack': pack
                }

                self.csv_products.append(product)

            print(f"Loaded {len(self.csv_products)} Arabic products from CSV")
        except Exception as e:
            print(f"Error loading CSV: {e}")

    def get_random_price(self, category=None):
        """Generate a random price based on category"""
        if category == "Water" or category == "Soft Drinks & Juices":
            return round(random.uniform(5, 30), 2)
        elif category == "Vegetables & Fruits":
            return round(random.uniform(10, 50), 2)
        elif category == "Cleaning Supplies":
            return round(random.uniform(30, 200), 2)
        elif "Tins" in str(category) or "Rice" in str(category):
            return round(random.uniform(15, 80), 2)
        else:
            return round(random.uniform(5, 500), 2)

    def generate_products_from_csv(self, num_products):
        """Generate products using the loaded CSV data"""
        products = []

        if not self.csv_products:
            print("No Arabic CSV products loaded. Using random generation instead.")
            return self.generate_arabic_products(num_products)

        # Select random products from the loaded CSV data
        selected_products = random.choices(
            self.csv_products,
            k=min(num_products, len(self.csv_products))
        )

        # Fill any remaining products with random data if needed
        if len(selected_products) < num_products:
            additional_needed = num_products - len(selected_products)
            # Repeat some products with different quantities
            additional = random.choices(self.csv_products, k=additional_needed)
            selected_products.extend(additional)

        # Generate product entries
        for i, product in enumerate(selected_products):
            quantity = random.randint(1, 10)
            price = self.get_random_price(product.get('category'))
            total = round(price * quantity, 2)

            # Format product name
            product_name = product['name']

            # Add brand if available and it's in Arabic
            if product.get('brand') and not pd.isna(product['brand']) and product['brand'] != '':
                if any('\u0600' <= c <= '\u06FF' for c in product['brand']) and product['brand'] not in product_name:
                    product_name = f"{product['brand']} {product_name}"

            # Weight is intentionally not included in the product name

            products.append({
                'description': product_name,
                'quantity': quantity,
                'unit_price': f"{price:.2f} ريال",
                'unit_price_numeric': price,
                'total': f"{total:.2f} ريال",
                'total_numeric': total,
                'row_index': i,
                'pack': product.get('pack', ''),
                'unit': product.get('unit', '')
            })

        return products

    def generate_arabic_products(self, num_products):
        """Generate only Arabic product data"""
        products = []
        for i in range(num_products):
            price = round(random.uniform(5, 500), 2)
            quantity = random.randint(1, 10)
            total = round(price * quantity, 2)

            # Randomly decide the product name structure
            name_style = random.randint(1, 4)
            if name_style == 1:
                product_name = f"{random.choice(self.brands)} {random.choice(self.product_types)}"
            elif name_style == 2:
                product_name = f"{random.choice(self.product_types)} {random.choice(self.colors)}"
            elif name_style == 3:
                product_name = f"{random.choice(self.brands)} {random.choice(self.product_types)}"
            else:
                product_name = f"{random.choice(self.product_types)}"

            products.append({
                'description': product_name,
                'quantity': quantity,
                'unit_price': f"{price:.2f} ريال",
                'unit_price_numeric': price,
                'total': f"{total:.2f} ريال",
                'total_numeric': total,
                'row_index': i,
                'pack': random.choice(['كيس', 'علبة', 'زجاجة', 'عبوة', 'باكو']),
                'unit': random.choice(['جم', 'مل', 'كجم', ''])
            })
        return products

    def generate_products(self, num_products):
        """Original method to generate random product data - maintained for compatibility"""
        products = []
        for i in range(num_products):
            price = round(random.uniform(5, 5000), 2)
            quantity = random.randint(1, 50)
            total = round(price * quantity, 2)

            # Randomly decide the product name structure
            name_style = random.randint(1, 4)
            if name_style == 1:
                product_name = f"{random.choice(self.brands)} {random.choice(self.product_types)} {random.choice(self.adjectives)}"
            elif name_style == 2:
                product_name = f"{random.choice(self.product_types)} {random.choice(self.colors)} {random.choice(self.adjectives)}"
            elif name_style == 3:
                product_name = f"{random.choice(self.brands)} {random.choice(self.colors)} {random.choice(self.product_types)}"
            else:
                product_name = f"{random.choice(self.product_types)} {random.choice(self.adjectives)}"

            products.append({
                'description': product_name,
                'quantity': quantity,
                'unit_price': f"{price:.2f} ريال",
                'unit_price_numeric': price,
                'total': f"{total:.2f} ريال",
                'total_numeric': total,
                'row_index': i
            })
        return products


# Example usage
if __name__ == "__main__":
    # Initialize with CSV path
    generator = ProductGenerator(csv_path="products.csv")

    # Generate products using CSV data - Arabic only
    csv_products = generator.generate_products_from_csv(15)
    print("\nArabic Products from CSV:")
    for p in csv_products:
        print(f"{p['description']} - {p['quantity']} x {p['unit_price']} = {p['total']}")

    # Generate Arabic-only random products
    arabic_products = generator.generate_arabic_products(5)
    print("\nRandomly Generated Arabic Products:")
    for p in arabic_products:
        print(f"{p['description']} - {p['quantity']} x {p['unit_price']} = {p['total']}")