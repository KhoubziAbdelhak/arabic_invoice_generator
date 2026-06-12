import random
from config.config import *
from data.products import ProductGenerator
from utils.document_processor import DocumentProcessor
from utils.image_processor import ImageProcessor
from utils.annotation_processor import AnnotationProcessor
from utils.lilt_annotation_processor import LiLTAnnotationProcessor
from faker import Faker
import os
import re
from datetime import datetime, timedelta
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP


# ---------------------------------------------------------------------------
# Helpers for realistic Saudi / Arabic fake data
# ---------------------------------------------------------------------------

SAR = "ر.س"

SAUDI_VENDOR_PROFILES = [
    {
        "ar": "مؤسسة نجم الخليج للتجارة",
        "en": "Gulf Star Trading Est.",
        "address": "الرياض، حي الملز، شارع الأمير عبدالمحسن بن عبدالعزيز",
        "branch": "الرياض - الملز",
        "domain": "gulfstar-sa.com",
    },
    {
        "ar": "شركة روافد الحجاز للمقاولات",
        "en": "Rawafed Al-Hijaz Contracting Co.",
        "address": "جدة، حي السلامة، طريق المدينة المنورة",
        "branch": "جدة - السلامة",
        "domain": "rawafed-hijaz.com",
    },
    {
        "ar": "مركز التميز لقطع غيار السيارات",
        "en": "Excellence Auto Parts Center",
        "address": "الدمام، حي الخالدية، شارع الملك سعود",
        "branch": "الدمام - الخالدية",
        "domain": "excellenceparts.sa",
    },
    {
        "ar": "مؤسسة مدار التقنية للحلول المكتبية",
        "en": "Madar Tech Office Solutions",
        "address": "مكة المكرمة، حي العوالي، شارع إبراهيم الجفالي",
        "branch": "مكة - العوالي",
        "domain": "madartech.sa",
    },
    {
        "ar": "شركة أفق المدينة للخدمات اللوجستية",
        "en": "Madinah Horizon Logistics Co.",
        "address": "المدينة المنورة، حي قباء، طريق الأمير عبدالمجيد",
        "branch": "المدينة - قباء",
        "domain": "horizonlogistics.sa",
    },
]

SAUDI_CUSTOMER_NAMES = [
    "شركة مدى الإمداد التجارية",
    "مؤسسة حلول البناء الحديثة",
    "مركز رواد الصيانة",
    "شركة اليمامة للتوريدات",
    "مؤسسة عبدالعزيز السهلي التجارية",
    "شركة مسار الخليج للتشغيل",
    "مؤسسة صالح العتيبي للمقاولات",
    "مركز الوفاء لخدمات السيارات",
    "شركة أجيال التقنية المحدودة",
    "مؤسسة دار النخبة للتجارة",
]

SAUDI_PERSON_NAMES = [
    "محمد عبدالله القحطاني",
    "عبدالرحمن صالح الشهري",
    "خالد سعد العتيبي",
    "سارة ناصر الحربي",
    "نورة فهد الدوسري",
    "أحمد إبراهيم الزهراني",
    "ريم عبدالله الغامدي",
    "ماجد علي المطيري",
]

SAUDI_ADDRESSES = [
    "الرياض، حي العليا، شارع موسى بن نصير",
    "جدة، حي الروضة، شارع الأمير سعود الفيصل",
    "الخبر، حي الثقبة، شارع مكة",
    "الدمام، حي الفيصلية، طريق الملك فهد",
    "مكة المكرمة، حي الشرائع، شارع عمر قاضي",
    "المدينة المنورة، حي العزيزية، طريق الهجرة",
    "الطائف، حي شهار، شارع الجيش",
    "بريدة، حي الريان، طريق عثمان بن عفان",
]

SAUDI_CITY_POSTCODES = [
    "الرياض 12241",
    "جدة 23434",
    "الدمام 32241",
    "الخبر 34623",
    "مكة المكرمة 24372",
    "المدينة المنورة 42313",
    "الطائف 26523",
    "بريدة 52361",
]

AUTO_PART_ITEMS = [
    ("فلتر زيت أصلي تويوتا", 28, 55, "قطعة"),
    ("زيت محرك 5W-30 عبوة 4 لتر", 95, 165, "عبوة"),
    ("تيل فرامل أمامي", 120, 260, "طقم"),
    ("بطارية سيارة 70 أمبير", 280, 520, "قطعة"),
    ("مساعد أمامي يمين", 220, 480, "قطعة"),
    ("سير مكينة", 45, 130, "قطعة"),
    ("فلتر مكيف", 35, 85, "قطعة"),
    ("طقم مساحات زجاج", 25, 70, "طقم"),
    ("كفر مقاس 17", 310, 690, "قطعة"),
    ("بواجي إيريديوم", 85, 210, "طقم"),
    ("رديتر ماء", 340, 780, "قطعة"),
    ("حساس أكسجين", 180, 430, "قطعة"),
]

GENERAL_ITEMS = [
    ("توريد ورق تصوير A4", 18, 32, "كرتون"),
    ("خدمة صيانة دورية", 150, 450, "خدمة"),
    ("حبر طابعة ليزر أسود", 180, 360, "قطعة"),
    ("مستلزمات تعبئة وتغليف", 25, 95, "حزمة"),
    ("اشتراك دعم فني شهري", 250, 900, "شهر"),
    ("توريد كابلات شبكة", 12, 35, "قطعة"),
    ("كرسي مكتب شبكي", 220, 580, "قطعة"),
    ("لوحة مفاتيح عربية", 45, 120, "قطعة"),
    ("خدمة نقل داخل المدينة", 120, 380, "رحلة"),
    ("مواد تنظيف للمكاتب", 35, 160, "عبوة"),
    ("صيانة أجهزة حاسب", 90, 320, "خدمة"),
    ("تركيب نقاط شبكة", 75, 210, "نقطة"),
]

SAUDI_VENDOR_PROFILES += [
    {
        "ar": "شركة اليسر للتجهيزات المكتبية",
        "en": "Al-Yusr Office Supplies Co.",
        "address": "الرياض، حي السليمانية، شارع الضباب",
        "branch": "الرياض - السليمانية",
        "domain": "alyusr-office.sa",
    },
    {
        "ar": "مؤسسة رواحل الشرق للنقل",
        "en": "Rawahil East Transport Est.",
        "address": "الخبر، حي الراكة، طريق الأمير فيصل بن فهد",
        "branch": "الخبر - الراكة",
        "domain": "rawahil-east.sa",
    },
    {
        "ar": "شركة مسارات الصيانة للتشغيل",
        "en": "Masarat Maintenance Operations",
        "address": "جدة، حي النزهة، شارع حراء",
        "branch": "جدة - النزهة",
        "domain": "masarat-ops.com",
    },
    {
        "ar": "مؤسسة البيان للمواد الغذائية",
        "en": "Al-Bayan Food Supplies Est.",
        "address": "الدمام، حي الشاطئ، شارع الخليج",
        "branch": "الدمام - الشاطئ",
        "domain": "albayanfoods.sa",
    },
    {
        "ar": "شركة أساس التقنية للأنظمة",
        "en": "Asas Tech Systems Co.",
        "address": "الرياض، حي الصحافة، طريق أنس بن مالك",
        "branch": "الرياض - الصحافة",
        "domain": "asastech.sa",
    },
    {
        "ar": "مركز الرافعة لقطع الغيار",
        "en": "Al-Rafiah Spare Parts Center",
        "address": "مكة المكرمة، حي الكعكية، شارع الحج",
        "branch": "مكة - الكعكية",
        "domain": "rafiahparts.com",
    },
    {
        "ar": "مؤسسة مرسى الخليج للمقاولات",
        "en": "Gulf Marsa Contracting Est.",
        "address": "ينبع، حي السميري، طريق الملك عبدالعزيز",
        "branch": "ينبع - السميري",
        "domain": "gulfmarsa.sa",
    },
    {
        "ar": "شركة نماء المدينة للتوريد",
        "en": "Nama Madinah Supply Co.",
        "address": "المدينة المنورة، حي العريض، شارع السلام",
        "branch": "المدينة - العريض",
        "domain": "namasupply.sa",
    },
]

SAUDI_CUSTOMER_NAMES += [
    "شركة روابي نجد للخدمات",
    "مؤسسة أركان الخليج للتجارة",
    "مركز الفارس للصيانة السريعة",
    "شركة العطاء الذكي للتقنية",
    "مؤسسة بدر الشمال للمقاولات",
    "شركة وادي الحجاز للتوريد",
    "مركز النخبة للعناية بالسيارات",
    "مؤسسة عمران المدينة التجارية",
    "شركة جسور الشرقية للتشغيل",
    "مؤسسة البدر للتموين",
    "شركة ركن الجودة للخدمات",
    "مركز المسار الحديث",
]

SAUDI_PERSON_NAMES += [
    "يوسف صالح القحطاني",
    "عبدالله ماجد الحربي",
    "فهد ناصر السبيعي",
    "تركي عبدالرحمن الغامدي",
    "عبدالعزيز سالم الزهراني",
    "منى خالد الدوسري",
    "لطيفة محمد الشهراني",
    "هند فهد العتيبي",
    "وليد إبراهيم المطيري",
    "سلمان علي القرني",
]

SAUDI_ADDRESSES += [
    "الرياض، حي النرجس، طريق أبي بكر الصديق",
    "جدة، حي الصفا، شارع الأربعين",
    "الدمام، حي أحد، شارع عمر بن الخطاب",
    "الخبر، حي العليا، طريق الملك سلمان",
    "مكة المكرمة، حي الزاهر، شارع الستين",
    "المدينة المنورة، حي الدفاع، طريق الملك عبدالله",
    "الطائف، حي الفيصلية، شارع وادي وج",
    "الأحساء، حي الهفوف، شارع الظهران",
    "تبوك، حي المروج، طريق الملك فهد",
    "حائل، حي المنتزه، شارع الأمير سلطان",
]

SAUDI_CITY_POSTCODES += [
    "ينبع 46424",
    "الأحساء 36361",
    "تبوك 47312",
    "حائل 55425",
    "خميس مشيط 62461",
    "جازان 82723",
]

AUTO_PART_ITEMS += [
    ("كمبروسر مكيف", 650, 1450, "قطعة"),
    ("دينمو شحن", 420, 980, "قطعة"),
    ("طرمبة بنزين", 260, 740, "قطعة"),
    ("مقص أمامي", 190, 520, "قطعة"),
    ("قماشات فرامل خلفية", 95, 240, "طقم"),
    ("كلتش مروحة", 180, 430, "قطعة"),
    ("فلتر هواء مكينة", 25, 90, "قطعة"),
    ("وجه غطاء بلوف", 35, 115, "قطعة"),
    ("جلدة مقص", 22, 75, "قطعة"),
    ("شمعة أمامية يمين", 260, 850, "قطعة"),
    ("صدام أمامي", 480, 1300, "قطعة"),
    ("مراية جانبية كهربائية", 190, 620, "قطعة"),
]

GENERAL_ITEMS += [
    ("توريد أجهزة راوتر مكتبية", 180, 520, "قطعة"),
    ("رخصة برنامج محاسبي", 450, 1250, "رخصة"),
    ("خدمة أرشفة مستندات", 300, 950, "خدمة"),
    ("صيانة مكيفات سبليت", 120, 380, "خدمة"),
    ("توريد مصابيح LED", 18, 65, "قطعة"),
    ("أعمال سباكة خفيفة", 150, 600, "خدمة"),
    ("تجهيز طاولة اجتماع", 650, 1800, "قطعة"),
    ("توريد مياه شرب مكتبية", 12, 28, "كرتون"),
    ("خدمة تنظيف شهرية", 700, 2200, "شهر"),
    ("تركيب كاميرا مراقبة", 220, 750, "نقطة"),
    ("خدمة استضافة بريد إلكتروني", 180, 480, "شهر"),
    ("توريد أحبار ملونة", 210, 540, "طقم"),
]

SAUDI_CITIES = [
    "الرياض", "جدة", "الدمام", "الخبر", "مكة المكرمة", "المدينة المنورة",
    "الطائف", "بريدة", "ينبع", "الأحساء", "تبوك", "حائل", "خميس مشيط",
    "جازان", "نجران", "أبها", "الجبيل", "القطيف", "عرعر", "سكاكا",
]

SAUDI_DISTRICTS = [
    "العليا", "الملز", "السليمانية", "الروضة", "السلامة", "الخالدية",
    "النزهة", "الفيصلية", "العزيزية", "النسيم", "الصفا", "الريان",
    "الشاطئ", "الواحة", "النرجس", "الصحافة", "المروج", "الربوة",
    "الشفا", "قباء", "العوالي", "الراكة", "الثقبة", "الزاهر",
]

SAUDI_STREETS = [
    "طريق الملك فهد", "طريق الملك عبدالله", "طريق الأمير محمد بن فهد",
    "شارع الأمير سلطان", "شارع التحلية", "شارع الستين", "شارع حراء",
    "شارع فلسطين", "شارع عمر بن الخطاب", "شارع عثمان بن عفان",
    "شارع الأمير نايف", "شارع الملك سعود", "طريق المدينة المنورة",
    "شارع الأمير عبدالمجيد", "طريق الهجرة", "شارع أبي بكر الصديق",
    "شارع أنس بن مالك", "شارع موسى بن نصير", "طريق خريص", "طريق الدمام",
]

BUSINESS_SECTORS = [
    ("auto_parts", "قطع غيار السيارات", "Auto Parts", "parts"),
    ("office", "التجهيزات المكتبية", "Office Supplies", "office"),
    ("it", "تقنية المعلومات", "IT Solutions", "tech"),
    ("construction", "مواد البناء", "Building Materials", "build"),
    ("electrical", "المواد الكهربائية", "Electrical Supplies", "electric"),
    ("plumbing", "الأدوات الصحية", "Plumbing Supplies", "plumbing"),
    ("logistics", "الخدمات اللوجستية", "Logistics Services", "logistics"),
    ("cleaning", "خدمات النظافة", "Cleaning Services", "clean"),
    ("food", "المواد الغذائية", "Food Supplies", "food"),
    ("medical", "المستلزمات الطبية", "Medical Supplies", "medical"),
    ("safety", "معدات السلامة", "Safety Equipment", "safety"),
    ("printing", "الطباعة والتغليف", "Printing & Packaging", "print"),
    ("furniture", "الأثاث المكتبي", "Office Furniture", "furniture"),
    ("hvac", "التكييف والتبريد", "HVAC Services", "hvac"),
    ("maintenance", "الصيانة والتشغيل", "Maintenance Operations", "ops"),
    ("telecom", "الاتصالات والشبكات", "Telecom & Networks", "networks"),
]

BUSINESS_PREFIXES = [
    "رواد", "نجوم", "أفق", "نماء", "مدار", "مسار", "ركن", "أساس",
    "روافد", "منار", "إمداد", "مجد", "جسور", "التميز", "اليسر",
    "القمة", "البدر", "النخبة", "الوفاء", "المحور", "العطاء", "الريادة",
]

BUSINESS_NOUNS = [
    "الخليج", "نجد", "الحجاز", "المدينة", "الشرق", "الغرب", "الشمال",
    "الجنوب", "الوطن", "الصفوة", "البيان", "المستقبل", "الابتكار",
    "الاعتماد", "المسار", "المنار", "الجزيرة", "التكامل", "الجودة",
]

LEGAL_FORMS = ["شركة", "مؤسسة", "مركز", "مجموعة", "مصنع", "معرض"]

EN_PREFIXES = [
    "Gulf", "Najd", "Hijaz", "Riyadh", "Madinah", "East", "West",
    "Prime", "Elite", "Future", "Quality", "United", "Smart", "Nama",
]

SECTOR_ITEM_CATALOG = {
    "office": [
        ("ملفات أرشفة مقاس A4", 12, 45, "حزمة"),
        ("دفاتر فواتير مطبوعة", 25, 90, "دفتر"),
        ("آلة تغليف حراري", 180, 620, "قطعة"),
        ("ورق حراري لنقاط البيع", 8, 28, "رول"),
        ("أقلام حبر أزرق", 18, 55, "علبة"),
        ("آلة حاسبة مكتبية", 35, 160, "قطعة"),
        ("خزنة ملفات معدنية", 420, 1350, "قطعة"),
        ("حامل شاشة مكتبي", 65, 240, "قطعة"),
    ],
    "it": [
        ("جهاز حاسب مكتبي Core i5", 1450, 3200, "جهاز"),
        ("شاشة LED مقاس 24 بوصة", 420, 980, "قطعة"),
        ("قرص تخزين SSD سعة 1TB", 220, 520, "قطعة"),
        ("نقطة وصول لاسلكية", 180, 690, "قطعة"),
        ("اشتراك حماية ضد الفيروسات", 95, 360, "رخصة"),
        ("خدمة نسخ احتياطي سحابي", 150, 780, "شهر"),
        ("تركيب خادم ملفات", 900, 3500, "خدمة"),
        ("كابل HDMI عالي السرعة", 18, 75, "قطعة"),
    ],
    "construction": [
        ("أسمنت مقاوم للأملاح", 18, 32, "كيس"),
        ("حديد تسليح 12 ملم", 2100, 3150, "طن"),
        ("بلك إسمنتي عازل", 2, 6, "قطعة"),
        ("دهان داخلي أبيض", 95, 260, "جالون"),
        ("ألواح جبس مقاومة للرطوبة", 24, 58, "لوح"),
        ("رمل مغسول", 250, 650, "رد"),
        ("خرسانة جاهزة C30", 190, 310, "متر مكعب"),
        ("مواد عزل حراري", 35, 120, "متر"),
    ],
    "electrical": [
        ("قاطع كهربائي 32 أمبير", 22, 95, "قطعة"),
        ("سلك كهرباء 6 ملم", 120, 280, "لفة"),
        ("لوحة توزيع 12 خط", 160, 480, "قطعة"),
        ("كشاف LED خارجي", 35, 180, "قطعة"),
        ("مقبس جداري مزدوج", 8, 35, "قطعة"),
        ("حساس حركة للإنارة", 45, 160, "قطعة"),
        ("بطارية UPS", 260, 950, "قطعة"),
        ("منظم جهد كهربائي", 190, 820, "قطعة"),
    ],
    "plumbing": [
        ("خلاط مغسلة كروم", 85, 340, "قطعة"),
        ("ماسورة PPR مقاس 25", 9, 32, "متر"),
        ("محبس زاوية", 12, 55, "قطعة"),
        ("سخان مياه كهربائي", 320, 980, "قطعة"),
        ("طقم تمديدات صرف", 45, 180, "طقم"),
        ("مضخة ضغط مياه", 420, 1450, "قطعة"),
        ("فلتر مياه منزلي", 160, 620, "قطعة"),
        ("ليّ مرن ستانلس", 10, 45, "قطعة"),
    ],
    "logistics": [
        ("خدمة نقل داخل المدينة", 120, 420, "رحلة"),
        ("تغليف كراتين للشحن", 4, 18, "كرتون"),
        ("تخزين مستودع مبرد", 180, 780, "يوم"),
        ("تحميل وتنزيل بضائع", 150, 620, "خدمة"),
        ("شحن سريع بين المدن", 35, 210, "طرد"),
        ("تأمين شحنة", 25, 180, "وثيقة"),
        ("خدمة توصيل مستندات", 18, 75, "طلب"),
        ("تجهيز منصة شحن", 220, 900, "خدمة"),
    ],
    "cleaning": [
        ("منظف أرضيات مركز", 28, 95, "عبوة"),
        ("أكياس نفايات كبيرة", 16, 65, "رول"),
        ("مناشف ورقية للمكاتب", 22, 85, "كرتون"),
        ("معقم أسطح", 18, 70, "عبوة"),
        ("خدمة تنظيف واجهات", 450, 1800, "خدمة"),
        ("مكنسة كهربائية صناعية", 380, 1450, "قطعة"),
        ("قفازات تنظيف", 9, 40, "علبة"),
        ("صابون سائل لليدين", 20, 75, "جالون"),
    ],
    "food": [
        ("أرز بسمتي 10 كيلو", 48, 92, "كيس"),
        ("زيت طبخ 1.8 لتر", 14, 32, "عبوة"),
        ("سكر أبيض 5 كيلو", 16, 35, "كيس"),
        ("مياه شرب 330 مل", 8, 22, "كرتون"),
        ("قهوة عربية فاخرة", 45, 180, "كيلو"),
        ("تمر سكري فاخر", 30, 120, "علبة"),
        ("حليب طويل الأجل", 25, 70, "كرتون"),
        ("مواد تموين ضيافة", 65, 240, "حزمة"),
    ],
    "medical": [
        ("كمامات طبية", 12, 55, "علبة"),
        ("قفازات نيتريل", 18, 75, "علبة"),
        ("مطهر طبي 5 لتر", 45, 160, "جالون"),
        ("جهاز قياس حرارة", 35, 220, "قطعة"),
        ("شاش طبي معقم", 9, 38, "عبوة"),
        ("سرير فحص طبي", 950, 3200, "قطعة"),
        ("جهاز قياس ضغط", 85, 390, "قطعة"),
        ("حقيبة إسعافات أولية", 55, 260, "حقيبة"),
    ],
    "safety": [
        ("خوذة سلامة بيضاء", 18, 65, "قطعة"),
        ("سترة عاكسة", 12, 48, "قطعة"),
        ("طفاية حريق بودرة", 95, 320, "قطعة"),
        ("لوحة إرشادية تحذيرية", 25, 110, "لوحة"),
        ("حذاء سلامة", 85, 260, "زوج"),
        ("نظارة حماية", 10, 45, "قطعة"),
        ("حزام أمان للعمل المرتفع", 180, 620, "قطعة"),
        ("فحص أنظمة إنذار الحريق", 350, 1600, "خدمة"),
    ],
    "printing": [
        ("طباعة بروشورات ملونة", 0.8, 3.5, "نسخة"),
        ("تصميم ختم رسمي", 45, 180, "خدمة"),
        ("طباعة رول أب", 120, 360, "قطعة"),
        ("ملصقات باركود", 18, 85, "رول"),
        ("تغليف حراري للبطاقات", 1.5, 7, "بطاقة"),
        ("طباعة كروت عمل", 45, 160, "علبة"),
        ("تصميم هوية بصرية", 650, 3200, "خدمة"),
        ("أكياس ورقية مطبوعة", 2.5, 12, "قطعة"),
    ],
    "furniture": [
        ("مكتب إداري خشبي", 450, 1800, "قطعة"),
        ("كرسي انتظار", 140, 520, "قطعة"),
        ("طاولة اجتماعات", 900, 4200, "قطعة"),
        ("دولاب ملفات", 350, 1250, "قطعة"),
        ("وحدة أدراج متحركة", 180, 680, "قطعة"),
        ("كنبة استقبال", 650, 2800, "قطعة"),
        ("مقسم مكاتب", 220, 950, "قطعة"),
        ("تركيب أثاث مكتبي", 150, 750, "خدمة"),
    ],
    "hvac": [
        ("مكيف سبليت 18 وحدة", 1350, 2800, "جهاز"),
        ("تنظيف فلتر مكيف", 45, 180, "خدمة"),
        ("غاز تبريد R410A", 120, 380, "عبوة"),
        ("صيانة مكيف مركزي", 450, 2400, "خدمة"),
        ("ثرموستات رقمي", 95, 360, "قطعة"),
        ("مروحة شفط", 160, 650, "قطعة"),
        ("عقد صيانة سنوي", 1200, 6500, "عقد"),
        ("تمديد نحاس للمكيف", 45, 160, "متر"),
    ],
    "maintenance": [
        ("خدمة صيانة عامة", 180, 900, "خدمة"),
        ("زيارة فني طوارئ", 120, 480, "زيارة"),
        ("إصلاح باب زجاجي", 220, 980, "خدمة"),
        ("صيانة مصعد شهرية", 600, 2200, "شهر"),
        ("فحص تمديدات كهربائية", 250, 1200, "خدمة"),
        ("دهان مكتب داخلي", 450, 2500, "خدمة"),
        ("إصلاح تسرب مياه", 180, 850, "خدمة"),
        ("توريد قطع صيانة متنوعة", 35, 420, "قطعة"),
    ],
    "telecom": [
        ("جهاز هاتف مكتبي IP", 160, 520, "جهاز"),
        ("سنترال سحابي شهري", 220, 850, "شهر"),
        ("شريحة بيانات أعمال", 45, 180, "شريحة"),
        ("كيابل ألياف بصرية", 35, 190, "متر"),
        ("تركيب راك شبكة", 350, 1600, "خدمة"),
        ("سويتش شبكة 24 منفذ", 480, 1900, "قطعة"),
        ("فحص نقاط اتصال", 20, 90, "نقطة"),
        ("اشتراك إنترنت أعمال", 250, 1200, "شهر"),
    ],
}

AUTO_PART_ITEMS += [
    ("فلتر قير أوتوماتيك", 75, 210, "قطعة"),
    ("وجه رأس مكينة", 95, 380, "قطعة"),
    ("طرمبة ماء", 140, 420, "قطعة"),
    ("فحمات دينمو", 25, 90, "طقم"),
    ("جلدة عامود توازن", 18, 65, "قطعة"),
    ("حساس ABS", 110, 360, "قطعة"),
    ("رديتر مكيف", 240, 720, "قطعة"),
    ("لمبة LED أمامية", 35, 160, "زوج"),
    ("زيت قير CVT", 55, 180, "علبة"),
    ("غطاء رديتر", 18, 55, "قطعة"),
    ("فحمات فرامل سيراميك", 130, 320, "طقم"),
    ("جلدة مكينة", 85, 260, "قطعة"),
    ("كمبيوتر مكينة مستعمل", 650, 2100, "قطعة"),
    ("حساس حرارة ماء", 45, 160, "قطعة"),
    ("مبرد زيت مكينة", 170, 540, "قطعة"),
]

for _items in SECTOR_ITEM_CATALOG.values():
    GENERAL_ITEMS += _items

DESCRIPTION_SUFFIXES = [
    "مطابق للمواصفات", "درجة أولى", "ضمان سنة", "توريد وتركيب",
    "شامل الفحص", "حسب طلب العميل", "موديل 2025", "مقاس كبير",
    "رقم تشغيلي 01", "لون أبيض", "لون أسود", "للاستخدام المكتبي",
]

ARABIC_LABEL_FIXES = {
    "اإلجمالي": "الإجمالي",
    "الفرعي اإلجمالي": "الإجمالي الفرعي",
    "الوحدة سعر": "سعر الوحدة",
    "الشحن بيانات": "بيانات الشحن",
    "البائع اسم": "اسم البائع",
    "الفاتورة رقم": "رقم الفاتورة",
    "الضريبي الرقم": "الرقم الضريبي",
    "ضريبية فاتورة": "فاتورة ضريبية",
}


def _money(value):
    amount = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return amount


def _format_sar(value, symbol=SAR):
    amount = f"{_money(value):,.2f}"
    styles = [
        f"{amount} {symbol}",
        f"{symbol} {amount}",
        f"{amount} ريال",
        f"{amount} ريال سعودي",
        f"{amount} SAR",
    ]
    return random.choice(styles)


def _clean_arabic_text(text):
    text = str(text or "")
    text = text.replace("\u0640", "")
    text = re.sub(r"\s+", " ", text).strip()
    for bad, good in ARABIC_LABEL_FIXES.items():
        text = text.replace(bad, good)
    return text


def _saudi_vat_number(fake):
    """
    Generate a plausible Saudi VAT number.
    Real format: 15 digits, starts with '3', ends with '3'.
    Example: 310122393500003
    """
    middle = fake.numerify(text="#############")   # 13 random digits
    return f"3{middle}3"


def _invoice_ref(fake):
    """Varied invoice-reference styles."""
    year = random.choice([2024, 2025, 2026])
    styles = [
        f"INV-{year}-{fake.numerify('######')}",
        f"SA-{year}-{fake.numerify('#####')}",
        f"{year}/{fake.numerify('######')}",
        f"فاتورة-{year}-{fake.numerify('####')}",
        f"SI-{fake.numerify('########')}",
        f"TAX-{year}{fake.numerify('#####')}",
        f"{fake.numerify('####')}-{year}",
        f"{year}-{fake.numerify('##')}-{fake.numerify('#####')}",
        f"رقم-{fake.numerify('######')}",
    ]
    return random.choice(styles)


def _arabic_date(fake):
    """
    Return a date string in common invoice formats.
    Biased toward recent years for realism.
    """
    start = datetime(2024, 1, 1)
    end = datetime(2026, 5, 31)
    delta = end - start
    random_day = start + timedelta(days=random.randint(0, delta.days))
    formats = ["%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y"]
    return random_day.strftime(random.choice(formats))


def _phone_sa(fake):
    """Saudi-style phone number."""
    formats = [
        f"+966 5{fake.numerify('# ### ####')}",
        f"05{fake.numerify('########')}",
        f"+966-{fake.numerify('##-###-####')}",
    ]
    return random.choice(formats)


def _compact_phone_sa(fake):
    """Saudi mobile number without separators, like POS invoice headers."""
    return f"05{fake.numerify('########')}"


def _fax_sa(fake):
    formats = [
        f"+966 1{fake.numerify('# ### ####')}",
        f"01{fake.numerify('########')}",
    ]
    return random.choice(formats)


def _cr_number(fake):
    return fake.numerify("##########")


def _iban_sa(fake):
    return f"SA{fake.numerify('##')} {fake.numerify('####')} {fake.numerify('####')} {fake.numerify('####')} {fake.numerify('####')} {fake.numerify('####')}"


def _zatca_uuid(fake):
    return (
        f"{fake.hexify('^^^^^^^^')}-"
        f"{fake.hexify('^^^^')}-"
        f"{fake.hexify('^^^^')}-"
        f"{fake.hexify('^^^^')}-"
        f"{fake.hexify('^^^^^^^^^^^^')}"
    )


def _payment_terms():
    return random.choice([
        "الدفع نقدًا عند الاستلام",
        "الدفع خلال 7 أيام من تاريخ الفاتورة",
        "الدفع خلال 15 يومًا من تاريخ الفاتورة",
        "الدفع خلال 30 يومًا من تاريخ الفاتورة",
        "تحويل بنكي على الحساب المعتمد",
        "مدى / بطاقة ائتمانية",
        "سداد عبر نقطة البيع",
    ])


def _item_code(fake):
    """Car-spare-part style item code seen on POS invoices."""
    return f"{fake.numerify('#####')}-{fake.numerify('#####')}-{random.choice(['M', 'A', 'B', 'R'])}"


def _sample_vendor():
    if random.random() < 0.55:
        return _generate_synthetic_vendor()
    vendor = random.choice(SAUDI_VENDOR_PROFILES).copy()
    vendor.setdefault("sector_key", _infer_sector_key(vendor))
    return vendor


def _sample_customer():
    city = random.choice(SAUDI_CITIES)
    district = random.choice(SAUDI_DISTRICTS)
    street = random.choice(SAUDI_STREETS)
    if random.random() < 0.55:
        sector_key, sector_ar, _, _ = random.choice(BUSINESS_SECTORS)
        company = _synthetic_company_name(sector_ar)
    else:
        company = random.choice(SAUDI_CUSTOMER_NAMES)
    return {
        "name": random.choice(SAUDI_PERSON_NAMES),
        "company": company,
        "address": f"{city}، حي {district}، {street}",
        "city_postcode": _city_postcode(city),
    }


def _city_postcode(city):
    matching = [item for item in SAUDI_CITY_POSTCODES if item.startswith(city)]
    if matching:
        return random.choice(matching)
    return f"{city} {random.randint(10000, 89999)}"


def _infer_sector_key(vendor):
    domain = str(vendor.get("domain", ""))
    name = str(vendor.get("ar", ""))
    if "part" in domain or "قطع" in name or "سيارات" in name:
        return "auto_parts"
    if "office" in domain or "مكتبية" in name:
        return "office"
    if "tech" in domain or "تقنية" in name:
        return "it"
    if "logistic" in domain or "نقل" in name:
        return "logistics"
    if "food" in domain or "غذائية" in name:
        return "food"
    if "maintenance" in domain or "صيانة" in name:
        return "maintenance"
    return random.choice([item[0] for item in BUSINESS_SECTORS])


def _synthetic_company_name(sector_ar):
    legal = random.choice(LEGAL_FORMS)
    prefix = random.choice(BUSINESS_PREFIXES)
    noun = random.choice(BUSINESS_NOUNS)
    endings = [
        sector_ar,
        f"{sector_ar} المحدودة",
        f"{sector_ar} والتوريدات",
        f"{sector_ar} والخدمات",
        f"{sector_ar} التجارية",
    ]
    return f"{legal} {prefix} {noun} {random.choice(endings)}"


def _generate_synthetic_vendor():
    sector_key, sector_ar, sector_en, slug = random.choice(BUSINESS_SECTORS)
    city = random.choice(SAUDI_CITIES)
    district = random.choice(SAUDI_DISTRICTS)
    street = random.choice(SAUDI_STREETS)
    prefix_ar = random.choice(BUSINESS_PREFIXES)
    noun_ar = random.choice(BUSINESS_NOUNS)
    prefix_en = random.choice(EN_PREFIXES)
    suffix = random.choice(["Co.", "Est.", "LLC", "Trading", "Services"])
    domain_name = f"{slug}-{random.randint(10, 999)}{random.choice(['sa', 'ksa', 'hub'])}"
    tld = random.choice([".sa", ".com", ".com.sa"])
    return {
        "ar": _synthetic_company_name(sector_ar),
        "en": f"{prefix_en} {sector_en} {suffix}",
        "address": f"{city}، حي {district}، {street}",
        "branch": f"{city} - {district}",
        "domain": f"{domain_name}{tld}",
        "sector_key": sector_key,
    }


def _vary_description(description, unit):
    parts = [_clean_arabic_text(description)]
    if random.random() < 0.35:
        parts.append(random.choice(DESCRIPTION_SUFFIXES))
    if random.random() < 0.18:
        parts.append(random.choice([
            f"كود {random.randint(1000, 9999)}",
            f"REF-{random.randint(10000, 99999)}",
            f"دفعة {random.randint(1, 24):02d}",
            f"{unit} معتمد",
        ]))
    return " - ".join(parts)


def _build_product(description, price_low, price_high, unit, row_index):
    unit_price = _money(random.uniform(price_low, price_high))
    quantity = random.randint(1, 4 if unit in {"خدمة", "شهر", "رحلة"} else 8)
    total = unit_price * quantity
    return {
        "description": _vary_description(description, unit),
        "quantity": quantity,
        "unit_price": _format_sar(unit_price),
        "unit_price_numeric": float(unit_price),
        "total": _format_sar(total),
        "total_numeric": float(total),
        "row_index": row_index,
        "pack": unit,
        "unit": unit,
    }


def _generate_realistic_products(product_generator, num_products, template_style, vendor=None):
    """Use curated Saudi invoice items first; keep CSV generator as fallback."""
    sector_key = (vendor or {}).get("sector_key")
    if template_style in {"auto_parts", "al_murabaha"} or sector_key == "auto_parts":
        source_items = AUTO_PART_ITEMS
    elif sector_key in SECTOR_ITEM_CATALOG:
        source_items = SECTOR_ITEM_CATALOG[sector_key] + random.sample(GENERAL_ITEMS, k=min(12, len(GENERAL_ITEMS)))
    else:
        source_items = GENERAL_ITEMS

    if num_products <= len(source_items):
        selected_items = random.sample(source_items, k=num_products)
    else:
        selected_items = random.choices(source_items, k=num_products)

    products = [
        _build_product(description, low, high, unit, row_index=i)
        for i, (description, low, high, unit) in enumerate(selected_items)
    ]

    if products:
        return products

    return product_generator.generate_products_from_csv(num_products)


def _add_pos_product_fields(products, discount_rate):
    """Add tokens needed by the competitive-price POS template."""
    code_fake = Faker("en_US")
    colors = ["أبيض", "أسود", "فضي", "رمادي", "أزرق", "أحمر"]
    for product in products:
        unit_price = float(product["unit_price_numeric"])
        quantity = int(product["quantity"])
        gross = round(unit_price * quantity, 2)
        net = round(gross * (1 - discount_rate / 100), 2)
        product["item_code"] = _item_code(code_fake)
        product["location"] = random.choice(["A1", "A2", "B1", "B3", "C2", "المخزن"])
        product["discount_percent"] = f"{discount_rate:.2f}"
        product["line_subtotal"] = f"{gross:.2f}"
        product["total"] = _format_sar(net)
        product["total_numeric"] = net
        product["model"] = str(random.randint(2018, 2026))
        product["color"] = random.choice(colors)
        product["chassis_no"] = f"{random.choice(['JT', 'KM', 'MA'])}{code_fake.numerify('#############')}"
    return products


def _template_style(template_name):
    if template_name in {"generated_templates_1.docx", "template_al_murabaha.docx", "template_medra_cars.docx"}:
        return "al_murabaha"
    auto_parts_templates = {
        "auto_parts_tax_invoice_template.docx",
        "competitive_price_establishment_template.docx",
    }
    if template_name in auto_parts_templates:
        return "auto_parts"
    return "standard"


ARABIC_SPECIAL_INSTRUCTIONS = [
    "يرجى تسليم الفاتورة مع الشحنة.",
    "الدفع خلال 30 يومًا من تاريخ الإصدار.",
    "يرجى التأكد من سلامة المنتجات قبل الاستلام.",
    "لا تقبل المرتجعات بعد فتح العبوة.",
    "يُرجى التواصل مع خدمة العملاء عند وجود أي مشكلة.",
    "يجب ختم الفاتورة عند الاستلام.",
    "الشحن غير شامل رسوم التفريغ.",
    "يرجى مطابقة الكميات مع أمر الشراء.",
    "يتم التسليم خلال أيام العمل الرسمية فقط.",
    "جميع الأسعار تشمل ضريبة القيمة المضافة.",
    "يُمنع الإرجاع أو الاستبدال بعد مرور 7 أيام من الاستلام.",
    "يرجى الاحتفاظ بهذه الفاتورة للرجوع إليها عند الحاجة.",
    "يشترط تقديم الفاتورة الأصلية عند طلب الضمان.",
    "الأسعار المدرجة بالريال السعودي وقابلة للتغيير دون إشعار مسبق.",
    "جميع المنتجات مغلفة ومختومة، لا نتحمل مسؤولية فك التغليف.",
    "يتم احتساب الضريبة وفقًا للوائح هيئة الزكاة والضريبة والجمارك.",
    "للاستفسار عن الفاتورة يرجى الإشارة إلى رقم الفاتورة.",
    "التأخر في السداد يستوجب رسوم إضافية بنسبة 2% شهريًا.",
    "يرجى الدفع عبر التحويل البنكي على الحساب المعتمد.",
    "هذه الفاتورة صادرة إلكترونيًا وسارية المفعول دون توقيع.",
]


def _special_instructions():
    """Pick 1–3 instructions and join them."""
    k = random.randint(1, 3)
    return " ".join(random.sample(ARABIC_SPECIAL_INSTRUCTIONS, k=k))


def _validate_invoice_data(invoice_data):
    warnings = []
    required = [
        "invoice_ref", "seller_name", "seller_vat_number", "issue_datetime",
        "customer_name", "subtotal", "tax", "total", "products"
    ]
    for key in required:
        if not invoice_data.get(key):
            warnings.append(f"missing:{key}")

    date_value = str(invoice_data.get("issue_datetime"))
    if not any(_parse_ok(date_value, fmt) for fmt in ("%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y")):
        warnings.append("invalid_date")

    products = invoice_data.get("products", [])
    subtotal = _money(sum(Decimal(str(p.get("unit_price_numeric", 0))) * int(p.get("quantity", 0)) for p in products))
    discount_amount = _money(str(invoice_data.get("_discount_amount_numeric", 0)))
    net_amount = subtotal - discount_amount
    tax = _money(net_amount * Decimal(str(VAT_RATE)))
    total = net_amount + tax
    if abs(float(subtotal) - float(invoice_data.get("_subtotal_numeric", 0))) > 0.01:
        warnings.append("subtotal_mismatch")
    if abs(float(tax) - float(invoice_data.get("_tax_numeric", 0))) > 0.01:
        warnings.append("tax_mismatch")
    if abs(float(total) - float(invoice_data.get("_total_numeric", 0))) > 0.01:
        warnings.append("total_mismatch")
    return warnings


def _parse_ok(value, fmt):
    try:
        datetime.strptime(value, fmt)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_dataset():
    seed = os.environ.get("INVOICE_GENERATOR_SEED")
    if seed is not None:
        seed_int = int(seed)
        random.seed(seed_int)
        Faker.seed(seed_int)
        print(f"Using seed: {seed_int}")

    fake = Faker("ar_SA")
    product_generator = ProductGenerator()
    invoice_count = int(
        os.environ.get(
            "INVOICE_GENERATOR_COUNT",
            NUM_INVOICES_TO_GENERATE,
        )
    )
    stats = {
        "attempted": 0,
        "generated": 0,
        "templates": Counter(),
        "validation_warnings": Counter(),
        "field_coverage": Counter(),
    }

    template_files = []
    skipped_templates = []
    for f in sorted(os.listdir(TEMPLATE_DIR)):
        if not f.endswith(".docx") or f.startswith(".~lock"):
            continue
        template_path = os.path.join(TEMPLATE_DIR, f)
        if validate_template(template_path):
            template_files.append(f)
        else:
            skipped_templates.append(f)

    if skipped_templates:
        print("Skipping invalid/incomplete templates:", ", ".join(skipped_templates))
    if not template_files:
        raise RuntimeError("No valid DOCX templates found for dataset generation.")
    print("Using valid templates:", ", ".join(template_files))

    padding_digits = len(str(invoice_count))

    for i in range(invoice_count):
        invoice_num = str(i).zfill(padding_digits)
        try:
            stats["attempted"] += 1
            print(f"\nProcessing invoice {invoice_num}...")
            docx_path = os.path.join(DOCX_DIR, f"invoice_{invoice_num}.docx")
            selected_template = random.choice(template_files)
            template_style = _template_style(selected_template)
            stats["templates"][selected_template] += 1

            # ±40 % variation in product count, rounded (not truncated)
            num_products = max(1, round(NUM_PRODUCTS_PER_INVOICE * (1 + random.uniform(-0.4, 0.4))))

            # Ensure seller and company are distinct
            vendor = _sample_vendor()
            company = vendor["ar"]
            seller = vendor["ar"]

            products = _generate_realistic_products(product_generator, num_products, template_style, vendor)

            subtotal = _money(sum(Decimal(str(p["unit_price_numeric"])) * int(p["quantity"]) for p in products))
            discount_rate = random.choice([0, 0.5, 1, 2, 5]) if template_style in {"auto_parts", "al_murabaha"} else 0
            if template_style in {"auto_parts", "al_murabaha"}:
                products = _add_pos_product_fields(products, discount_rate)
            discount_amount = _money(subtotal * Decimal(str(discount_rate)) / Decimal("100"))
            net_amount = subtotal - discount_amount
            tax = _money(net_amount * Decimal(str(VAT_RATE)))
            total = net_amount + tax

            # Ensure shipping and billing addresses differ
            customer = _sample_customer()
            recipient_name = customer["name"]
            recipient_company = customer["company"]
            street_address = customer["address"]
            city_postcode = customer["city_postcode"]
            recipient_phone   = _phone_sa(fake)

            shipping = _sample_customer()
            shipping_name = shipping["name"]
            shipping_company = shipping["company"]
            # Keep trying until shipping address differs from billing
            shipping_street = shipping["address"]
            while shipping_street == street_address:
                shipping_street = random.choice(SAUDI_ADDRESSES)
            shipping_city = shipping["city_postcode"]
            shipping_phone   = _phone_sa(fake)

            invoice_data = {
                # Header / seller
                "invoice_ref":        _invoice_ref(fake),
                "company_name":       company,
                "seller_name":        seller,
                "seller_address":     vendor["address"],
                "seller_vat_number":  _saudi_vat_number(fake),
                "seller_cr_number":   _cr_number(fake),
                "seller_iban":        _iban_sa(fake),
                "issue_datetime":     _arabic_date(fake),
                "issue_time":         fake.time(pattern="%H:%M:%S"),
                "supply_date":        _arabic_date(fake),
                "due_date":           _arabic_date(fake),
                "email":              f"sales@{vendor['domain']}",
                "website":            vendor["domain"],
                "phone_number":       _phone_sa(fake),
                "fax_number":         _fax_sa(fake),
                "branch_name":        vendor["branch"],
                "invoice_type":       random.choice(["نقدي", "آجل", "شبكة", "فاتورة ضريبية", "فاتورة ضريبية مبسطة"]),
                "invoice_number":     fake.numerify("#####"),
                "zatca_uuid":         _zatca_uuid(fake),
                "qr_placeholder":     random.choice(["[QR CODE]", "[رمز QR]", "رمز الاستجابة السريعة"]),
                "payment_terms":      _payment_terms(),
                "payment_method":     random.choice(["نقدًا", "شبكة", "تحويل بنكي", "مدى", "فيزا", "آجل"]),
                "po_number":          random.choice([fake.numerify("PO-######"), fake.numerify("طلب-#####"), ""]),
                "delivery_note":      random.choice([fake.numerify("DN-######"), fake.numerify("إذن-#####"), ""]),
                "vat_category":       random.choice(["S - خاضع للضريبة 15%", "Z - ضريبة صفرية", "E - معفى"]),
                "page_number":        "1 / 1",
                "cashier_id":         fake.numerify("00#"),
                "customer_number":    fake.numerify("#####"),
                "customer_vat_number": _saudi_vat_number(fake),
                "customer_cr_number": _cr_number(fake),
                "customer_name":      recipient_company,
                "customer_address":   street_address,
                "pos_phone_number":   _compact_phone_sa(fake),
                "pos_phone_number_2": _compact_phone_sa(fake),
                "pos_phone_number_3": _compact_phone_sa(fake),
                "side_serial_left":   fake.numerify("###############"),
                "side_serial_right":  fake.numerify("SA35 #### #### #### #### ####"),
                "seller_trade_name_ar": vendor["ar"],
                "seller_trade_name_en": vendor["en"],
                "seller_location_line": vendor["address"],

                # Totals
                "subtotal": _format_sar(subtotal),
                "discount": _format_sar(discount_amount),
                "net_amount": _format_sar(net_amount),
                "tax": _format_sar(tax),
                "total": _format_sar(total),
                "_subtotal_numeric": float(subtotal),
                "_discount_amount_numeric": float(discount_amount),
                "_tax_numeric": float(tax),
                "_total_numeric": float(total),

                # Products (used by DocumentProcessor internally)
                "products": products,

                # Billing address
                "recipient_name":    recipient_name,
                "recipient_company": recipient_company,
                "street_address":    street_address,
                "city_postcode":     city_postcode,
                "recipient_phone":   recipient_phone,

                # Shipping address
                "shipping_recipient_name":    shipping_name,
                "shipping_recipient_company": shipping_company,
                "shipping_street_address":    shipping_street,
                "shipping_city_postcode":     shipping_city,
                "shipping_recipient_phone":   shipping_phone,

                # Misc
                "special_instructions": _special_instructions(),
                "template_name": selected_template,
                "template_style": template_style,
            }

            validation_warnings = _validate_invoice_data(invoice_data)
            for warning in validation_warnings:
                stats["validation_warnings"][warning] += 1
            for key, value in invoice_data.items():
                if value not in (None, "", []):
                    stats["field_coverage"][key] += 1

            # Merge uppercase variants so templates using {{COMPANY_NAME}} also work
            invoice_data.update({k.upper(): v for k, v in invoice_data.items()})

            print("Generated invoice data")

            print(f"Using template: {selected_template}  →  {docx_path}")

            DocumentProcessor.create_invoice(
                os.path.join(TEMPLATE_DIR, selected_template),
                invoice_data,
                docx_path,
            )
            print("Created DOCX successfully")

            print("Converting to PDF...")
            pdf_path = DocumentProcessor.convert_to_pdf(docx_path, PDF_DIR)

            if pdf_path and os.path.exists(pdf_path):
                print(f"Created PDF at: {pdf_path}")

                print("Converting to images...")
                image_paths = ImageProcessor.pdf_to_images(
                    pdf_path, IMAGES_DIR, format="jpg", quality=75
                )

                processor = AnnotationProcessor()
                lilt_processor = LiLTAnnotationProcessor()
                print("Generating annotations...")
                for j, image_path in enumerate(image_paths):
                    # Generate standard annotations
                    annotations = processor.process_pdf(
                        pdf_path=pdf_path,
                        image_path=image_path,
                        invoice_data=invoice_data,
                        page_index=j,
                    )
                    annotation_name = (
                        f"invoice_{invoice_num}.json"
                        if len(image_paths) == 1
                        else f"invoice_{invoice_num}_page_{j + 1:02d}.json"
                    )
                    annotation_path = os.path.join(
                        ANNOTATIONS_DIR, annotation_name
                    )
                    processor.save_annotations(annotation_path)
                    
                    # Generate LiLT-compatible annotations
                    try:
                        lilt_annotation = lilt_processor.process_pdf_for_lilt(
                            pdf_path=pdf_path,
                            image_path=image_path,
                            invoice_data=invoice_data,
                            page_index=j,
                        )
                        if lilt_annotation:
                            lilt_annotation_name = (
                                f"invoice_{invoice_num}.json"
                                if len(image_paths) == 1
                                else f"invoice_{invoice_num}_page_{j + 1:02d}.json"
                            )
                            lilt_annotation_path = os.path.join(
                                LILT_ANNOTATIONS_DIR, lilt_annotation_name
                            )
                            lilt_processor.save_lilt_annotation(
                                lilt_annotation, lilt_annotation_path
                            )
                            print(f"LiLT annotation saved at: {lilt_annotation_path}")
                    except Exception as e:
                        print(f"Warning: Failed to generate LiLT annotation: {e}")

                print(f"Annotations saved at: {ANNOTATIONS_DIR}")
                print(f"LiLT annotations saved at: {LILT_ANNOTATIONS_DIR}")
                stats["generated"] += 1
            else:
                print(f"Failed to generate PDF for invoice {invoice_num}")

            os.remove(docx_path)
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)

        except Exception as e:
            print(f"Error processing invoice {invoice_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nGeneration summary")
    print(f"Samples attempted : {stats['attempted']}")
    print(f"Samples generated : {stats['generated']}")
    print("Templates used    :", dict(stats["templates"]))
    print("Validation warnings:", dict(stats["validation_warnings"]))
    required_fields = [
        "invoice_ref", "seller_name", "seller_vat_number", "issue_datetime",
        "customer_name", "subtotal", "tax", "total", "products"
    ]
    print("Required field coverage:", {key: stats["field_coverage"].get(key, 0) for key in required_fields})


if __name__ == "__main__":
    generate_dataset()
