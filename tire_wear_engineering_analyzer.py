"""
اپلیکیشن مهندسی تحلیل فرسایش تایر و تخمین عمر مفید باقی‌مانده (RUL)
نسخه بهینه‌سازی شده برای موبایل و دسکتاپ (Mobile-First Responsive Edition)
بازبینی و ارتقا یافته بر اساس استانداردهای Agency-Agents (Code Reviewer & Frontend Developer)

پیش‌نیازها برای اجرا در ترمینال:
pip install streamlit opencv-python-headless numpy pillow matplotlib

نحوه اجرا:
streamlit run tire_wear_analyzer.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# =====================================================================
# تنظیمات اولیه صفحه (سازگار با نمایشگر موبایل و دسکتاپ)
# =====================================================================
st.set_page_config(
    page_title="سامانه مهندسی تحلیل فرسایش تایر",
    page_icon="🛞",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================================================================
# استایل واکنش‌گرا (Mobile-First Responsive CSS & RTL Fix)
# =====================================================================
st.markdown("""
<style>
    /* تنظیمات پایه راست‌چین بدون تخریب کامپوننت‌های داخلی Streamlit */
    .block-container {
        direction: rtl;
        text-align: right;
        font-family: 'Tahoma', 'Vazirmatn', sans-serif;
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 960px;
    }

    h1, h2, h3, h4, p, label, .stMarkdown {
        direction: rtl;
        text-align: right;
    }

    /* هدر اپلیکیشن موبایل */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: #ffffff;
        padding: 18px 20px;
        border-radius: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    .app-header h1 {
        font-size: 1.35rem !important;
        margin: 0 0 6px 0 !important;
        color: #ffffff !important;
        line-height: 1.4;
    }
    .app-header p {
        font-size: 0.85rem !important;
        color: #cbd5e1 !important;
        margin: 0 !important;
    }

    /* بهینه‌سازی منوی تب‌ها برای لمس راحت روی موبایل (Touch Targets) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: rgba(148, 163, 184, 0.12);
        padding: 6px;
        border-radius: 14px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1 1 auto;
        min-height: 44px;
        border-radius: 10px !important;
        padding: 8px 12px !important;
        font-size: 0.9rem !important;
        font-weight: bold !important;
        justify-content: center;
        text-align: center;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35);
    }

    /* شبکه کارت‌های متریک واکنش‌گرا (۴ ستون در دسکتاپ، ۲ ستون در موبایل) */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 16px 0;
        direction: rtl;
    }
    .metric-card {
        background: rgba(37, 99, 235, 0.07);
        border: 1px solid rgba(37, 99, 235, 0.25);
        border-radius: 14px;
        padding: 14px 10px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-title {
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 6px;
        font-weight: 600;
        text-align: center !important;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #1e40af;
        direction: ltr;
        text-align: center !important;
    }
    .metric-sub {
        font-size: 0.72rem;
        color: #475569;
        margin-top: 4px;
        text-align: center !important;
    }

    /* بنر وضعیت ایمنی تایر */
    .status-banner {
        color: white;
        padding: 14px 16px;
        border-radius: 14px;
        font-size: 1rem;
        font-weight: bold;
        text-align: center !important;
        margin: 12px 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.12);
    }

    /* بهینه‌سازی اختصاصی برای صفحه‌نمایش موبایل */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
            padding-top: 0.6rem !important;
        }
        .app-header {
            padding: 14px 16px;
            border-radius: 14px;
        }
        .app-header h1 {
            font-size: 1.12rem !important;
        }
        .app-header p {
            font-size: 0.78rem !important;
        }
        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .metric-value {
            font-size: 1.2rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem !important;
            padding: 6px 8px !important;
            min-height: 42px;
        }
        /* بزرگ‌تر کردن دکمه‌ها و کنترل‌های رادیویی برای لمس موبایل */
        .stRadio > div {
            gap: 8px;
        }
        button[kind="primary"], button[kind="secondary"] {
            min-height: 48px !important;
            width: 100% !important;
            border-radius: 12px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# کلاس‌ها و ساختارهای داده مهندسی خودرو
# =====================================================================
@dataclass
class TireSpecs:
    """مشخصات استاندارد مهندسی تایر"""
    new_tread_depth_mm: float = 8.0      # عمق آج تایر نو (استاندارد سواری: 8 میلی‌متر)
    legal_min_depth_mm: float = 1.6      # حد مجاز قانونی فرسایش (شاخص TWI: 1.6 میلی‌متر)
    nominal_lifespan_km: int = 50000     # عمر اسمی تایر برحسب کیلومتر
    driving_aggressiveness: float = 1.0  # ضریب شرایط رانندگی و جاده (0.7 تا 1.5)


@dataclass
class AnalysisResult:
    """خروجی تحلیل پردازش تصویر و محاسبات مهندسی"""
    estimated_depth_mm: float
    wear_percentage: float
    remaining_life_percentage: float
    remaining_km: int
    edge_density: float
    groove_ratio: float
    surface_anomaly_score: float
    brightness_level: float
    status_label: str
    status_color: str
    processed_images: Dict[str, np.ndarray]


# =====================================================================
# موتور پردازش تصویر و بینایی ماشین (Computer Vision Engine)
# =====================================================================
class TireVisionAnalyzer:
    """
    موتور تحلیل تصویر آج تایر با اصلاح نسبت تصویر (Square Center Crop)
    و مقاومت در برابر تغییرات نوری محیط عکسبرداری موبایل.
    """
    # ثوابت مرجع مهندسی (رفع مشکل Magic Numbers در بازبینی کد)
    REF_GROOVE_RATIO: float = 0.28
    REF_EDGE_DENSITY: float = 0.12
    ABSOLUTE_MIN_DEPTH_MM: float = 0.5
    STANDARD_SIZE: int = 500

    def __init__(self, specs: TireSpecs):
        self.specs = specs

    @staticmethod
    def validate_and_crop_square(image_rgb: np.ndarray) -> np.ndarray:
        """
        اعتبارسنجی ابعاد تصویر و برش مربعی از مرکز (Center Square Crop)
        برای جلوگیری از اعوجاج هندسی عکس‌های مستطیلی دوربین موبایل
        """
        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("تصویر ورودی خالی یا نامعتبر است.")

        h, w = image_rgb.shape[:2]
        if h < 120 or w < 120:
            raise ValueError("رزولوشن تصویر بسیار پایین است (حداقل ۱۲۰×۱۲۰ پیکسل نیاز است).")

        min_dim = min(h, w)
        start_y = (h - min_dim) // 2
        start_x = (w - min_dim) // 2
        square_crop = image_rgb[start_y:start_y + min_dim, start_x:start_x + min_dim]
        return square_crop

    def preprocess_image(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """برش مربعی، تغییر سایز استاندارد، انتخاب ROI و نرمال‌سازی نوری"""
        square_img = self.validate_and_crop_square(image_rgb)
        resized = cv2.resize(square_img, (self.STANDARD_SIZE, self.STANDARD_SIZE), interpolation=cv2.INTER_AREA)

        # انتخاب ناحیه مرکزی (ROI) برای حذف حاشیه و پس‌زمینه تصویر
        h, w, _ = resized.shape
        margin_h, margin_w = int(h * 0.15), int(w * 0.15)
        roi_rgb = resized[margin_h:h - margin_h, margin_w:w - margin_w]

        # تبدیل به سطح خاکستری و محاسبه روشنایی میانگین
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        brightness = float(np.mean(gray))

        # بهبود کنتراست محلی (CLAHE) برای حذف سایه‌های محیطی
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # فیلتر دوطرفه (Bilateral) برای حذف نویز سطحی با حفظ لبه‌های آج
        filtered = cv2.bilateralFilter(enhanced_gray, d=9, sigmaColor=75, sigmaSpace=75)

        return roi_rgb, filtered, brightness

    def extract_features(self, filtered_gray: np.ndarray) -> Tuple[float, float, float, Dict[str, np.ndarray]]:
        """استخراج ویژگی‌های هندسی و بافتی از سطح تایر"""
        # ۱. استخراج شیارهای عمیق (Groove Segmentation)
        groove_mask = cv2.adaptiveThreshold(
            filtered_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31, C=8
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        groove_mask_clean = cv2.morphologyEx(groove_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        groove_ratio = float(np.sum(groove_mask_clean > 0)) / float(groove_mask_clean.size)

        # ۲. تحلیل تراکم لبه‌های آج (Edge Density)
        edges = cv2.Canny(filtered_gray, threshold1=50, threshold2=150)
        edge_density = float(np.sum(edges > 0)) / float(edges.size)

        # ۳. تحلیل گرادیان عمق (Sobel Depth Proxy)
        sobelx = cv2.Sobel(filtered_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(filtered_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_norm = np.uint8(255 * gradient_magnitude / (np.max(gradient_magnitude) + 1e-5))

        # ۴. تشخیص ترک‌خوردگی‌های ریز و پوسیدگی سطحی
        laplacian = np.abs(cv2.Laplacian(filtered_gray, cv2.CV_64F))
        anomaly_score = float(np.mean(laplacian))

        processed_visuals = {
            "enhanced": filtered_gray,
            "groove_mask": groove_mask_clean,
            "edges": edges,
            "gradient": gradient_norm
        }

        return groove_ratio, edge_density, anomaly_score, processed_visuals

    def analyze(self, image_rgb: np.ndarray, calibration_factor: float = 1.0) -> AnalysisResult:
        """محاسبه نهایی عمق آج، درصد فرسایش و عمر باقی‌مانده"""
        roi_rgb, filtered_gray, brightness = self.preprocess_image(image_rgb)
        groove_ratio, edge_density, anomaly_score, visuals = self.extract_features(filtered_gray)
        visuals["roi_rgb"] = roi_rgb

        # نرمال‌سازی شاخص‌های نوری نسبت به مقادیر مرجع استاندارد
        normalized_groove = np.clip(groove_ratio / self.REF_GROOVE_RATIO, 0.0, 1.0)
        normalized_edge = np.clip(edge_density / self.REF_EDGE_DENSITY, 0.0, 1.0)

        health_index = (0.65 * normalized_groove + 0.35 * normalized_edge) * calibration_factor
        health_index = float(np.clip(health_index, 0.0, 1.0))

        # نگاشت شاخص سلامت به عمق فیزیکی آج (بر حسب میلی‌متر)
        min_d = self.specs.legal_min_depth_mm
        max_d = max(self.specs.new_tread_depth_mm, min_d + 0.5)

        estimated_depth_mm = self.ABSOLUTE_MIN_DEPTH_MM + health_index * (max_d - self.ABSOLUTE_MIN_DEPTH_MM)

        # محاسبه درصد عمر مفید باقی‌مانده (بر اساس حد مجاز قانونی TWI)
        usable_range = max(max_d - min_d, 0.1)
        usable_remaining = max(0.0, estimated_depth_mm - min_d)
        remaining_life_pct = float(np.clip((usable_remaining / usable_range) * 100.0, 0.0, 100.0))
        wear_pct = 100.0 - remaining_life_pct

        # محاسبه کیلومتر باقی‌مانده
        remaining_km = int(
            (remaining_life_pct / 100.0) *
            (self.specs.nominal_lifespan_km / self.specs.driving_aggressiveness)
        )

        # تعیین وضعیت ایمنی مهندسی
        if remaining_life_pct >= 65:
            status_label = "✅ عالی (استاندارد و ایمن)"
            status_color = "#16a34a"
        elif remaining_life_pct >= 30:
            status_label = "⚠️ متوسط (نیازمند بررسی دوره‌ای)"
            status_color = "#d97706"
        elif remaining_life_pct > 5:
            status_label = "🟠 فرسوده (نزدیک به مرز قانونی TWI - تعویض به‌زودی)"
            status_color = "#ea580c"
        else:
            status_label = "🚨 بحرانی و خطرناک (غیرمجاز - نیازمند تعویض فوری!)"
            status_color = "#dc2626"

        return AnalysisResult(
            estimated_depth_mm=round(estimated_depth_mm, 2),
            wear_percentage=round(wear_pct, 1),
            remaining_life_percentage=round(remaining_life_pct, 1),
            remaining_km=remaining_km,
            edge_density=round(edge_density * 100, 2),
            groove_ratio=round(groove_ratio * 100, 2),
            surface_anomaly_score=round(anomaly_score, 2),
            brightness_level=round(brightness, 1),
            status_label=status_label,
            status_color=status_color,
            processed_images=visuals
        )


# =====================================================================
# شبیه‌ساز تصویر تایر (برای تست فوری روی موبایل و دسکتاپ)
# =====================================================================
def generate_synthetic_tire(wear_level: str) -> np.ndarray:
    """تولید تصویر مصنوعی از بافت و شیارهای تایر برای تست الگوریتم"""
    img = np.ones((500, 500, 3), dtype=np.uint8) * 110

    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 12, (500, 500)).astype(np.int16)
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c].astype(np.int16) + noise, 40, 180).astype(np.uint8)

    if wear_level == "نو (سالم)":
        groove_intensity = 25
        groove_width = 16
        lateral_step = 45
    elif wear_level == "کارکرده (متوسط)":
        groove_intensity = 60
        groove_width = 10
        lateral_step = 65
    else:
        groove_intensity = 95
        groove_width = 4
        lateral_step = 110

    for x in [140, 220, 280, 360]:
        cv2.rectangle(
            img,
            (x - groove_width, 0),
            (x + groove_width, 500),
            (groove_intensity, groove_intensity, groove_intensity),
            -1
        )

    for y in range(20, 480, lateral_step):
        thickness = max(2, groove_width // 2)
        cv2.line(img, (60, y), (140, y + 25), (groove_intensity, groove_intensity, groove_intensity), thickness)
        cv2.line(img, (140, y + 10), (220, y - 10), (groove_intensity, groove_intensity, groove_intensity), thickness)
        cv2.line(img, (280, y - 10), (360, y + 10), (groove_intensity, groove_intensity, groove_intensity), thickness)
        cv2.line(img, (360, y + 25), (440, y), (groove_intensity, groove_intensity, groove_intensity), thickness)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


# =====================================================================
# کامپوننت منوی تنظیمات موبایل‌فرست (جایگزین سایدبار مخفی در موبایل)
# =====================================================================
def render_mobile_friendly_menu() -> Tuple[TireSpecs, float]:
    """
    ایجاد منوی تنظیمات کالیبراسیون در داخل بدنه اصلی صفحه (Expander)
    تا کاربران موبایل بدون نیاز به جستجوی دکمه مخفی سایدبار به تمام پارامترها دسترسی داشته باشند.
    """
    with st.expander("⚙️ منوی تنظیمات مهندسی و نوع تایر (برای تغییر ضربه بزنید)", expanded=False):
        st.caption("پارامترهای استاندارد خودرو و شرایط رانندگی خود را در این منو تنظیم کنید:")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            tire_category = st.selectbox(
                "نوع تایر خودرو:",
                ["تایر سواری تابستانی / چهارفصل", "تایر سواری زمستانی (برفی)", "تایر شاسی‌بلند (SUV / آفرود)"]
            )
            default_depth = 8.0
            if "زمستانی" in tire_category:
                default_depth = 9.5
            elif "شاسی‌بلند" in tire_category:
                default_depth = 10.5

            new_depth = st.slider("عمق آج تایر نو (mm):", 6.0, 14.0, default_depth, 0.5)
            legal_min = st.slider("حداقل مجاز قانونی TWI (mm):", 1.0, 3.0, 1.6, 0.1)

        with m_col2:
            nominal_km = st.number_input(
                "عمر اسمی تایر نو (کیلومتر):",
                min_value=20000, max_value=100000, value=50000, step=5000
            )
            driving_style = st.select_slider(
                "شرایط جاده و سبک رانندگی:",
                options=["آرام (بزرگراهی)", "معمولی (شهری)", "پرفشار / ناهموار"],
                value="معمولی (شهری)"
            )
            calibration = st.slider(
                "ضریب کالیبراسیون نوری دوربین:",
                0.5, 1.5, 1.0, 0.05,
                help="اگر محیط عکسبرداری خیلی تاریک یا پرنور است، حساسیت تشخیص شیار را تنظیم کنید."
            )

    aggressiveness_map = {
        "آرام (بزرگراهی)": 0.85,
        "معمولی (شهری)": 1.0,
        "پرفشار / ناهموار": 1.35
    }

    specs = TireSpecs(
        new_tread_depth_mm=new_depth,
        legal_min_depth_mm=legal_min,
        nominal_lifespan_km=nominal_km,
        driving_aggressiveness=aggressiveness_map[driving_style]
    )
    return specs, calibration


# =====================================================================
# رابط کاربری اصلی (Mobile-First UI)
# =====================================================================
def main():
    # ۱. هدر واکنش‌گرا
    st.markdown("""
    <div class="app-header">
        <h1>🛞 سامانه مهندسی تحلیل فرسایش تایر</h1>
        <p>تشخیص هوشمند عمق آج، درصد سایش و تخمین کیلومتر باقی‌مانده با پردازش تصویر</p>
    </div>
    """, unsafe_allow_html=True)

    # ۲. منوی تنظیمات بهینه شده برای موبایل (در بالای صفحه)
    specs, calibration = render_mobile_friendly_menu()

    # ۳. منوی دریافت تصویر با انتخابگر لمسی
    st.markdown("##### 📸 ۱. روش دریافت تصویر تایر:")
    input_method = st.radio(
        "منبع تصویر:",
        ["🧪 نمونه آزمایشی", "📷 دوربین موبایل", "🖼️ آپلود از گالری"],
        horizontal=True,
        label_visibility="collapsed"
    )

    input_image_rgb: Optional[np.ndarray] = None

    if input_method == "🖼️ آپلود از گالری":
        uploaded_file = st.file_uploader(
            "یک تصویر واضح از نمای روبروی آج تایر انتخاب کنید:",
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            try:
                pil_img = Image.open(uploaded_file).convert("RGB")
                input_image_rgb = np.array(pil_img)
            except Exception as err:
                st.error(f"خطا در خواندن فایل تصویر: {err}")

    elif input_method == "📷 دوربین موبایل":
        camera_photo = st.camera_input("دوربین گوشی را روبروی آج تایر نگه دارید:")
        if camera_photo is not None:
            try:
                pil_img = Image.open(camera_photo).convert("RGB")
                input_image_rgb = np.array(pil_img)
            except Exception as err:
                st.error(f"خطا در دریافت تصویر دوربین: {err}")

    else:
        sample_type = st.selectbox(
            "وضعیت تایر آزمایشی را انتخاب کنید:",
            ["نو (سالم)", "کارکرده (متوسط)", "فرسوده (صاف)"]
        )
        input_image_rgb = generate_synthetic_tire(sample_type)

    # ۴. اجرای تحلیل و نمایش نتایج در منوی تب‌بندی شده موبایل
    if input_image_rgb is not None:
        try:
            analyzer = TireVisionAnalyzer(specs)
            result = analyzer.analyze(input_image_rgb, calibration_factor=calibration)
        except ValueError as val_err:
            st.warning(f"⚠️ {val_err}")
            return
        except Exception as ex:
            st.error(f"خطا در پردازش تصویر: {ex}")
            return

        # هشدار هوشمند روشنایی محیط برای عکسبرداری موبایل
        if result.brightness_level < 45:
            st.warning("💡 نور تصویر بسیار کم است. پیشنهاد می‌شود فلش موبایل را روشن کنید یا ضریب کالیبراسیون را در منوی تنظیمات افزایش دهید.")
        elif result.brightness_level > 215:
            st.warning("☀️ تابش نور مستقیم روی تایر زیاد است. لطفاً زاویه دوربین را کمی تغییر دهید تا سایه شیارها مشخص شود.")

        # منوی تب‌بندی شده اصلی برای جلوگیری از اسکرول طولانی در موبایل
        tab_summary, tab_vision, tab_curve = st.tabs([
            "📊 گزارش وضعیت",
            "🔍 مراحل بینایی ماشین",
            "📈 نمودار استهلاک"
        ])

        # --- تب اول: داشبورد خلاصه مهندسی (واکنش‌گرا ۲×۲ در موبایل) ---
        with tab_summary:
            st.markdown(
                f'<div class="status-banner" style="background-color: {result.status_color};">'
                f'{result.status_label}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">عمق تخمینی آج</div>
                    <div class="metric-value">{result.estimated_depth_mm} mm</div>
                    <div class="metric-sub">حد مجاز: {specs.legal_min_depth_mm} mm</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">درصد فرسایش</div>
                    <div class="metric-value">{result.wear_percentage}%</div>
                    <div class="metric-sub">نسبت به تایر نو</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">عمر مفید باقی‌مانده</div>
                    <div class="metric-value">{result.remaining_life_percentage}%</div>
                    <div class="metric-sub">تا شاخص قانونی TWI</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">پیمایش قابل انجام</div>
                    <div class="metric-value">{result.remaining_km:,}</div>
                    <div class="metric-sub">کیلومتر تخمینی</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**نوار سلامت فیزیکی آج تایر:**")
            st.progress(int(result.remaining_life_percentage))

            # نمایش تصویر برش‌خورده و ماسک شیار در کنار هم
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.image(
                    result.processed_images["roi_rgb"],
                    caption="ناحیه مرکزی بررسی‌شده (بدون اعوجاج)",
                    use_container_width=True
                )
            with p_col2:
                st.image(
                    result.processed_images["groove_mask"],
                    caption=f"شیارهای تشخیص‌داده‌شده ({result.groove_ratio}%)",
                    use_container_width=True,
                    clamp=True
                )

        # --- تب دوم: جزئیات پردازش تصویر ---
        with tab_vision:
            st.caption("خروجی گام‌به‌گام فیلترهای پردازش تصویر OpenCV روی عکس تایر:")
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.image(
                    result.processed_images["enhanced"],
                    caption="۱. یکنواخت‌سازی نور محیط (CLAHE)",
                    use_container_width=True,
                    clamp=True
                )
                st.image(
                    result.processed_images["edges"],
                    caption=f"۳. نقشه لبه‌های آج Canny (تراکم: {result.edge_density}%)",
                    use_container_width=True,
                    clamp=True
                )
            with v_col2:
                st.image(
                    result.processed_images["groove_mask"],
                    caption=f"۲. ماسک باینری شیارها (نسبت: {result.groove_ratio}%)",
                    use_container_width=True,
                    clamp=True
                )
                st.image(
                    result.processed_images["gradient"],
                    caption=f"۴. گرادیان عمق Sobel (شاخص بافت: {result.surface_anomaly_score})",
                    use_container_width=True,
                    clamp=True
                )

        # --- تب سوم: منحنی مهندسی استهلاک و طول عمر ---
        with tab_curve:
            st.caption("پیش‌بینی روند کاهش عمق آج بر حسب کیلومتر پیمایش تا رسیدن به مرز قانونی تعویض (TWI):")
            fig, ax = plt.subplots(figsize=(7, 3.6), dpi=110)
            max_km = specs.nominal_lifespan_km / specs.driving_aggressiveness
            km_axis = np.linspace(0, max_km, 100)
            depth_curve = specs.new_tread_depth_mm - (
                (specs.new_tread_depth_mm - specs.legal_min_depth_mm) * (km_axis / max(km_axis[-1], 1.0))
            )

            current_driven_km = max(0.0, km_axis[-1] - result.remaining_km)

            ax.plot(km_axis, depth_curve, label="Standard Wear Curve", color="#2563eb", linewidth=2.5)
            ax.axhline(specs.legal_min_depth_mm, color="#dc2626", linestyle="--", label=f"Legal TWI ({specs.legal_min_depth_mm}mm)")
            ax.scatter(
                [current_driven_km],
                [result.estimated_depth_mm],
                color="#f97316",
                s=100,
                zorder=5,
                label="Current Tire State"
            )

            ax.set_xlabel("Driven Distance (km)")
            ax.set_ylabel("Tread Depth (mm)")
            ax.set_title("Tire Tread Depth vs. Driven Distance")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()

            st.pyplot(fig)
            # آزادسازی حافظه رم سرور (رفع خطای بحرانی Memory Leak در بازبینی کد)
            plt.close(fig)


if __name__ == "__main__":
    main()
