"""
اپلیکیشن مهندسی تحلیل فرسایش تایر و تخمین عمر مفید باقی‌مانده (RUL)
Tire Wear & Remaining Useful Life (RUL) Engineering Analyzer

پیش‌نیازها برای اجرا در ترمینال:
pip install streamlit opencv-python numpy pillow matplotlib

نحوه اجرا:
streamlit run tire_wear_analyzer.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict

# =====================================================================
# تنظیمات اولیه صفحه و استایل مهندسی (راست‌چین برای زبان فارسی)
# =====================================================================
st.set_page_config(
    page_title="سامانه مهندسی تحلیل فرسایش تایر",
    page_icon="🛞",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        direction: rtl;
        text-align: right;
        font-family: 'Tahoma', sans-serif;
    }
    h1, h2, h3, p, div {
        direction: rtl;
        text-align: right;
    }
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.3);
        padding: 15px;
        border-radius: 12px;
        text-align: center !important;
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
    status_label: str
    status_color: str
    processed_images: Dict[str, np.ndarray]


# =====================================================================
# موتور پردازش تصویر و بینایی ماشین (Computer Vision Engine)
# =====================================================================
class TireVisionAnalyzer:
    """
    موتور تحلیل تصویر آج تایر با استفاده از تکنیک‌های پردازش تصویر:
    1. یکنواخت‌سازی نور محیط با الگوریتم CLAHE
    2. استخراج شیارهای اصلی آج با آستانه‌گذاری تطبیقی (Adaptive Thresholding)
    3. تحلیل بافت و تراکم لبه‌ها (Canny & Sobel Gradient) برای تشخیص صاف‌شدگی
    4. تخمین عمق آج و محاسبه فرسایش بر اساس نسبت سایه شیارها و بافت سطحی
    """

    def __init__(self, specs: TireSpecs):
        self.specs = specs

    def preprocess_image(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """تغییر سایز استاندارد، برش ناحیه مرکزی تایر (ROI) و حذف نویز"""
        # تغییر ابعاد به اندازه استاندارد برای پایداری محاسبات
        resized = cv2.resize(image_rgb, (500, 500))
        
        # انتخاب ناحیه مرکزی (ROI) برای حذف حاشیه و پس‌زمینه تصویر
        h, w, _ = resized.shape
        margin_h, margin_w = int(h * 0.15), int(w * 0.15)
        roi_rgb = resized[margin_h:h-margin_h, margin_w:w-margin_w]
        
        # تبدیل به سطح خاکستری
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        
        # بهبود کنتراست محلی (CLAHE) برای حذف اثر سایه و نور غیریکنواخت
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # فیلتر دوطرفه (Bilateral) برای حذف نویز سطح لاستیک با حفظ لبه‌های تیز آج
        filtered = cv2.bilateralFilter(enhanced_gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        return roi_rgb, filtered

    def extract_features(self, filtered_gray: np.ndarray) -> Tuple[float, float, float, Dict[str, np.ndarray]]:
        """استخراج ویژگی‌های هندسی و بافتی از سطح تایر"""
        
        # 1. استخراج شیارهای عمیق (Groove Segmentation)
        # شیارهای عمیق آج در تصویر به صورت نواحی تیره دیده می‌شوند
        groove_mask = cv2.adaptiveThreshold(
            filtered_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31, C=8
        )
        
        # عملیات مورفولوژی برای حذف نقاط ریز و نگه‌داشتن شیارهای واقعی آج
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        groove_mask_clean = cv2.morphologyEx(groove_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # محاسبه نسبت مساحت شیارها به کل سطح تایر
        groove_ratio = np.sum(groove_mask_clean > 0) / groove_mask_clean.size

        # 2. تحلیل تراکم لبه‌های آج (Edge Density)
        edges = cv2.Canny(filtered_gray, threshold1=50, threshold2=150)
        edge_density = np.sum(edges > 0) / edges.size

        # 3. تحلیل گرادیان و کنتراست عمق (Sobel Depth Proxy)
        sobelx = cv2.Sobel(filtered_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(filtered_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_norm = np.uint8(255 * gradient_magnitude / (np.max(gradient_magnitude) + 1e-5))
        
        # 4. تشخیص ترک‌خوردگی‌های ریز و پوسیدگی سطحی (High-Frequency Anomalies)
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
        roi_rgb, filtered_gray = self.preprocess_image(image_rgb)
        groove_ratio, edge_density, anomaly_score, visuals = self.extract_features(filtered_gray)
        visuals["roi_rgb"] = roi_rgb

        # ترکیب وزن‌دار شاخص‌های نوری برای تخمین سلامت آج (بین 0 تا 1)
        # در یک تایر نو: groove_ratio معمولاً حدود 0.22 تا 0.35 و edge_density حدود 0.08 تا 0.16 است
        normalized_groove = np.clip(groove_ratio / 0.28, 0.0, 1.0)
        normalized_edge = np.clip(edge_density / 0.12, 0.0, 1.0)
        
        health_index = (0.65 * normalized_groove + 0.35 * normalized_edge) * calibration_factor
        health_index = float(np.clip(health_index, 0.0, 1.0))

        # نگاشت شاخص سلامت به عمق فیزیکی آج (بر حسب میلی‌متر)
        min_d = self.specs.legal_min_depth_mm
        max_d = self.specs.new_tread_depth_mm
        
        # امکان رسیدن به کمتر از حد قانونی در صورت صاف بودن کامل تایر (تا 0.5 میلی‌متر)
        absolute_min = 0.5
        estimated_depth_mm = absolute_min + health_index * (max_d - absolute_min)

        # محاسبه درصد عمر مفید باقی‌مانده (بر اساس حد مجاز قانونی TWI)
        usable_range = max_d - min_d
        usable_remaining = max(0.0, estimated_depth_mm - min_d)
        remaining_life_pct = float(np.clip((usable_remaining / usable_range) * 100.0, 0.0, 100.0))
        wear_pct = 100.0 - remaining_life_pct

        # محاسبه کیلومتر باقی‌مانده با در نظر گرفتن ضریب رانندگی
        remaining_km = int(
            (remaining_life_pct / 100.0) * 
            (self.specs.nominal_lifespan_km / self.specs.driving_aggressiveness)
        )

        # تعیین وضعیت ایمنی مهندسی
        if remaining_life_pct >= 65:
            status_label = "عالی (استاندارد و ایمن)"
            status_color = "green"
        elif remaining_life_pct >= 30:
            status_label = "متوسط (نیازمند بررسی دوره‌ای)"
            status_color = "orange"
        elif remaining_life_pct > 5:
            status_label = "فرسوده (نزدیک به شاخص TWI - تعویض به‌زودی)"
            status_color = "red"
        else:
            status_label = "بحرانی و خطرناک (غیرمجاز - نیازمند تعویض فوری!)"
            status_color = "darkred"

        return AnalysisResult(
            estimated_depth_mm=round(estimated_depth_mm, 2),
            wear_percentage=round(wear_pct, 1),
            remaining_life_percentage=round(remaining_life_pct, 1),
            remaining_km=remaining_km,
            edge_density=round(edge_density * 100, 2),
            groove_ratio=round(groove_ratio * 100, 2),
            surface_anomaly_score=round(anomaly_score, 2),
            status_label=status_label,
            status_color=status_color,
            processed_images=visuals
        )


# =====================================================================
# شبیه‌ساز تصویر تایر (برای تست فوری برنامه بدون نیاز به عکس اولیه)
# =====================================================================
def generate_synthetic_tire(wear_level: str) -> np.ndarray:
    """تولید تصویر مصنوعی از بافت و شیارهای تایر برای تست الگوریتم"""
    img = np.ones((500, 500, 3), dtype=np.uint8) * 110
    
    # ایجاد بافت لاستیک پایه
    noise = np.random.normal(0, 12, (500, 500)).astype(np.int16)
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
    else:  # فرسوده و صاف
        groove_intensity = 95
        groove_width = 4
        lateral_step = 110

    # ترسیم شیارهای طولی اصلی (Main Longitudinal Grooves)
    for x in [140, 220, 280, 360]:
        cv2.rectangle(img, (x - groove_width, 0), (x + groove_width, 500), 
                      (groove_intensity, groove_intensity, groove_intensity), -1)

    # ترسیم آج‌های عرضی و مورب (Lateral Tread Blocks)
    for y in range(20, 480, lateral_step):
        thickness = max(2, groove_width // 2)
        cv2.line(img, (60, y), (140, y + 25), (groove_intensity, groove_intensity, groove_intensity), thickness)
        cv2.line(img, (140, y + 10), (220, y - 10), (groove_intensity, groove_intensity, groove_intensity), thickness)
        cv2.line(img, (280, y - 10), (360, y + 10), (groove_intensity, groove_intensity, groove_intensity), thickness)
        cv2.line(img, (360, y + 25), (440, y), (groove_intensity, groove_intensity, groove_intensity), thickness)

    # مات‌سازی ملایم برای شبیه‌سازی تصویر دوربین واقعی
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


# =====================================================================
# رابط کاربری گرافیکی (Streamlit UI)
# =====================================================================
def main():
    st.title("🛞 سامانه مهندسی تشخیص فرسایش تایر و تخمین عمر مفید (RUL)")
    st.markdown(
        "این نرم‌افزار با استفاده از **پردازش تصویر دیجیتال (OpenCV)**، تراکم شیارها، لبه‌های آج و کنتراست عمق را از روی تصویر تایر استخراج کرده و میزان سایش و کیلومتر قابل پیمایش باقی‌مانده را محاسبه می‌کند."
    )

    # --- نوار کناری: پارامترهای کالیبراسیون مهندسی ---
    with st.sidebar:
        st.header("⚙️ تنظیمات و پارامترهای مهندسی")
        tire_category = st.selectbox(
            "نوع تایر خودرو:",
            ["تایر سواری تابستانی / چهارفصل", "تایر سواری زمستانی (برفی)", "تایر شاسی‌بلند (SUV / آفرود)"]
        )

        default_depth = 8.0
        if "زمستانی" in tire_category:
            default_depth = 9.5
        elif "شاسی‌بلند" in tire_category:
            default_depth = 10.5

        new_depth = st.slider("عمق آج تایر نو (میلی‌متر):", 6.0, 14.0, default_depth, 0.5)
        legal_min = st.slider("حداقل عمق مجاز قانونی TWI (میلی‌متر):", 1.0, 3.0, 1.6, 0.1)
        nominal_km = st.number_input("عمر اسمی تایر نو (کیلومتر):", min_value=20000, max_value=100000, value=50000, step=5000)
        
        driving_style = st.select_slider(
            "شرایط جاده و سبک رانندگی:",
            options=["آرام (بزرگراهی)", "معمولی (شهری)", "پرفشار / جاده ناهموار"],
            value="معمولی (شهری)"
        )
        aggressiveness_map = {"آرام (بزرگراهی)": 0.85, "معمولی (شهری)": 1.0, "پرفشار / جاده ناهموار": 1.35}
        
        calibration = st.slider(
            "ضریب کالیبراسیون نوری دوربین:",
            0.5, 1.5, 1.0, 0.05,
            help="در صورتی که نور محیط بسیار کم یا زیاد است، با این ضریب حساسیت تشخیص شیار را کالیبره کنید."
        )

    specs = TireSpecs(
        new_tread_depth_mm=new_depth,
        legal_min_depth_mm=legal_min,
        nominal_lifespan_km=nominal_km,
        driving_aggressiveness=aggressiveness_map[driving_style]
    )

    # --- بخش دریافت تصویر (آپلود، دوربین یا شبیه‌ساز) ---
    st.subheader("📸 ۱. ورود تصویر آج تایر")
    input_method = st.radio(
        "روش دریافت تصویر را انتخاب کنید:",
        ["آپلود عکس از گالری", "عکسبرداری مستقیم با دوربین", "استفاده از نمونه آزمایشی (شبیه‌ساز)"],
        horizontal=True
    )

    input_image_rgb = None

    if input_method == "آپلود عکس از گالری":
        uploaded_file = st.file_uploader("یک تصویر واضح از نمای روبروی آج تایر انتخاب کنید:", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            input_image_rgb = np.array(pil_img)

    elif input_method == "عکسبرداری مستقیم با دوربین":
        camera_photo = st.camera_input("دوربین را روبروی آج تایر نگه دارید و عکس بگیرید:")
        if camera_photo is not None:
            pil_img = Image.open(camera_photo).convert("RGB")
            input_image_rgb = np.array(pil_img)

    else:
        sample_type = st.selectbox("حالت نمونه آزمایشی را انتخاب کنید:", ["نو (سالم)", "کارکرده (متوسط)", "فرسوده (صاف)"])
        input_image_rgb = generate_synthetic_tire(sample_type)

    # --- اجرای تحلیل در صورت وجود تصویر ---
    if input_image_rgb is not None:
        analyzer = TireVisionAnalyzer(specs)
        result = analyzer.analyze(input_image_rgb, calibration_factor=calibration)

        st.markdown("---")
        st.subheader("📊 ۲. گزارش مهندسی وضعیت تایر")

        # نمایش کارت وضعیت کلی
        st.markdown(
            f"""
            <div style="background-color: {result.status_color}; color: white; padding: 14px; border-radius: 10px; font-size: 18px; font-weight: bold; text-align: center; margin-bottom: 20px;">
                وضعیت نهایی تایر: {result.status_label}
            </div>
            """,
            unsafe_allow_html=True
        )

        # متریک‌های اصلی در 4 ستون
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("عمق تخمینی آج فعلی", f"{result.estimated_depth_mm} mm", f"حداقل مجاز: {specs.legal_min_depth_mm} mm")
        with col2:
            st.metric("درصد فرسایش تایر", f"{result.wear_percentage} %")
        with col3:
            st.metric("عمر مفید باقی‌مانده", f"{result.remaining_life_percentage} %")
        with col4:
            st.metric("پیمایش تخمینی باقی‌مانده", f"{result.remaining_km:,} km")

        # نوار پیشرفت عمر تایر
        st.write("**نمای گرافیکی سلامت آج:**")
        st.progress(int(result.remaining_life_percentage))

        # --- نمایش مراحل پردازش تصویر ---
        st.markdown("---")
        st.subheader("🔍 ۳. خروجی مراحل بینایی ماشین (Computer Vision Pipeline)")
        
        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        with v_col1:
            st.image(result.processed_images["roi_rgb"], caption="۱. ناحیه مورد بررسی (ROI)", use_container_width=True)
        with v_col2:
            st.image(result.processed_images["enhanced"], caption="۲. نرمال‌سازی نوری (CLAHE)", use_container_width=True, clamp=True)
        with v_col3:
            st.image(result.processed_images["groove_mask"], caption=f"۳. ماسک شیارها (نسبت: {result.groove_ratio}%)", use_container_width=True, clamp=True)
        with v_col4:
            st.image(result.processed_images["edges"], caption=f"۴. نقشه لبه‌های آج (تراکم: {result.edge_density}%)", use_container_width=True, clamp=True)

        # --- نمودار منحنی استهلاک مهندسی ---
        st.markdown("---")
        st.subheader("📈 ۴. منحنی استهلاک و پیش‌بینی عمر (RUL Curve)")
        
        fig, ax = plt.subplots(figsize=(8, 3.5))
        km_axis = np.linspace(0, specs.nominal_lifespan_km / specs.driving_aggressiveness, 100)
        depth_curve = specs.new_tread_depth_mm - (
            (specs.new_tread_depth_mm - specs.legal_min_depth_mm) * (km_axis / km_axis[-1])
        )
        
        current_driven_km = km_axis[-1] - result.remaining_km
        
        ax.plot(km_axis, depth_curve, label="Linear Wear Model", color="#1f77b4", linewidth=2.5)
        ax.axhline(specs.legal_min_depth_mm, color="red", linestyle="--", label="Legal Limit (TWI = 1.6mm)")
        ax.scatter([current_driven_km], [result.estimated_depth_mm], color="darkorange", s=90, zorder=5, label="Current Tire State")
        
        ax.set_xlabel("Driven Distance (km)")
        ax.set_ylabel("Tread Depth (mm)")
        ax.set_title("Tire Tread Depth Degradation vs. Distance")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        st.pyplot(fig)


if __name__ == "__main__":
    main()