"""
app.py - ThermaVision AI: Infrared Image Enhancement and Colorization
Main Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from enhancement import run_pipeline, compute_metrics
from colorization import colorize_for_display, get_colormap_names

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ThermaVision AI",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark thermal aesthetic
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base / Background ─────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0a0e1a 0%, #0d1b2a 50%, #0a1520 100%);
    color: #e8edf5;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d19 0%, #0b1622 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Hero banner ───────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0d2137 0%, #0a1a2e 40%, #051020 100%);
    border: 1px solid #1a4a7a;
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(255,100,0,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Courier New', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #ff6b35;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: clamp(1.6rem, 4vw, 2.6rem);
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.02em;
    color: #f0f4ff;
    margin: 0 0 1rem;
}
.hero-title span { color: #ff6b35; }
.hero-desc {
    font-size: 0.98rem;
    color: #8da9c4;
    max-width: 680px;
    line-height: 1.65;
}
.hero-badge {
    display: inline-block;
    margin-top: 1.2rem;
    background: rgba(255,107,53,0.15);
    border: 1px solid rgba(255,107,53,0.4);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.75rem;
    color: #ff9966;
    font-family: 'Courier New', monospace;
    letter-spacing: 0.1em;
}

/* ── Section headings ──────────────────────────────────────── */
.section-label {
    font-family: 'Courier New', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4a7fa5;
    margin-bottom: 0.3rem;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #c8ddf0;
    margin: 0 0 1.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1a3a5c;
}

/* ── Image cards ───────────────────────────────────────────── */
.img-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 0.8rem;
    margin-bottom: 0.8rem;
}
.img-label {
    font-family: 'Courier New', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #ff6b35;
    margin-bottom: 0.45rem;
}

/* ── Metric cards ──────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid #1e3f6a;
    border-radius: 10px;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: #6a9bbf !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #f0f4ff !important; font-size: 1.5rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Sidebar ───────────────────────────────────────────────── */
.sidebar-title {
    font-family: 'Courier New', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a7fa5;
    margin-bottom: 1rem;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,107,53,0.05) !important;
    border: 1.5px dashed #3a6a9a !important;
    border-radius: 10px !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #0d1e30 !important;
    border: 1px solid #1e3f6a !important;
    border-radius: 8px !important;
    color: #c8ddf0 !important;
}

/* ── Download buttons ──────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #1a3a5c, #0d2a45) !important;
    border: 1px solid #2a5a8a !important;
    color: #c8ddf0 !important;
    border-radius: 8px !important;
    width: 100%;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    transition: all 0.2s ease;
}
[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, #ff6b35, #cc4a1a) !important;
    border-color: #ff6b35 !important;
    color: #ffffff !important;
}

/* ── Divider ───────────────────────────────────────────────── */
hr { border-color: #1a3a5c !important; margin: 1.5rem 0 !important; }

/* ── Info box ──────────────────────────────────────────────── */
[data-testid="stInfo"] {
    background: rgba(26, 74, 122, 0.25) !important;
    border: 1px solid #1e4a7a !important;
    border-radius: 8px !important;
    color: #8db8d8 !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: convert NumPy array → PNG bytes for download
# ──────────────────────────────────────────────────────────────────────────────
def array_to_png_bytes(img_array: np.ndarray) -> bytes:
    """Convert a NumPy image array (grayscale or RGB) to PNG bytes."""
    if len(img_array.shape) == 2:
        pil_img = Image.fromarray(img_array, mode="L")
    else:
        pil_img = Image.fromarray(img_array.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def labeled_image(label: str, img_array: np.ndarray, caption: str = ""):
    """Display an image inside a styled card with a monospace label."""
    st.markdown(f'<div class="img-label">{label}</div>', unsafe_allow_html=True)
    cmap = "gray" if len(img_array.shape) == 2 else None
    st.image(img_array, caption=caption, use_container_width=True, clamp=True)


# ──────────────────────────────────────────────────────────────────────────────
# Hero Banner
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🛰 ISRO Hackathon · Remote Sensing · Computer Vision</div>
    <div class="hero-title">ThermaVision <span>AI</span></div>
    <div class="hero-desc">
        Infrared image enhancement and false-color thermal colorization for improved
        object interpretation. Upload a raw thermal or near-infrared image and the pipeline
        denoises, sharpens, and applies perceptual colormaps — making scene details
        that are invisible to the human eye instantly interpretable.
    </div>
    <div class="hero-badge">▸ CLAHE · NLM DENOISING · EDGE SHARPENING · PSEUDO-COLOR MAPPING</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — upload & controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙ Controls</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Thermal / IR Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a grayscale or color infrared image. JPG, JPEG, or PNG formats accepted.",
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-title">🎨 Colorization</div>', unsafe_allow_html=True)

    colormap_choice = st.selectbox(
        "Select Colormap",
        options=get_colormap_names(),
        index=0,
        help=(
            "Inferno – black→red→yellow (perceptually uniform)\n"
            "Jet – blue→green→red (classic thermal)\n"
            "Plasma – purple→orange→yellow\n"
            "Turbo – improved rainbow with better perceptual range"
        ),
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#4a7fa5; line-height:1.7;">
    <b style="color:#6a9bbf;">Pipeline stages</b><br>
    1 · Grayscale conversion<br>
    2 · Non-Local Means denoising<br>
    3 · CLAHE contrast enhancement<br>
    4 · Edge sharpening kernel<br>
    5 · Pseudo-color mapping
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem; color:#3a6080;">
    ThermaVision AI · v1.0<br>
    Built with OpenCV + Streamlit
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main content — only rendered after an image is uploaded
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file is None:
    # Empty state prompt
    st.info(
        "👆  Upload a thermal or infrared image using the sidebar panel to begin. "
        "The pipeline will run automatically."
    )
    st.stop()

# ── Load image ──────────────────────────────────────────────────────────────
file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

if bgr_image is None:
    st.error("Could not decode the uploaded file. Please upload a valid JPG or PNG image.")
    st.stop()

# Convert to RGB for display; keep BGR for OpenCV processing
rgb_original = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) if len(bgr_image.shape) == 3 else bgr_image

# ── Run enhancement pipeline ─────────────────────────────────────────────────
with st.spinner("Running enhancement pipeline…"):
    pipeline = run_pipeline(bgr_image)

gray_img     = pipeline["gray"]
denoised_img = pipeline["denoised"]
clahe_img    = pipeline["clahe"]
sharpened_img = pipeline["sharpened"]

# ── Run colorization ─────────────────────────────────────────────────────────
with st.spinner(f"Applying {colormap_choice} colormap…"):
    colorized_img = colorize_for_display(sharpened_img, colormap_choice)

# ── Compute metrics ──────────────────────────────────────────────────────────
metrics = compute_metrics(gray_img, sharpened_img)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 · Enhancement Pipeline
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Module 01</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Enhancement Pipeline</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    labeled_image("Original (Grayscale)", gray_img)
    st.caption("Raw sensor output converted to single-channel.")

with col2:
    labeled_image("Denoised", denoised_img)
    st.caption("Non-Local Means — reduces thermal sensor noise while preserving edges.")

with col3:
    labeled_image("Contrast Enhanced (CLAHE)", clahe_img)
    st.caption("Local histogram equalization — improves visibility in low-contrast regions.")

with col4:
    labeled_image("Sharpened", sharpened_img)
    st.caption("Laplacian sharpening kernel — accentuates object boundaries.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 · Thermal Colorization
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">Module 02</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-title">Thermal Colorization — {colormap_choice}</div>',
    unsafe_allow_html=True,
)

col_a, col_b = st.columns([2, 1])
with col_a:
    labeled_image(f"False-Color Output · {colormap_choice} Colormap", colorized_img)
    st.caption(
        "Pseudo-color mapping translates grayscale intensity values to a perceptual color scale, "
        "making temperature gradients and object boundaries immediately visible."
    )

with col_b:
    # Quick colormap guide
    cm_desc = {
        "Inferno": "Perceptually uniform. Black (cold) → deep red → orange → bright yellow (hot). Ideal for scientific reporting.",
        "Jet": "Classic rainbow thermal palette. Blue (cold) → cyan → green → yellow → red (hot). High contrast, widely recognized.",
        "Plasma": "Purple → magenta → orange → yellow. Smooth gradients; excellent for subtle temperature differences.",
        "Turbo": "Improved full-spectrum rainbow. Better perceptual uniformity than Jet; great for detecting fine thermal features.",
    }
    st.markdown(f"""
    <div style="background:rgba(255,107,53,0.07); border:1px solid rgba(255,107,53,0.25);
                border-radius:10px; padding:1rem; margin-top:1.5rem;">
        <div style="font-family:'Courier New',monospace; font-size:0.65rem;
                    letter-spacing:0.15em; color:#ff6b35; margin-bottom:0.5rem;">
            COLORMAP INFO
        </div>
        <div style="font-size:0.85rem; color:#8db8d8; line-height:1.6;">
            <b style="color:#c8ddf0;">{colormap_choice}</b><br>{cm_desc[colormap_choice]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # All 4 colormaps small preview
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:\'Courier New\',monospace; font-size:0.65rem; '
        'letter-spacing:0.15em; color:#4a7fa5; margin-bottom:0.4rem;">ALL COLORMAPS</div>',
        unsafe_allow_html=True,
    )
    for cm_name in get_colormap_names():
        preview = colorize_for_display(sharpened_img, cm_name)
        active = "🔹 " if cm_name == colormap_choice else "   "
        st.image(preview, caption=f"{active}{cm_name}", use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 · Comparison Dashboard
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">Module 03</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Comparison Dashboard</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Original IR Image**")
    st.image(gray_img, use_container_width=True, clamp=True)
    st.caption("Raw input — unprocessed grayscale thermal image.")

with c2:
    st.markdown("**Enhanced IR Image**")
    st.image(sharpened_img, use_container_width=True, clamp=True)
    st.caption("After denoising + CLAHE + sharpening — improved structural detail.")

with c3:
    st.markdown(f"**Colorized IR Image ({colormap_choice})**")
    st.image(colorized_img, use_container_width=True, clamp=True)
    st.caption("False-color mapping applied to enhanced image for human interpretation.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 · Metrics
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">Module 04</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Image Quality Metrics</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric(
        label="Mean Pixel Intensity",
        value=f"{metrics['mean_intensity']}",
        help="Average brightness of the enhanced image (0–255 scale).",
    )

with m2:
    delta_val = metrics["contrast_improvement"]
    st.metric(
        label="Contrast Improvement",
        value=f"{abs(delta_val):.1f}%",
        delta=f"+{delta_val:.1f}%" if delta_val >= 0 else f"{delta_val:.1f}%",
        delta_color="normal",
        help="Percentage change in standard deviation of pixel values (contrast proxy).",
    )

with m3:
    st.metric(
        label="Image Resolution",
        value=metrics["resolution"],
        help="Width × Height of the processed image in pixels.",
    )

with m4:
    st.metric(
        label="Original Contrast (σ)",
        value=f"{metrics['original_contrast']}",
        help="Standard deviation of the original grayscale image — measures spread of intensity values.",
    )

with m5:
    st.metric(
        label="Enhanced Contrast (σ)",
        value=f"{metrics['enhanced_contrast']}",
        help="Standard deviation after the full enhancement pipeline.",
    )


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 · Download
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">Module 05</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Download Results</div>', unsafe_allow_html=True)

d1, d2, d3 = st.columns(3)

enhanced_bytes = array_to_png_bytes(sharpened_img)
colorized_bytes = array_to_png_bytes(colorized_img)
original_gray_bytes = array_to_png_bytes(gray_img)

with d1:
    st.download_button(
        label="⬇ Download Enhanced Image",
        data=enhanced_bytes,
        file_name="thermavision_enhanced.png",
        mime="image/png",
        help="Download the sharpened, contrast-enhanced grayscale image.",
    )

with d2:
    st.download_button(
        label=f"⬇ Download Colorized ({colormap_choice})",
        data=colorized_bytes,
        file_name=f"thermavision_colorized_{colormap_choice.lower()}.png",
        mime="image/png",
        help="Download the false-color thermal image with the selected colormap.",
    )

with d3:
    st.download_button(
        label="⬇ Download Original Grayscale",
        data=original_gray_bytes,
        file_name="thermavision_original_gray.png",
        mime="image/png",
        help="Download the original image converted to grayscale.",
    )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:0.72rem; color:#2a5070; padding:0.5rem 0 1rem;">
    ThermaVision AI · Infrared Enhancement &amp; Colorization ·
    Built with OpenCV, NumPy &amp; Streamlit ·
    ISRO Hackathon Prototype
</div>
""", unsafe_allow_html=True)
