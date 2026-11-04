from __future__ import annotations

import base64
import html
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

# =============================================================================
# LANDSCAPE AI CLASSIFIER
# Streamlit interface for the Intel Image Classification DL experiments.
# The app reads the actual Keras checkpoints and evaluation artefacts produced
# by Intel_Landscape_Classification_Tahap1_2.ipynb. No dummy predictions or
# fabricated evaluation metrics are displayed.
# =============================================================================
st.set_page_config(
    page_title="Landscape AI Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
LOGO_PATH = ASSETS / "logo.png"  # Logo IMAGE A08 asli dari pengguna
IMAGE_SIZE = (150, 150)
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_META = {
    "buildings": ("🏙️", "Bangunan dan area perkotaan"),
    "forest": ("🌲", "Hutan dan vegetasi lebat"),
    "glacier": ("🧊", "Gletser dan bentang es"),
    "mountain": ("⛰️", "Pegunungan dan perbukitan"),
    "sea": ("🌊", "Laut dan pesisir"),
    "street": ("🛣️", "Jalan dan lingkungan perkotaan"),
}
MODEL_PATHS = {
    "Custom CNN": MODELS / "custom_cnn_best.keras",
    "MobileNetV2": MODELS / "mobilenetv2_best.keras",
}
EXPERIMENTS = {
    "E1_custom_cnn_no_augmentation": {
        "short": "E1 · Custom CNN tanpa augmentasi",
        "model": "Custom CNN",
        "treatment": "Tanpa augmentasi",
        "epochs": "Maks. 30",
        "lr": "0,001",
        "purpose": "Baseline awal untuk mengukur kemampuan CNN kustom tanpa augmentasi data.",
    },
    "E2_custom_cnn_with_augmentation": {
        "short": "E2 · Custom CNN dengan augmentasi",
        "model": "Custom CNN",
        "treatment": "Dengan augmentasi",
        "epochs": "Maks. 30",
        "lr": "0,001",
        "purpose": "Mengukur pengaruh augmentasi terhadap generalisasi Custom CNN.",
    },
    "E3_mobilenetv2_feature_extraction": {
        "short": "E3 · MobileNetV2 feature extraction",
        "model": "MobileNetV2",
        "treatment": "Augmentasi + backbone dibekukan",
        "epochs": "Maks. 15",
        "lr": "0,001",
        "purpose": "Melatih classifier baru dengan fitur pretrained ImageNet tanpa memperbarui backbone.",
    },
    "E4_mobilenetv2_finetuning": {
        "short": "E4 · MobileNetV2 fine-tuning",
        "model": "MobileNetV2",
        "treatment": "Augmentasi + fine-tuning 30 layer terakhir*",
        "epochs": "Maks. 10",
        "lr": "0,00001",
        "purpose": "Menyesuaikan sebagian backbone pretrained pada karakteristik dataset pemandangan.",
    },
}
for directory in (MODELS, RESULTS, ASSETS):
    directory.mkdir(parents=True, exist_ok=True)


def data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    mime = "image/svg+xml" if suffix == ".svg" else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


LOGO_URI = data_uri(LOGO_PATH)

# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------
CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root{--ink:#14251d;--muted:#718078;--green:#176b4c;--green2:#23845e;--mint:#e8f4ec;--line:#e1ebe3;--paper:#f5f8f4;--gold:#d9bc78}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(circle at 86% 0%,rgba(214,238,220,.58),transparent 27%),linear-gradient(180deg,#f7faf7 0%,#f2f6f2 100%);color:var(--ink)}
.block-container{max-width:1480px;padding-top:1.45rem;padding-bottom:3rem}
h1,h2,h3,h4{font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;color:var(--ink)}
p,li,label{color:#586b61}
[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"],#MainMenu,footer{visibility:hidden!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#edf5ef 0%,#e7f0e9 100%);border-right:1px solid #dce8df}
[data-testid="stSidebar"] .block-container{padding-top:1rem}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#62766b}
[data-testid="stSidebar"] [data-testid="stRadio"] label{font-weight:650;padding:.28rem .2rem}
.stButton>button,.stDownloadButton>button{border-radius:13px;border:1px solid #176b4c;background:#176b4c;color:#fff;font-weight:700;min-height:2.8rem;padding:.55rem 1.05rem;box-shadow:0 7px 18px rgba(23,107,76,.12);transition:all .18s ease}
.stButton>button:hover,.stDownloadButton>button:hover{background:#10573d;border-color:#10573d;color:#fff;transform:translateY(-1px)}
.stButton>button:disabled{background:#dce7df;border-color:#dce7df;color:#819087;box-shadow:none}
.stButton>button,.stDownloadButton>button{color:#ffffff!important;opacity:1!important}
div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button{color:#fff!important;opacity:1!important}
div[data-testid="stButton"] > button p, div[data-testid="stButton"] > button span, div[data-testid="stButton"] > button div, div[data-testid="stDownloadButton"] > button p, div[data-testid="stDownloadButton"] > button span, div[data-testid="stDownloadButton"] > button div{color:#fff!important;opacity:1!important;font-weight:750!important}
div[data-testid="stButton"] > button:disabled, div[data-testid="stDownloadButton"] > button:disabled{background:#dce7df!important;border-color:#dce7df!important;color:#52645a!important;opacity:1!important}
div[data-testid="stButton"] > button:disabled p, div[data-testid="stButton"] > button:disabled span, div[data-testid="stDownloadButton"] > button:disabled p, div[data-testid="stDownloadButton"] > button:disabled span{color:#52645a!important;opacity:1!important}
.stButton>button p,.stButton>button span,.stDownloadButton>button p,.stDownloadButton>button span{color:#ffffff!important;opacity:1!important;font-weight:750!important}
.stButton>button:disabled,.stDownloadButton>button:disabled{color:#52645a!important;opacity:1!important}
.stButton>button:disabled p,.stButton>button:disabled span,.stDownloadButton>button:disabled p,.stDownloadButton>button:disabled span{color:#52645a!important}
[data-testid="stProgressBar"]>div{background:#e2ece4;border-radius:99px}
.lac-onboarding{max-width:900px;margin:1.1rem auto 0}
.lac-onboard-logo img{width:126px;height:126px;object-fit:contain;background:#fff;border:1px solid #e8eee8;border-radius:22px;padding:3px;box-shadow:0 10px 24px rgba(23,64,42,.08)}

[data-testid="stFileUploader"]{background:#fbfdfb;border:1.5px dashed #9cbaa8;border-radius:18px;padding:.8rem 1rem}
[data-testid="stFileUploader"] section{border:0;background:transparent}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:.85rem 1rem;box-shadow:0 5px 18px rgba(20,55,38,.035)}
[data-testid="stMetricLabel"] p{color:#718078!important;font-size:.78rem}
[data-testid="stMetricValue"]{font-family:'Manrope',sans-serif;color:#176b4c;font-weight:800}
[data-testid="stAlert"]{border-radius:14px}
[data-testid="stImage"] img{border-radius:16px}
[data-testid="stProgressBar"]>div>div{background:#23845e}
hr{border-color:#e4ece5}
.lac-brand{display:flex;align-items:center;gap:.72rem;padding:.25rem .1rem 1rem}
.lac-brand img{width:57px;height:57px;object-fit:contain;border-radius:13px;background:white;border:1px solid #e1e9e2;padding:3px}
.lac-brand-name{font-family:'Manrope',sans-serif;color:#153d2d;font-weight:800;font-size:1.02rem;line-height:1.15}
.lac-brand-sub{font-size:10px;color:#819188;margin-top:4px;letter-spacing:.03em}
.lac-eyebrow{font-size:10px;font-weight:800;letter-spacing:.18em;text-transform:uppercase;color:#4f9871}
.lac-hero{position:relative;overflow:hidden;border-radius:27px;padding:clamp(1.45rem,3.2vw,3rem);background:linear-gradient(118deg,#102f25 0%,#174c37 57%,#267957 100%);color:#fff;box-shadow:0 18px 48px rgba(18,66,46,.15);isolation:isolate}
.lac-hero:before{content:'';position:absolute;z-index:-1;width:440px;height:440px;right:-75px;top:-190px;border-radius:50%;background:radial-gradient(circle,rgba(197,231,204,.24),rgba(197,231,204,0) 68%)}
.lac-hero h1{color:white;font-size:clamp(2rem,4vw,3.4rem);font-weight:800;line-height:1.03;letter-spacing:-.065em;margin:.65rem 0 .9rem;max-width:760px}
.lac-hero p{color:rgba(245,255,248,.82);font-size:14px;line-height:1.8;max-width:800px;margin:0}
.lac-hero .eyebrow{color:#b8e2c6;font-size:10px;font-weight:800;letter-spacing:.19em;text-transform:uppercase}
.lac-hero-art{position:absolute;right:3%;bottom:5%;width:min(32%,310px);opacity:.9;filter:drop-shadow(0 16px 22px rgba(0,0,0,.13));pointer-events:none}
.lac-hero-art svg{width:100%;height:auto}
.lac-pills{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1.3rem}
.lac-pill{display:inline-flex;align-items:center;gap:.35rem;border:1px solid rgba(223,247,229,.23);background:rgba(255,255,255,.09);border-radius:99px;padding:.4rem .7rem;color:#e7f5ea;font-size:10px;font-weight:750}
.lac-head{display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;margin:1.55rem 0 .8rem}
.lac-head h2{font-size:1.28rem;margin:0;color:#18382b}
.lac-head p{margin:.25rem 0 0;color:#7a8981;font-size:12px;line-height:1.55}
.lac-card{background:#fff;border:1px solid var(--line);border-radius:19px;padding:1.1rem 1.15rem;box-shadow:0 8px 24px rgba(28,65,45,.04);height:100%}
.lac-card-dark{background:linear-gradient(145deg,#183d2e,#225d42);border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:1.25rem;color:white;height:100%;box-shadow:0 12px 28px rgba(20,65,44,.12)}
.lac-card-dark h3,.lac-card-dark h2{color:#fff}
.lac-card-dark p{color:#d0e6d7;line-height:1.7}
.lac-kicker{font-size:9px;font-weight:850;letter-spacing:.14em;text-transform:uppercase;color:#6f9d80}
.lac-card-title{font-family:'Manrope',sans-serif;font-size:1.1rem;font-weight:800;color:#183b2c;margin:.35rem 0 .45rem}
.lac-muted{color:#728178;font-size:12px;line-height:1.7}
.lac-class-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}
.lac-class-card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:.9rem .55rem;text-align:center;box-shadow:0 6px 18px rgba(28,65,45,.03)}
.lac-class-emoji{font-size:25px;margin-bottom:.35rem}.lac-class-name{font-weight:800;color:#294b39;font-size:12px}
.lac-status{display:flex;align-items:flex-start;gap:.8rem;padding:1rem 1.05rem;border-radius:16px;background:#fff8e9;border:1px solid #f0e0bb;color:#715a25}
.lac-status strong{display:block;color:#5e4a1c;margin-bottom:.25rem}.lac-status p{font-size:12px;line-height:1.65;color:#806d42;margin:0}
.lac-status-ok{background:#edf8ef;border-color:#d6ebd9;color:#28643e}.lac-status-ok strong,.lac-status-ok p{color:#28643e}
.lac-model-box{height:100%;border:1px solid var(--line);border-radius:19px;background:#fff;padding:1.1rem;box-shadow:0 8px 22px rgba(28,65,45,.04)}
.lac-model-name{font-family:'Manrope',sans-serif;font-size:1.1rem;font-weight:850;color:#193c2d;margin:.35rem 0}
.lac-badge{display:inline-block;padding:.32rem .58rem;border-radius:99px;background:#e8f4eb;color:#28724d;font-size:9px;font-weight:850;letter-spacing:.1em;text-transform:uppercase}
.lac-arch{display:flex;flex-direction:column;gap:5px;margin-top:.75rem}
.lac-arch-step{background:#f1f7f2;border:1px solid #e1ece3;border-radius:10px;padding:.55rem .7rem;text-align:center;color:#315a41;font-size:11px;font-weight:750}
.lac-arch-arrow{text-align:center;color:#80a48c;font-size:13px;line-height:1}
.lac-experiment{background:#fff;border:1px solid var(--line);border-radius:16px;padding:.95rem 1rem;height:100%}
.lac-exp-code{font-family:'Manrope',sans-serif;font-weight:900;color:#176b4c;font-size:1.2rem}
.lac-exp-title{font-weight:800;color:#213e30;margin:.15rem 0 .45rem;font-size:13px}
.lac-exp-meta{font-size:11px;color:#75847b;line-height:1.65}
.lac-prob-row{display:grid;grid-template-columns:90px 1fr 55px;gap:9px;align-items:center;margin:.48rem 0;font-size:11px}
.lac-prob-track{height:8px;background:#edf2ee;border-radius:20px;overflow:hidden}.lac-prob-fill{height:100%;border-radius:20px;background:linear-gradient(90deg,#3f9b6c,#176b4c)}
.lac-step{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1rem;height:100%}.lac-step-num{width:30px;height:30px;display:grid;place-items:center;border-radius:10px;background:#e2f2e7;color:#176b4c;font-weight:900;margin-bottom:.65rem}
.lac-footer{color:#94a198;text-align:center;font-size:10px;padding:2rem 0 .5rem}
.lac-onboarding{background:linear-gradient(140deg,#f0f8f1 0%,#fff 55%,#f4f6ed 100%);border:1px solid #deebe1;border-radius:26px;padding:clamp(1.2rem,4vw,2.7rem);box-shadow:0 18px 50px rgba(22,65,43,.08)}
.lac-onboard-logo{display:flex;justify-content:center;margin:0 auto 1rem}.lac-onboard-logo img{width:108px;height:108px;object-fit:contain;background:#fff;border:1px solid #e8eee8;border-radius:22px;padding:5px;box-shadow:0 10px 24px rgba(23,64,42,.08)}
.lac-splash{position:fixed;inset:0;z-index:999999;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:13px;background:radial-gradient(circle at 50% 32%,#2d7957 0%,#174a36 48%,#0d2f23 100%);color:#fff;animation:splashFade 2.7s ease forwards;pointer-events:none}
.lac-splash-logo{width:116px;height:116px;display:grid;place-items:center;background:#fff;border-radius:25px;padding:6px;box-shadow:0 18px 44px rgba(0,0,0,.2)}.lac-splash-logo img{width:100%;height:100%;object-fit:contain;border-radius:17px}
.lac-splash-name{font-family:'Manrope',sans-serif;font-size:25px;font-weight:850;color:white}.lac-splash-sub{color:rgba(244,255,247,.74);font-size:9px;letter-spacing:.18em;text-transform:uppercase}
@keyframes splashFade{0%,82%{opacity:1;visibility:visible}100%{opacity:0;visibility:hidden}}
@media(max-width:950px){.lac-hero-art{opacity:.25;width:40%;right:-2%}.lac-class-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:650px){.block-container{padding-left:1rem;padding-right:1rem}.lac-hero{border-radius:21px}.lac-hero h1{font-size:2.05rem}.lac-hero-art{display:none}.lac-class-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.lac-prob-row{grid-template-columns:75px 1fr 48px}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# State and helpers
# -----------------------------------------------------------------------------
DEFAULT_STATE = {
    "splash_seen": False,
    "onboarding_done": False,
    "onboarding_step": 0,
    "nav_page": "Beranda",
    "last_prediction": None,
    "upload_key": 0,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def esc(value: Any) -> str:
    return html.escape(str(value))


def title_case(value: str) -> str:
    return str(value).replace("_", " ").title()


def render_hero(eyebrow: str, title: str, description: str, pills: list[str] | None = None) -> None:
    pill_html = "".join(f'<span class="lac-pill">{esc(x)}</span>' for x in (pills or []))
    title_html = "<br>".join(esc(part) for part in title.splitlines())
    art = """<div class="lac-hero-art"><svg viewBox="0 0 420 250" xmlns="http://www.w3.org/2000/svg"><circle cx="305" cy="66" r="39" fill="#d9edbd" opacity=".9"/><path d="M12 220 118 80l72 92 67-120 145 168Z" fill="#b8d9bc" opacity=".78"/><path d="m118 80 30 39-28-10-16 19-22-2Zm139-28 34 48-34-13-18 20-25-6Z" fill="#f4f8e8"/><path d="M0 225q65-52 135-20t130 0 155 6v39H0Z" fill="#6da982"/><path d="M0 235q75-32 151 0t141-4 128 0v19H0Z" fill="#327b56"/></svg></div>"""
    st.markdown(
        f'<section class="lac-hero"><div class="eyebrow">{esc(eyebrow)}</div><h1>{title_html}</h1><p>{esc(description)}</p><div class="lac-pills">{pill_html}</div>{art}</section>',
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="lac-head"><div><h2>{esc(title)}</h2><p>{esc(subtitle)}</p></div></div>', unsafe_allow_html=True)


def model_file_status() -> dict[str, bool]:
    return {name: path.exists() for name, path in MODEL_PATHS.items()}


def result_file_status() -> dict[str, bool]:
    return {
        "test_metrics_all_experiments.csv": (RESULTS / "test_metrics_all_experiments.csv").exists(),
        "training_curves_all_experiments.png": (RESULTS / "training_curves_all_experiments.png").exists(),
        "confusion_matrix_best_model.png": (RESULTS / "confusion_matrix_best_model.png").exists(),
        "experiment_summary.json": (RESULTS / "experiment_summary.json").exists(),
        "model_selection.json": (RESULTS / "model_selection.json").exists(),
    }


def get_class_names() -> list[str]:
    path = RESULTS / "class_names.json"
    if path.exists():
        try:
            names = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(names, list) and len(names) == 6:
                return [str(name).lower() for name in names]
        except (OSError, json.JSONDecodeError):
            pass
    return CLASS_NAMES


@st.cache_resource(show_spinner=False)
def load_keras_model(path_string: str):
    # Import lazily so the UI and research pages work without TensorFlow installed.
    import tensorflow as tf
    return tf.keras.models.load_model(path_string, compile=False)


def predict_one(model: Any, image: Image.Image) -> np.ndarray:
    # Samakan resize dengan pipeline notebook: tf.image.resize(..., antialias=True).
    # Kedua model menyimpan normalisasi masing-masing sebagai layer internal.
    import tensorflow as tf
    rgb_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    resized = tf.image.resize(rgb_array, IMAGE_SIZE, method="bilinear", antialias=True).numpy()
    batch = resized[None, ...]  # float32, rentang pixel 0..255; jangan normalisasi lagi di aplikasi
    probabilities = np.asarray(model.predict(batch, verbose=0))[0]
    if probabilities.ndim != 1 or probabilities.size != 6:
        raise ValueError(f"Output model harus berisi enam skor kelas; shape ditemukan {probabilities.shape}.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model menghasilkan nilai yang tidak valid (NaN/Inf).")
    return probabilities.astype(float)


def render_model_architecture(model_name: str) -> None:
    if model_name == "Custom CNN":
        steps = [
            "Input RGB · 150 × 150 × 3",
            "Augmentasi (khusus E2)",
            "Rescaling · 1/255",
            "Conv Block 1 · 32 filter × 2 Conv2D",
            "BatchNorm + ReLU + MaxPooling",
            "Conv Block 2 · 64 filter × 2 Conv2D",
            "BatchNorm + ReLU + MaxPooling",
            "Conv Block 3 · 128 filter × 2 Conv2D",
            "BatchNorm + ReLU + MaxPooling",
            "Global Average Pooling",
            "Dense 128 · ReLU + Dropout 0,40",
            "Output · Softmax 6 kelas",
        ]
    else:
        steps = [
            "Input RGB · 150 × 150 × 3",
            "Augmentasi data training",
            "Rescaling · [0, 255] → [-1, 1]",
            "MobileNetV2 · ImageNet weights",
            "Feature maps · include_top=False",
            "Global Average Pooling",
            "Dropout 0,30",
            "Dense 128 · ReLU",
            "Output · Softmax 6 kelas",
        ]
    rows = []
    for index, item in enumerate(steps):
        rows.append(f'<div class="lac-arch-step">{esc(item)}</div>')
        if index < len(steps) - 1:
            rows.append('<div class="lac-arch-arrow">↓</div>')
    st.markdown('<div class="lac-arch">' + "".join(rows) + "</div>", unsafe_allow_html=True)


def build_pdf(payload: dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=17 * mm, leftMargin=17 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="LACSmall", parent=styles["Normal"], fontSize=8, leading=12, textColor=colors.HexColor("#64756c")))
    styles.add(ParagraphStyle(name="LACGreen", parent=styles["Heading2"], textColor=colors.HexColor("#176b4c"), spaceBefore=10, spaceAfter=6))
    story = [Paragraph("Landscape AI Classifier", styles["Title"]), Paragraph("Laporan hasil klasifikasi citra pemandangan dengan Deep Learning", styles["LACSmall"]), Spacer(1, 6 * mm)]
    if LOGO_PATH.exists():
        try:
            story.append(RLImage(str(LOGO_PATH), width=24 * mm, height=24 * mm, kind="proportional"))
        except Exception:
            pass
    story.extend([
        Paragraph(f"Waktu analisis: {esc(payload['created_at'])}", styles["Normal"]),
        Paragraph(f"Nama file: {esc(payload['filename'])}", styles["Normal"]),
        Spacer(1, 4 * mm),
    ])
    if payload.get("image_bytes"):
        try:
            story.extend([RLImage(io.BytesIO(payload["image_bytes"]), width=78 * mm, height=52 * mm, kind="proportional"), Spacer(1, 4 * mm)])
        except Exception:
            pass
    rows = [["Model Deep Learning", "Kelas prediksi", "Skor Softmax"]]
    for model_name, item in payload["predictions"].items():
        rows.append([model_name, title_case(item["class_name"]), f"{item['confidence'] * 100:.2f}%"])
    table = Table(rows, colWidths=[49 * mm, 65 * mm, 43 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176b4c")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#dce8df")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f7f3")]), ("PAD", (0, 0), (-1, -1), 8),
    ]))
    story.extend([Paragraph("Ringkasan prediksi", styles["LACGreen"]), table])
    for model_name, item in payload["predictions"].items():
        story.extend([Spacer(1, 4 * mm), Paragraph(f"Distribusi skor — {esc(model_name)}", styles["LACGreen"])])
        prob_rows = [["Kategori", "Skor"]]
        for index in np.argsort(item["probabilities"])[::-1]:
            prob_rows.append([title_case(payload["class_names"][int(index)]), f"{item['probabilities'][int(index)] * 100:.2f}%"])
        prob_table = Table(prob_rows, colWidths=[95 * mm, 45 * mm])
        prob_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5f2e8")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#dce8df")), ("PAD", (0, 0), (-1, -1), 6),
        ]))
        story.append(prob_table)
    story.extend([Spacer(1, 7 * mm), Paragraph("Catatan: skor Softmax merupakan keluaran relatif model dan bukan jaminan kebenaran yang sudah terkalibrasi. Model hanya dilatih untuk enam kelas Intel Image Classification.", styles["LACSmall"])])
    doc.build(story)
    return buffer.getvalue()


def render_onboarding() -> None:
    steps = [
        ("01 / KENALI PROYEK", "Dua arsitektur, satu tujuan", "Proyek ini membandingkan Custom CNN yang dilatih dari awal dengan MobileNetV2 yang memanfaatkan bobot pretrained ImageNet.", "🧠", "Fokus penelitian: klasifikasi enam kategori pemandangan."),
        ("02 / COBA PREDIKSI", "Masukkan satu citra", "Unggah gambar JPG, JPEG, atau PNG. Setelah checkpoint hasil training Colab tersedia, kedua model akan memproses gambar yang sama secara otomatis.", "🖼️", "Preprocessing pada aplikasi mengikuti layer preprocessing di dalam model."),
        ("03 / BACA HASIL", "Bandingkan dan evaluasi", "Lihat prediksi kedua model, distribusi skor enam kelas, serta hasil evaluasi eksperimen E1 sampai E4. Unduh ringkasan prediksi sebagai PDF.", "📊", "Metrik penelitian hanya ditampilkan setelah hasil evaluasi aktual tersedia."),
    ]
    index = max(0, min(2, int(st.session_state.onboarding_step)))
    eyebrow, title, body, icon, tip = steps[index]
    _, center, _ = st.columns([1, 2.8, 1])
    with center:
        logo = f'<div class="lac-onboard-logo"><img src="{LOGO_URI}" alt="IMAGE A08 logo"></div>' if LOGO_URI else ""
        st.markdown(
            '<div class="lac-onboarding">' + logo +
            f'<div class="lac-eyebrow" style="text-align:center">{esc(eyebrow)}</div>' +
            f'<div style="text-align:center;font-size:32px;margin:.8rem 0 .2rem">{icon}</div>' +
            f'<h1 style="text-align:center;font-size:2rem;margin:.3rem 0 .7rem">{esc(title)}</h1>' +
            f'<p style="text-align:center;line-height:1.8;max-width:620px;margin:0 auto 1.1rem">{esc(body)}</p>' +
            f'<div class="lac-card" style="background:#f4f9f4;border-color:#dcebe0;text-align:center;font-size:12px;color:#4c6c58">{esc(tip)}</div></div>',
            unsafe_allow_html=True,
        )
        st.progress((index + 1) / 3)
        back, skip, next_col = st.columns([1, 1, 1.4])
        with back:
            if index > 0 and st.button("← Kembali", width="stretch"):
                st.session_state.onboarding_step -= 1
                st.rerun()
        with skip:
            if st.button("Lewati", width="stretch"):
                st.session_state.onboarding_done = True
                st.rerun()
        with next_col:
            if st.button("Mulai aplikasi →" if index == 2 else "Lanjut →", type="primary", width="stretch"):
                if index == 2:
                    st.session_state.onboarding_done = True
                else:
                    st.session_state.onboarding_step += 1
                st.rerun()
        st.markdown('<div class="lac-footer">LANDSCAPE AI CLASSIFIER · PANDUAN PENGGUNAAN</div>', unsafe_allow_html=True)


# First-visit splash. It is a short visual transition, not a blocking sleep.
if not st.session_state.splash_seen:
    st.session_state.splash_seen = True
    logo_html = f'<div class="lac-splash-logo"><img src="{LOGO_URI}" alt="IMAGE A08 logo"></div>' if LOGO_URI else ""
    st.markdown(
        f'<div class="lac-splash">{logo_html}<div class="lac-splash-name">Landscape AI Classifier</div><div class="lac-splash-sub">Deep Learning · Image Recognition</div></div>',
        unsafe_allow_html=True,
    )

if not st.session_state.onboarding_done:
    render_onboarding()
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
if "_nav_target" in st.session_state:
    st.session_state.nav_page = st.session_state.pop("_nav_target")
with st.sidebar:
    if LOGO_PATH.exists():
        left, right = st.columns([.72, 1.8], vertical_alignment="center")
        with left:
            st.image(str(LOGO_PATH), width=58)
        with right:
            st.markdown('<div class="lac-brand-name">Landscape AI</div><div class="lac-brand-sub">DEEP LEARNING CLASSIFIER</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="lac-brand-name">Landscape AI Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="lac-eyebrow" style="margin:.75rem 0 .4rem">NAVIGASI PROYEK</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigasi",
        ["Beranda", "Prediksi AI", "Arsitektur & Eksperimen", "Hasil Evaluasi", "Metodologi"],
        label_visibility="collapsed",
        key="nav_page",
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    statuses = model_file_status()
    for model_name, is_ready in statuses.items():
        color = "#176b4c" if is_ready else "#a47727"
        label = "Checkpoint tersedia" if is_ready else "Menunggu checkpoint"
        st.markdown(
            f'<div style="font-size:11px;margin:.45rem 0;color:#62766b"><b style="color:{color}">{"●" if is_ready else "○"} {esc(model_name)}</b><br><span style="font-size:10px">{label}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("↗ Lihat panduan penggunaan", width="stretch"):
        st.session_state.onboarding_done = False
        st.session_state.onboarding_step = 0
        st.rerun()
    st.markdown('<div style="font-size:10px;color:#91a198;line-height:1.7;margin-top:1rem">PROYEK DEEP LEARNING · UTS<br>Intel Image Classification<br>Custom CNN × MobileNetV2</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Page 1: Home / research dashboard
# -----------------------------------------------------------------------------
if page == "Beranda":
    render_hero(
        "DEEP LEARNING RESEARCH PROJECT",
        "Membaca pemandangan.\nMembandingkan cara AI belajar.",
        "Landscape AI Classifier adalah implementasi penelitian klasifikasi citra enam kelas yang membandingkan CNN kustom dengan transfer learning MobileNetV2. Halaman ini menghubungkan demonstrasi aplikasi dengan rancangan eksperimen dan evaluasi model.",
        ["Intel Image Classification", "Custom CNN", "MobileNetV2 · ImageNet"],
    )
    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kategori", "6", "Multikelas")
    m2.metric("Arsitektur", "2", "CNN & MobileNetV2")
    m3.metric("Skenario eksperimen", "4", "E1 sampai E4")
    m4.metric("Ukuran input", "150 × 150", "RGB")

    section("Fokus penelitian", "Bukan hanya mengenali isi gambar, tetapi membandingkan pendekatan pembelajaran model.")
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown(
            '<div class="lac-card-dark"><div class="lac-kicker" style="color:#b9e5c7">PERTANYAAN EKSPERIMEN</div>'
            '<h2 style="font-size:1.55rem;margin:.45rem 0 .7rem">Pendekatan mana yang lebih efektif?</h2>'
            '<p>Eksperimen membandingkan CNN yang belajar dari bobot awal acak dengan MobileNetV2 yang membawa fitur visual pretrained ImageNet. Pengaruh augmentasi dan fine-tuning juga diuji secara terpisah.</p>'
            '<div class="lac-pills"><span class="lac-pill">Accuracy</span><span class="lac-pill">Macro Precision</span><span class="lac-pill">Macro Recall</span><span class="lac-pill">Macro F1</span></div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="lac-card"><div class="lac-kicker">ALUR DEEP LEARNING</div><div class="lac-card-title">Dari citra ke keputusan model</div>'
            '<div class="lac-arch"><div class="lac-arch-step">Citra RGB · 150 × 150</div><div class="lac-arch-arrow">↓</div>'
            '<div class="lac-arch-step">Preprocessing & augmentasi training</div><div class="lac-arch-arrow">↓</div>'
            '<div class="lac-arch-step">Custom CNN / MobileNetV2</div><div class="lac-arch-arrow">↓</div>'
            '<div class="lac-arch-step">Dense classifier · Softmax 6 kelas</div><div class="lac-arch-arrow">↓</div>'
            '<div class="lac-arch-step">Prediksi · Evaluasi · Analisis error</div></div></div>',
            unsafe_allow_html=True,
        )

    section("Enam kelas target", "Kelas yang dipelajari model sesuai label pada Intel Image Classification.")
    class_html = '<div class="lac-class-grid">'
    for name, (emoji, description) in CLASS_META.items():
        class_html += f'<div class="lac-class-card"><div class="lac-class-emoji">{emoji}</div><div class="lac-class-name">{title_case(name)}</div><div style="font-size:10px;color:#849189;margin-top:.25rem">{esc(description)}</div></div>'
    class_html += "</div>"
    st.markdown(class_html, unsafe_allow_html=True)

    section("Dua pendekatan yang dibandingkan", "Arsitektur dan peran masing-masing model dalam penelitian.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="lac-model-box"><span class="lac-badge">Baseline · From scratch</span><div class="lac-model-name">Custom CNN</div><p class="lac-muted">Tiga blok convolution (32, 64, 128 filter), Batch Normalization, ReLU, MaxPooling, Global Average Pooling, Dense 128, Dropout, dan Softmax enam kelas.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="lac-model-box"><span class="lac-badge">Transfer learning</span><div class="lac-model-name">MobileNetV2</div><p class="lac-muted">Backbone pretrained ImageNet dengan include_top=False, Global Average Pooling, Dropout 0,30, Dense 128, dan Softmax enam kelas. Diuji melalui feature extraction serta fine-tuning.</p></div>', unsafe_allow_html=True)

    st.write("")
    a, b = st.columns(2)
    with a:
        if st.button("Buka studio prediksi →", type="primary", width="stretch"):
            st.session_state._nav_target = "Prediksi AI"
            st.rerun()
    with b:
        if st.button("Pelajari arsitektur & eksperimen →", width="stretch"):
            st.session_state._nav_target = "Arsitektur & Eksperimen"
            st.rerun()

# -----------------------------------------------------------------------------
# Page 2: AI inference
# -----------------------------------------------------------------------------
elif page == "Prediksi AI":
    render_hero(
        "MODEL INFERENCE STUDIO",
        "Satu citra.\nDua model Deep Learning.",
        "Unggah gambar untuk menjalankan inference menggunakan checkpoint Custom CNN dan MobileNetV2 hasil training Google Colab. Aplikasi menampilkan kelas prediksi dan distribusi skor enam kelas.",
        ["RGB · 150 × 150", "Custom CNN", "MobileNetV2"],
    )
    st.write("")
    statuses = model_file_status()
    ready_count = sum(statuses.values())
    col1, col2, col3 = st.columns(3)
    col1.metric("Checkpoint terpasang", f"{ready_count}/2", "Model hasil Colab")
    col2.metric("Kelas output", "6", "Softmax")
    col3.metric("Input model", "150 × 150", "3 kanal RGB")

    # Upload tetap ditampilkan walaupun checkpoint belum ada. Dengan demikian pengguna
    # dapat melihat preprocessing dan penjelasan jalur Deep Learning pada gambar input.
    if ready_count == 2:
        st.markdown('<div class="lac-status lac-status-ok"><div style="font-size:22px">✓</div><div><strong>Kedua checkpoint terpasang</strong><p>Custom CNN dan MobileNetV2 siap melakukan inference otomatis pada gambar yang sama.</p></div></div>', unsafe_allow_html=True)
    else:
        missing = [f"models/{MODEL_PATHS[name].name}" for name, ok in statuses.items() if not ok]
        st.markdown(
            '<div class="lac-status"><div style="font-size:22px">🧠</div><div><strong>Mode penjelasan aktif · checkpoint menunggu hasil Colab</strong>'
            f'<p>Area upload tetap aktif. Gambar yang diunggah di sini adalah satu citra uji untuk inference, bukan dataset training. Aplikasi akan menampilkan preprocessing dan menjelaskan jalur Custom CNN serta MobileNetV2. Prediksi numerik baru dijalankan setelah kedua bobot model hasil training tersedia: {esc(", ".join(missing))}.</p></div></div>',
            unsafe_allow_html=True,
        )

    section("1 · Masukkan citra uji", "Unggah satu gambar pemandangan. Input yang sama akan disiapkan untuk dua arsitektur dengan normalisasi internal yang berbeda.")
    uploaded = st.file_uploader(
        "Pilih gambar pemandangan (JPG / JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key=f"predict_upload_{st.session_state.upload_key}",
    )
    image_obj = None
    if uploaded is not None:
        try:
            image_obj = Image.open(uploaded).convert("RGB")
            current_signature = hashlib.sha256(uploaded.getvalue()).hexdigest()
            if st.session_state.get("last_prediction_signature") != current_signature:
                st.session_state.last_prediction = None
            preview, details = st.columns([1.05, .95], gap="large")
            with preview:
                st.image(image_obj, caption="Citra input untuk proses Deep Learning", width="stretch")
            with details:
                st.markdown(
                    '<div class="lac-card"><div class="lac-kicker">TAHAP A · INPUT & PREPROCESSING</div>'
                    '<div class="lac-card-title">Informasi tensor masukan</div>'
                    f'<p class="lac-muted"><b>Nama file:</b> {esc(uploaded.name)}<br>'
                    f'<b>Resolusi asli:</b> {image_obj.width} × {image_obj.height} piksel<br>'
                    '<b>Mode warna:</b> RGB (3 kanal)<br>'
                    '<b>Resize:</b> 150 × 150 piksel<br>'
                    '<b>Bentuk batch:</b> (1, 150, 150, 3)<br>'
                    '<b>Tipe data:</b> float32 · rentang awal 0–255</p>'
                    '<hr><div class="lac-kicker">NORMALISASI PER MODEL</div>'
                    '<p class="lac-muted"><b>Custom CNN:</b> Rescaling 1/255 → [0, 1].<br>'
                    '<b>MobileNetV2:</b> Rescaling 1/127,5 dengan offset −1 → [−1, 1].</p></div>',
                    unsafe_allow_html=True,
                )

            section("2 · Bagaimana gambar diproses oleh model?", "Penjelasan berikut mengikuti arsitektur yang benar-benar didefinisikan di notebook Colab, bukan prediksi atau angka simulasi.")
            cnn_col, mobile_col = st.columns(2, gap="large")
            with cnn_col:
                st.markdown(
                    '<div class="lac-model-box"><span class="lac-badge">MODEL BASELINE · FROM SCRATCH</span>'
                    '<div class="lac-model-name">Custom CNN</div>'
                    '<p class="lac-muted">Model belajar filter konvolusi dari bobot awal acak menggunakan gambar training Intel.</p>'
                    '<div class="lac-arch">'
                    '<div class="lac-arch-step">Input · 150 × 150 × 3</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">Rescaling · 1/255</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">Conv2D · 32 filter × 2</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">BatchNorm · ReLU · MaxPooling</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">Conv2D · 64 filter × 2</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">BatchNorm · ReLU · MaxPooling</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">Conv2D · 128 filter × 2</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">Global Average Pooling</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">Dense 128 · ReLU · Dropout 0,40</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">Softmax · 6 kelas</div></div></div>',
                    unsafe_allow_html=True,
                )
            with mobile_col:
                st.markdown(
                    '<div class="lac-model-box"><span class="lac-badge">MODEL UTAMA · TRANSFER LEARNING</span>'
                    '<div class="lac-model-name">MobileNetV2</div>'
                    '<p class="lac-muted">Backbone menggunakan bobot pretrained ImageNet. E3 membekukan backbone; E4 membuka 30 layer terakhir untuk fine-tuning (Batch Normalization tetap dibekukan).</p>'
                    '<div class="lac-arch">'
                    '<div class="lac-arch-step">Input · 150 × 150 × 3</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">Rescaling · [0,255] → [−1,1]</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">MobileNetV2 · ImageNet weights</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">Depthwise separable convolution</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">Inverted residual & linear bottleneck</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">Global Average Pooling</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">Dropout 0,30 · Dense 128 ReLU</div><div class="lac-arch-arrow">↓</div>'
                    '<div class="lac-arch-step">Softmax · 6 kelas</div></div></div>',
                    unsafe_allow_html=True,
                )

            section("3 · Hubungan dengan eksperimen Tahap 2", "Satu gambar input akan diuji menggunakan checkpoint model terbaik dari masing-masing keluarga arsitektur.")
            exp_cols = st.columns(4, gap="small")
            experiment_cards = [
                ("E1", "Custom CNN", "Tanpa augmentasi", "Maks. 30 epoch · Adam 0,001"),
                ("E2", "Custom CNN", "Dengan augmentasi", "Maks. 30 epoch · Adam 0,001"),
                ("E3", "MobileNetV2", "Feature extraction", "Backbone frozen · maks. 15 epoch"),
                ("E4", "MobileNetV2", "Fine-tuning", "30 layer terakhir · LR 0,00001"),
            ]
            for col, (code, model, treatment, detail) in zip(exp_cols, experiment_cards):
                with col:
                    st.markdown(
                        f'<div class="lac-experiment"><div class="lac-exp-code">{code}</div><div class="lac-exp-title">{model}</div><div class="lac-exp-meta">{treatment}<br>{detail}</div></div>',
                        unsafe_allow_html=True,
                    )

            if ready_count < 2:
                st.warning("Gambar sudah dapat diperiksa dan jalur model sudah dijelaskan. Untuk menghitung kelas serta skor Softmax yang nyata, jalankan notebook E1–E4 di Google Colab, lalu letakkan custom_cnn_best.keras dan mobilenetv2_best.keras di folder models/.")
            else:
                st.success("Kedua checkpoint tersedia. Tombol di bawah akan menjalankan inference sungguhan pada Custom CNN dan MobileNetV2.")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"File tidak dapat dibaca sebagai gambar: {exc}")
            image_obj = None
    else:
        st.markdown(
            '<div class="lac-step"><div class="lac-step-num">01</div><b>Upload sebuah gambar pemandangan</b>'
            '<p class="lac-muted">Setelah gambar dimasukkan, aplikasi akan menampilkan resolusi, konversi RGB, bentuk tensor, normalisasi, penjelasan lapisan Custom CNN, MobileNetV2, dan empat eksperimen E1–E4.</p></div>',
            unsafe_allow_html=True,
        )

    run = st.button("▶ Jalankan inference kedua model", type="primary", disabled=(image_obj is None or ready_count != 2), width="stretch")
    if run and image_obj is not None:
        try:
            outputs = {}
            with st.spinner("Memuat checkpoint dan menjalankan forward pass..."):
                for model_name, model_path in MODEL_PATHS.items():
                    keras_model = load_keras_model(str(model_path))
                    scores = predict_one(keras_model, image_obj)
                    idx = int(np.argmax(scores))
                    outputs[model_name] = {
                        "class_name": get_class_names()[idx],
                        "confidence": float(scores[idx]),
                        "probabilities": [float(x) for x in scores],
                    }
            st.session_state.last_prediction_signature = hashlib.sha256(uploaded.getvalue()).hexdigest()
            st.session_state.last_prediction = {
                "filename": uploaded.name,
                "created_at": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "image_bytes": uploaded.getvalue(),
                "class_names": get_class_names(),
                "predictions": outputs,
            }
        except ImportError:
            st.error("TensorFlow belum terpasang. Aktifkan environment model dan jalankan: py -m pip install -r requirements-models.txt")
        except Exception as exc:
            st.error(f"Inference gagal. Periksa versi TensorFlow/Keras dan kompatibilitas checkpoint. Detail: {exc}")

    payload = st.session_state.last_prediction
    if payload:
        section("Hasil inference", "Ini adalah keluaran aktual dari model yang tersimpan, bukan contoh atau angka ilustrasi.")
        first, second = st.columns(2, gap="large")
        for col, model_name, tag in ((first, "Custom CNN", "BASELINE · TRAINED FROM SCRATCH"), (second, "MobileNetV2", "TRANSFER LEARNING · IMAGENET")):
            item = payload["predictions"][model_name]
            with col:
                order = np.argsort(np.asarray(item["probabilities"]))[::-1]
                bars = []
                for idx in order:
                    score = max(0.0, min(100.0, float(item["probabilities"][int(idx)]) * 100))
                    bars.append(f'<div class="lac-prob-row"><span>{esc(title_case(payload["class_names"][int(idx)]))}</span><div class="lac-prob-track"><div class="lac-prob-fill" style="width:{score:.2f}%"></div></div><b style="text-align:right">{score:.1f}%</b></div>')
                st.markdown(
                    f'<div class="lac-model-box"><span class="lac-badge">{esc(tag)}</span><div class="lac-model-name" style="font-size:1.55rem;margin-top:.8rem">{esc(title_case(item["class_name"]))}</div>'
                    f'<div style="font:800 1.7rem Manrope;color:#176b4c">{item["confidence"] * 100:.2f}%</div><div class="lac-muted">Skor Softmax tertinggi</div><hr><div class="lac-kicker">DISTRIBUSI SKOR PER KELAS</div>{"".join(bars)}</div>',
                    unsafe_allow_html=True,
                )
        p1 = payload["predictions"]["Custom CNN"]
        p2 = payload["predictions"]["MobileNetV2"]
        if p1["class_name"] == p2["class_name"]:
            st.success(f"Kedua model menghasilkan kelas prediksi yang sama: **{title_case(p1['class_name'])}**.")
        else:
            st.warning(f"Prediksi berbeda. Custom CNN: **{title_case(p1['class_name'])}**; MobileNetV2: **{title_case(p2['class_name'])}**.")
        st.caption("Skor Softmax adalah keluaran relatif model. Skor tersebut tidak otomatis berarti tingkat kebenaran yang sudah terkalibrasi.")
        report_pdf = build_pdf(payload)
        safe_name = Path(payload["filename"]).stem.replace(" ", "_")[:40]
        dl, reset = st.columns(2)
        with dl:
            st.download_button("↓ Unduh laporan prediksi PDF", data=report_pdf, file_name=f"landscape_prediction_{safe_name}.pdf", mime="application/pdf", width="stretch")
        with reset:
            if st.button("↻ Analisis gambar lain", width="stretch"):
                st.session_state.last_prediction = None
                st.session_state.last_prediction_signature = None
                st.session_state.upload_key += 1
                st.rerun()

# -----------------------------------------------------------------------------
# Page 3: Architecture and experiment plan
# -----------------------------------------------------------------------------
elif page == "Arsitektur & Eksperimen":
    render_hero(
        "MODEL DESIGN & EXPERIMENTS",
        "Lihat bagaimana\nmodel dibangun dan diuji.",
        "Halaman ini menjelaskan komponen arsitektur, preprocessing yang tertanam di model, dan empat skenario eksperimen yang dijalankan pada notebook Google Colab.",
        ["Custom CNN", "MobileNetV2", "E1–E4"],
    )
    section("Arsitektur baseline · Custom CNN", "Model dibangun dari awal sehingga bobot convolution dipelajari dari dataset Intel.")
    c1, c2 = st.columns([1.05, .95], gap="large")
    with c1:
        st.markdown('<div class="lac-card"><span class="lac-badge">Baseline · From scratch</span><div class="lac-card-title">Custom Convolutional Neural Network</div><p class="lac-muted">Tiga blok convolution mengekstraksi fitur secara bertingkat. Setiap blok memiliki dua Conv2D 3×3, Batch Normalization, aktivasi ReLU, MaxPooling, dan Dropout.</p></div>', unsafe_allow_html=True)
        render_model_architecture("Custom CNN")
    with c2:
        st.markdown('<div class="lac-kicker">KONFIGURASI CUSTOM CNN</div><div class="lac-card-title">Parameter arsitektur</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            ["Input", "150 × 150 × 3"], ["Filter convolution", "32 → 64 → 128"], ["Kernel", "3 × 3"],
            ["Aktivasi", "ReLU"], ["Normalisasi", "Batch Normalization + Rescaling 1/255"],
            ["Pooling", "MaxPooling 2 × 2"], ["Classifier", "GAP → Dense 128 → Dropout 0,40"],
            ["Output", "Softmax · 6 kelas"], ["Optimizer", "Adam · LR 0,001"],
        ], columns=["Komponen", "Konfigurasi"]), width="stretch", hide_index=True)

    section("Arsitektur utama · MobileNetV2", "Memanfaatkan representasi visual ImageNet dan diuji melalui dua tahap transfer learning.")
    m1, m2 = st.columns([1.05, .95], gap="large")
    with m1:
        st.markdown('<div class="lac-card"><span class="lac-badge">Transfer learning · ImageNet</span><div class="lac-card-title">MobileNetV2</div><p class="lac-muted">Backbone menggunakan bobot ImageNet dengan include_top=False. Bagian classifier bawaan diganti dengan Global Average Pooling, Dropout 0,30, Dense 128 ReLU, dan output Softmax enam kelas.</p></div>', unsafe_allow_html=True)
        render_model_architecture("MobileNetV2")
    with m2:
        st.markdown('<div class="lac-card"><div class="lac-kicker">DUA FASE TRANSFER LEARNING</div><div class="lac-card-title">Feature extraction vs fine-tuning</div><b>Fase E3 · Feature extraction</b><p class="lac-muted">Seluruh backbone MobileNetV2 dibekukan. Hanya classifier baru yang dilatih dengan Adam, learning rate 0,001, maksimal 15 epoch.</p><hr><b>Fase E4 · Fine-tuning</b><p class="lac-muted">Dimulai dari checkpoint terbaik E3. Tiga puluh layer terakhir backbone dibuka kecuali Batch Normalization, lalu model di-compile ulang dengan Adam, learning rate 0,00001, maksimal 10 epoch.</p></div>', unsafe_allow_html=True)

    section("Matriks eksperimen E1–E4", "Perbandingan dirancang agar pengaruh augmentasi dan transfer learning dapat dianalisis.")
    exp_cols = st.columns(4, gap="small")
    for col, (code, meta) in zip(exp_cols, EXPERIMENTS.items()):
        with col:
            st.markdown(
                f'<div class="lac-experiment"><div class="lac-exp-code">{code.split("_")[0]}</div><div class="lac-exp-title">{esc(meta["model"])}</div>'
                f'<div class="lac-exp-meta"><b>Perlakuan</b><br>{esc(meta["treatment"])}<br><br><b>Maks. epoch</b>: {esc(meta["epochs"])}<br><b>Learning rate</b>: {esc(meta["lr"])}</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(meta["purpose"])
    st.info("Semua eksperimen memakai batch size 32, input RGB 150 × 150, label enam kelas, split stratified 80:20 dari seg_train, dan test set seg_test yang disimpan untuk evaluasi akhir. Early stopping dan checkpoint menggunakan validation loss.")

# -----------------------------------------------------------------------------
# Page 4: Actual performance artefacts
# -----------------------------------------------------------------------------
elif page == "Hasil Evaluasi":
    render_hero(
        "EXPERIMENT RESULTS & MODEL EVALUATION",
        "Kinerja model harus\ndibuktikan dengan data.",
        "Halaman ini membaca metrik test, riwayat training, confusion matrix, dan pemilihan model yang dihasilkan notebook. Sebelum artefak tersedia, rancangan eksperimen ditampilkan tanpa mengarang angka.",
        ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
    )
    files = result_file_status()
    metrics_path = RESULTS / "test_metrics_all_experiments.csv"
    validation_path = RESULTS / "validation_metrics.csv"
    curves_path = RESULTS / "training_curves_all_experiments.png"
    cm_path = RESULTS / "confusion_matrix_best_model.png"
    selection_path = RESULTS / "model_selection.json"
    summary_path = RESULTS / "experiment_summary.json"
    history_paths = {code: RESULTS / f"{code}_history.csv" for code in EXPERIMENTS}
    available_histories = {code: path for code, path in history_paths.items() if path.exists()}
    summary_data = {}
    if summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary_data = {}

    status_cols = st.columns(4)
    status_cols[0].metric("File metrik test", "Tersedia" if files["test_metrics_all_experiments.csv"] else "Belum ada")
    status_cols[1].metric("Riwayat training", f"{len(available_histories)}/4", "CSV per eksperimen")
    status_cols[2].metric("Training curves", "Tersedia" if files["training_curves_all_experiments.png"] else "Belum ada")
    status_cols[3].metric("Confusion matrix", "Tersedia" if files["confusion_matrix_best_model.png"] else "Belum ada")

    section("Seleksi model pada validation set", "Validation Macro F1 digunakan untuk memilih checkpoint; test set tidak dipakai untuk memilih model.")
    if validation_path.exists():
        try:
            validation_df = pd.read_csv(validation_path)
            validation_view = validation_df.copy()
            if "experiment" in validation_view.columns:
                validation_view["experiment"] = validation_view["experiment"].map(lambda x: EXPERIMENTS.get(str(x), {}).get("short", str(x)))
            st.dataframe(validation_view, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"File validation metrics tidak dapat dibaca: {exc}")
    else:
        st.info("Validation metrics akan muncul setelah empat checkpoint dievaluasi pada validation set di Colab. Pemilihan model dilakukan melalui validation Macro F1.")

    section("Perbandingan metrik test", "Metrik ini dihasilkan setelah empat eksperimen dievaluasi pada test set yang sama.")
    if metrics_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_path)
            required = {"experiment", "accuracy", "macro_precision", "macro_recall", "macro_f1"}
            if required.issubset(metrics_df.columns):
                display_df = metrics_df.copy()
                display_df["experiment"] = display_df["experiment"].map(lambda x: EXPERIMENTS.get(str(x), {}).get("short", str(x)))
                metric_key = st.selectbox("Pilih metrik visualisasi", ["accuracy", "macro_precision", "macro_recall", "macro_f1"], index=3, format_func=lambda x: {"accuracy": "Accuracy", "macro_precision": "Macro Precision", "macro_recall": "Macro Recall", "macro_f1": "Macro F1-score"}[x])
                st.bar_chart(display_df.set_index("experiment")[[metric_key]], width="stretch")
                st.dataframe(display_df, width="stretch", hide_index=True)
            else:
                st.warning("Format CSV belum sesuai. Kolom yang diperlukan: experiment, accuracy, macro_precision, macro_recall, macro_f1.")
        except Exception as exc:
            st.error(f"File metrik tidak dapat dibaca: {exc}")
    else:
        st.markdown('<div class="lac-status"><div style="font-size:22px">📈</div><div><strong>Belum ada angka hasil training</strong><p>Jalankan notebook Colab sampai evaluasi test. Setelah itu salin <b>results/test_metrics_all_experiments.csv</b> ke folder results aplikasi. Nilai accuracy dan Macro F1 tidak dibuat sebagai contoh agar laporan tetap valid secara akademis.</p></div></div>', unsafe_allow_html=True)

    section("Kurva pembelajaran", "Riwayat accuracy dan loss train/validation digunakan untuk mengamati konvergensi serta indikasi overfitting.")
    if curves_path.exists():
        st.image(str(curves_path), width="stretch")
    elif available_histories:
        choice = st.selectbox("Pilih eksperimen untuk melihat training history", list(available_histories), format_func=lambda x: EXPERIMENTS[x]["short"])
        history_df = pd.read_csv(available_histories[choice])
        if "epoch" not in history_df.columns:
            history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
        if {"accuracy", "val_accuracy"}.issubset(history_df.columns):
            st.markdown("**Accuracy**")
            st.line_chart(history_df.set_index("epoch")[["accuracy", "val_accuracy"]], width="stretch")
        if {"loss", "val_loss"}.issubset(history_df.columns):
            st.markdown("**Loss**")
            st.line_chart(history_df.set_index("epoch")[["loss", "val_loss"]], width="stretch")
        st.dataframe(history_df, width="stretch", hide_index=True)
    else:
        st.info("CSV riwayat training belum ditemukan. Notebook menyimpan satu file history untuk setiap eksperimen E1, E2, E3, dan E4 ke folder results.")

    section("Confusion matrix", "Membaca kelas yang tepat dan kelas yang sering tertukar pada model terbaik yang dipilih dari validation Macro F1.")
    if cm_path.exists():
        st.image(str(cm_path), width="stretch")
    else:
        st.info("Confusion matrix akan tampil setelah notebook menyelesaikan evaluasi test dan menyimpan confusion_matrix_best_model.png.")

    if selection_path.exists():
        try:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            section("Pemilihan model", "Model final dipilih menggunakan validation Macro F1, bukan berdasarkan test set.")
            chosen = selection.get("best_overall_experiment", "Belum tercatat")
            st.success(f"Model terbaik berdasarkan validation Macro F1: **{EXPERIMENTS.get(chosen, {}).get('short', chosen)}**")
            st.json(selection)
        except (OSError, json.JSONDecodeError):
            st.warning("File model_selection.json tidak dapat dibaca.")

    if summary_data:
        with st.expander("Informasi run eksperimen dari Colab"):
            st.json({k: v for k, v in summary_data.items() if k not in {"test_metrics"}})

# -----------------------------------------------------------------------------
# Page 5: methodology and dataset
# -----------------------------------------------------------------------------
elif page == "Metodologi":
    render_hero(
        "RESEARCH METHODOLOGY",
        "Bagaimana data\ndiproses dan diuji.",
        "Rangkuman metodologi yang menjadi dasar notebook: sumber data, pembagian dataset, preprocessing, augmentasi, optimasi, dan metrik evaluasi.",
        ["Supervised learning", "Stratified split", "Reproducible experiment"],
    )
    section("Dataset penelitian", "Intel Image Classification menyediakan enam kelas citra pemandangan dan lingkungan.")
    d1, d2, d3 = st.columns(3)
    d1.metric("Kelas target", "6", "Multiclass classification")
    d2.metric("Resolusi input", "150 × 150", "RGB")
    d3.metric("Folder supervised", "seg_train + seg_test", "seg_pred tidak untuk evaluasi")
    st.markdown('<div class="lac-card"><div class="lac-card-title">Sumber dataset</div><p class="lac-muted">Intel Image Classification — Kaggle. Notebook mengaudit file pada seg_train dan seg_test, memeriksa gambar yang tidak dapat dibaca, dan mendeteksi hash identik lintas split untuk mengurangi risiko exact-duplicate leakage.</p><a href="https://www.kaggle.com/datasets/puneet6060/intel-image-classification" target="_blank">Buka dataset di Kaggle ↗</a></div>', unsafe_allow_html=True)

    section("Distribusi kelas hasil audit", "Jika notebook sudah dijalankan, aplikasi akan membaca jumlah gambar aktual setelah pemeriksaan file rusak dan duplikasi.")
    class_summary_path = RESULTS / "dataset_class_summary.csv"
    if class_summary_path.exists():
        try:
            class_df = pd.read_csv(class_summary_path, index_col=0)
            st.bar_chart(class_df, width="stretch")
            st.dataframe(class_df, width="stretch")
        except Exception as exc:
            st.warning(f"Ringkasan distribusi kelas tidak dapat dibaca: {exc}")
    else:
        st.info("Distribusi kelas aktual belum tersedia. Notebook akan menghasilkan dataset_class_summary.csv setelah audit dataset selesai.")

    if summary_data := (json.loads((RESULTS / "experiment_summary.json").read_text(encoding="utf-8")) if (RESULTS / "experiment_summary.json").exists() else {}):
        split_summary = {key: summary_data.get(key) for key in ("train_count", "validation_count", "test_count") if summary_data.get(key) is not None}
        if split_summary:
            st.markdown("**Jumlah data setelah audit dan split**")
            q1, q2, q3 = st.columns(3)
            q1.metric("Training", f"{split_summary.get('train_count', 0):,}")
            q2.metric("Validation", f"{split_summary.get('validation_count', 0):,}")
            q3.metric("Testing", f"{split_summary.get('test_count', 0):,}")

    section("Pembagian data dan preprocessing", "Test set asli tetap terpisah dan tidak dipakai untuk tuning model.")
    split_cols = st.columns(3)
    split_cols[0].markdown('<div class="lac-step"><div class="lac-step-num">01</div><b>Training · 80%</b><p class="lac-muted">Diambil dari seg_train setelah audit, menggunakan stratified split.</p></div>', unsafe_allow_html=True)
    split_cols[1].markdown('<div class="lac-step"><div class="lac-step-num">02</div><b>Validation · 20%</b><p class="lac-muted">Digunakan untuk early stopping, pemilihan checkpoint, dan seleksi model.</p></div>', unsafe_allow_html=True)
    split_cols[2].markdown('<div class="lac-step"><div class="lac-step-num">03</div><b>Testing · seg_test</b><p class="lac-muted">Digunakan sekali untuk evaluasi akhir setelah konfigurasi dikunci.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="lac-card"><div class="lac-card-title">Pipeline input</div><p class="lac-muted">Baca file → decode sebagai RGB → resize 150 × 150 → cast float32 dengan rentang pixel 0–255 → batch 32 → prefetch. Normalisasi spesifik arsitektur dilakukan sebagai layer di dalam model.</p><ul><li><b>Custom CNN:</b> Rescaling 1/255 sehingga pixel berada pada rentang 0–1.</li><li><b>MobileNetV2:</b> Rescaling 1/127,5 dengan offset −1 sehingga pixel berada pada rentang −1 sampai 1.</li></ul></div>', unsafe_allow_html=True)

    section("Augmentasi data training", "Augmentasi hanya diterapkan pada pipeline training, bukan validation atau testing.")
    aug = st.columns(4)
    for col, title, detail in zip(aug, ["Horizontal flip", "Rotation", "Zoom", "Contrast"], ["Membalik citra secara horizontal.", "RandomRotation 0,05.", "RandomZoom 0,10.", "RandomContrast 0,10."]):
        with col:
            st.markdown(f'<div class="lac-step"><div class="lac-step-num">✦</div><b>{esc(title)}</b><p class="lac-muted">{esc(detail)}</p></div>', unsafe_allow_html=True)

    section("Konfigurasi training", "Nilai berikut mengikuti notebook eksperimen.")
    st.dataframe(pd.DataFrame([
        ["Random seed", "42"], ["Batch size", "32"], ["Loss", "Sparse Categorical Crossentropy"],
        ["Optimizer", "Adam"], ["E1–E2 learning rate", "0,001"], ["E3 learning rate", "0,001"],
        ["E4 learning rate", "0,00001"], ["Callback", "ModelCheckpoint · EarlyStopping · ReduceLROnPlateau"],
        ["Seleksi model", "Validation Macro F1"], ["Metrik akhir", "Accuracy · Macro Precision · Macro Recall · Macro F1"],
    ], columns=["Komponen", "Konfigurasi"]), width="stretch", hide_index=True)

    section("Artefak yang dibaca aplikasi", "Salin hasil dari Google Drive landscape_dl_project ke folder aplikasi yang sesuai.")
    st.code(
        "landscape_ai_classifier_ui/\n"
        "├── models/\n"
        "│   ├── custom_cnn_best.keras\n"
        "│   └── mobilenetv2_best.keras\n"
        "└── results/\n"
        "    ├── class_names.json\n"
        "    ├── test_metrics_all_experiments.csv\n"
        "    ├── validation_metrics.csv\n"
        "    ├── training_curves_all_experiments.png\n"
        "    ├── confusion_matrix_best_model.png\n"
        "    ├── model_selection.json\n"
        "    └── experiment_summary.json",
        language="text",
    )
    st.caption("Notebook Colab juga menyimpan riwayat terpisah E1–E4 dengan pola nama <experiment>_history.csv.")

st.markdown('<div class="lac-footer">LANDSCAPE AI CLASSIFIER · INTEL IMAGE CLASSIFICATION · CUSTOM CNN × MOBILENETV2</div>', unsafe_allow_html=True)
