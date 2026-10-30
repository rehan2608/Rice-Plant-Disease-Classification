"""F2.py — Compact Rice Disease Classification Frontend.
Run with:  streamlit run F2.py  (or click 'Run' directly in IDE)
"""

from streamlit.runtime import exists

# ── Self-Launching Wrapper: Clicking 'Run' opens the website ──
if not exists():
    import os, sys, subprocess
    check = subprocess.run(["lsof", "-i", ":8501"], capture_output=True, text=True)
    os.system("open http://localhost:8501 2>/dev/null")
    if "LISTEN" in check.stdout:
        print("🚀 Opening http://localhost:8501 in your browser...")
        sys.exit(0)
    else:
        print("🚀 Starting Streamlit and opening http://localhost:8501...")
        os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", __file__, "--server.port=8501"])

import io, json, time
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st
from PIL import Image
from backend.service import get_service, DISEASE_KNOWLEDGE_BASE

# ── Config & Styling ──────────────────────────────────────────
st.set_page_config(page_title="🌾 Rice Disease AI", page_icon="🌾", layout="wide")
st.markdown("""<style>
.big{font-size:2.2rem;font-weight:900;color:#065f46;margin:0}
.sub{color:#6b7280;margin-bottom:1rem}
.card{background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:1rem;margin-bottom:.8rem}
</style>""", unsafe_allow_html=True)

service = get_service()

# ── Header ───────────────────────────────────────────────────
st.markdown('<p class="big">🌾 Rice Plant Disease Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">7,168-D Feature Fusion (Color Correlogram + MCLBP L3) & OvR SVM · Nature Sci Rep (2024)</p>', unsafe_allow_html=True)

# ── Sidebar: Model Benchmarks ────────────────────────────────
with st.sidebar:
    st.header("📊 Model Performance")
    try:
        metrics_path = Path(__file__).resolve().parent / "outputs/evaluation/test_metrics.json"
        if not metrics_path.exists():
            metrics_path = Path("outputs/evaluation/test_metrics.json")
        with open(metrics_path) as f:
            m = json.load(f)
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{m['accuracy']:.1f}%")
        c2.metric("Macro F1", f"{m['macro_f1']:.1f}%")
        c1.metric("Balanced Acc", f"{m['balanced_accuracy']:.1f}%")
        c2.metric("MCC", f"{m['macro_mcc']:.1f}%")

        st.divider()
        st.subheader("🔢 Confusion Matrix")
        classes = m["classes"]
        st.dataframe(pd.DataFrame(np.array(m["confusion_matrix"]), index=classes, columns=classes), use_container_width=True)

        st.divider()
        st.subheader("📋 Per-Class Metrics")
        rows = [{"Class": f"{c} ({DISEASE_KNOWLEDGE_BASE[c]['full_name']})", "Prec%": round(m["per_class"][c]["precision"], 1), "Rec%": round(m["per_class"][c]["recall"], 1), "F1%": round(m["per_class"][c]["f1_score"], 1)} for c in classes]
        st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)
    except Exception as e:
        st.warning(f"Metrics unavailable: {e}")
    st.caption("7,168-D features (256 CC + 6,912 MCLBP L3) · OvR RBF SVM")

# ── Main Layout ──────────────────────────────────────────────
left, right = st.columns([1, 1.3], gap="large")

with left:
    mode = st.radio("Source:", ["🗂️ Test Sample", "📤 Upload Image"], horizontal=True)
    img_bytes, src_name = None, None

    if mode == "🗂️ Test Sample":
        samples = service.list_available_samples()
        c = st.selectbox("Disease class:", list(samples.keys()), format_func=lambda k: f"{k} — {DISEASE_KNOWLEDGE_BASE[k]['full_name']}")
        if samples.get(c):
            samp = st.selectbox("Image:", samples[c], format_func=lambda s: s["filename"])
            p = service.get_sample_image_path(c, samp["filename"])
            if p:
                with open(p, "rb") as fp:
                    img_bytes = fp.read()
                src_name = f"{c}/{samp['filename']}"
    else:
        f = st.file_uploader("Drop a rice leaf image", type=["jpg", "jpeg", "png"])
        if f:
            img_bytes, src_name = f.read(), f.name

    if img_bytes:
        st.image(Image.open(io.BytesIO(img_bytes)), caption=src_name, use_container_width=True)

with right:
    if not img_bytes:
        st.info("👈 Select a sample or upload an image to classify.")
    else:
        with st.spinner("Extracting features & classifying…"):
            res = service.classify_image(img_bytes, source_name=src_name)

        if not res.get("success"):
            st.error(res.get("message", "Classification failed."))
        else:
            p, r, d, t = res["prediction"], res["ranked_classes"], res["disease_details"], res["timings"]

            # Diagnosis Card
            st.markdown(f"""<div class="card">
            <span style="background:#065f46;color:#fff;padding:2px 8px;border-radius:4px;font-weight:700">{p['class']}</span>
            <span style="float:right;color:#059669;font-size:1.6rem;font-weight:800">{p['confidence_percent']:.1f}%</span>
            <h2 style="color:#065f46;margin:.2rem 0 0">{p['full_name']}</h2>
            <p style="color:#6b7280;font-style:italic;margin:0">{p['scientific_name']}</p>
            <p style="margin:.3rem 0 0;font-size:.85rem"><b>Pathogen:</b> {p['pathogen_type']} · <b>Severity:</b> {p['severity_level']}</p>
            </div>""", unsafe_allow_html=True)

            # Probabilities Bar Chart
            st.bar_chart(pd.DataFrame({"Disease": [f"{x['full_name']} ({x['class']})" for x in r], "Confidence %": [x["confidence_percent"] for x in r]}), x="Disease", y="Confidence %", color="#10b981", horizontal=True)

            # Disease Tabs
            t1, t2, t3 = st.tabs(["🔬 Symptoms", "🌦️ Conditions", "🛡️ Management"])
            with t1:
                for s in d.get("primary_symptoms", []): st.markdown(f"- {s}")
            with t2:
                st.write(d.get("favorable_conditions", "N/A"))
            with t3:
                m = d.get("management", {})
                for k, icon in [("cultural","🌾"), ("chemical","🧪"), ("biological","🦠"), ("resistant_cultivars","🌱")]:
                    st.markdown(f"**{icon} {k.replace('_',' ').title()}:** {m.get(k, 'N/A')}")

            # Latency & Download
            st.caption(f"⏱ Latency: {t['total_ms']}ms (Pre: {t.get('preprocessing_ms','-')}ms, Feat: {t.get('feature_extraction_ms','-')}ms, Inf: {t.get('inference_ms','-')}ms) · Dims: {res['feature_specs']['total_dimensions']}")
            st.download_button("📥 Download Report (JSON)", json.dumps(res, indent=2), f"diagnosis_{p['class']}_{int(time.time())}.json", "application/json")
