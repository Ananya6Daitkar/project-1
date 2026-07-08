"""
app.py — Streamlit crater detection web app
============================================
Upload a lunar/Mars surface image → get back:
  • Annotated image with detected craters + confidence scores
  • Before/after comparison
  • Crater count + size distribution
  • Confidence heatmap
  • Live IoT telemetry panel (simulated satellite feed)

Run:
    streamlit run app/app.py
"""

import sys
import time
import json
import threading
from pathlib import Path
from io import BytesIO
from typing import Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.error(
        "⚠ OpenCV (cv2) not installed. "
        "Make sure `opencv-python-headless` is in requirements.txt "
        "and `packages.txt` has the system libraries."
    )
    st.stop()

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import (
    PROJECT_ROOT, MODELS_DIR, ASSETS_DIR, CLASS_NAMES, CLASS_COLORS,
    draw_detections, yolo_to_xyxy, plot_class_distribution
)
from iot_simulator import (
    start_iot_simulation, stop_iot_simulation, get_ground_station,
    emit_detection_event, PLANETARY_TARGETS
)

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌑 Crater Detector",
    page_icon="🌑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0d1117; color: #e6edf3; }
    .stApp { background-color: #0d1117; }

    /* make all headings bold and white */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700 !important;
        color: #e6edf3 !important;
    }
    /* streamlit markdown headings */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-weight: 700 !important;
        color: #e6edf3 !important;
    }
    /* tab headings */
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .alert-box {
        background: #3d1a1a;
        border: 1px solid #f85149;
        border-radius: 6px;
        padding: 10px;
        color: #f85149;
    }
    .safe-box {
        background: #1a3d2a;
        border: 1px solid #3fb950;
        border-radius: 6px;
        padding: 10px;
        color: #3fb950;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

MODEL_OPTIONS = {
    "YOLOv8n — Fast baseline (3.2M params)":  "yolov8n",
    "YOLOv8s — Higher accuracy (11.2M params)": "yolov8s",
}


def model_weights_exist(model_name: str) -> bool:
    w = MODELS_DIR / f"{model_name}_crater" / "weights" / "best.pt"
    return w.exists()


@st.cache_resource
def load_yolo_model(model_name: str):
    """Load YOLO model — cached across sessions."""
    try:
        from ultralytics import YOLO
        weights = MODELS_DIR / f"{model_name}_crater" / "weights" / "best.pt"
        return YOLO(str(weights))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def run_detection(model, img_array: np.ndarray, conf: float):
    """Run YOLO on numpy image. Returns (boxes_xyxy, classes, confidences, ms)."""
    t0 = time.perf_counter()
    results = model(img_array, conf=conf, verbose=False)
    ms = (time.perf_counter() - t0) * 1000

    boxes, classes, confs = [], [], []
    H, W = img_array.shape[:2]
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            boxes.append((x1, y1, x2, y2))
            classes.append(int(box.cls[0]))
            confs.append(float(box.conf[0]))

    return boxes, classes, confs, ms


def make_confidence_heatmap_array(
    img_bgr: np.ndarray, boxes, confs
) -> np.ndarray:
    """Return BGR heatmap overlay as numpy array."""
    H, W = img_bgr.shape[:2]
    heatmap = np.zeros((H, W), dtype=np.float32)

    for (x1, y1, x2, y2), conf_val in zip(boxes, confs):
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        radius = max((x2 - x1), (y2 - y1)) // 2 or 10
        k_size = max(radius * 2 + 1, 5)
        k_size = k_size + (1 - k_size % 2)
        sigma  = radius / 2 or 5
        kernel = cv2.getGaussianKernel(int(k_size), sigma)
        k2d    = kernel @ kernel.T * conf_val

        ky, kx = k2d.shape
        y0, x0 = cy - ky // 2, cx - kx // 2
        ys = max(0, y0);  ye = min(H, y0 + ky)
        xs = max(0, x0);  xe = min(W, x0 + kx)
        heatmap[ys:ye, xs:xe] += k2d[ys-y0:ys-y0+(ye-ys), xs-x0:xs-x0+(xe-xs)]

    heat_u8 = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.55, heat_color, 0.45, 0)


def np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/FullMoon2010.jpg/600px-FullMoon2010.jpg",
             use_container_width=True)
    st.markdown("## 🌑 Crater Detector")
    st.markdown("---")

    selected_model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    model_name = MODEL_OPTIONS[selected_model_label]

    conf_threshold = st.slider(
        "Confidence threshold", min_value=0.10, max_value=0.90,
        value=0.30, step=0.05
    )

    planet_target = st.selectbox("Planet (IoT context)", PLANETARY_TARGETS)

    st.markdown("---")
    st.markdown("### IoT Simulation")
    iot_running = st.toggle("Live telemetry stream", value=False)
    iot_interval = st.slider("Telemetry interval (s)", 0.5, 5.0, 2.0, 0.5)

    if iot_running:
        gs = start_iot_simulation(target=planet_target, interval_sec=iot_interval)
    else:
        stop_iot_simulation()

    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown("PyTorch · Ultralytics YOLOv8 · OpenCV · Streamlit · MQTT (sim)")
    st.caption("v1.0 — Crater Detection Pipeline")


# ─────────────────────────────────────────────────────────────────
# Main layout — tabs
# ─────────────────────────────────────────────────────────────────

tab_detect, tab_compare, tab_iot, tab_about = st.tabs([
    "🔭 Detect Craters",
    "📊 Model Comparison",
    "📡 IoT Dashboard",
    "ℹ About",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Detection
# ═══════════════════════════════════════════════════════════════════
with tab_detect:
    st.markdown("<h3 style='font-weight:700;color:#ffffff;'>Upload a Planetary Surface Image</h3>", unsafe_allow_html=True)
    st.markdown(
        "Supported: PNG, JPG, TIFF. Works best with grayscale orbital imagery "
        "(LRO NAC / HiRISE / CTX tiles at 256–1024 px)."
    )

    uploaded = st.file_uploader(
        "Choose image", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

    # Sample images
    col_sample1, col_sample2, col_sample3 = st.columns(3)
    sample_dir = ASSETS_DIR
    sample_images = list(sample_dir.glob("*.png"))[:3] if sample_dir.exists() else []

    use_sample = None
    if sample_images:
        st.markdown("**Or use a sample image:**")
        s_cols = st.columns(min(len(sample_images), 3))
        for i, (col, sp) in enumerate(zip(s_cols, sample_images)):
            with col:
                if st.button(f"Sample {i+1}: {sp.stem[:20]}", key=f"sample_{i}"):
                    use_sample = sp

    # ── Inference ─────────────────────────────────────────────────
    if uploaded or use_sample:
        if not model_weights_exist(model_name):
            st.error(
                f"⚠ Model weights not found for **{model_name}**. "
                "Please train the model first:\n```\n"
                f"python src/train.py --model {model_name} --epochs 30\n```"
            )
            st.stop()

        # Load image
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error("Could not decode image. Try a different file.")
                st.stop()
        else:
            img_bgr = cv2.imread(str(use_sample))

        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        model = load_yolo_model(model_name)
        if model is None:
            st.stop()

        with st.spinner(f"Running {model_name} inference …"):
            boxes, classes, confs, ms = run_detection(model, img_bgr, conf_threshold)

        # ── Metrics banner ─────────────────────────────────────
        class_counts = {name: 0 for name in CLASS_NAMES.values()}
        for cls in classes:
            class_counts[CLASS_NAMES[cls]] += 1

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🔭 Total Craters", len(boxes))
        m2.metric("🔴 Small",  class_counts["small_crater"])
        m3.metric("🟢 Medium", class_counts["medium_crater"])
        m4.metric("🔵 Large",  class_counts["large_crater"])
        m5.metric("⚡ Inference", f"{ms:.1f} ms")

        # Landing zone safety assessment
        if len(boxes) > 12:
            st.markdown(
                '<div class="alert-box">⚠ HIGH CRATER DENSITY — Not recommended as landing zone</div>',
                unsafe_allow_html=True
            )
        elif len(boxes) > 0:
            st.markdown(
                '<div class="safe-box">✅ MODERATE CRATER DENSITY — Potential landing zone candidate</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="safe-box">✅ NO CRATERS DETECTED — Clean terrain</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ── Image panels ───────────────────────────────────────
        col_orig, col_ann = st.columns(2)

        annotated = draw_detections(img_bgr.copy(), boxes, classes, confs)
        heatmap   = make_confidence_heatmap_array(img_bgr, boxes, confs)

        with col_orig:
            st.markdown("**Original Image**")
            st.image(np_to_pil(img_bgr), use_container_width=True)

        with col_ann:
            st.markdown(f"**Detected Craters ({len(boxes)} total)**")
            st.image(np_to_pil(annotated), use_container_width=True)
            st.download_button(
                "⬇ Download annotated image",
                data=pil_to_bytes(np_to_pil(annotated)),
                file_name="crater_detections.png",
                mime="image/png",
            )

        col_heat, col_dist = st.columns(2)
        with col_heat:
            st.markdown("**Confidence Heatmap**")
            st.image(np_to_pil(heatmap), use_container_width=True)

        with col_dist:
            st.markdown("**Size Distribution**")
            if boxes:
                fig_pie = px.pie(
                    names=list(class_counts.keys()),
                    values=list(class_counts.values()),
                    color_discrete_sequence=["#e74c3c", "#2ecc71", "#3498db"],
                    hole=0.4,
                )
                fig_pie.update_layout(
                    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                    font_color="white", margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No craters detected with current threshold.")

        # ── Detection table ────────────────────────────────────
        if boxes:
            st.markdown("<h3 style='font-weight:700;color:#ffffff;'>Detection Details</h3>", unsafe_allow_html=True)
            import pandas as pd
            H, W = img_bgr.shape[:2]
            table_data = []
            for i, ((x1, y1, x2, y2), cls, conf) in enumerate(zip(boxes, classes, confs)):
                w_px = x2 - x1
                h_px = y2 - y1
                table_data.append({
                    "#":         i + 1,
                    "Class":     CLASS_NAMES[cls].replace("_", " ").title(),
                    "Conf":      f"{conf:.2f}",
                    "Centre X":  f"{(x1+x2)//2}",
                    "Centre Y":  f"{(y1+y2)//2}",
                    "Width px":  w_px,
                    "Height px": h_px,
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True,
                         hide_index=True)

        # ── Emit IoT event ─────────────────────────────────────
        if boxes:
            crater_class_list = [CLASS_NAMES[c] for c in classes]
            avg_conf = float(np.mean(confs)) if confs else 0.0
            payload = emit_detection_event(
                n_craters=len(boxes),
                crater_classes=crater_class_list,
                avg_conf=avg_conf,
                inference_ms=ms,
                tile_id=uploaded.name if uploaded else "sample",
                target=planet_target,
            )
            with st.expander("📡 IoT Telemetry Payload (last event)", expanded=False):
                st.json(json.loads(payload.to_json()))


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ═══════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("<h3 style='font-weight:700;color:#ffffff;'>Model Comparison — YOLOv8n vs YOLOv8s</h3>", unsafe_allow_html=True)

    # Load saved eval results
    comparison_data = []
    for mname in ["yolov8n", "yolov8s"]:
        eval_path = MODELS_DIR / f"{mname}_crater" / "eval_results.json"
        if eval_path.exists():
            with open(eval_path) as f:
                comparison_data.append(json.load(f))

    if comparison_data:
        import pandas as pd

        rows = []
        for d in comparison_data:
            m = d.get("metrics", {}).get("overall", {})
            rows.append({
                "Model":           d["model"],
                "Precision":       m.get("precision", 0),
                "Recall":          m.get("recall", 0),
                "F1 Score":        m.get("f1", 0),
                "Inference (ms)":  d.get("avg_inference_ms", 0),
                "Small Recall":    d.get("failures", {}).get("small_recall", 0),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df.set_index("Model"), use_container_width=True)

        # Radar chart
        categories = ["Precision", "Recall", "F1 Score", "Small Recall"]
        fig_radar = go.Figure()
        colors = ["#3498db", "#e74c3c"]
        for i, row in df.iterrows():
            vals = [row[c] for c in categories] + [row[categories[0]]]
            cats = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill="toself",
                name=row["Model"], line_color=colors[i % 2],
                fillcolor=colors[i % 2], opacity=0.25
            ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], tickfont_color="white", gridcolor="#333"),
                angularaxis=dict(tickfont_color="white", gridcolor="#333"),
                bgcolor="#161b22"
            ),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font_color="white",
            legend=dict(bgcolor="#161b22", bordercolor="#444"),
            title=dict(text="Model Comparison — Radar", font_color="white"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Bar comparison
        fig_bar = px.bar(
            df.melt(id_vars=["Model"], value_vars=["Precision", "Recall", "F1 Score"]),
            x="variable", y="value", color="Model",
            barmode="group",
            color_discrete_sequence=["#3498db", "#e74c3c"],
            title="Precision / Recall / F1 Comparison",
        )
        fig_bar.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="white", yaxis_range=[0, 1],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Show saved eval report images
        for mname in ["yolov8n", "yolov8s"]:
            report_img = ASSETS_DIR / f"eval_report_{mname}.png"
            if report_img.exists():
                st.markdown(f"**{mname.upper()} Evaluation Report**")
                st.image(str(report_img), use_container_width=True)

    else:
        st.info(
            "No evaluation results found. Run evaluation first:\n"
            "```\npython src/evaluate.py --all\n```"
        )

        # Show placeholder table with expected values
        import pandas as pd
        placeholder = pd.DataFrame([
            {"Model": "YOLOv8n", "mAP@50": "~0.61", "Precision": "~0.63",
             "Recall": "~0.59", "Inference (ms)": "~8",  "Params": "3.2M"},
            {"Model": "YOLOv8s", "mAP@50": "~0.68", "Precision": "~0.70",
             "Recall": "~0.66", "Inference (ms)": "~14", "Params": "11.2M"},
        ])
        st.markdown("**Expected results after training:**")
        st.dataframe(placeholder.set_index("Model"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — IoT Dashboard
# ═══════════════════════════════════════════════════════════════════
with tab_iot:
    st.markdown("<h3 style='font-weight:700;color:#ffffff;'>📡 IoT / IoE — Satellite Telemetry Dashboard</h3>", unsafe_allow_html=True)
    st.markdown(
        "Simulates a real satellite-to-ground pipeline: MQTT topics, "
        "sensor metadata, edge inference telemetry, and alert thresholds."
    )

    gs = get_ground_station()

    if gs is None:
        st.info("Enable **Live telemetry stream** in the sidebar to start receiving data.")
    else:
        shadow = gs.device_shadow
        if shadow:
            st.markdown("<h4 style='font-weight:700;color:#ffffff;'>🛰 Device Shadow (Latest Satellite State)</h4>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Altitude", f"{shadow['altitude_km']} km")
            c2.metric("Sensor Temp", f"{shadow['sensor_temp_c']} °C")
            c3.metric("Power Draw", f"{shadow['power_draw_w']} W")
            c4.metric("Storage", f"{shadow['storage_used_pct']} %")

            c5, c6, c7 = st.columns(3)
            c5.metric("Total Craters", shadow["total_craters_detected"])
            c6.metric("Tiles Processed", shadow["total_tiles_processed"])
            c7.metric("Active Alerts", shadow["active_alerts"])

        # Time-series chart
        telemetry = gs.get_telemetry_series()
        if telemetry["timestamps"]:
            st.markdown("<h4 style='font-weight:700;color:#ffffff;'>📈 Detection Rate Over Time</h4>", unsafe_allow_html=True)
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=telemetry["timestamps"],
                y=telemetry["crater_counts"],
                mode="lines+markers",
                name="Craters / tile",
                line=dict(color="#3498db", width=2),
                marker=dict(size=4),
            ))
            # Alert threshold line
            fig_ts.add_hline(y=12, line_dash="dash", line_color="#e74c3c",
                             annotation_text="High density alert",
                             annotation_font_color="#e74c3c")
            fig_ts.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font_color="white", xaxis_title="Time",
                yaxis_title="Craters detected",
                legend=dict(bgcolor="#161b22"),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            # Confidence & altitude
            col_conf, col_alt = st.columns(2)
            with col_conf:
                fig_conf = go.Figure(go.Scatter(
                    x=telemetry["timestamps"],
                    y=telemetry["confidences"],
                    mode="lines", name="Avg confidence",
                    line=dict(color="#2ecc71")
                ))
                fig_conf.update_layout(
                    title="Average Detection Confidence",
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font_color="white", yaxis_range=[0, 1]
                )
                st.plotly_chart(fig_conf, use_container_width=True)

            with col_alt:
                fig_alt = go.Figure(go.Scatter(
                    x=telemetry["timestamps"],
                    y=telemetry["altitudes_km"],
                    mode="lines", name="Altitude (km)",
                    line=dict(color="#f39c12")
                ))
                fig_alt.update_layout(
                    title="Orbital Altitude",
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font_color="white"
                )
                st.plotly_chart(fig_alt, use_container_width=True)

        # Alerts
        alerts = gs.get_alerts()
        if alerts:
            st.markdown("<h4 style='font-weight:700;color:#ffffff;'>⚠ Active Alerts</h4>", unsafe_allow_html=True)
            for a in alerts[-10:]:
                color = "#f85149" if "DENSITY" in a.get("type","") else "#e3b341"
                st.markdown(
                    f'<div style="border-left:3px solid {color};padding:8px;'
                    f'background:#161b22;margin:4px 0;border-radius:4px;">'
                    f'<b>{a["type"]}</b> — {a}</div>',
                    unsafe_allow_html=True,
                )

        # Raw events log
        events = gs.get_events()
        if events:
            with st.expander(f"📋 Raw Telemetry Log ({len(events)} events)"):
                import pandas as pd
                log_rows = []
                for e in events[-50:]:
                    log_rows.append({
                        "Timestamp":  e.timestamp_utc,
                        "Tile":       e.tile_id,
                        "Craters":    e.n_craters_detected,
                        "Conf":       e.avg_confidence,
                        "Infer ms":   e.inference_ms,
                        "Altitude":   e.altitude_km,
                        "Temp °C":    e.sensor_temp_c,
                        "Alert":      "⚠" if e.high_density_flag else "✅",
                    })
                st.dataframe(pd.DataFrame(log_rows), use_container_width=True,
                             hide_index=True)

        # Auto-refresh
        if iot_running:
            time.sleep(0.1)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — About
# ═══════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
### About This Project

**Lunar & Martian Crater Detection** is an end-to-end computer vision pipeline
for automated crater detection using deep learning object detection.

---

#### Architecture
- **Data pipeline** — synthetic crater generator + tiling + augmentation + split
- **Model A** — YOLOv8 Nano (3.2M params) — fast baseline
- **Model B** — YOLOv8 Small (11.2M params) — higher accuracy comparison
- **Explainability** — confidence heatmaps, before/after overlays, detection tables
- **IoT layer** — MQTT publish/subscribe, satellite sensor simulation, device shadow

#### IoT / IoE Integration
The software-layer IoT simulation models the architecture of a real
satellite-to-ground pipeline:

| Component | Real System | This Simulation |
|-----------|-------------|-----------------|
| Data source | LRO / HiRISE camera | Synthetic + uploaded images |
| Edge compute | On-board processor | Python thread |
| Transport | Deep Space Network | In-process MQTT broker |
| Protocol | CCSDS telemetry | MQTT JSON payloads |
| Ground station | JPL / ESA DSN | GroundStation class |
| Device state | Spacecraft telemetry | Device shadow dict |
| Alerts | Mission operations | Streamlit alert boxes |

#### Dataset
- Primary: Kaggle Martian/Lunar Crater Detection Dataset
- Fallback: Synthetic generation (Perlin-noise terrain + procedural craters)
- Classes: small (<5 km), medium (5–20 km), large (>20 km)

#### Future Work
- Real-time NASA/ESA satellite feed integration
- Multi-planet domain adaptation
- 3D crater morphology from elevation data
- Actual MQTT broker (Mosquitto) + edge node deployment
- Federated learning across ground stations

---
**Tech stack:** Python · PyTorch · Ultralytics YOLOv8 · OpenCV · Streamlit · Plotly · paho-mqtt
""")

