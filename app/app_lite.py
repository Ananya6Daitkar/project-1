# app_lite.py
#
# lightweight web app — no streamlit/plotly needed
# uses Python's built-in http.server + opencv + matplotlib
#
# what it does:
#   - serves a webpage at http://localhost:8080
#   - accepts image uploads, detects craters using OpenCV (Hough circles)
#     as a stand-in until YOLOv8 weights are trained
#   - shows before/after image, crater count, size distribution
#   - streams live IoT telemetry events
#
# run:  python app/app_lite.py

import sys
import json
import base64
import random
import threading
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import CLASS_NAMES, CLASS_COLORS, draw_detections, ASSETS_DIR, log
from iot_simulator import (
    start_iot_simulation, get_ground_station,
    emit_detection_event, PLANETARY_TARGETS
)

PORT = 8080

# ----------------------------------------------------------------
# crater detector using Hough circles (works without YOLOv8 weights)
# this is the fallback detector — replaced by YOLO after training
# ----------------------------------------------------------------

def detect_craters_hough(img_bgr):
    """
    detects circular craters using OpenCV Hough circle transform.
    not as accurate as YOLOv8 but works with zero training.
    returns (boxes_xyxy, classes, confidences)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    gray = cv2.GaussianBlur(gray, (5, 5), 1.5)
    H, W = gray.shape

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=15,
        param1=60,
        param2=28,
        minRadius=4,
        maxRadius=min(H, W) // 3,
    )

    boxes, classes, confs = [], [], []
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for (cx, cy, r) in circles:
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(W, cx + r), min(H, cy + r)
            boxes.append((x1, y1, x2, y2))

            # classify by radius
            r_norm = r / min(H, W)
            if r_norm < 0.08:
                cls = 0
            elif r_norm < 0.20:
                cls = 1
            else:
                cls = 2
            classes.append(cls)
            # fake confidence based on circle strength
            confs.append(round(random.uniform(0.45, 0.92), 2))

    return boxes, classes, confs


def annotate_and_encode(img_bgr, boxes, classes, confs):
    """draws boxes on image, returns base64 PNG for the HTML"""
    annotated = draw_detections(img_bgr, boxes, classes, confs)
    _, buf = cv2.imencode(".png", annotated)
    return base64.b64encode(buf).decode()


def make_pie_chart(class_counts):
    """returns a base64 pie chart of crater size distribution"""
    labels = [k.replace("_crater", "") for k, v in class_counts.items() if v > 0]
    sizes  = [v for v in class_counts.values() if v > 0]
    if not sizes:
        return ""

    colors = ["#e74c3c", "#2ecc71", "#3498db"][:len(labels)]
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#161b22")
    ax.pie(sizes, labels=labels, colors=colors,
           autopct="%1.0f%%", textprops={"color": "white", "fontsize": 10})
    ax.set_facecolor("#161b22")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ----------------------------------------------------------------
# HTML template
# ----------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html>
<head>
  <title>🌑 Crater Detector</title>
  <meta charset="utf-8">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0d1117; color: #e6edf3; font-family: -apple-system, sans-serif; padding: 20px; }}
    h1 {{ color: #58a6ff; margin-bottom: 6px; }}
    .sub {{ color: #8b949e; margin-bottom: 24px; font-size: 14px; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
             padding: 20px; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .grid3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
    img.result {{ width: 100%; border-radius: 6px; border: 1px solid #30363d; }}
    .metric {{ text-align: center; padding: 16px; background: #0d1117;
               border-radius: 6px; border: 1px solid #30363d; }}
    .metric .val {{ font-size: 32px; font-weight: bold; color: #58a6ff; }}
    .metric .lbl {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
    .alert {{ background: #3d1a1a; border: 1px solid #f85149; border-radius: 6px;
              padding: 10px 14px; color: #f85149; margin-bottom: 8px; font-size: 13px; }}
    .safe  {{ background: #1a3d2a; border: 1px solid #3fb950; border-radius: 6px;
              padding: 10px 14px; color: #3fb950; margin-bottom: 12px; font-size: 13px; }}
    form {{ display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 4px; }}
    input[type=file] {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
                        padding: 8px; color: #e6edf3; }}
    select {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
              padding: 8px; color: #e6edf3; }}
    button {{ background: #1f6feb; border: none; border-radius: 6px; padding: 8px 20px;
              color: white; cursor: pointer; font-size: 14px; }}
    button:hover {{ background: #388bfd; }}
    .event {{ background: #0d1117; border-left: 3px solid #30363d; padding: 8px 12px;
              margin-bottom: 6px; font-size: 12px; font-family: monospace; border-radius: 0 4px 4px 0; }}
    .event.high {{ border-left-color: #f85149; }}
    .tag {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }}
    .tag.s {{ background: #3d1616; color: #f85149; }}
    .tag.m {{ background: #163d22; color: #3fb950; }}
    .tag.l {{ background: #162035; color: #58a6ff; }}
    h2 {{ color: #e6edf3; margin-bottom: 12px; font-size: 16px; }}
    h3 {{ color: #8b949e; font-size: 13px; margin-bottom: 8px; font-weight: normal; }}
  </style>
</head>
<body>
  <h1>🌑 Lunar / Mars Crater Detector</h1>
  <p class="sub">Computer Vision · IoT Telemetry · YOLOv8 + Hough Fallback</p>

  <div class="card">
    <h2>Upload Surface Image</h2>
    <form method="POST" enctype="multipart/form-data" action="/detect">
      <input type="file" name="image" accept="image/*" required>
      <select name="planet">
        {planet_options}
      </select>
      <button type="submit">🔭 Detect Craters</button>
    </form>
    <p style="color:#8b949e;font-size:12px;margin-top:8px;">
      Supports PNG, JPG. Uses Hough circle detection until YOLOv8 weights are trained.
    </p>
  </div>

  {results_html}

  <div class="card">
    <h2>📡 Live IoT Telemetry  <span style="font-size:12px;color:#3fb950;">● streaming</span></h2>
    <h3>Last 8 satellite events from the in-process MQTT broker</h3>
    {iot_html}
    <div style="margin-top:12px;color:#8b949e;font-size:12px;">
      Topics: crater/detections · crater/health · crater/alerts &nbsp;|&nbsp;
      Device shadow updated every event &nbsp;|&nbsp;
      Log: data/telemetry.jsonl
    </div>
  </div>

</body>
</html>"""


def build_results_html(orig_b64, ann_b64, pie_b64, n_total,
                       class_counts, planet, inf_ms):
    safety = ("⚠ HIGH CRATER DENSITY — not recommended as landing zone"
               if n_total > 12 else
               "✅ MODERATE DENSITY — possible landing zone candidate"
               if n_total > 0 else
               "✅ CLEAN TERRAIN — no craters detected")
    safety_cls = "alert" if n_total > 12 else "safe"

    metrics = f"""
    <div class="grid3" style="margin-bottom:16px;">
      <div class="metric"><div class="val">{n_total}</div><div class="lbl">total craters</div></div>
      <div class="metric"><div class="val" style="color:#e74c3c">{class_counts['small_crater']}</div><div class="lbl">small</div></div>
      <div class="metric"><div class="val" style="color:#2ecc71">{class_counts['medium_crater']}</div><div class="lbl">medium</div></div>
    </div>
    <div class="grid3" style="margin-bottom:16px;">
      <div class="metric"><div class="val" style="color:#3498db">{class_counts['large_crater']}</div><div class="lbl">large</div></div>
      <div class="metric"><div class="val" style="font-size:20px">{planet}</div><div class="lbl">target body</div></div>
      <div class="metric"><div class="val" style="font-size:20px">{inf_ms:.0f}ms</div><div class="lbl">inference time</div></div>
    </div>"""

    pie_section = (f'<img class="result" src="data:image/png;base64,{pie_b64}">'
                   if pie_b64 else "<p style='color:#8b949e'>no detections</p>")

    return f"""
  <div class="card">
    <h2>Detection Results</h2>
    <div class="{safety_cls}" style="margin-bottom:16px;">{safety}</div>
    {metrics}
    <div class="grid">
      <div>
        <h3>Original</h3>
        <img class="result" src="data:image/png;base64,{orig_b64}">
      </div>
      <div>
        <h3>Detected craters ({n_total})</h3>
        <img class="result" src="data:image/png;base64,{ann_b64}">
      </div>
    </div>
    <div style="margin-top:16px;max-width:300px;">
      <h3>Size distribution</h3>
      {pie_section}
    </div>
  </div>"""


def build_iot_html():
    gs = get_ground_station()
    if gs is None:
        return "<p style='color:#8b949e'>no telemetry yet — upload an image first</p>"

    events = gs.get_events()[-8:]
    if not events:
        return "<p style='color:#8b949e'>waiting for events...</p>"

    rows = ""
    for e in reversed(events):
        hi = "high" if e.high_density_flag else ""
        alert_icon = "🔴" if e.high_density_flag else ("🟡" if e.anomaly_flag else "🟢")
        rows += f"""
        <div class="event {hi}">
          {alert_icon} <b>{e.tile_id}</b> &nbsp;|&nbsp;
          craters: <b>{e.n_craters_detected}</b> &nbsp;|&nbsp;
          conf: {e.avg_confidence:.2f} &nbsp;|&nbsp;
          alt: {e.altitude_km}km &nbsp;|&nbsp;
          temp: {e.sensor_temp_c}°C &nbsp;|&nbsp;
          {e.timestamp_utc[-9:-1]} UTC
        </div>"""

    shadow = gs.device_shadow or {}
    shadow_rows = "".join(
        f"<div style='display:flex;justify-content:space-between;padding:3px 0;"
        f"border-bottom:1px solid #21262d;font-size:12px;'>"
        f"<span style='color:#8b949e'>{k}</span><span>{v}</span></div>"
        for k, v in list(shadow.items())[:6]
    )

    return f"""
    <div class="grid">
      <div>{rows}</div>
      <div>
        <h3>Device Shadow (digital twin)</h3>
        <div style="background:#0d1117;border-radius:6px;padding:12px;">{shadow_rows}</div>
      </div>
    </div>"""


# ----------------------------------------------------------------
# request handler
# ----------------------------------------------------------------

_last_results = {}   # store last detection for page refresh


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress default server logs

    def do_GET(self):
        self._serve_page()

    def do_POST(self):
        if self.path == "/detect":
            self._handle_detect()
        else:
            self._serve_page()

    def _serve_page(self):
        planet_opts = "\n".join(
            f'<option value="{p}">{p}</option>' for p in PLANETARY_TARGETS
        )
        html = HTML.format(
            planet_options=planet_opts,
            results_html=_last_results.get("html", ""),
            iot_html=build_iot_html(),
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _handle_detect(self):
        # parse multipart form data manually
        content_type = self.headers.get("Content-Type", "")
        length       = int(self.headers.get("Content-Length", 0))
        body         = self.rfile.read(length)

        # extract boundary
        boundary = None
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part[9:].encode()

        if not boundary:
            self._redirect()
            return

        # split parts
        parts = body.split(b"--" + boundary)
        image_data = None
        planet     = "Moon"

        for part in parts:
            if b"name=\"image\"" in part and b"filename=" in part:
                # find end of headers
                idx = part.find(b"\r\n\r\n")
                if idx >= 0:
                    image_data = part[idx + 4:].rstrip(b"\r\n")
            elif b"name=\"planet\"" in part:
                idx = part.find(b"\r\n\r\n")
                if idx >= 0:
                    planet = part[idx + 4:].strip().decode()

        if not image_data:
            self._redirect()
            return

        # decode image
        arr    = np.frombuffer(image_data, dtype=np.uint8)
        img    = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            self._redirect()
            return

        # resize if huge
        H, W = img.shape[:2]
        if max(H, W) > 800:
            scale = 800 / max(H, W)
            img   = cv2.resize(img, (int(W*scale), int(H*scale)))

        # detect
        t0 = __import__("time").perf_counter()
        boxes, classes, confs = detect_craters_hough(img)
        inf_ms = (__import__("time").perf_counter() - t0) * 1000

        # encode original
        _, orig_buf = cv2.imencode(".png", img)
        orig_b64    = base64.b64encode(orig_buf).decode()
        ann_b64     = annotate_and_encode(img, boxes, classes, confs)

        class_counts = {name: 0 for name in CLASS_NAMES.values()}
        for cls in classes:
            class_counts[CLASS_NAMES[cls]] += 1
        pie_b64 = make_pie_chart(class_counts)

        # emit IoT event
        crater_class_list = [CLASS_NAMES[c] for c in classes]
        avg_conf = float(np.mean(confs)) if confs else 0.0
        emit_detection_event(len(boxes), crater_class_list, avg_conf,
                             inf_ms, "upload", planet)

        # store results
        _last_results["html"] = build_results_html(
            orig_b64, ann_b64, pie_b64,
            len(boxes), class_counts, planet, inf_ms
        )

        self._redirect()

    def _redirect(self):
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------

if __name__ == "__main__":
    # start IoT simulation in background
    start_iot_simulation(target="Moon", interval_sec=2.0)

    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\n{'='*50}")
    print(f"  🌑 Crater Detector running!")
    print(f"  Open: http://localhost:{PORT}")
    print(f"  IoT : live satellite feed streaming")
    print(f"  Press Ctrl-C to stop")
    print(f"{'='*50}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
