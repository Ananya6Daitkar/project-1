"""
iot_dashboard.py — Standalone IoT telemetry dashboard
======================================================
Run independently for a dedicated IoT monitor view:
    streamlit run app/iot_dashboard.py

Features:
  - Real-time MQTT-style event feed
  - Satellite health monitoring
  - Crater density alerts
  - Orbital trajectory simulation
  - Coverage map (lat/lon scatter)
"""

import sys
import json
import time
import random
from pathlib import Path
from typing import List, Dict

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from iot_simulator import (
    start_iot_simulation, stop_iot_simulation, get_ground_station,
    PLANETARY_TARGETS, SensorPayload
)

st.set_page_config(
    page_title="🛰 IoT Telemetry — Crater Mission",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp { background-color: #070d19; color: #e6edf3; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛰 Mission Control")
    target = st.selectbox("Target Body", PLANETARY_TARGETS)
    interval = st.slider("Telemetry interval (s)", 0.5, 5.0, 1.5, 0.5)
    streaming = st.toggle("Stream active", value=True)
    refresh_rate = st.slider("Dashboard refresh (s)", 1, 10, 3)

if streaming:
    gs = start_iot_simulation(target=target, interval_sec=interval)
else:
    stop_iot_simulation()
    gs = get_ground_station()

# ─────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown(f"# 🛰 Crater Mission — IoT Telemetry Dashboard")
    st.markdown(f"**Target:** {target}  |  **Protocol:** MQTT (simulated)  |  "
                f"**Transport:** In-process broker")
with col_status:
    if streaming and gs:
        st.markdown("""<div style="background:#1a3d2a;border:1px solid #3fb950;
            border-radius:8px;padding:12px;text-align:center;margin-top:12px;">
            🟢 STREAMING</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:#3d1a1a;border:1px solid #f85149;
            border-radius:8px;padding:12px;text-align:center;margin-top:12px;">
            🔴 OFFLINE</div>""", unsafe_allow_html=True)

st.markdown("---")

if gs is None:
    st.info("Start the telemetry stream using the sidebar toggle.")
    st.stop()

# ─────────────────────────────────────────────────────────────────
# Live metrics
# ─────────────────────────────────────────────────────────────────
shadow = gs.device_shadow
events = gs.get_events()
alerts = gs.get_alerts()
telemetry = gs.get_telemetry_series()

if shadow:
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🛰 Altitude", f"{shadow['altitude_km']} km")
    m2.metric("🌡 Sensor Temp", f"{shadow['sensor_temp_c']} °C",
              delta=f"{shadow['sensor_temp_c'] - 20:.1f}" if shadow['sensor_temp_c'] > 80 else None,
              delta_color="inverse")
    m3.metric("⚡ Power", f"{shadow['power_draw_w']} W")
    m4.metric("💾 Storage", f"{shadow['storage_used_pct']} %")
    m5.metric("🔭 Total Craters", shadow["total_craters_detected"])
    m6.metric("📁 Tiles", shadow["total_tiles_processed"])

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# Charts row
# ─────────────────────────────────────────────────────────────────
col_ts, col_map = st.columns([2, 1])

with col_ts:
    if telemetry["timestamps"]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=telemetry["timestamps"],
            y=telemetry["crater_counts"],
            mode="lines+markers",
            name="Craters / tile",
            line=dict(color="#58a6ff", width=2),
            marker=dict(size=5, color=["#f85149" if c > 12 else "#58a6ff"
                                        for c in telemetry["crater_counts"]]),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
        ))
        fig.add_hline(y=12, line_dash="dash", line_color="#f85149",
                      annotation_text="Alert threshold",
                      annotation_font_color="#f85149",
                      annotation_position="top right")
        fig.update_layout(
            title="Real-time Crater Detection Rate",
            paper_bgcolor="#070d19", plot_bgcolor="#0d1526",
            font_color="#c9d1d9", xaxis_title="Time (UTC)",
            yaxis_title="Craters detected",
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

with col_map:
    if events:
        lats = [e.latitude_deg for e in events[-200:]]
        lons = [e.longitude_deg for e in events[-200:]]
        counts = [e.n_craters_detected for e in events[-200:]]

        fig_map = go.Figure(go.Scattergeo(
            lat=lats, lon=lons,
            mode="markers",
            marker=dict(
                size=[max(4, min(c*1.5, 20)) for c in counts],
                color=counts,
                colorscale="Reds",
                cmin=0, cmax=20,
                colorbar=dict(title="Craters", titlefont_color="white",
                              tickfont_color="white"),
                line=dict(width=0.3, color="#0d1526"),
            ),
            text=[f"Craters: {c}" for c in counts],
        ))
        fig_map.update_geos(
            bgcolor="#0d1526",
            showland=True, landcolor="#1c2333",
            showocean=True, oceancolor="#0d1526",
            showcoastlines=True, coastlinecolor="#30363d",
            projection_type="natural earth",
        )
        fig_map.update_layout(
            title=f"{target} Coverage Map",
            paper_bgcolor="#070d19",
            font_color="#c9d1d9",
            geo=dict(bgcolor="#0d1526"),
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# Alerts panel
# ─────────────────────────────────────────────────────────────────
st.markdown("### ⚠ Alert Feed")
if alerts:
    for a in reversed(alerts[-8:]):
        atype = a.get("type", "UNKNOWN")
        color = "#f85149" if "DENSITY" in atype else "#e3b341"
        icon  = "🔴" if "DENSITY" in atype else "🟡"
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:8px 12px;'
            f'background:#0d1526;margin:4px 0;border-radius:0 6px 6px 0;">'
            f'{icon} <b>{atype}</b> — Tile: {a.get("tile_id","?")}  '
            f'Lat:{a.get("lat","?")}  Lon:{a.get("lon","?")}  '
            f'Craters:{a.get("count","?")}</div>',
            unsafe_allow_html=True,
        )
else:
    st.success("✅ No active alerts")

# ─────────────────────────────────────────────────────────────────
# MQTT-style message log
# ─────────────────────────────────────────────────────────────────
st.markdown("### 📨 MQTT Message Log (last 20 events)")
if events:
    rows = []
    for e in events[-20:]:
        rows.append({
            "Topic":     "crater/detections",
            "Timestamp": e.timestamp_utc[-8:],    # HH:MM:SSZ
            "Tile":      e.tile_id,
            "Body":      e.target_body,
            "Craters":   e.n_craters_detected,
            "Conf":      f"{e.avg_confidence:.2f}",
            "Infer ms":  f"{e.inference_ms:.1f}",
            "Temp °C":   f"{e.sensor_temp_c:.1f}",
            "⚠":         "🔴" if e.high_density_flag else ("🟡" if e.anomaly_flag else "🟢"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

# Auto-refresh
if streaming:
    time.sleep(0.1)
    st.rerun()
