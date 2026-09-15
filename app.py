"""
CCTV AI Vision System — Streamlit Surveillance Console & Cybersecurity Hub
===========================================================================
Real-time Human Tracking, Face Tracking, ANPR OCR & 10-Vector Cybersecurity Health Monitor.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Mute backend warnings
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anpr_detector import ANPRDetector
from src.cyber_health_monitor import cyber_monitor
from src.face_tracker import FastFaceTracker
from src.human_detector import HumanTracker
from src.plate_recorder import PlateRecorder


# =============================================================================
# Streamlit Page Config & Custom Styling
# =============================================================================

st.set_page_config(
    page_title="CCTV AI Vision Hub | Multi-AI & Cybersecurity",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Dark Futuristic Theme */
    .stApp {
        background-color: #0b0d13;
        color: #e2e8f0;
        font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
    }
    
    .cctv-header {
        background: linear-gradient(135deg, rgba(20, 24, 36, 0.9) 0%, rgba(13, 16, 24, 0.95) 100%);
        border: 1px solid rgba(0, 255, 140, 0.25);
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .cctv-title {
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 1px;
        color: #f8fafc;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .status-live {
        background: rgba(0, 255, 140, 0.15);
        border: 1px solid #00ff8c;
        color: #00ff8c;
    }
    
    .status-stopped {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #ef4444;
    }
    
    .metric-card {
        background: rgba(18, 22, 33, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #00ff8c;
        margin: 4px 0 0 0;
        font-family: 'Courier New', monospace;
    }
    
    .metric-lbl {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }
    
    .security-alert {
        background: rgba(180, 20, 30, 0.85);
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        color: #ffffff;
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .cyber-card {
        background: rgba(16, 20, 30, 0.90);
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .cyber-card-ok { border-left: 4px solid #00ff8c; }
    .cyber-card-warn { border-left: 4px solid #f59e0b; }
    .cyber-card-crit { border-left: 4px solid #ef4444; }
    
    .plate-card {
        background: rgba(22, 27, 40, 0.9);
        border-left: 4px solid #00ff8c;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Courier New', monospace;
    }
    
    .plate-text {
        font-size: 16px;
        font-weight: 700;
        color: #f1f5f9;
        background: #0f172a;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helper Utilities & Loaders
# =============================================================================

def get_camera_names() -> List[str]:
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        return graph.get_input_devices()
    except Exception:
        return []


@st.cache_resource(show_spinner="Loading YOLO ANPR & EasyOCR...")
def load_anpr_detector(device: str, enable_ocr: bool) -> ANPRDetector:
    return ANPRDetector(device=device, enable_ocr=enable_ocr)


@st.cache_resource(show_spinner="Loading YuNet Face Detector & SFace Recognizer...")
def load_face_trackers() -> Tuple[FastFaceTracker, FastFaceTracker]:
    return FastFaceTracker(max_alerts_per_person=3), FastFaceTracker(max_alerts_per_person=3)


@st.cache_resource(show_spinner="Loading YOLO Human Tracker Checkpoint...")
def load_human_trackers(device: str) -> Tuple[HumanTracker, HumanTracker]:
    return HumanTracker(device=device), HumanTracker(device=device)


class DualCameraManager:
    def __init__(self, cam0_id: int = 0, cam1_id: int = 1, width: int = 640, height: int = 480) -> None:
        self.cam0_id = cam0_id
        self.cam1_id = cam1_id
        self.width = width
        self.height = height
        self.cap0: Optional[cv2.VideoCapture] = None
        self.cap1: Optional[cv2.VideoCapture] = None
        self.dev0_name = f"Camera {cam0_id}"
        self.dev1_name = f"Camera {cam1_id}"
        self.is_cam0_active = False
        self.is_cam1_active = False
        self.init_cameras()

    def init_cameras(self) -> None:
        names = get_camera_names()
        if self.cam0_id < len(names):
            self.dev0_name = names[self.cam0_id]
        if self.cam1_id < len(names):
            self.dev1_name = names[self.cam1_id]

        try:
            self.cap0 = cv2.VideoCapture(self.cam0_id, cv2.CAP_DSHOW)
            if not self.cap0.isOpened():
                self.cap0 = cv2.VideoCapture(self.cam0_id)
            if self.cap0.isOpened():
                self.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(5):
                    r, f = self.cap0.read()
                    if r and f is not None:
                        self.is_cam0_active = True
                        break
                    time.sleep(0.03)
        except Exception:
            self.is_cam0_active = False

        time.sleep(0.15)

        try:
            self.cap1 = cv2.VideoCapture(self.cam1_id, cv2.CAP_DSHOW)
            if not self.cap1.isOpened():
                self.cap1 = cv2.VideoCapture(self.cam1_id)
            if self.cap1.isOpened():
                self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(5):
                    r, f = self.cap1.read()
                    if r and f is not None:
                        self.is_cam1_active = True
                        break
                    time.sleep(0.03)
        except Exception:
            self.is_cam1_active = False

    def read_sync(self) -> Tuple[bool, Optional[np.ndarray], bool, Optional[np.ndarray]]:
        g0 = self.cap0.grab() if (self.is_cam0_active and self.cap0 and self.cap0.isOpened()) else False
        g1 = self.cap1.grab() if (self.is_cam1_active and self.cap1 and self.cap1.isOpened()) else False

        r0, f0 = self.cap0.retrieve() if g0 else (False, None)
        r1, f1 = self.cap1.retrieve() if g1 else (False, None)
        return r0, f0, r1, f1

    def release(self) -> None:
        if self.cap0 and self.cap0.isOpened():
            self.cap0.release()
        if self.cap1 and self.cap1.isOpened():
            self.cap1.release()
        self.is_cam0_active = False
        self.is_cam1_active = False


def create_placeholder_frame(width: int, height: int, title: str) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (20, 22, 28)

    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (35, 38, 48), 1)
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (35, 38, 48), 1)

    cx, cy = width // 2, height // 2
    cv2.rectangle(frame, (cx - 150, cy - 35), (cx + 150, cy + 35), (30, 32, 42), -1)
    cv2.rectangle(frame, (cx - 150, cy - 35), (cx + 150, cy + 35), (0, 140, 255), 2)
    cv2.putText(
        frame, title, (cx - 120, cy + 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2, cv2.LINE_AA,
    )
    return frame


def draw_header_banner(
    frame: np.ndarray,
    label: str,
    device_name: str,
    humans: int,
    faces: int,
    plates: int,
    is_live: bool = True,
) -> np.ndarray:
    h, w = frame.shape[:2]
    banner_h = 34

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (16, 18, 24), -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
    cv2.line(frame, (0, banner_h), (w, banner_h), (45, 50, 65), 1)

    dot_color = (0, 255, 120) if is_live else (0, 80, 255)
    cv2.circle(frame, (16, banner_h // 2), 5, dot_color, -1)

    title = f"{label.upper()} | {device_name}"
    cv2.putText(
        frame, title, (30, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 245), 1, cv2.LINE_AA,
    )

    badge_text = f"HUMANS: {humans}  |  FACES: {faces}  |  PLATES: {plates}"
    (bw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    badge_x = w - bw - 15
    cv2.rectangle(frame, (badge_x - 8, 5), (w - 8, 28), (28, 32, 44), -1)
    cv2.rectangle(frame, (badge_x - 8, 5), (w - 8, 28), (60, 70, 95), 1)
    cv2.putText(
        frame, badge_text, (badge_x, 21),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA,
    )
    return frame


def combine_dual_feeds(
    frame0: np.ndarray,
    frame1: np.ndarray,
    dev0_name: str,
    dev1_name: str,
    h_count0: int,
    f_count0: int,
    p_count0: int,
    h_count1: int,
    f_count1: int,
    p_count1: int,
    target_h: int = 480,
) -> np.ndarray:
    h0, w0 = frame0.shape[:2]
    w0_res = int(w0 * (target_h / h0))
    f0_res = cv2.resize(frame0, (w0_res, target_h))

    h1, w1 = frame1.shape[:2]
    w1_res = int(w1 * (target_h / h1))
    f1_res = cv2.resize(frame1, (w1_res, target_h))

    draw_header_banner(f0_res, "Feed A", dev0_name, h_count0, f_count0, p_count0)
    draw_header_banner(f1_res, "Feed B", dev1_name, h_count1, f_count1, p_count1)

    combined = np.hstack((f0_res, f1_res))
    cv2.line(combined, (w0_res, 0), (w0_res, target_h), (0, 255, 140), 2)
    return combined


# =============================================================================
# Streamlit App Execution Flow
# =============================================================================

def main():
    if "streaming" not in st.session_state:
        st.session_state.streaming = False
    if "latest_plates" not in st.session_state:
        st.session_state.latest_plates = []

    is_live = st.session_state.streaming
    status_html = (
        '<span class="status-badge status-live">● LIVE STREAMING</span>'
        if is_live
        else '<span class="status-badge status-stopped">○ STREAM IDLE</span>'
    )
    
    st.markdown(
        f"""
        <div class="cctv-header">
            <div>
                <h1 class="cctv-title">🛡️ CCTV AI Surveillance & Cybersecurity Hub</h1>
                <p style="color: #94a3b8; margin: 4px 0 0 0; font-size: 14px;">
                    Human Tracking, SFace Offline Face Recognition, ANPR OCR & 10-Vector Cybersecurity Health Monitor
                </p>
            </div>
            <div>{status_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### ⚙️ Surveillance Control Panel")

    source_type = st.sidebar.selectbox(
        "Input Source Mode",
        options=["Physical Live Webcams", "Video File Upload", "RTSP / IP Camera Stream"],
        index=0,
    )

    cam0_id = 0
    cam1_id = 1
    uploaded_video_path = None
    rtsp_url = ""

    device_names = get_camera_names()
    if source_type == "Physical Live Webcams":
        col_c0, col_c1 = st.sidebar.columns(2)
        with col_c0:
            cam0_id = st.number_input("Primary Cam", min_value=0, max_value=10, value=0, step=1)
        with col_c1:
            cam1_id = st.number_input("Secondary Cam", min_value=0, max_value=10, value=1, step=1)

        camera_view_mode = st.sidebar.radio(
            "Camera Viewport Layout",
            options=["Dual Side-by-Side View", f"Camera {cam0_id} Only", f"Camera {cam1_id} Only"],
            index=0,
        )
        if camera_view_mode == "Dual Side-by-Side View":
            layout_key = "dual"
        elif camera_view_mode == f"Camera {cam0_id} Only":
            layout_key = "0"
        else:
            layout_key = "1"

    elif source_type == "Video File Upload":
        layout_key = "file"
        video_file = st.sidebar.file_uploader("Upload Surveillance Footage", type=["mp4", "avi", "mov", "mkv"])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            uploaded_video_path = tfile.name
    else:
        layout_key = "rtsp"
        rtsp_url = st.sidebar.text_input("RTSP Stream URL", value="rtsp://")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 AI Detection Modes")

    detection_mode = st.sidebar.radio(
        "Active AI Pipeline",
        options=[
            "All-In-One (Humans + Faces + Plates)",
            "Human Tracking Only",
            "Face Recognition Only",
            "Number Plates (ANPR) Only",
            "Dual AI (Faces + Plates)",
        ],
        index=0,
    )
    if detection_mode == "Human Tracking Only":
        ai_mode = "humans"
    elif detection_mode == "Face Recognition Only":
        ai_mode = "faces"
    elif detection_mode == "Number Plates (ANPR) Only":
        ai_mode = "plates"
    elif detection_mode == "Dual AI (Faces + Plates)":
        ai_mode = "both"
    else:
        ai_mode = "all"

    with st.sidebar.expander("🔧 Model Sensitivity & Hardware", expanded=False):
        device = st.selectbox("Inference Accelerator", options=["cuda:0", "cpu"] if torch.cuda.is_available() else ["cpu"], index=0)
        enable_ocr = st.checkbox("Enable EasyOCR Text Extraction", value=True)
        conf_human = st.slider("Human Detector Conf Threshold", min_value=0.10, max_value=0.90, value=0.40, step=0.05)
        conf_anpr = st.slider("ANPR Detection Conf Threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
        conf_face = st.slider("Face Detector Conf Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
        cooldown_sec = st.slider("Plate Deduplication Cooldown (s)", min_value=5, max_value=120, value=45, step=5)

    plate_recorder = PlateRecorder(cooldown_seconds=cooldown_sec)

    # Top Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        fps_metric = st.empty()
        fps_metric.markdown('<div class="metric-card"><p class="metric-lbl">Live FPS</p><p class="metric-val">0.0</p></div>', unsafe_allow_html=True)
    with m_col2:
        humans_metric = st.empty()
        humans_metric.markdown('<div class="metric-card"><p class="metric-lbl">Humans</p><p class="metric-val">0</p></div>', unsafe_allow_html=True)
    with m_col3:
        faces_metric = st.empty()
        faces_metric.markdown('<div class="metric-card"><p class="metric-lbl">Faces</p><p class="metric-val">0</p></div>', unsafe_allow_html=True)
    with m_col4:
        plates_metric = st.empty()
        plates_metric.markdown('<div class="metric-card"><p class="metric-lbl">Plates</p><p class="metric-val">0</p></div>', unsafe_allow_html=True)
    with m_col5:
        total_plates_metric = st.empty()
        total_plates_metric.markdown(f'<div class="metric-card"><p class="metric-lbl">Logged</p><p class="metric-val">{len(plate_recorder.recent_logs)}</p></div>', unsafe_allow_html=True)
    with m_col6:
        gpu_status_metric = st.empty()
        serv_stat = cyber_monitor.metrics.get("service", None)
        h_grade = serv_stat.status if serv_stat else "OPTIMAL"
        h_color = "#00ff8c" if h_grade == "OPTIMAL" else ("#f59e0b" if h_grade == "WARNING" else "#ef4444")
        gpu_status_metric.markdown(f'<div class="metric-card"><p class="metric-lbl">Cyber Health</p><p class="metric-val" style="color: {h_color};">{h_grade}</p></div>', unsafe_allow_html=True)

    st.write("")

    tab_live, tab_records, tab_cyber = st.tabs([
        "📺 Live Surveillance Feed",
        "📋 Plate Records Database",
        "🛡️ Cybersecurity & System Health",
    ])

    with tab_cyber:
        st.markdown("### 🛡️ IBVAP Cybersecurity & System Health Center")
        st.caption("Continuous real-time verification of 10 system and biometric security vectors.")

        audit = cyber_monitor.run_full_audit()
        c_cols = st.columns(2)

        for idx, (k, m) in enumerate(audit.items()):
            col_target = c_cols[idx % 2]
            card_class = "cyber-card-ok" if m.status == "OK" else ("cyber-card-warn" if m.status == "WARNING" else "cyber-card-crit")
            pill_color = "#00ff8c" if m.status == "OK" else ("#f59e0b" if m.status == "WARNING" else "#ef4444")

            with col_target:
                st.markdown(
                    f"""
                    <div class="cyber-card {card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; font-size: 15px; color: #f8fafc;">{m.name}</span>
                            <span style="background: rgba(0,0,0,0.4); border: 1px solid {pill_color}; color: {pill_color}; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 700;">{m.status}</span>
                        </div>
                        <div style="font-size: 13px; color: #94a3b8; margin-top: 6px;">{m.details}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("#### 🔐 Cryptographic Model Fingerprints (SHA-256)")
        if cyber_monitor.baseline_hashes:
            hash_df = pd.DataFrame([
                {"Model File": Path(k).name, "Path": k, "SHA-256 Checksum": v}
                for k, v in cyber_monitor.baseline_hashes.items()
            ])
            st.dataframe(hash_df, use_container_width=True, height=200)

        st.markdown("#### 📜 Security Event Audit Logs")
        if cyber_monitor.security_logs:
            st.dataframe(pd.DataFrame(cyber_monitor.security_logs), use_container_width=True, height=180)
        else:
            st.info("No security violations or alerts recorded in current session.")

    with tab_records:
        csv_file_path = PROJECT_ROOT / "records" / "number_plates" / "plate_records.csv"
        if csv_file_path.exists():
            try:
                df = pd.read_csv(csv_file_path)
                if not df.empty:
                    r_col1, r_col2, r_col3 = st.columns([2, 2, 1])
                    with r_col1:
                        search_query = st.text_input("🔍 Search Plate Number", value="")
                    with r_col2:
                        unique_cams = ["All"] + list(df["Camera"].dropna().unique())
                        selected_cam = st.selectbox("Filter by Camera", options=unique_cams)
                    with r_col3:
                        st.write("")
                        st.write("")
                        st.download_button(
                            "📥 Download CSV",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name="plate_records.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                    filtered_df = df.copy()
                    if search_query:
                        filtered_df = filtered_df[filtered_df["PlateNumber"].astype(str).str.contains(search_query.upper(), na=False)]
                    if selected_cam != "All":
                        filtered_df = filtered_df[filtered_df["Camera"] == selected_cam]

                    st.dataframe(filtered_df.iloc[::-1], use_container_width=True, height=350)
                else:
                    st.info("No license plate records logged yet.")
            except Exception as e:
                st.error(f"Error loading records: {e}")
        else:
            st.info("No license plate records CSV found.")

    with tab_live:
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
        with btn_col1:
            start_btn = st.button("▶️ Start Live Stream", type="primary", use_container_width=True)
        with btn_col2:
            stop_btn = st.button("⏹️ Stop Stream", type="secondary", use_container_width=True)

        if start_btn:
            st.session_state.streaming = True
        if stop_btn:
            st.session_state.streaming = False

        security_alert_placeholder = st.empty()

        feed_col, alerts_col = st.columns([3, 1])

        with feed_col:
            video_placeholder = st.empty()
            video_placeholder.info("Click **Start Live Stream** above to activate camera feeds and AI recognition.")

        with alerts_col:
            st.markdown("#### 🚨 Real-Time ANPR Alerts")
            alerts_placeholder = st.empty()

        if st.session_state.streaming:
            anpr_detector = load_anpr_detector(device=device, enable_ocr=enable_ocr)
            anpr_detector.conf_threshold = conf_anpr
            face_tracker_0, face_tracker_1 = load_face_trackers()
            face_tracker_0.conf_threshold = conf_face
            face_tracker_1.conf_threshold = conf_face
            human_tracker_0, human_tracker_1 = load_human_trackers(device=device)
            human_tracker_0.conf_threshold = conf_human
            human_tracker_1.conf_threshold = conf_human

            cam_manager = None
            cap_single = None

            if layout_key in ["dual", "0", "1"]:
                cam_manager = DualCameraManager(cam0_id=cam0_id, cam1_id=cam1_id)
            elif layout_key == "file":
                if uploaded_video_path and os.path.exists(uploaded_video_path):
                    cap_single = cv2.VideoCapture(uploaded_video_path)
                else:
                    st.warning("Please upload a valid video file in the sidebar.")
                    st.session_state.streaming = False
                    return
            elif layout_key == "rtsp":
                if rtsp_url and rtsp_url != "rtsp://":
                    cap_single = cv2.VideoCapture(rtsp_url)
                else:
                    st.warning("Please enter a valid RTSP stream URL in the sidebar.")
                    st.session_state.streaming = False
                    return

            prev_time = time.time()

            try:
                while st.session_state.streaming:
                    h_count0, f_count0, p_count0 = 0, 0, 0
                    h_count1, f_count1, p_count1 = 0, 0, 0
                    alert_unk0, alert_unk1 = 0, 0
                    total_unk0, total_unk1 = 0, 0
                    out_frame = None

                    if cam_manager is not None:
                        ret0, raw0, ret1, raw1 = cam_manager.read_sync()

                        if not ret0 or raw0 is None:
                            raw0 = create_placeholder_frame(640, 480, f"Cam {cam0_id} Offline")
                        if not ret1 or raw1 is None:
                            raw1 = create_placeholder_frame(640, 480, f"Cam {cam1_id} Offline")

                        out_cam0 = raw0.copy()
                        out_cam1 = raw1.copy()

                        process_cam0 = (layout_key in ["dual", "0"]) and ret0
                        process_cam1 = (layout_key in ["dual", "1"]) and ret1

                        # Human Tracking
                        if ai_mode in ["humans", "all"]:
                            if process_cam0:
                                out_cam0, _, h_count0 = human_tracker_0.process_frame(out_cam0)
                            if process_cam1:
                                out_cam1, _, h_count1 = human_tracker_1.process_frame(out_cam1)

                        # Face Recognition
                        if ai_mode in ["faces", "both", "all"]:
                            if process_cam0:
                                out_cam0, f_count0 = face_tracker_0.process_frame(out_cam0)
                                alert_unk0 = face_tracker_0.active_alerting_unknown_count
                                total_unk0 = face_tracker_0.total_unknown_count
                            if process_cam1:
                                out_cam1, f_count1 = face_tracker_1.process_frame(out_cam1)
                                alert_unk1 = face_tracker_1.active_alerting_unknown_count
                                total_unk1 = face_tracker_1.total_unknown_count

                        # ANPR
                        if ai_mode in ["plates", "both", "all"]:
                            if process_cam0:
                                out_cam0, p_dets0, p_count0 = anpr_detector.process_frame(out_cam0)
                                for det in p_dets0:
                                    p_text = det.get("text", "")
                                    if p_text:
                                        is_new, cleaned = plate_recorder.log_plate(
                                            plate_text=p_text,
                                            confidence=det.get("conf", 0.0),
                                            camera_name=f"Cam {cam0_id} ({cam_manager.dev0_name})",
                                            plate_crop_bgr=det.get("crop"),
                                        )
                                        if is_new:
                                            st.session_state.latest_plates.insert(
                                                0,
                                                {"text": cleaned, "time": datetime.now().strftime("%H:%M:%S"), "cam": f"Cam {cam0_id}"}
                                            )
                                            if len(st.session_state.latest_plates) > 6:
                                                st.session_state.latest_plates.pop()

                            if process_cam1:
                                out_cam1, p_dets1, p_count1 = anpr_detector.process_frame(out_cam1)
                                for det in p_dets1:
                                    p_text = det.get("text", "")
                                    if p_text:
                                        is_new, cleaned = plate_recorder.log_plate(
                                            plate_text=p_text,
                                            confidence=det.get("conf", 0.0),
                                            camera_name=f"Cam {cam1_id} ({cam_manager.dev1_name})",
                                            plate_crop_bgr=det.get("crop"),
                                        )
                                        if is_new:
                                            st.session_state.latest_plates.insert(
                                                0,
                                                {"text": cleaned, "time": datetime.now().strftime("%H:%M:%S"), "cam": f"Cam {cam1_id}"}
                                            )
                                            if len(st.session_state.latest_plates) > 6:
                                                st.session_state.latest_plates.pop()

                        if layout_key == "dual":
                            out_frame = combine_dual_feeds(
                                out_cam0, out_cam1,
                                dev0_name=cam_manager.dev0_name,
                                dev1_name=cam_manager.dev1_name,
                                h_count0=h_count0, f_count0=f_count0, p_count0=p_count0,
                                h_count1=h_count1, f_count1=f_count1, p_count1=p_count1,
                            )
                        elif layout_key == "0":
                            out_frame = out_cam0
                            draw_header_banner(out_frame, "Feed A", cam_manager.dev0_name, h_count0, f_count0, p_count0)
                        else:  # "1"
                            out_frame = out_cam1
                            draw_header_banner(out_frame, "Feed B", cam_manager.dev1_name, h_count1, f_count1, p_count1)

                    elif cap_single is not None:
                        ret, raw = cap_single.read()
                        if not ret or raw is None:
                            if layout_key == "file":
                                cap_single.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                continue
                            else:
                                break

                        out_cam0 = raw.copy()
                        if ai_mode in ["humans", "all"]:
                            out_cam0, _, h_count0 = human_tracker_0.process_frame(out_cam0)
                        if ai_mode in ["faces", "both", "all"]:
                            out_cam0, f_count0 = face_tracker_0.process_frame(out_cam0)
                            alert_unk0 = face_tracker_0.active_alerting_unknown_count
                            total_unk0 = face_tracker_0.total_unknown_count
                        if ai_mode in ["plates", "both", "all"]:
                            out_cam0, p_dets0, p_count0 = anpr_detector.process_frame(out_cam0)
                            for det in p_dets0:
                                p_text = det.get("text", "")
                                if p_text:
                                    is_new, cleaned = plate_recorder.log_plate(
                                        plate_text=p_text,
                                        confidence=det.get("conf", 0.0),
                                        camera_name="Video Source",
                                        plate_crop_bgr=det.get("crop"),
                                    )
                                    if is_new:
                                        st.session_state.latest_plates.insert(
                                            0,
                                            {"text": cleaned, "time": datetime.now().strftime("%H:%M:%S"), "cam": "Stream"}
                                        )
                                        if len(st.session_state.latest_plates) > 6:
                                            st.session_state.latest_plates.pop()

                        draw_header_banner(out_cam0, "Source", "Active Feed", h_count0, f_count0, p_count0)
                        out_frame = out_cam0

                    curr_time = time.time()
                    fps = 1.0 / max(0.001, curr_time - prev_time)
                    prev_time = curr_time

                    total_humans = h_count0 + h_count1
                    total_faces = f_count0 + f_count1
                    active_alerts = alert_unk0 + alert_unk1
                    total_unknown_faces = total_unk0 + total_unk1
                    total_plates = p_count0 + p_count1

                    if active_alerts > 0:
                        cyber_monitor.record_unauthorized_face_event(active_alerts, "Live Surveillance Feed")


                    fps_metric.markdown(f'<div class="metric-card"><p class="metric-lbl">Live FPS</p><p class="metric-val">{fps:.1f}</p></div>', unsafe_allow_html=True)
                    humans_metric.markdown(f'<div class="metric-card"><p class="metric-lbl">Humans</p><p class="metric-val">{total_humans}</p></div>', unsafe_allow_html=True)
                    faces_metric.markdown(f'<div class="metric-card"><p class="metric-lbl">Faces</p><p class="metric-val">{total_faces}</p></div>', unsafe_allow_html=True)
                    plates_metric.markdown(f'<div class="metric-card"><p class="metric-lbl">Plates</p><p class="metric-val">{total_plates}</p></div>', unsafe_allow_html=True)
                    total_plates_metric.markdown(f'<div class="metric-card"><p class="metric-lbl">Logged</p><p class="metric-val">{len(plate_recorder.recent_logs)}</p></div>', unsafe_allow_html=True)

                    if st.session_state.latest_plates:
                        alerts_html = ""
                        for p in st.session_state.latest_plates:
                            alerts_html += f"""
                            <div class="plate-card">
                                <div>
                                    <span class="plate-text">{p['text']}</span>
                                    <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">{p['cam']}</div>
                                </div>
                                <div style="font-size: 12px; color: #00ff8c; font-weight: 600;">{p['time']}</div>
                            </div>
                            """
                        alerts_placeholder.markdown(alerts_html, unsafe_allow_html=True)
                    else:
                        alerts_placeholder.caption("No number plates logged in current session.")

                    if out_frame is not None:
                        frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    time.sleep(0.005)

            finally:
                if cam_manager is not None:
                    cam_manager.release()
                if cap_single is not None and cap_single.isOpened():
                    cap_single.release()


if __name__ == "__main__":
    main()
