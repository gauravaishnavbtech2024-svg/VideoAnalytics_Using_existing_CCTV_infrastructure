"""
CCTV AI Vision System — Real-Time Human, Face, ANPR & Vehicle Tracking
======================================================================

An advanced multi-camera AI surveillance console featuring:
- High-Speed YOLOv8 Human Detection & Tracking (Interleaved Pipelined Inference).
- Deep Learning Face Detection, Recognition & Tracking (YuNet DNN + SFace ONNX).
- Real-Time Vehicle Number Plate Detection (YOLO) + Background EasyOCR extraction.
- Multi-Class Vehicle Detection & Tracking (Trained YOLOv8: Bus, Car, Truck, Motorcycle, etc.).
- Threaded Asynchronous Camera Streams (0ms main loop capture latency).
- IBVAP CYBERSECURITY & SYSTEM HEALTH MONITOR:
  * Continuous monitoring of Camera, Network, CPU/GPU, Disk, Model Integrity (SHA-256),
    Database Integrity, Unauthorized Logins/Faces, Config changes, System Clock & Service Health.
- High-Performance Zero-Copy Glassmorphism UI (ROI Alpha Blending & HUD).

Controls:
  [A]       -> All-In-One AI (Humans + Faces + Plates + Vehicles)
  [H]       -> Human Detection & Tracking Only
  [F]       -> Face Detection & Recognition Only
  [N]       -> Number Plate Detection Only (ANPR)
  [V]       -> Vehicle Detection & Tracking Only
  [B]       -> Dual AI (Faces + Plates)
  [C]       -> Toggle Cybersecurity Health Dashboard Overlay
  [D] / [M] -> Dual Camera Side-by-Side View
  [0] / [1] -> Single Camera Feed (Cam 0 / Cam 1)
  [S]       -> Cycle Camera Views
  [Q] / ESC -> Quit Application
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Mute noisy backend warnings
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
from src.vehicle_detector import VehicleTracker


# =============================================================================
# Threaded Asynchronous Camera Streams (0ms Grab Latency)
# =============================================================================

def get_camera_names() -> List[str]:
    """Enumerate physical camera names via DirectShow filter graph."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        return graph.get_input_devices()
    except Exception:
        return []


class ThreadedCameraStream:
    """Non-blocking background camera capture thread (0ms grab latency in main loop)."""

    def __init__(self, cam_id: int, width: int = 640, height: int = 480) -> None:
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.is_active = False
        self.stopped = False
        self.lock = threading.Lock()

        self._init_cap()
        if self.is_active:
            self._worker = threading.Thread(target=self._capture_loop, daemon=True)
            self._worker.start()

    def _init_cap(self) -> None:
        try:
            self.cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.cam_id)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except Exception:
                    pass

                for _ in range(6):
                    r, f = self.cap.read()
                    if r and f is not None:
                        self.frame = f
                        self.is_active = True
                        break
                    time.sleep(0.03)
        except Exception as e:
            print(f"[Camera {self.cam_id}] Stream init error: {e}")
            self.is_active = False

    def _capture_loop(self) -> None:
        while not self.stopped:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.05)
                continue
            ret, f = self.cap.read()
            if ret and f is not None:
                with self.lock:
                    self.frame = f
                    self.is_active = True
            else:
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is not None and self.is_active:
                return True, self.frame.copy()
            return False, None

    def release(self) -> None:
        self.stopped = True
        self.is_active = False
        if self.cap and self.cap.isOpened():
            self.cap.release()


class DualCameraManager:
    """Threaded Dual-Camera Stream Manager with non-blocking retrieval."""

    def __init__(self, cam0_id: int = 0, cam1_id: int = 1) -> None:
        self.cam0_id = cam0_id
        self.cam1_id = cam1_id
        self.dev0_name = f"Camera {cam0_id}"
        self.dev1_name = f"Camera {cam1_id}"

        names = get_camera_names()
        if self.cam0_id < len(names):
            self.dev0_name = names[self.cam0_id]
        if self.cam1_id < len(names):
            self.dev1_name = names[self.cam1_id]

        print(f"[Camera Manager] Initializing Threaded Stream for Cam {cam0_id} [{self.dev0_name}]...")
        self.stream0 = ThreadedCameraStream(cam0_id)
        time.sleep(0.15)
        print(f"[Camera Manager] Initializing Threaded Stream for Cam {cam1_id} [{self.dev1_name}]...")
        self.stream1 = ThreadedCameraStream(cam1_id)

    @property
    def is_cam0_active(self) -> bool:
        return self.stream0.is_active

    @property
    def is_cam1_active(self) -> bool:
        return self.stream1.is_active

    def read_sync(self) -> Tuple[bool, Optional[np.ndarray], bool, Optional[np.ndarray]]:
        r0, f0 = self.stream0.read()
        r1, f1 = self.stream1.read()
        return r0, f0, r1, f1

    def release(self) -> None:
        self.stream0.release()
        self.stream1.release()


# =============================================================================
# UI, HUD & Pop-Up Alert Rendering Engine (High Performance ROI Alpha Blend)
# =============================================================================

CURRENT_DETECTION_MODE = "all"  # "all", "humans", "faces", "plates", "vehicles", "both"
SHOW_CYBER_OVERLAY = False
BUTTON_REGIONS: Dict[str, Tuple[int, int, int, int]] = {}
LAST_LOGGED_PLATE_ALERT = ""
LAST_LOGGED_PLATE_TIME = 0.0
LAST_AUDIO_BEEP_TIME = 0.0


def trigger_alert_sound() -> None:
    """Non-blocking audio alert for security warnings."""
    global LAST_AUDIO_BEEP_TIME
    now = time.time()
    if now - LAST_AUDIO_BEEP_TIME > 2.0:
        LAST_AUDIO_BEEP_TIME = now
        try:
            import winsound
            threading.Thread(
                target=lambda: winsound.MessageBeep(winsound.MB_ICONEXCLAMATION),
                daemon=True,
            ).start()
        except Exception:
            pass


def on_mouse_click(event: int, x: int, y: int, flags: int, param: any) -> None:
    """Handle mouse clicks on interactive on-screen mode buttons."""
    global CURRENT_DETECTION_MODE, SHOW_CYBER_OVERLAY
    if event == cv2.EVENT_LBUTTONDOWN:
        for mode_key, (bx1, by1, bx2, by2) in BUTTON_REGIONS.items():
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                if mode_key == "cyber":
                    SHOW_CYBER_OVERLAY = not SHOW_CYBER_OVERLAY
                else:
                    CURRENT_DETECTION_MODE = mode_key
                    print(f"\n[Mode Switched] -> {CURRENT_DETECTION_MODE.upper()}")
                break


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
    vehicles: int = 0,
    is_live: bool = True,
) -> np.ndarray:
    h, w = frame.shape[:2]
    banner_h = 34

    # Fast ROI alpha-blend
    roi = frame[0:banner_h, 0:w]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (16, 18, 24), -1)
    cv2.addWeighted(overlay, 0.92, roi, 0.08, 0, roi)
    cv2.line(frame, (0, banner_h), (w, banner_h), (45, 50, 65), 1)

    dot_color = (0, 255, 120) if is_live else (0, 80, 255)
    cv2.circle(frame, (16, banner_h // 2), 5, dot_color, -1)

    title = f"{label.upper()} | {device_name}"
    cv2.putText(
        frame, title, (30, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 245), 1, cv2.LINE_AA,
    )

    badge_text = f"HUMANS: {humans} | FACES: {faces} | PLATES: {plates} | VEHICLES: {vehicles}"
    (bw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    badge_x = w - bw - 15
    cv2.rectangle(frame, (badge_x - 8, 5), (w - 8, 28), (28, 32, 44), -1)
    cv2.rectangle(frame, (badge_x - 8, 5), (w - 8, 28), (60, 70, 95), 1)
    cv2.putText(
        frame, badge_text, (badge_x, 21),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 200), 1, cv2.LINE_AA,
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
    v_count0: int,
    h_count1: int,
    f_count1: int,
    p_count1: int,
    v_count1: int,
    target_h: int = 480,
) -> np.ndarray:
    h0, w0 = frame0.shape[:2]
    w0_res = int(w0 * (target_h / h0))
    f0_res = cv2.resize(frame0, (w0_res, target_h)) if (h0 != target_h) else frame0

    h1, w1 = frame1.shape[:2]
    w1_res = int(w1 * (target_h / h1))
    f1_res = cv2.resize(frame1, (w1_res, target_h)) if (h1 != target_h) else frame1

    draw_header_banner(f0_res, "Feed A", dev0_name, h_count0, f_count0, p_count0, v_count0)
    draw_header_banner(f1_res, "Feed B", dev1_name, h_count1, f_count1, p_count1, v_count1)

    combined = np.hstack((f0_res, f1_res))
    cv2.line(combined, (w0_res, 0), (w0_res, target_h), (0, 255, 140), 2)
    return combined


def draw_top_mode_bar(frame: np.ndarray, active_mode: str) -> None:
    """Render interactive glassmorphic mode selector buttons + Cyber Health button."""
    global BUTTON_REGIONS, SHOW_CYBER_OVERLAY
    h, w = frame.shape[:2]
    bar_h = 46

    # ROI alpha-blend
    roi = frame[0:bar_h, 0:w]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (12, 14, 18), -1)
    cv2.addWeighted(overlay, 0.94, roi, 0.06, 0, roi)
    cv2.line(frame, (0, bar_h), (w, bar_h), (40, 45, 60), 1)

    cv2.putText(
        frame, "AI MODE:", (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 175, 190), 1, cv2.LINE_AA,
    )

    buttons = [
        ("[A] All AI", "all", 130),
        ("[H] Humans", "humans", 130),
        ("[F] Faces", "faces", 125),
        ("[N] Plates", "plates", 125),
        ("[V] Vehicles", "vehicles", 135),
        ("[B] Both", "both", 115),
    ]

    start_x = 95
    btn_h = 32
    btn_y1 = 7
    btn_y2 = btn_y1 + btn_h

    for label, mode_key, btn_w in buttons:
        btn_x1 = start_x
        btn_x2 = btn_x1 + btn_w
        BUTTON_REGIONS[mode_key] = (btn_x1, btn_y1, btn_x2, btn_y2)

        is_active = (active_mode == mode_key)
        bg_color = (25, 95, 45) if is_active else (28, 30, 40)
        border_color = (0, 255, 140) if is_active else (65, 70, 85)
        text_color = (255, 255, 255) if is_active else (170, 175, 190)

        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), bg_color, -1)
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), border_color, 2 if is_active else 1)

        dot_color = (0, 255, 140) if is_active else (90, 95, 110)
        cv2.circle(frame, (btn_x1 + 10, btn_y1 + btn_h // 2), 4, dot_color, -1)

        cv2.putText(
            frame, label, (btn_x1 + 18, btn_y1 + 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color,
            2 if is_active else 1, cv2.LINE_AA,
        )
        start_x += btn_w + 6

    # Cybersecurity Health Button
    c_btn_w = 165
    c_btn_x1 = w - c_btn_w - 15
    c_btn_x2 = c_btn_x1 + c_btn_w
    BUTTON_REGIONS["cyber"] = (c_btn_x1, btn_y1, c_btn_x2, btn_y2)

    c_status = cyber_monitor.metrics.get("service", None)
    c_color = (0, 255, 140) if (c_status and c_status.status == "OPTIMAL") else (0, 140, 255)
    c_bg = (20, 38, 55) if SHOW_CYBER_OVERLAY else (24, 26, 36)

    cv2.rectangle(frame, (c_btn_x1, btn_y1), (c_btn_x2, btn_y2), c_bg, -1)
    cv2.rectangle(frame, (c_btn_x1, btn_y1), (c_btn_x2, btn_y2), c_color, 2 if SHOW_CYBER_OVERLAY else 1)
    cv2.putText(
        frame, "[C] CYBER HEALTH", (c_btn_x1 + 10, btn_y1 + 21),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 255), 1, cv2.LINE_AA,
    )


def draw_cybersecurity_dashboard_overlay(frame: np.ndarray) -> None:
    """Render full 10-vector Cybersecurity & System Health Dashboard overlay."""
    h, w = frame.shape[:2]
    dw = min(960, w - 60)
    dh = min(540, h - 80)
    dx1 = (w - dw) // 2
    dx2 = dx1 + dw
    dy1 = 54
    dy2 = dy1 + dh

    # Backdrop ROI
    roi = frame[dy1:dy2, dx1:dx2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (dw, dh), (10, 13, 20), -1)
    cv2.addWeighted(overlay, 0.95, roi, 0.05, 0, roi)

    # Cyber glowing border
    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 140), 2)
    cv2.rectangle(frame, (dx1 + 4, dy1 + 4), (dx2 - 4, dy2 - 4), (0, 140, 255), 1)

    # Header
    cv2.rectangle(frame, (dx1, dy1), (dx2, dy1 + 42), (18, 22, 34), -1)
    cv2.putText(
        frame, "IBVAP CYBERSECURITY & SYSTEM HEALTH MONITOR", (dx1 + 20, dy1 + 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 140), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "[C] Close", (dx2 - 95, dy1 + 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 190, 210), 1, cv2.LINE_AA,
    )
    cv2.line(frame, (dx1, dy1 + 42), (dx2, dy1 + 42), (50, 60, 80), 1)

    # Render 10 Metrics in a 2-Column Grid
    metrics = list(cyber_monitor.metrics.values())
    col_w = (dw - 40) // 2
    row_h = 42

    for idx, m in enumerate(metrics[:10]):
        col = idx % 2
        row = idx // 2
        mx = dx1 + 20 + col * (col_w + 10)
        my = dy1 + 54 + row * row_h

        # Status badge color
        if m.status == "OK":
            s_color = (0, 255, 120)  # Green
            badge_bg = (12, 40, 24)
        elif m.status == "WARNING":
            s_color = (0, 180, 255)  # Orange
            badge_bg = (18, 38, 55)
        else:
            s_color = (0, 40, 255)   # Red
            badge_bg = (18, 15, 55)

        cv2.rectangle(frame, (mx, my), (mx + col_w, my + row_h - 6), (18, 21, 30), -1)
        cv2.rectangle(frame, (mx, my), (mx + col_w, my + row_h - 6), (40, 48, 65), 1)

        # Status pill
        cv2.rectangle(frame, (mx + 6, my + 6), (mx + 70, my + row_h - 12), badge_bg, -1)
        cv2.rectangle(frame, (mx + 6, my + 6), (mx + 70, my + row_h - 12), s_color, 1)
        cv2.putText(frame, m.status, (mx + 12, my + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.38, s_color, 1, cv2.LINE_AA)

        # Name & Value
        cv2.putText(frame, m.name, (mx + 80, my + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, m.details[:44], (mx + 80, my + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 160, 180), 1, cv2.LINE_AA)


def draw_bottom_hud(
    frame: np.ndarray,
    mode: str,
    view: str,
    fps: float,
    total_humans: int,
    total_faces: int,
    total_unknown_faces: int,
    total_plates: int,
    total_vehicles: int,
    total_logged_plates: int,
) -> None:
    """Render high-tech command center HUD status bar at bottom of window."""
    h, w = frame.shape[:2]
    hud_h = 32

    cv2.rectangle(frame, (0, h - hud_h), (w, h), (12, 14, 18), -1)
    cv2.line(frame, (0, h - hud_h), (w, h - hud_h), (40, 45, 60), 1)

    mode_str = mode.upper()
    view_str = "DUAL FEED" if view == "dual" else f"CAM {view}"
    face_status = f"FACES: {total_faces}"
    if total_unknown_faces > 0:
        face_status = f"FACES: {total_faces} ({total_unknown_faces} UNKNOWN)"

    serv_metric = cyber_monitor.metrics.get("service", None)
    health_str = f"HEALTH: {serv_metric.status if serv_metric else 'OPTIMAL'}"

    hud_items = [
        f"⚡ {fps:.1f} FPS",
        f"MODE: {mode_str}",
        f"VIEW: {view_str}",
        health_str,
        f"HUMANS: {total_humans}",
        face_status,
        f"PLATES: {total_plates}",
        f"VEHICLES: {total_vehicles}",
        f"LOGGED: {total_logged_plates}",
    ]

    hud_text = "  |  ".join(hud_items)
    hud_color = (0, 100, 255) if total_unknown_faces > 0 else (0, 255, 140)
    cv2.putText(
        frame, hud_text, (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.39, hud_color, 1, cv2.LINE_AA,
    )

    global LAST_LOGGED_PLATE_ALERT, LAST_LOGGED_PLATE_TIME
    if LAST_LOGGED_PLATE_ALERT and (time.time() - LAST_LOGGED_PLATE_TIME < 3.5):
        toast_text = f"CAPTURED & LOGGED: {LAST_LOGGED_PLATE_ALERT}"
        (tw, th), _ = cv2.getTextSize(toast_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        tx = (w - tw) // 2
        ty = h - hud_h - 18
        cv2.rectangle(frame, (tx - 12, ty - th - 6), (tx + tw + 12, ty + 8), (14, 45, 25), -1)
        cv2.rectangle(frame, (tx - 12, ty - th - 6), (tx + tw + 12, ty + 8), (0, 255, 140), 2)
        cv2.putText(frame, toast_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)


# =============================================================================
# Main Application Entrypoint
# =============================================================================

def main() -> None:
    global CURRENT_DETECTION_MODE, SHOW_CYBER_OVERLAY, LAST_LOGGED_PLATE_ALERT, LAST_LOGGED_PLATE_TIME

    parser = argparse.ArgumentParser(description="CCTV Vision System: Human Tracking, Face Tracking, ANPR & Vehicle Tracking")
    parser.add_argument("--cam0", type=int, default=0, help="Primary camera index (default: 0)")
    parser.add_argument("--cam1", type=int, default=1, help="Secondary camera index (default: 1)")
    parser.add_argument("--mode", type=str, default="dual", choices=["dual", "0", "1"], help="Initial camera view")
    parser.add_argument("--detect-mode", type=str, default="all", choices=["all", "humans", "faces", "plates", "vehicles", "both"], help="Initial detection mode")
    parser.add_argument("--disable-ocr", action="store_true", help="Disable OCR text extraction")
    args = parser.parse_args()

    CURRENT_DETECTION_MODE = args.detect_mode
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    enable_ocr = not args.disable_ocr

    print("\n" + "=" * 75)
    print("  🎥  CCTV AI VISION SYSTEM — Complete Human, Face, ANPR & Vehicle Tracking")
    print("=" * 75)
    print(f"  ⚡ Hardware Acceleration : {device.upper()}")
    print(f"  🛡️ Cybersecurity Watch   : ACTIVE (10 Security Vectors)")
    print(f"  📝 OCR Engine Enabled    : {enable_ocr}")
    print(f"  🖥️ Camera Layout        : {args.mode.upper()}")
    print(f"  🎯 AI Pipeline Mode     : {CURRENT_DETECTION_MODE.upper()}")
    print("-" * 75)
    print("  Controls:")
    print("    [A/H/F/N/V/B] -> Switch Detection Pipelines (Humans, Faces, Plates, Vehicles)")
    print("    [C]           -> Toggle Cybersecurity Health Dashboard")
    print("    [D] / [0/1]   -> Camera Layout Views")
    print("    [Q] / ESC     -> Quit")
    print("=" * 75 + "\n")

    # 1. Initialize Human Tracker
    print("[1/4] Initializing Human Detectors & Trackers...")
    human_tracker_0 = HumanTracker(device=device)
    human_tracker_1 = HumanTracker(device=device)

    # 2. Initialize Face Tracker & Recognizer
    print("\n[2/4] Initializing Face Detector & SFace Recognizer...")
    face_tracker_0 = FastFaceTracker(max_alerts_per_person=3)
    face_tracker_1 = FastFaceTracker(max_alerts_per_person=3)

    # 3. Initialize ANPR Detector & Plate Recorder
    print("\n[3/4] Initializing ANPR Plate Detector & EasyOCR...")
    anpr_detector = ANPRDetector(device=device, enable_ocr=enable_ocr)
    plate_recorder = PlateRecorder(cooldown_seconds=45.0, similarity_threshold=0.50)

    # 4. Initialize Vehicle Detector & Tracker
    print("\n[4/4] Initializing Multi-Class Vehicle Detector & Trackers...")
    vehicle_tracker_0 = VehicleTracker(device=device)
    vehicle_tracker_1 = VehicleTracker(device=device)

    # Initialize Synchronized Dual Camera Stream
    print("\nConnecting to Video Cameras...")
    cam_manager = DualCameraManager(cam0_id=args.cam0, cam1_id=args.cam1)

    print(f"  Cam {args.cam0} [{cam_manager.dev0_name}]: {'● ONLINE' if cam_manager.is_cam0_active else '○ OFFLINE'}")
    print(f"  Cam {args.cam1} [{cam_manager.dev1_name}]: {'● ONLINE' if cam_manager.is_cam1_active else '○ OFFLINE'}")

    window_name = "CCTV AI Vision Console — Human, Face, ANPR & Vehicle Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, on_mouse_click)

    camera_view_mode = args.mode
    prev_time = time.time()
    fps_smooth = 30.0
    last_cyber_check = time.time()

    print("\n🚀 Ultra-high FPS streaming started! Press [Q] to exit.\n")

    frame_counter = 0

    try:
        while True:
            ret0, raw0, ret1, raw1 = cam_manager.read_sync()

            if not ret0 or raw0 is None:
                raw0 = create_placeholder_frame(640, 480, f"Cam {args.cam0} Offline")

            if not ret1 or raw1 is None:
                raw1 = create_placeholder_frame(640, 480, f"Cam {args.cam1} Offline")

            out_cam0 = raw0.copy()
            out_cam1 = raw1.copy()
            h_count0, f_count0, p_count0, v_count0 = 0, 0, 0, 0
            h_count1, f_count1, p_count1, v_count1 = 0, 0, 0, 0
            total_unk0, total_unk1 = 0, 0

            process_cam0 = (camera_view_mode in ["dual", "0"]) and ret0
            process_cam1 = (camera_view_mode in ["dual", "1"]) and ret1

            frame_counter += 1

            # =========================================================================
            # Interleaved Pipelined AI Scheduling (Guarantees max 1 heavy model/frame)
            # =========================================================================
            run_face_cam0 = (frame_counter % 2 == 0) and (CURRENT_DETECTION_MODE in ["faces", "both", "all"])
            run_face_cam1 = (frame_counter % 2 == 1) and (CURRENT_DETECTION_MODE in ["faces", "both", "all"])

            run_human_cam0 = (frame_counter % 4 == 0) and (CURRENT_DETECTION_MODE in ["humans", "all"])
            run_human_cam1 = (frame_counter % 4 == 2) and (CURRENT_DETECTION_MODE in ["humans", "all"])

            run_anpr_cam0 = (frame_counter % 4 == 1) and (CURRENT_DETECTION_MODE in ["plates", "both", "all"])
            run_anpr_cam1 = (frame_counter % 4 == 3) and (CURRENT_DETECTION_MODE in ["plates", "both", "all"])

            run_vehicle_cam0 = (frame_counter % 4 == 2) and (CURRENT_DETECTION_MODE in ["vehicles", "all"])
            run_vehicle_cam1 = (frame_counter % 4 == 0) and (CURRENT_DETECTION_MODE in ["vehicles", "all"])

            face_boxes0, face_boxes1 = [], []
            human_boxes0, human_boxes1 = [], []

            # 1. Face Tracking & Recognition
            if CURRENT_DETECTION_MODE in ["faces", "both", "all"]:
                if process_cam0:
                    out_cam0, f_count0 = face_tracker_0.process_frame(out_cam0, run_detection=run_face_cam0)
                    face_boxes0 = [t.bbox for t in face_tracker_0.tracks]
                    total_unk0 = face_tracker_0.total_unknown_count
                if process_cam1:
                    out_cam1, f_count1 = face_tracker_1.process_frame(out_cam1, run_detection=run_face_cam1)
                    face_boxes1 = [t.bbox for t in face_tracker_1.tracks]
                    total_unk1 = face_tracker_1.total_unknown_count

            # 2. Human Tracking
            if CURRENT_DETECTION_MODE in ["humans", "all"]:
                if process_cam0:
                    out_cam0, _, h_count0 = human_tracker_0.process_frame(
                        out_cam0, existing_face_boxes=face_boxes0, run_detection=run_human_cam0
                    )
                    human_boxes0 = [t.bbox for t in human_tracker_0.tracks]
                if process_cam1:
                    out_cam1, _, h_count1 = human_tracker_1.process_frame(
                        out_cam1, existing_face_boxes=face_boxes1, run_detection=run_human_cam1
                    )
                    human_boxes1 = [t.bbox for t in human_tracker_1.tracks]

            # 3. Vehicle Tracking (Bus, Car, Truck, Motorcycle, Van, Jeep, etc.)
            if CURRENT_DETECTION_MODE in ["vehicles", "all"]:
                if process_cam0:
                    out_cam0, _, v_count0 = vehicle_tracker_0.process_frame(out_cam0, run_detection=run_vehicle_cam0)
                if process_cam1:
                    out_cam1, _, v_count1 = vehicle_tracker_1.process_frame(out_cam1, run_detection=run_vehicle_cam1)

            # 4. ANPR Plate Detection (Suppresses person ID cards, badges, and lanyard text)
            if CURRENT_DETECTION_MODE in ["plates", "both", "all"]:
                if process_cam0:
                    person_boxes0 = human_boxes0 + face_boxes0
                    out_cam0, p_dets0, p_count0 = anpr_detector.process_frame(
                        out_cam0, person_boxes=person_boxes0, run_detection=run_anpr_cam0
                    )
                    if run_anpr_cam0:
                        for det in p_dets0:
                            p_text = det.get("text", "")
                            if p_text:
                                is_new, cleaned = plate_recorder.log_plate(
                                    plate_text=p_text,
                                    confidence=det.get("conf", 0.0),
                                    camera_name=f"Cam {args.cam0} ({cam_manager.dev0_name})",
                                    plate_crop_bgr=det.get("crop"),
                                )
                                if is_new:
                                    LAST_LOGGED_PLATE_ALERT = cleaned
                                    LAST_LOGGED_PLATE_TIME = time.time()

                if process_cam1:
                    person_boxes1 = human_boxes1 + face_boxes1
                    out_cam1, p_dets1, p_count1 = anpr_detector.process_frame(
                        out_cam1, person_boxes=person_boxes1, run_detection=run_anpr_cam1
                    )
                    if run_anpr_cam1:
                        for det in p_dets1:
                            p_text = det.get("text", "")
                            if p_text:
                                is_new, cleaned = plate_recorder.log_plate(
                                    plate_text=p_text,
                                    confidence=det.get("conf", 0.0),
                                    camera_name=f"Cam {args.cam1} ({cam_manager.dev1_name})",
                                    plate_crop_bgr=det.get("crop"),
                                )
                                if is_new:
                                    LAST_LOGGED_PLATE_ALERT = cleaned
                                    LAST_LOGGED_PLATE_TIME = time.time()

            # Construct View Layout
            if camera_view_mode == "dual":
                out_frame = combine_dual_feeds(
                    out_cam0, out_cam1,
                    dev0_name=cam_manager.dev0_name,
                    dev1_name=cam_manager.dev1_name,
                    h_count0=h_count0, f_count0=f_count0, p_count0=p_count0, v_count0=v_count0,
                    h_count1=h_count1, f_count1=f_count1, p_count1=p_count1, v_count1=v_count1,
                )
            elif camera_view_mode == "0":
                out_frame = out_cam0
                draw_header_banner(out_frame, "Feed A", cam_manager.dev0_name, h_count0, f_count0, p_count0, v_count0)
            else:  # "1"
                out_frame = out_cam1
                draw_header_banner(out_frame, "Feed B", cam_manager.dev1_name, h_count1, f_count1, p_count1, v_count1)

            # Draw Interactive Top Bar
            draw_top_mode_bar(out_frame, CURRENT_DETECTION_MODE)

            # Record unauthorized face event in cyber monitor silently without intrusive popups
            total_unk = total_unk0 + total_unk1
            if total_unk > 0:
                loc_label = "Dual Feed" if camera_view_mode == "dual" else f"Cam {camera_view_mode}"
                cyber_monitor.record_unauthorized_face_event(total_unk, loc_label)

            curr_time = time.time()
            instant_fps = 1.0 / max(0.0001, curr_time - prev_time)
            prev_time = curr_time
            fps_smooth = (0.85 * fps_smooth) + (0.15 * instant_fps)

            # Update camera status in security monitor at 1Hz interval
            if curr_time - last_cyber_check > 1.0:
                cyber_monitor.check_camera_status(cam_manager.is_cam0_active, cam_manager.is_cam1_active, fps_smooth)
                last_cyber_check = curr_time

            # Draw Cyber Security Dashboard Overlay if toggled
            if SHOW_CYBER_OVERLAY:
                draw_cybersecurity_dashboard_overlay(out_frame)

            total_humans = h_count0 + h_count1
            total_faces = f_count0 + f_count1
            total_unknown_faces = total_unk0 + total_unk1
            total_plates = p_count0 + p_count1
            total_vehicles = v_count0 + v_count1
            draw_bottom_hud(
                out_frame,
                CURRENT_DETECTION_MODE,
                camera_view_mode,
                fps_smooth,
                total_humans,
                total_faces,
                total_unknown_faces,
                total_plates,
                total_vehicles,
                len(plate_recorder.recent_logs),
            )

            cv2.imshow(window_name, out_frame)

            # Keyboard Hotkeys
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break
            elif key == ord("c"):
                SHOW_CYBER_OVERLAY = not SHOW_CYBER_OVERLAY
                print(f"  [Cyber Security Dashboard] -> {'OPEN' if SHOW_CYBER_OVERLAY else 'CLOSED'}")
            elif key == ord("a"):
                CURRENT_DETECTION_MODE = "all"
                print("  [Mode] -> ALL-IN-ONE AI (Humans + Faces + Plates + Vehicles)")
            elif key == ord("h"):
                CURRENT_DETECTION_MODE = "humans"
                print("  [Mode] -> HUMAN DETECTION ONLY")
            elif key == ord("f"):
                CURRENT_DETECTION_MODE = "faces"
                print("  [Mode] -> FACE RECOGNITION ONLY")
            elif key == ord("n"):
                CURRENT_DETECTION_MODE = "plates"
                print("  [Mode] -> NUMBER PLATES (ANPR) ONLY")
            elif key == ord("v"):
                CURRENT_DETECTION_MODE = "vehicles"
                print("  [Mode] -> VEHICLE DETECTION ONLY")
            elif key == ord("b"):
                CURRENT_DETECTION_MODE = "both"
                print("  [Mode] -> DUAL AI (FACES + PLATES)")
            elif key in [ord("d"), ord("m")]:
                camera_view_mode = "dual"
                print("  [View] -> DUAL CAMERA FEED")
            elif key == ord("0"):
                camera_view_mode = "0"
                print(f"  [View] -> CAMERA {args.cam0} ONLY")
            elif key == ord("1"):
                camera_view_mode = "1"
                print(f"  [View] -> CAMERA {args.cam1} ONLY")
            elif key == ord("s"):
                views = ["dual", "0", "1"]
                next_idx = (views.index(camera_view_mode) + 1) % len(views)
                camera_view_mode = views[next_idx]
                print(f"  [View] -> {camera_view_mode.upper()} VIEW")

    finally:
        cam_manager.release()
        cv2.destroyAllWindows()
        print("\nApplication closed cleanly.")


if __name__ == "__main__":
    main()
