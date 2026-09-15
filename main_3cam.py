"""
CCTV AI Vision System — Real-Time Human Tracking, Face Tracking & ANPR (3 Cameras)
====================================================================================

An advanced triple-camera AI surveillance console featuring:
- Trained YOLOv8 Human Detection & Tracking.
- Deep Learning Face Detection, Recognition & Tracking (YuNet DNN + SFace ONNX).
- Real-Time Vehicle Number Plate Detection (YOLO) + EasyOCR text extraction.
- IBVAP CYBERSECURITY & SYSTEM HEALTH MONITOR:
  * Continuous monitoring of Camera, Network, CPU/GPU, Disk, Model Integrity (SHA-256),
    Database Integrity, Unauthorized Logins/Faces, Config changes, System Clock & Service Health.
- Interactive Glassmorphism UI (On-Screen Mode Buttons & HUD).
- Synchronized Triple-Camera DirectShow Streaming with zero thread contention.

Controls:
  [A]       -> All-In-One AI (Humans + Faces + Plates)
  [H]       -> Human Detection & Tracking Only
  [F]       -> Face Detection & Recognition Only
  [N]       -> Number Plate Detection Only (ANPR)
  [B]       -> Dual AI (Faces + Plates)
  [C]       -> Toggle Cybersecurity Health Dashboard Overlay
  [T]       -> Triple Camera View (All 3)
  [D]       -> Dual View (Cam 0 + Cam 1)
  [0/1/2]   -> Single Camera Feed (Cam 0 / Cam 1 / Cam 2)
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


# =============================================================================
# Synchronized Triple Camera Manager
# =============================================================================

def get_camera_names() -> List[str]:
    """Enumerate physical camera names via DirectShow filter graph."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        return graph.get_input_devices()
    except Exception:
        return []


class TripleCameraManager:
    """Synchronized 3-Camera Stream Manager using DirectShow grab & retrieve."""

    def __init__(self, cam0_id: int = 0, cam1_id: int = 1, cam2_id: int = 2) -> None:
        self.cam_ids = [cam0_id, cam1_id, cam2_id]
        self.caps: List[Optional[cv2.VideoCapture]] = [None, None, None]
        self.dev_names: List[str] = [f"Camera {cam0_id}", f"Camera {cam1_id}", f"Camera {cam2_id}"]
        self.is_active: List[bool] = [False, False, False]

        self.init_cameras()

    def init_cameras(self) -> None:
        """Initialize all 3 cameras with staggered DirectShow binding."""
        names = get_camera_names()
        for i, cam_id in enumerate(self.cam_ids):
            if cam_id < len(names):
                self.dev_names[i] = names[cam_id]

        for i, cam_id in enumerate(self.cam_ids):
            try:
                cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    for _ in range(8):
                        r, f = cap.read()
                        if r and f is not None:
                            self.is_active[i] = True
                            break
                        time.sleep(0.04)
                self.caps[i] = cap
            except Exception as e:
                print(f"[Camera {cam_id}] Init error: {e}")
                self.is_active[i] = False

            if i < 2:
                time.sleep(0.20)

    def read_sync(self) -> List[Tuple[bool, Optional[np.ndarray]]]:
        """Synchronously grab and retrieve frames from all 3 cameras."""
        grabs = []
        for i in range(3):
            if self.is_active[i] and self.caps[i] and self.caps[i].isOpened():
                grabs.append(self.caps[i].grab())
            else:
                grabs.append(False)

        results = []
        for i in range(3):
            if grabs[i]:
                r, f = self.caps[i].retrieve()
                results.append((r, f))
            else:
                results.append((False, None))

        return results

    def release(self) -> None:
        """Release hardware handles."""
        for i in range(3):
            if self.caps[i] and self.caps[i].isOpened():
                self.caps[i].release()
            self.is_active[i] = False


# =============================================================================
# UI, HUD & Pop-Up Alert Rendering Engine
# =============================================================================

CURRENT_DETECTION_MODE = "all"  # "all", "humans", "faces", "plates", "both"
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
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 245), 1, cv2.LINE_AA,
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


def combine_triple_feeds(
    frames: List[np.ndarray],
    dev_names: List[str],
    h_counts: List[int],
    f_counts: List[int],
    p_counts: List[int],
    target_h: int = 360,
) -> np.ndarray:
    """Combine 3 camera feeds into a single horizontal strip with headers."""
    resized = []
    labels = ["Feed A", "Feed B", "Feed C"]

    for i, frame in enumerate(frames):
        fh, fw = frame.shape[:2]
        new_w = int(fw * (target_h / fh))
        r = cv2.resize(frame, (new_w, target_h))
        draw_header_banner(r, labels[i], dev_names[i], h_counts[i], f_counts[i], p_counts[i])
        resized.append(r)

    combined = np.hstack(resized)

    # Draw vertical separator lines
    x_offset = 0
    for i in range(len(resized) - 1):
        x_offset += resized[i].shape[1]
        cv2.line(combined, (x_offset, 0), (x_offset, target_h), (0, 255, 140), 2)

    return combined


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


def draw_top_mode_bar(frame: np.ndarray, active_mode: str) -> None:
    """Render interactive glassmorphic mode selector buttons + Cyber Health button."""
    global BUTTON_REGIONS, SHOW_CYBER_OVERLAY
    h, w = frame.shape[:2]
    bar_h = 46

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (12, 14, 18), -1)
    cv2.addWeighted(overlay, 0.94, frame, 0.06, 0, frame)
    cv2.line(frame, (0, bar_h), (w, bar_h), (40, 45, 60), 1)

    cv2.putText(
        frame, "AI MODE:", (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 175, 190), 1, cv2.LINE_AA,
    )

    buttons = [
        ("[A] All AI", "all", 150),
        ("[H] Humans", "humans", 145),
        ("[F] Faces", "faces", 140),
        ("[N] Plates", "plates", 140),
        ("[B] Both", "both", 130),
    ]

    start_x = 105
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
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_color,
            2 if is_active else 1, cv2.LINE_AA,
        )
        start_x += btn_w + 8

    # Cybersecurity Health Button
    c_btn_w = 175
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
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 255), 1, cv2.LINE_AA,
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

    # Backdrop
    overlay = frame.copy()
    cv2.rectangle(overlay, (dx1, dy1), (dx2, dy2), (10, 13, 20), -1)
    cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)

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
    total_logged_plates: int,
) -> None:
    """Render high-tech command center HUD status bar at bottom of window."""
    h, w = frame.shape[:2]
    hud_h = 32

    cv2.rectangle(frame, (0, h - hud_h), (w, h), (12, 14, 18), -1)
    cv2.line(frame, (0, h - hud_h), (w, h - hud_h), (40, 45, 60), 1)

    mode_str = mode.upper()
    view_map = {"triple": "TRIPLE", "dual": "DUAL", "0": "CAM 0", "1": "CAM 1", "2": "CAM 2"}
    view_str = view_map.get(view, view.upper())

    face_status = f"FACES: {total_faces}"
    if total_unknown_faces > 0:
        face_status = f"FACES: {total_faces} ({total_unknown_faces} UNK)"

    serv_metric = cyber_monitor.metrics.get("service", None)
    health_str = f"HEALTH: {serv_metric.status if serv_metric else 'OPTIMAL'}"

    hud_items = [
        f"{fps:.0f} FPS",
        f"MODE: {mode_str}",
        f"VIEW: {view_str}",
        health_str,
        f"HUMANS: {total_humans}",
        face_status,
        f"PLATES: {total_plates}",
        f"LOGGED: {total_logged_plates}",
        "[C] Cyber",
    ]

    hud_text = "  |  ".join(hud_items)
    hud_color = (0, 100, 255) if total_unknown_faces > 0 else (0, 255, 140)
    cv2.putText(
        frame, hud_text, (14, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, hud_color, 1, cv2.LINE_AA,
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
# Main Application Entrypoint (3 Cameras)
# =============================================================================

def main() -> None:
    global CURRENT_DETECTION_MODE, SHOW_CYBER_OVERLAY, LAST_LOGGED_PLATE_ALERT, LAST_LOGGED_PLATE_TIME

    parser = argparse.ArgumentParser(description="CCTV Vision System: 3-Camera Human Tracking, Face Tracking & ANPR")
    parser.add_argument("--cam0", type=int, default=0, help="Camera 0 index (default: 0)")
    parser.add_argument("--cam1", type=int, default=1, help="Camera 1 index (default: 1)")
    parser.add_argument("--cam2", type=int, default=2, help="Camera 2 index (default: 2)")
    parser.add_argument("--mode", type=str, default="triple", choices=["triple", "dual", "0", "1", "2"], help="Initial camera view")
    parser.add_argument("--detect-mode", type=str, default="all", choices=["all", "humans", "faces", "plates", "both"], help="Initial detection mode")
    parser.add_argument("--disable-ocr", action="store_true", help="Disable OCR text extraction")
    args = parser.parse_args()

    CURRENT_DETECTION_MODE = args.detect_mode
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    enable_ocr = not args.disable_ocr

    print("\n" + "=" * 75)
    print("  🎥  CCTV AI VISION SYSTEM — 3-Camera Human, Face & ANPR Tracking")
    print("=" * 75)
    print(f"  ⚡ GPU Acceleration   : {device.upper()}")
    print(f"  🛡️ Cybersecurity Watch : ACTIVE (10 Security Vectors)")
    print(f"  📝 OCR Engine Enabled  : {enable_ocr}")
    print(f"  🖥️ Camera Count       : 3 (Cam {args.cam0}, Cam {args.cam1}, Cam {args.cam2})")
    print(f"  🎯 AI Pipeline Mode   : {CURRENT_DETECTION_MODE.upper()}")
    print("-" * 75)
    print("  Controls:")
    print("    [A/H/F/N/B] -> Switch Detection Pipelines")
    print("    [C]         -> Toggle Cybersecurity Health Dashboard")
    print("    [T]         -> Triple Camera View (All 3)")
    print("    [D]         -> Dual Camera View (Cam 0 + 1)")
    print("    [0/1/2]     -> Single Camera Feed")
    print("    [S]         -> Cycle Camera Views")
    print("    [Q] / ESC   -> Quit")
    print("=" * 75 + "\n")

    # 1. Initialize Human Trackers (one per camera)
    print("[1/3] Initializing Human Detectors & Trackers...")
    human_trackers = [HumanTracker(device=device) for _ in range(3)]

    # 2. Initialize Face Trackers (one per camera)
    print("\n[2/3] Initializing Face Detector & SFace Recognizer...")
    face_trackers = [FastFaceTracker(max_alerts_per_person=3) for _ in range(3)]

    # 3. Initialize shared ANPR Detector & Plate Recorder
    print("\n[3/3] Initializing ANPR Plate Detector & EasyOCR...")
    anpr_detector = ANPRDetector(device=device, enable_ocr=enable_ocr)
    plate_recorder = PlateRecorder(cooldown_seconds=45.0, similarity_threshold=0.50)

    # Initialize Synchronized Triple Camera Stream
    print("\nConnecting to Video Cameras...")
    cam_manager = TripleCameraManager(cam0_id=args.cam0, cam1_id=args.cam1, cam2_id=args.cam2)

    for i, cam_id in enumerate(cam_manager.cam_ids):
        status = '● ONLINE' if cam_manager.is_active[i] else '○ OFFLINE'
        print(f"  Cam {cam_id} [{cam_manager.dev_names[i]}]: {status}")

    window_name = "CCTV AI Vision Console — 3-Camera Human, Face & ANPR Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1440, 540)
    cv2.setMouseCallback(window_name, on_mouse_click)

    camera_view_mode = args.mode
    prev_time = time.time()

    print("\n🚀 Streaming started! Press [Q] to exit.\n")

    # Frame-skip counters for FPS optimization:
    # - Faces: every frame (lightweight YuNet DNN ~8ms)
    # - Humans: every 2nd frame (YOLO ~20ms)
    # - ANPR:   every 5th frame (YOLO + OCR ~30ms, plates are static)
    frame_counter = 0
    cached_h_counts = [0, 0, 0]
    cached_p_counts = [0, 0, 0]
    cached_anpr_masks: List[Optional[np.ndarray]] = [None, None, None]

    try:
        while True:
            cam_results = cam_manager.read_sync()  # List of (ret, frame)

            raws: List[np.ndarray] = []
            rets: List[bool] = []
            for i, (ret, frame) in enumerate(cam_results):
                if not ret or frame is None:
                    raws.append(create_placeholder_frame(640, 480, f"Cam {cam_manager.cam_ids[i]} Offline"))
                    rets.append(False)
                else:
                    raws.append(frame)
                    rets.append(True)

            outs = [r.copy() for r in raws]
            h_counts = [0, 0, 0]
            f_counts = [0, 0, 0]
            p_counts = [0, 0, 0]
            total_unks = [0, 0, 0]

            # Determine which cameras to process based on view mode
            if camera_view_mode == "triple":
                process_flags = [rets[0], rets[1], rets[2]]
            elif camera_view_mode == "dual":
                process_flags = [rets[0], rets[1], False]
            else:
                cam_idx = int(camera_view_mode)
                process_flags = [False, False, False]
                if cam_idx < 3:
                    process_flags[cam_idx] = rets[cam_idx]

            frame_counter += 1
            run_humans = (frame_counter % 2 == 0)  # Every 2nd frame
            run_anpr = (frame_counter % 5 == 0)     # Every 5th frame (3 cams = heavier load)

            # 1. Face Tracking & Recognition — runs EVERY frame (YuNet is ~8ms)
            face_boxes = [[], [], []]
            if CURRENT_DETECTION_MODE in ["faces", "both", "all"]:
                for i in range(3):
                    if process_flags[i]:
                        outs[i], f_counts[i] = face_trackers[i].process_frame(outs[i])
                        face_boxes[i] = [t.bbox for t in face_trackers[i].tracks]
                        total_unks[i] = face_trackers[i].total_unknown_count

            # 2. Human Tracking — runs every 2nd frame
            if CURRENT_DETECTION_MODE in ["humans", "all"]:
                if run_humans:
                    for i in range(3):
                        if process_flags[i]:
                            outs[i], _, h_counts[i] = human_trackers[i].process_frame(
                                outs[i], existing_face_boxes=face_boxes[i]
                            )
                            cached_h_counts[i] = h_counts[i]
                else:
                    h_counts = cached_h_counts.copy()

            # 3. ANPR Plate Detection — runs every 5th frame
            if CURRENT_DETECTION_MODE in ["plates", "both", "all"]:
                if run_anpr:
                    for i in range(3):
                        if process_flags[i]:
                            anpr_ann, p_dets, p_count = anpr_detector.process_frame(raws[i])
                            cached_anpr_masks[i] = cv2.absdiff(anpr_ann, raws[i])
                            outs[i] = cv2.add(outs[i], cached_anpr_masks[i])
                            cached_p_counts[i] = p_count
                            p_counts[i] = p_count

                            for det in p_dets:
                                p_text = det.get("text", "")
                                if p_text:
                                    is_new, cleaned = plate_recorder.log_plate(
                                        plate_text=p_text,
                                        confidence=det.get("conf", 0.0),
                                        camera_name=f"Cam {cam_manager.cam_ids[i]} ({cam_manager.dev_names[i]})",
                                        plate_crop_bgr=det.get("crop"),
                                    )
                                    if is_new:
                                        LAST_LOGGED_PLATE_ALERT = cleaned
                                        LAST_LOGGED_PLATE_TIME = time.time()
                else:
                    p_counts = cached_p_counts.copy()
                    for i in range(3):
                        if cached_anpr_masks[i] is not None and process_flags[i]:
                            outs[i] = cv2.add(outs[i], cached_anpr_masks[i])

            # Construct View Layout
            if camera_view_mode == "triple":
                out_frame = combine_triple_feeds(
                    outs,
                    dev_names=cam_manager.dev_names,
                    h_counts=h_counts,
                    f_counts=f_counts,
                    p_counts=p_counts,
                )
            elif camera_view_mode == "dual":
                out_frame = combine_dual_feeds(
                    outs[0], outs[1],
                    dev0_name=cam_manager.dev_names[0],
                    dev1_name=cam_manager.dev_names[1],
                    h_count0=h_counts[0], f_count0=f_counts[0], p_count0=p_counts[0],
                    h_count1=h_counts[1], f_count1=f_counts[1], p_count1=p_counts[1],
                )
            else:
                cam_idx = int(camera_view_mode)
                out_frame = outs[cam_idx]
                labels = ["Feed A", "Feed B", "Feed C"]
                draw_header_banner(
                    out_frame, labels[cam_idx], cam_manager.dev_names[cam_idx],
                    h_counts[cam_idx], f_counts[cam_idx], p_counts[cam_idx],
                )

            # Draw Interactive Top Bar
            draw_top_mode_bar(out_frame, CURRENT_DETECTION_MODE)

            # Record unauthorized face event in cyber monitor
            total_alerts = sum(total_unks)
            if total_alerts > 0:
                loc_label = "Triple Feed" if camera_view_mode == "triple" else (
                    "Dual Feed" if camera_view_mode == "dual" else f"Cam {camera_view_mode}"
                )
                cyber_monitor.record_unauthorized_face_event(total_alerts, loc_label)

            curr_time = time.time()
            fps = 1.0 / max(0.001, curr_time - prev_time)
            prev_time = curr_time

            # Update camera status in security monitor
            cyber_monitor.check_camera_status(
                cam_manager.is_active[0], cam_manager.is_active[1], fps
            )

            # Draw Cyber Security Dashboard Overlay if toggled
            if SHOW_CYBER_OVERLAY:
                draw_cybersecurity_dashboard_overlay(out_frame)

            total_humans = sum(h_counts)
            total_faces = sum(f_counts)
            total_unknown_faces = sum(total_unks)
            total_plates = sum(p_counts)
            draw_bottom_hud(
                out_frame,
                CURRENT_DETECTION_MODE,
                camera_view_mode,
                fps,
                total_humans,
                total_faces,
                total_unknown_faces,
                total_plates,
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
                print("  [Mode] -> ALL-IN-ONE AI")
            elif key == ord("h"):
                CURRENT_DETECTION_MODE = "humans"
                print("  [Mode] -> HUMAN DETECTION ONLY")
            elif key == ord("f"):
                CURRENT_DETECTION_MODE = "faces"
                print("  [Mode] -> FACE RECOGNITION ONLY")
            elif key == ord("n"):
                CURRENT_DETECTION_MODE = "plates"
                print("  [Mode] -> NUMBER PLATES (ANPR) ONLY")
            elif key == ord("b"):
                CURRENT_DETECTION_MODE = "both"
                print("  [Mode] -> DUAL AI (FACES + PLATES)")
            elif key == ord("t"):
                camera_view_mode = "triple"
                print("  [View] -> TRIPLE CAMERA FEED")
            elif key in [ord("d"), ord("m")]:
                camera_view_mode = "dual"
                print("  [View] -> DUAL CAMERA FEED (Cam 0 + 1)")
            elif key == ord("0"):
                camera_view_mode = "0"
                print(f"  [View] -> CAMERA {args.cam0} ONLY")
            elif key == ord("1"):
                camera_view_mode = "1"
                print(f"  [View] -> CAMERA {args.cam1} ONLY")
            elif key == ord("2"):
                camera_view_mode = "2"
                print(f"  [View] -> CAMERA {args.cam2} ONLY")
            elif key == ord("s"):
                views = ["triple", "dual", "0", "1", "2"]
                next_idx = (views.index(camera_view_mode) + 1) % len(views)
                camera_view_mode = views[next_idx]
                print(f"  [View] -> {camera_view_mode.upper()} VIEW")

    finally:
        cam_manager.release()
        cv2.destroyAllWindows()
        print("\nApplication closed cleanly.")


if __name__ == "__main__":
    main()
