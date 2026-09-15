
"""
CCTV AI Vision & Cybersecurity Platform — CustomTkinter Desktop Console
========================================================================

A state-of-the-art native desktop surveillance hub featuring:
- High-performance multi-threaded video streaming with zero UI freeze.
- Trained YOLOv8 Human Detection & Multi-person tracking.
- Deep Learning Face Recognition (YuNet DNN + SFace ONNX) with 3-alert limit for unknown faces.
- Real-Time Vehicle Number Plate Detection (YOLO) + EasyOCR text extraction & CSV recording.
- Full 10-Vector IBVAP Cybersecurity & System Health Monitor.
- Interactive Dark Glassmorphism CustomTkinter Desktop Interface.

Run:
  python desktop_ui.py
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
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

# Set CustomTkinter Theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")


def get_camera_names() -> List[str]:
    """Enumerate physical camera names via DirectShow filter graph."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        return graph.get_input_devices()
    except Exception:
        return []


class DualCameraManager:
    """Synchronized Multi-Camera Stream Manager using DirectShow grab & retrieve."""

    def __init__(self, cam0_id: int = 0, cam1_id: int = 1) -> None:
        self.cam0_id = cam0_id
        self.cam1_id = cam1_id
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
                self.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(6):
                    r, f = self.cap0.read()
                    if r and f is not None:
                        self.is_cam0_active = True
                        break
                    time.sleep(0.04)
        except Exception:
            self.is_cam0_active = False

        time.sleep(0.15)

        try:
            self.cap1 = cv2.VideoCapture(self.cam1_id, cv2.CAP_DSHOW)
            if not self.cap1.isOpened():
                self.cap1 = cv2.VideoCapture(self.cam1_id)
            if self.cap1.isOpened():
                self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(6):
                    r, f = self.cap1.read()
                    if r and f is not None:
                        self.is_cam1_active = True
                        break
                    time.sleep(0.04)
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


class SurveillanceApp(ctk.CTk):
    """Modern CustomTkinter CCTV AI Vision & Security Dashboard."""

    def __init__(self) -> None:
        super().__init__()

        self.title("CCTV AI Vision & Cybersecurity Console")
        self.geometry("1440x880")
        self.minsize(1200, 720)

        # Application State
        self.is_streaming = False
        self.camera_view_mode = "dual"  # "dual", "0", "1"
        self.detection_mode = "all"     # "all", "humans", "faces", "plates", "both"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.enable_ocr = True

        self.last_audio_time = 0.0
        self.prev_fps_time = time.time()
        self.fps_value = 0.0
        self.last_toast_text = ""
        self.last_toast_time = 0.0

        # AI Models
        self.human_tracker_0 = HumanTracker(device=self.device)
        self.human_tracker_1 = HumanTracker(device=self.device)
        self.face_tracker_0 = FastFaceTracker(max_alerts_per_person=3)
        self.face_tracker_1 = FastFaceTracker(max_alerts_per_person=3)
        self.anpr_detector = ANPRDetector(device=self.device, enable_ocr=self.enable_ocr)
        self.plate_recorder = PlateRecorder(cooldown_seconds=45.0, similarity_threshold=0.50)
        self.cam_manager: Optional[DualCameraManager] = None

        # Build UI Structure
        self._build_layout()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_layout(self) -> None:
        # Grid Configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 1. TOP HEADER & METRIC CARDS
        self.header_frame = ctk.CTkFrame(self, height=75, corner_radius=10, fg_color="#10141e")
        self.header_frame.grid(row=0, column=0, columnspan=2, padx=14, pady=(12, 6), sticky="nsew")
        self.header_frame.grid_columnconfigure(6, weight=1)

        # App Title
        title_label = ctk.CTkLabel(
            self.header_frame,
            text="🎥 CCTV AI SURVEILLANCE & CYBERSECURITY CONSOLE",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#00ff8c",
        )
        title_label.grid(row=0, column=0, padx=18, pady=16, sticky="w")

        # Metric Cards
        self.card_fps = self._create_metric_card(self.header_frame, "FPS", "0.0", 1)
        self.card_humans = self._create_metric_card(self.header_frame, "HUMANS", "0", 2)
        self.card_faces = self._create_metric_card(self.header_frame, "FACES", "0", 3)
        self.card_plates = self._create_metric_card(self.header_frame, "PLATES", "0", 4)
        self.card_logged = self._create_metric_card(self.header_frame, "LOGGED", "0", 5)
        self.card_cyber = self._create_metric_card(self.header_frame, "CYBER HEALTH", "OPTIMAL", 6, color="#00ff8c")

        # 2. LEFT CONTROL SIDEBAR
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=280, corner_radius=10, fg_color="#10141e")
        self.sidebar_frame.grid(row=1, column=0, padx=(14, 6), pady=(6, 12), sticky="nsew")

        # Stream Toggle
        self.btn_stream = ctk.CTkButton(
            self.sidebar_frame,
            text="▶️ START STREAMING",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#00aa5a",
            hover_color="#008848",
            height=40,
            command=self.toggle_streaming,
        )
        self.btn_stream.pack(fill="x", padx=12, pady=(10, 16))

        # AI Mode Selector
        lbl_ai = ctk.CTkLabel(self.sidebar_frame, text="AI DETECTION PIPELINE", font=ctk.CTkFont(size=12, weight="bold"), text_color="#94a3b8")
        lbl_ai.pack(anchor="w", padx=12, pady=(4, 2))

        self.seg_ai = ctk.CTkSegmentedButton(
            self.sidebar_frame,
            values=["All AI", "Humans", "Faces", "Plates", "Both"],
            command=self.change_ai_mode,
        )
        self.seg_ai.set("All AI")
        self.seg_ai.pack(fill="x", padx=12, pady=(2, 14))

        # Camera View Selector
        lbl_cam = ctk.CTkLabel(self.sidebar_frame, text="CAMERA VIEWPORT", font=ctk.CTkFont(size=12, weight="bold"), text_color="#94a3b8")
        lbl_cam.pack(anchor="w", padx=12, pady=(4, 2))

        self.seg_cam = ctk.CTkSegmentedButton(
            self.sidebar_frame,
            values=["Dual Feed", "Cam 0 Only", "Cam 1 Only"],
            command=self.change_camera_view,
        )
        self.seg_cam.set("Dual Feed")
        self.seg_cam.pack(fill="x", padx=12, pady=(2, 16))

        # Sliders & Thresholds
        lbl_thresh = ctk.CTkLabel(self.sidebar_frame, text="DETECTION SENSITIVITY", font=ctk.CTkFont(size=12, weight="bold"), text_color="#94a3b8")
        lbl_thresh.pack(anchor="w", padx=12, pady=(4, 2))

        # Human Conf
        self.lbl_human_conf = ctk.CTkLabel(self.sidebar_frame, text="Human Conf: 0.40", font=ctk.CTkFont(size=11), text_color="#cbd5e1")
        self.lbl_human_conf.pack(anchor="w", padx=12, pady=(4, 0))
        self.slider_human = ctk.CTkSlider(self.sidebar_frame, from_=0.10, to=0.90, number_of_steps=16, command=self._update_human_conf)
        self.slider_human.set(0.40)
        self.slider_human.pack(fill="x", padx=12, pady=(0, 10))

        # Face Conf
        self.lbl_face_conf = ctk.CTkLabel(self.sidebar_frame, text="Face Conf: 0.50", font=ctk.CTkFont(size=11), text_color="#cbd5e1")
        self.lbl_face_conf.pack(anchor="w", padx=12, pady=(4, 0))
        self.slider_face = ctk.CTkSlider(self.sidebar_frame, from_=0.10, to=0.90, number_of_steps=16, command=self._update_face_conf)
        self.slider_face.set(0.50)
        self.slider_face.pack(fill="x", padx=12, pady=(0, 10))

        # ANPR Conf
        self.lbl_anpr_conf = ctk.CTkLabel(self.sidebar_frame, text="Plate Conf: 0.35", font=ctk.CTkFont(size=11), text_color="#cbd5e1")
        self.lbl_anpr_conf.pack(anchor="w", padx=12, pady=(4, 0))
        self.slider_anpr = ctk.CTkSlider(self.sidebar_frame, from_=0.10, to=0.90, number_of_steps=16, command=self._update_anpr_conf)
        self.slider_anpr.set(0.35)
        self.slider_anpr.pack(fill="x", padx=12, pady=(0, 14))


        # OCR Checkbox
        self.chk_ocr = ctk.CTkCheckBox(self.sidebar_frame, text="Enable EasyOCR Text Extraction", command=self._toggle_ocr)
        self.chk_ocr.select()
        self.chk_ocr.pack(anchor="w", padx=12, pady=(4, 12))

        # Device Info
        dev_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=f"⚡ Hardware: {self.device.upper()}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38bdf8",
        )
        dev_label.pack(anchor="w", padx=12, pady=(10, 6))

        # 3. MAIN CENTER & RIGHT VIEWPORT
        self.main_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#10141e")
        self.main_frame.grid(row=1, column=1, padx=(6, 14), pady=(6, 12), sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=3)
        self.main_frame.grid_columnconfigure(1, weight=2)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Center Video Canvas Frame
        self.video_container = ctk.CTkFrame(self.main_frame, corner_radius=8, fg_color="#080a0f")
        self.video_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_container.grid_columnconfigure(0, weight=1)
        self.video_container.grid_rowconfigure(0, weight=1)

        self.video_canvas = ctk.CTkLabel(self.video_container, text="🎥 CAMERA FEED IDLE\nClick 'START STREAMING' to begin.", font=ctk.CTkFont(size=16))
        self.video_canvas.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")

        # Right Tabview (Plate Database & Cybersecurity Health)
        self.tabview = ctk.CTkTabview(self.main_frame, corner_radius=8, fg_color="#151924")
        self.tabview.grid(row=0, column=1, padx=(4, 10), pady=10, sticky="nsew")

        self.tab_plates = self.tabview.add("📋 Plate Records")
        self.tab_cyber = self.tabview.add("🛡️ Cybersecurity")
        self.tab_faces = self.tabview.add("👤 Known Faces")

        self._build_plate_tab()
        self._build_cyber_tab()
        self._build_faces_tab()

    def _create_metric_card(self, parent: ctk.CTkFrame, label: str, default_val: str, col: int, color: str = "#00ff8c") -> ctk.CTkLabel:
        frame = ctk.CTkFrame(parent, width=120, height=52, corner_radius=8, fg_color="#161b26")
        frame.grid(row=0, column=col, padx=6, pady=10)
        lbl = ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=10, weight="bold"), text_color="#94a3b8")
        lbl.pack(pady=(4, 0))
        val = ctk.CTkLabel(frame, text=default_val, font=ctk.CTkFont(size=16, weight="bold"), text_color=color)
        val.pack(pady=(0, 4))
        return val

    def _build_plate_tab(self) -> None:
        self.plate_scroll = ctk.CTkScrollableFrame(self.tab_plates, fg_color="#0e1118")
        self.plate_scroll.pack(fill="both", expand=True, padx=4, pady=4)
        self.refresh_plate_list()

    def _build_cyber_tab(self) -> None:
        self.cyber_scroll = ctk.CTkScrollableFrame(self.tab_cyber, fg_color="#0e1118")
        self.cyber_scroll.pack(fill="both", expand=True, padx=4, pady=4)
        self.refresh_cyber_list()

    def _build_faces_tab(self) -> None:
        self.faces_scroll = ctk.CTkScrollableFrame(self.tab_faces, fg_color="#0e1118")
        self.faces_scroll.pack(fill="both", expand=True, padx=4, pady=4)
        self.refresh_faces_list()

    def refresh_plate_list(self) -> None:
        for w in self.plate_scroll.winfo_children():
            w.destroy()

        csv_path = PROJECT_ROOT / "records" / "number_plates" / "plate_records.csv"
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = list(csv.reader(f))
                    rows = reader[1:] if len(reader) > 1 else []
                    rows.reverse()

                    for r in rows[:15]:
                        if len(r) >= 5:
                            ts, date, tm, pnum, cam = r[0], r[1], r[2], r[3], r[4]
                            card = ctk.CTkFrame(self.plate_scroll, corner_radius=6, fg_color="#181d28")
                            card.pack(fill="x", padx=4, pady=3)
                            
                            lbl_num = ctk.CTkLabel(card, text=f"🚗 {pnum}", font=ctk.CTkFont(size=13, weight="bold"), text_color="#f8fafc")
                            lbl_num.pack(side="left", padx=8, pady=6)

                            lbl_time = ctk.CTkLabel(card, text=f"{tm} | {cam.split('(')[0]}", font=ctk.CTkFont(size=10), text_color="#94a3b8")
                            lbl_time.pack(side="right", padx=8, pady=6)
            except Exception:
                pass

    def refresh_cyber_list(self) -> None:
        for w in self.cyber_scroll.winfo_children():
            w.destroy()

        audit = cyber_monitor.run_full_audit(
            cam0_active=self.cam_manager.is_cam0_active if self.cam_manager else False,
            cam1_active=self.cam_manager.is_cam1_active if self.cam_manager else False,
            fps=self.fps_value,
        )

        for k, m in audit.items():
            card = ctk.CTkFrame(self.cyber_scroll, corner_radius=6, fg_color="#181d28")
            card.pack(fill="x", padx=4, pady=3)

            s_col = "#00ff8c" if m.status == "OK" else ("#f59e0b" if m.status == "WARNING" else "#ef4444")
            
            top_line = ctk.CTkFrame(card, fg_color="transparent")
            top_line.pack(fill="x", padx=6, pady=(4, 0))
            
            lbl_name = ctk.CTkLabel(top_line, text=m.name, font=ctk.CTkFont(size=12, weight="bold"), text_color="#f8fafc")
            lbl_name.pack(side="left")

            lbl_stat = ctk.CTkLabel(top_line, text=f"● {m.status}", font=ctk.CTkFont(size=11, weight="bold"), text_color=s_col)
            lbl_stat.pack(side="right")

            lbl_det = ctk.CTkLabel(card, text=m.details, font=ctk.CTkFont(size=10), text_color="#94a3b8", anchor="w")
            lbl_det.pack(fill="x", padx=6, pady=(0, 4))

    def refresh_faces_list(self) -> None:
        for w in self.faces_scroll.winfo_children():
            w.destroy()

        knowns = self.face_tracker_0.known_database
        if knowns:
            for name, _ in knowns:
                card = ctk.CTkFrame(self.faces_scroll, corner_radius=6, fg_color="#181d28")
                card.pack(fill="x", padx=4, pady=3)
                lbl = ctk.CTkLabel(card, text=f"🟢 {name} (Authorized)", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00ff8c")
                lbl.pack(side="left", padx=8, pady=6)
        else:
            lbl = ctk.CTkLabel(self.faces_scroll, text="No known faces enrolled in known_faces/", text_color="#94a3b8")
            lbl.pack(padx=8, pady=8)

    # UI Callbacks
    def change_ai_mode(self, value: str) -> None:
        mode_map = {"All AI": "all", "Humans": "humans", "Faces": "faces", "Plates": "plates", "Both": "both"}
        self.detection_mode = mode_map.get(value, "all")

    def change_camera_view(self, value: str) -> None:
        view_map = {"Dual Feed": "dual", "Cam 0 Only": "0", "Cam 1 Only": "1"}
        self.camera_view_mode = view_map.get(value, "dual")

    def _update_human_conf(self, val: float) -> None:
        self.human_tracker_0.conf_threshold = val
        self.human_tracker_1.conf_threshold = val
        self.lbl_human_conf.configure(text=f"Human Conf: {val:.2f}")

    def _update_face_conf(self, val: float) -> None:
        self.face_tracker_0.conf_threshold = val
        self.face_tracker_1.conf_threshold = val
        self.lbl_face_conf.configure(text=f"Face Conf: {val:.2f}")

    def _update_anpr_conf(self, val: float) -> None:
        self.anpr_detector.conf_threshold = val
        self.lbl_anpr_conf.configure(text=f"Plate Conf: {val:.2f}")

    def _toggle_ocr(self) -> None:
        self.enable_ocr = bool(self.chk_ocr.get())
        self.anpr_detector.enable_ocr = self.enable_ocr

    def trigger_alert_beep(self) -> None:
        now = time.time()
        if now - self.last_audio_time > 2.0:
            self.last_audio_time = now
            try:
                import winsound
                threading.Thread(target=lambda: winsound.MessageBeep(winsound.MB_ICONEXCLAMATION), daemon=True).start()
            except Exception:
                pass

    def toggle_streaming(self) -> None:
        if not self.is_streaming:
            self.is_streaming = True
            self.btn_stream.configure(text="⏹️ STOP STREAMING", fg_color="#c22525", hover_color="#991b1b")
            self.cam_manager = DualCameraManager(cam0_id=0, cam1_id=1)
            self._update_frame_loop()
        else:
            self.is_streaming = False
            self.btn_stream.configure(text="▶️ START STREAMING", fg_color="#00aa5a", hover_color="#008848")
            if self.cam_manager:
                self.cam_manager.release()
                self.cam_manager = None
            self.video_canvas.configure(image="", text="🎥 CAMERA FEED STOPPED")

    def _update_frame_loop(self) -> None:
        if not self.is_streaming or self.cam_manager is None:
            return

        ret0, raw0, ret1, raw1 = self.cam_manager.read_sync()

        if not ret0 or raw0 is None:
            raw0 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(raw0, "Cam 0 Offline", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)

        if not ret1 or raw1 is None:
            raw1 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(raw1, "Cam 1 Offline", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)

        out_cam0 = raw0.copy()
        out_cam1 = raw1.copy()
        h_count0, f_count0, p_count0 = 0, 0, 0
        h_count1, f_count1, p_count1 = 0, 0, 0
        alert_unk0, alert_unk1 = 0, 0
        total_unk0, total_unk1 = 0, 0

        process_cam0 = (self.camera_view_mode in ["dual", "0"]) and ret0
        process_cam1 = (self.camera_view_mode in ["dual", "1"]) and ret1

        # 1. Face Recognition (Establishes primary identity tags: Green = Known, Red = UNKNOWN)
        face_boxes0, face_boxes1 = [], []
        if self.detection_mode in ["faces", "both", "all"]:
            if process_cam0:
                out_cam0, f_count0 = self.face_tracker_0.process_frame(out_cam0)
                face_boxes0 = [t.bbox for t in self.face_tracker_0.tracks]
                total_unk0 = self.face_tracker_0.total_unknown_count
            if process_cam1:
                out_cam1, f_count1 = self.face_tracker_1.process_frame(out_cam1)
                face_boxes1 = [t.bbox for t in self.face_tracker_1.tracks]
                total_unk1 = self.face_tracker_1.total_unknown_count

        # 2. Human Tracking (Suppresses duplicate tag if face is already identified)
        if self.detection_mode in ["humans", "all"]:
            if process_cam0:
                out_cam0, _, h_count0 = self.human_tracker_0.process_frame(out_cam0, existing_face_boxes=face_boxes0)
            if process_cam1:
                out_cam1, _, h_count1 = self.human_tracker_1.process_frame(out_cam1, existing_face_boxes=face_boxes1)


        # 3. ANPR Plate Detection — use RAW frames to avoid OCR reading annotation text
        if self.detection_mode in ["plates", "both", "all"]:
            if process_cam0:
                anpr_ann0, p_dets0, p_count0 = self.anpr_detector.process_frame(raw0)
                mask0 = cv2.absdiff(anpr_ann0, raw0)
                out_cam0 = cv2.add(out_cam0, mask0)
                for det in p_dets0:
                    p_text = det.get("text", "")
                    if p_text:
                        is_new, cleaned = self.plate_recorder.log_plate(
                            plate_text=p_text,
                            confidence=det.get("conf", 0.0),
                            camera_name="Cam 0",
                            plate_crop_bgr=det.get("crop"),
                        )
                        if is_new:
                            self.last_toast_text = cleaned
                            self.last_toast_time = time.time()
                            self.refresh_plate_list()

            if process_cam1:
                anpr_ann1, p_dets1, p_count1 = self.anpr_detector.process_frame(raw1)
                mask1 = cv2.absdiff(anpr_ann1, raw1)
                out_cam1 = cv2.add(out_cam1, mask1)
                for det in p_dets1:
                    p_text = det.get("text", "")
                    if p_text:
                        is_new, cleaned = self.plate_recorder.log_plate(
                            plate_text=p_text,
                            confidence=det.get("conf", 0.0),
                            camera_name="Cam 1",
                            plate_crop_bgr=det.get("crop"),
                        )
                        if is_new:
                            self.last_toast_text = cleaned
                            self.last_toast_time = time.time()
                            self.refresh_plate_list()


        # Combine View
        if self.camera_view_mode == "dual":
            out_frame = np.hstack((out_cam0, out_cam1))
        elif self.camera_view_mode == "0":
            out_frame = out_cam0
        else:
            out_frame = out_cam1

        # Silently record unauthorized face events without intrusive popups or beeps
        active_alerts = alert_unk0 + alert_unk1
        if active_alerts > 0:
            cyber_monitor.record_unauthorized_face_event(active_alerts, "Desktop Console")


        # Render Plate Toast
        if self.last_toast_text and (time.time() - self.last_toast_time < 3.5):
            h, w = out_frame.shape[:2]
            cv2.rectangle(out_frame, (w // 2 - 240, h - 55), (w // 2 + 240, h - 15), (15, 45, 25), -1)
            cv2.rectangle(out_frame, (w // 2 - 240, h - 55), (w // 2 + 240, h - 15), (0, 255, 140), 2)
            cv2.putText(
                out_frame,
                f"LOGGED TO CSV: {self.last_toast_text}",
                (w // 2 - 220, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # FPS Calculation
        curr_time = time.time()
        self.fps_value = 1.0 / max(0.001, curr_time - self.prev_fps_time)
        self.prev_fps_time = curr_time

        # Update Top Metric Cards
        total_humans = h_count0 + h_count1
        total_faces = f_count0 + f_count1
        total_unknown_faces = total_unk0 + total_unk1
        total_plates = p_count0 + p_count1

        self.card_fps.configure(text=f"{self.fps_value:.1f}")
        self.card_humans.configure(text=f"{total_humans}")
        face_txt = f"{total_faces} ({total_unknown_faces} UNK)" if total_unknown_faces > 0 else f"{total_faces}"
        self.card_faces.configure(text=face_txt, text_color="#ef4444" if total_unknown_faces > 0 else "#00ff8c")
        self.card_plates.configure(text=f"{total_plates}")
        self.card_logged.configure(text=f"{len(self.plate_recorder.recent_logs)}")

        # Convert frame for Tkinter Display
        vw = max(640, self.video_container.winfo_width() - 12)
        vh = max(400, self.video_container.winfo_height() - 12)
        fh, fw = out_frame.shape[:2]
        scale = min(vw / fw, vh / fh)
        nw, nh = int(fw * scale), int(fh * scale)

        resized = cv2.resize(out_frame, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(nw, nh))

        self.video_canvas.configure(image=ctk_img, text="")
        self.video_canvas.image = ctk_img


        # Periodic Cyber Health Check
        if int(time.time()) % 4 == 0:
            self.refresh_cyber_list()

        # Schedule Next Frame
        self.after(10, self._update_frame_loop)

    def on_close(self) -> None:
        self.is_streaming = False
        if self.cam_manager:
            self.cam_manager.release()
        self.destroy()


def main() -> None:
    app = SurveillanceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
