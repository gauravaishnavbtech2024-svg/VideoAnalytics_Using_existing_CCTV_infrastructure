"""
Multi-Camera Stream & AI Pipeline Manager for IBVAP Surveillance Backend.
========================================================================
Supports concurrent multi-camera ingestion (Dual Side-by-Side, Single Cam,
2x2 Grid), non-blocking threaded capture (0ms grab latency), and interleaved
multi-camera AI detection pipeline.
"""

from __future__ import annotations

import base64
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anpr_detector import ANPRDetector
from src.crowd_detector import CrowdDetector
from src.cyber_health_monitor import cyber_monitor
from src.db_manager import db_manager
from src.face_tracker import FastFaceTracker
from src.human_detector import HumanTracker
from src.plate_recorder import PlateRecorder
from src.vehicle_detector import VehicleTracker

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_camera_names() -> List[str]:
    """Enumerate physical camera names on Windows via DirectShow filter graph."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        return devices if devices else []
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
            print(f"[ThreadedCameraStream] Cam {self.cam_id} init error: {e}")
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


class MultiCamStreamManager:
    """Multi-Camera Concurrent Ingestion, Composition, and AI Analytics Engine."""

    def __init__(self, cam0_id: int = 0, cam1_id: int = 1, width: int = 640, height: int = 480) -> None:
        self.cam0_id = cam0_id
        self.cam1_id = cam1_id
        self.width = width
        self.height = height

        # Layout Modes: "dual", "0", "1"
        self.layout_mode = "dual"

        # Pipeline Feature Toggles
        self.enable_humans = True
        self.enable_faces = True
        self.enable_plates = True
        self.enable_vehicles = True
        self.enable_crowd = True
        self.conf_threshold = 0.35

        # Threading & Capture State
        self.running = False
        self.lock = threading.Lock()
        self.stream0: Optional[ThreadedCameraStream] = None
        self.stream1: Optional[ThreadedCameraStream] = None
        self.dev0_name = f"Camera {cam0_id}"
        self.dev1_name = f"Camera {cam1_id}"

        # Frames
        self.composite_frame: Optional[np.ndarray] = None
        self.frame_cam0: Optional[np.ndarray] = None
        self.frame_cam1: Optional[np.ndarray] = None

        # Telemetry & Stats
        self.fps_smooth = 0.0
        self.frame_counter = 0
        self.stats: Dict[str, Any] = {
            "fps": 0.0,
            "humans": 0,
            "faces": 0,
            "unknown_faces": 0,
            "plates": 0,
            "vehicles": 0,
            "crowd_level": "NORMAL",
            "crowd_count": 0,
            "layout": self.layout_mode,
            "device": DEVICE.upper(),
            "cam0": {"id": self.cam0_id, "name": self.dev0_name, "online": False},
            "cam1": {"id": self.cam1_id, "name": self.dev1_name, "online": False},
            "last_plate": "",
        }

        # Shared AI Model Singletons
        self.human_tracker_0: Optional[HumanTracker] = None
        self.human_tracker_1: Optional[HumanTracker] = None
        self.face_tracker_0: Optional[FastFaceTracker] = None
        self.face_tracker_1: Optional[FastFaceTracker] = None
        self.anpr_detector: Optional[ANPRDetector] = None
        self.plate_recorder: Optional[PlateRecorder] = None
        self.vehicle_tracker_0: Optional[VehicleTracker] = None
        self.vehicle_tracker_1: Optional[VehicleTracker] = None
        self.crowd_detector: Optional[CrowdDetector] = None

        self._init_models()

    def _init_models(self) -> None:
        """Initialize all AI model singletons."""
        print(f"[MultiCamStreamManager] Loading AI Detection Engines on {DEVICE.upper()}...")
        self.human_tracker_0 = HumanTracker(device=DEVICE)
        self.human_tracker_1 = HumanTracker(device=DEVICE)
        self.face_tracker_0 = FastFaceTracker(max_alerts_per_person=3)
        self.face_tracker_1 = FastFaceTracker(max_alerts_per_person=3)
        self.anpr_detector = ANPRDetector(device=DEVICE, enable_ocr=True)
        self.plate_recorder = PlateRecorder(cooldown_seconds=30.0)
        self.vehicle_tracker_0 = VehicleTracker(device=DEVICE)
        self.vehicle_tracker_1 = VehicleTracker(device=DEVICE)
        self.crowd_detector = CrowdDetector(device=DEVICE)
        print("[MultiCamStreamManager] All AI Models loaded.")

    def get_available_cameras(self) -> List[Dict[str, Any]]:
        """Return list of available physical cameras."""
        names = get_camera_names()
        results = []
        for idx, name in enumerate(names):
            results.append({
                "id": idx,
                "name": name,
                "is_cam0": (idx == self.cam0_id),
                "is_cam1": (idx == self.cam1_id),
            })
        if not results:
            # Fallback for devices without DirectShow names
            results = [
                {"id": 0, "name": "Camera 0 (Default)", "is_cam0": True, "is_cam1": False},
                {"id": 1, "name": "Camera 1 (Secondary)", "is_cam0": False, "is_cam1": True},
            ]
        return results

    def set_camera_ids(self, cam0: int, cam1: int) -> None:
        """Switch active camera inputs dynamically."""
        with self.lock:
            if self.cam0_id != cam0 or self.stream0 is None:
                if self.stream0:
                    self.stream0.release()
                self.cam0_id = cam0
                self.stream0 = ThreadedCameraStream(cam0, self.width, self.height)

            if self.cam1_id != cam1 or self.stream1 is None:
                if self.stream1:
                    self.stream1.release()
                self.cam1_id = cam1
                self.stream1 = ThreadedCameraStream(cam1, self.width, self.height)

            names = get_camera_names()
            self.dev0_name = names[cam0] if cam0 < len(names) else f"Camera {cam0}"
            self.dev1_name = names[cam1] if cam1 < len(names) else f"Camera {cam1}"
            print(f"[MultiCamStreamManager] Updated Cameras: Cam A={self.dev0_name}, Cam B={self.dev1_name}")

    def set_layout(self, layout: str) -> None:
        """Set layout mode: 'dual', '0', or '1'."""
        if layout in ["dual", "0", "1"]:
            self.layout_mode = layout
            self.stats["layout"] = layout
            print(f"[MultiCamStreamManager] Layout switched to: {layout.upper()}")

    def start(self) -> None:
        """Start threaded dual ingestion loop."""
        if self.running:
            return
        self.running = True

        names = get_camera_names()
        self.dev0_name = names[self.cam0_id] if self.cam0_id < len(names) else f"Camera {self.cam0_id}"
        self.dev1_name = names[self.cam1_id] if self.cam1_id < len(names) else f"Camera {self.cam1_id}"

        print(f"[MultiCamStreamManager] Starting Streams: Cam A [{self.dev0_name}], Cam B [{self.dev1_name}]...")
        self.stream0 = ThreadedCameraStream(self.cam0_id, self.width, self.height)
        time.sleep(0.15)
        self.stream1 = ThreadedCameraStream(self.cam1_id, self.width, self.height)

        self.worker_thread = threading.Thread(target=self._capture_and_process_loop, daemon=True)
        self.worker_thread.start()
        print("[MultiCamStreamManager] Multi-Camera worker thread active.")

    def _create_standby_frame(self, label: str) -> np.ndarray:
        """Generate animated synthetic CCTV feed when camera is in standby."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (14, 17, 24)

        for y in range(0, self.height, 40):
            cv2.line(frame, (0, y), (self.width, y), (25, 30, 42), 1)
        for x in range(0, self.width, 40):
            cv2.line(frame, (x, 0), (x, self.height), (25, 30, 42), 1)

        cx, cy = self.width // 2, self.height // 2
        cv2.circle(frame, (cx, cy), 40, (0, 255, 140), 1)
        cv2.putText(frame, label.upper(), (cx - 110, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 140), 1, cv2.LINE_AA)
        cv2.putText(frame, "STANDBY / NO SIGNAL", (cx - 95, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 155, 180), 1, cv2.LINE_AA)
        return frame

    def _draw_header_banner(
        self,
        frame: np.ndarray,
        label: str,
        dev_name: str,
        humans: int,
        faces: int,
        plates: int,
        vehicles: int,
        is_online: bool,
    ) -> np.ndarray:
        """Glassmorphic header banner for a camera feed."""
        h, w = frame.shape[:2]
        banner_h = 32
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (14, 18, 26), -1)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
        cv2.line(frame, (0, banner_h), (w, banner_h), (40, 50, 70), 1)

        dot_col = (0, 255, 120) if is_online else (0, 60, 240)
        cv2.circle(frame, (14, banner_h // 2), 4, dot_col, -1)

        title = f"{label.upper()} | {dev_name[:24]}"
        cv2.putText(frame, title, (26, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 245), 1, cv2.LINE_AA)

        badge_txt = f"H:{humans} F:{faces} P:{plates} V:{vehicles}"
        (bw, _), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        bx = max(10, w - bw - 12)
        cv2.rectangle(frame, (bx - 6, 4), (w - 6, 28), (22, 28, 40), -1)
        cv2.rectangle(frame, (bx - 6, 4), (w - 6, 28), (0, 255, 140), 1)
        cv2.putText(frame, badge_txt, (bx, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 180), 1, cv2.LINE_AA)
        return frame

    def _combine_dual_feeds(
        self,
        f0: np.ndarray,
        f1: np.ndarray,
        target_h: int = 480,
    ) -> np.ndarray:
        """Side-by-side dual camera view compositor."""
        h0, w0 = f0.shape[:2]
        w0_res = int(w0 * (target_h / h0))
        f0_res = cv2.resize(f0, (w0_res, target_h)) if (h0 != target_h) else f0

        h1, w1 = f1.shape[:2]
        w1_res = int(w1 * (target_h / h1))
        f1_res = cv2.resize(f1, (w1_res, target_h)) if (h1 != target_h) else f1

        combined = np.hstack((f0_res, f1_res))
        cv2.line(combined, (w0_res, 0), (w0_res, target_h), (0, 255, 140), 2)
        return combined

    def _capture_and_process_loop(self) -> None:
        """Continuous concurrent ingestion and AI inference loop."""
        prev_time = time.time()

        while self.running:
            r0, raw0 = self.stream0.read() if self.stream0 else (False, None)
            r1, raw1 = self.stream1.read() if self.stream1 else (False, None)

            if not r0 or raw0 is None:
                raw0 = self._create_standby_frame(f"Feed A: {self.dev0_name}")
            if not r1 or raw1 is None:
                raw1 = self._create_standby_frame(f"Feed B: {self.dev1_name}")

            out0 = raw0.copy()
            out1 = raw1.copy()
            self.frame_counter += 1

            # Interleaved Multi-Camera AI Execution
            h0, f0, p0, v0 = 0, 0, 0, 0
            h1, f1, p1, v1 = 0, 0, 0, 0
            unk0, unk1 = 0, 0
            human_boxes0, face_boxes0 = [], []
            human_boxes1, face_boxes1 = [], []

            run_ai_cam0 = (self.layout_mode in ["dual", "0"]) and r0
            run_ai_cam1 = (self.layout_mode in ["dual", "1"]) and r1

            # 1. Faces (YuNet + SFace)
            if self.enable_faces:
                if run_ai_cam0 and (self.frame_counter % 2 == 0):
                    out0, f0 = self.face_tracker_0.process_frame(out0, run_detection=True)
                    face_boxes0 = [t.bbox for t in self.face_tracker_0.tracks]
                    unk0 = self.face_tracker_0.total_unknown_count

                if run_ai_cam1 and (self.frame_counter % 2 == 1):
                    out1, f1 = self.face_tracker_1.process_frame(out1, run_detection=True)
                    face_boxes1 = [t.bbox for t in self.face_tracker_1.tracks]
                    unk1 = self.face_tracker_1.total_unknown_count

            # 2. Humans (YOLOv8)
            if self.enable_humans:
                if run_ai_cam0 and (self.frame_counter % 4 == 0):
                    out0, _, h0 = self.human_tracker_0.process_frame(out0, existing_face_boxes=face_boxes0, run_detection=True)
                    human_boxes0 = [t.bbox for t in self.human_tracker_0.tracks]

                if run_ai_cam1 and (self.frame_counter % 4 == 2):
                    out1, _, h1 = self.human_tracker_1.process_frame(out1, existing_face_boxes=face_boxes1, run_detection=True)
                    human_boxes1 = [t.bbox for t in self.human_tracker_1.tracks]

            # 3. Vehicles (YOLO Multi-Class)
            if self.enable_vehicles:
                if run_ai_cam0 and (self.frame_counter % 4 == 1):
                    out0, _, v0 = self.vehicle_tracker_0.process_frame(out0, run_detection=True)

                if run_ai_cam1 and (self.frame_counter % 4 == 3):
                    out1, _, v1 = self.vehicle_tracker_1.process_frame(out1, run_detection=True)

            # 4. License Plates (ANPR)
            if self.enable_plates:
                if run_ai_cam0 and (self.frame_counter % 6 == 0):
                    p_boxes0 = human_boxes0 + face_boxes0
                    out0, p_dets0, p0 = self.anpr_detector.process_frame(out0, person_boxes=p_boxes0, run_detection=True)
                    for det in p_dets0:
                        txt = det.get("text", "")
                        if txt and txt != "LICENSE PLATE":
                            is_new, cleaned = self.plate_recorder.log_plate(txt, det.get("conf", 0.0), self.dev0_name, det.get("crop"))
                            if is_new:
                                self.stats["last_plate"] = cleaned
                                db_manager.log_anpr(camera_id=f"Cam-{self.cam0_id}", plate_number=cleaned, confidence=det.get("conf", 0.0))

                if run_ai_cam1 and (self.frame_counter % 6 == 3):
                    p_boxes1 = human_boxes1 + face_boxes1
                    out1, p_dets1, p1 = self.anpr_detector.process_frame(out1, person_boxes=p_boxes1, run_detection=True)
                    for det in p_dets1:
                        txt = det.get("text", "")
                        if txt and txt != "LICENSE PLATE":
                            is_new, cleaned = self.plate_recorder.log_plate(txt, det.get("conf", 0.0), self.dev1_name, det.get("crop"))
                            if is_new:
                                self.stats["last_plate"] = cleaned
                                db_manager.log_anpr(camera_id=f"Cam-{self.cam1_id}", plate_number=cleaned, confidence=det.get("conf", 0.0))

            # Draw Individual Camera Banners
            out0 = self._draw_header_banner(out0, "Feed A", self.dev0_name, h0, f0, p0, v0, r0)
            out1 = self._draw_header_banner(out1, "Feed B", self.dev1_name, h1, f1, p1, v1, r1)

            # Compose Stream Based on Layout Mode
            if self.layout_mode == "dual":
                composite = self._combine_dual_feeds(out0, out1, target_h=480)
            elif self.layout_mode == "0":
                composite = out0
            else:
                composite = out1

            # Calculate FPS
            curr_time = time.time()
            instant_fps = 1.0 / max(0.0001, curr_time - prev_time)
            prev_time = curr_time
            self.fps_smooth = (0.85 * self.fps_smooth) + (0.15 * instant_fps)

            # Update Telemetry Stats
            with self.lock:
                self.frame_cam0 = out0
                self.frame_cam1 = out1
                self.composite_frame = composite
                self.stats.update({
                    "fps": round(self.fps_smooth, 1),
                    "humans": h0 + h1,
                    "faces": f0 + f1,
                    "unknown_faces": unk0 + unk1,
                    "plates": p0 + p1,
                    "vehicles": v0 + v1,
                    "layout": self.layout_mode,
                    "cam0": {"id": self.cam0_id, "name": self.dev0_name, "online": r0, "humans": h0, "faces": f0},
                    "cam1": {"id": self.cam1_id, "name": self.dev1_name, "online": r1, "humans": h1, "faces": f1},
                    "frame_count": self.frame_counter,
                })

            time.sleep(0.01)

    def get_latest_jpeg(self, which: str = "composite") -> bytes:
        """Encode requested frame as JPEG bytes."""
        with self.lock:
            if which == "0":
                frame = self.frame_cam0
            elif which == "1":
                frame = self.frame_cam1
            else:
                frame = self.composite_frame

        if frame is None:
            frame = self._create_standby_frame("Initializing Multi-Camera Stream")

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buffer.tobytes() if ret else b""

    def mjpeg_generator(self, which: str = "composite"):
        """Yield multipart JPEG chunks for MJPEG streaming."""
        while self.running:
            jpeg_bytes = self.get_latest_jpeg(which=which)
            if jpeg_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
            time.sleep(0.033)

    def update_toggles(self, data: Dict[str, Any]) -> None:
        """Update active AI pipeline toggles or camera layout."""
        if "layout" in data:
            self.set_layout(str(data["layout"]))
        if "humans" in data:
            self.enable_humans = bool(data["humans"])
        if "faces" in data:
            self.enable_faces = bool(data["faces"])
        if "plates" in data:
            self.enable_plates = bool(data["plates"])
        if "vehicles" in data:
            self.enable_vehicles = bool(data["vehicles"])
        if "crowd" in data:
            self.enable_crowd = bool(data["crowd"])
        if "conf_threshold" in data:
            self.conf_threshold = float(data["conf_threshold"])

    def stop(self) -> None:
        """Release camera streams and stop threads."""
        self.running = False
        if self.stream0:
            self.stream0.release()
        if self.stream1:
            self.stream1.release()
        print("[MultiCamStreamManager] Released all camera streams.")


# Global Multi-Camera Stream Manager Singleton
stream_manager = MultiCamStreamManager(cam0_id=0, cam1_id=1)
