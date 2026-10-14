"""
High-Speed YOLOv8 Multi-Class Vehicle Detector & Tracker Module.
================================================================

Optimized for real-time CCTV surveillance:
- CUDA FP16 acceleration.
- Multi-class vehicle tracking (Bus, Car, E-Bike, Jeep, Motorcycle, Tricycle, Truck, Van).
- Fast spatial IoU tracking with smooth bounding-box smoothing.
- Clean high-tech glassmorphic bounding box and badge rendering.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VEHICLE_MODEL = PROJECT_ROOT / "models" / "vehicle_detection" / "master_vehicle_model" / "weights" / "best.pt"

VEHICLE_EMOJIS = {
    "bus": "🚌 BUS",
    "car": "🚗 CAR",
    "e-bike": "🚲 E-BIKE",
    "jeep": "🚙 JEEP",
    "motorcycle": "🏍️ MOTORCYCLE",
    "tricycle": "🛺 TRICYCLE",
    "truck": "🚚 TRUCK",
    "van": "🚐 VAN",
}


@dataclass
class VehicleTrack:
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float = 0.0
    missing_frames: int = 0


class VehicleTracker:
    """High-speed YOLO Multi-Class Vehicle Detection and Tracking Engine."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.35,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.conf_threshold = conf_threshold
        self.device = device
        self.tracks: List[VehicleTrack] = []
        self.next_id = 1
        self.use_half = "cuda" in str(device).lower() and torch.cuda.is_available()

        # Load weights
        chosen_path = self._find_best_model_path(model_path)
        print(f"[VehicleTracker] Loading Trained Vehicle Model on {self.device} (FP16={self.use_half}): {chosen_path.name}")
        self.model = YOLO(str(chosen_path))
        if self.use_half:
            try:
                self.model.to(self.device).half()
            except Exception:
                pass

        print(f"[VehicleTracker] Vehicle Classes: {self.model.names}")

    def _find_best_model_path(self, model_path: str | Path | None) -> Path:
        if model_path:
            p = Path(model_path)
            if p.is_file():
                return p

        candidates = [
            PROJECT_ROOT / "models" / "vehicle_detection" / "master_vehicle_model" / "weights" / "best.pt",
            PROJECT_ROOT / "models" / "vehicle_detection" / "best.pt",
            PROJECT_ROOT / "models" / "vehicle_detection" / "master_vehicle_model" / "weights" / "last.pt",
            PROJECT_ROOT / "yolo11n.pt",
            PROJECT_ROOT / "yolo11s.pt",
            PROJECT_ROOT / "yolov8s.pt",
        ]
        for c in candidates:
            if c.is_file() and c.stat().st_size > 100000:
                return c
        return Path("yolo11n.pt")

    def _iou(self, b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        iw = max(0, xi2 - xi1)
        ih = max(0, yi2 - yi1)
        inter = iw * ih
        union = (w1 * h1) + (w2 * h2) - inter
        return inter / union if union > 0 else 0.0

    def detect_vehicles(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Fast YOLO inference returning (x, y, w, h, conf, class_name)."""
        if frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]
        detections = []

        try:
            results = self.model.predict(
                source=frame_bgr,
                conf=self.conf_threshold,
                device=self.device,
                imgsz=288,
                verbose=False,
            )

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else [0] * len(boxes)

                for box, conf, cls_id in zip(boxes, confs, cls_ids):
                    cls_name = self.model.names.get(int(cls_id), "Vehicle") if self.model.names else "Vehicle"
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    bw = x2 - x1
                    bh = y2 - y1

                    if bw > 25 and bh > 20:
                        detections.append((x1, y1, bw, bh, float(conf), cls_name))
        except Exception:
            pass

        return detections

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        run_detection: bool = True,
    ) -> Tuple[np.ndarray, List[Dict], int]:
        """
        Detect and track vehicles across frames with exactly ONE clean high-tech badge per vehicle.
        Returns: (annotated_frame_bgr, vehicle_records, count)
        """
        if frame_bgr is None:
            return frame_bgr, [], 0

        annotated = frame_bgr.copy()

        if run_detection:
            detections = self.detect_vehicles(frame_bgr)

            # Spatial tracking update
            updated_tracks: List[VehicleTrack] = []
            matched_detections = set()

            for track in self.tracks:
                best_iou = 0.0
                best_det_idx = -1
                for idx, det in enumerate(detections):
                    if idx in matched_detections:
                        continue
                    iou = self._iou(track.bbox, det[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = idx

                if best_iou >= 0.25 and best_det_idx != -1:
                    nx, ny, nw, nh, conf, cls_name = detections[best_det_idx]
                    ox, oy, ow, oh = track.bbox
                    track.bbox = (
                        int(0.6 * nx + 0.4 * ox),
                        int(0.6 * ny + 0.4 * oy),
                        int(0.6 * nw + 0.4 * ow),
                        int(0.6 * nh + 0.4 * oh),
                    )
                    track.confidence = conf
                    track.class_name = cls_name
                    track.missing_frames = 0
                    updated_tracks.append(track)
                    matched_detections.add(best_det_idx)
                else:
                    track.missing_frames += 1
                    if track.missing_frames < 6:
                        updated_tracks.append(track)

            # Add new tracks
            for idx, det in enumerate(detections):
                if idx not in matched_detections:
                    x, y, w, h, conf, cls_name = det
                    updated_tracks.append(
                        VehicleTrack(
                            track_id=self.next_id,
                            class_name=cls_name,
                            bbox=(x, y, w, h),
                            confidence=conf,
                        )
                    )
                    self.next_id += 1

            self.tracks = updated_tracks

        vehicle_records: List[Dict] = []

        # Render Bounding Boxes: High-tech Amber/Orange neon
        for track in self.tracks:
            x, y, w, h = track.bbox

            box_color = (0, 165, 255)  # Neon Amber / Orange
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

            # High-tech corner accents
            line_len = max(8, min(w, h) // 4)
            th = 3
            cv2.line(annotated, (x, y), (x + line_len, y), box_color, th)
            cv2.line(annotated, (x, y), (x, y + line_len), box_color, th)
            cv2.line(annotated, (x + w, y), (x + w - line_len, y), box_color, th)
            cv2.line(annotated, (x + w, y), (x + w, y + line_len), box_color, th)
            cv2.line(annotated, (x, y + h), (x + line_len, y + h), box_color, th)
            cv2.line(annotated, (x, y + h), (x, y + h - line_len), box_color, th)
            cv2.line(annotated, (x + w, y + h), (x + w - line_len, y + h), box_color, th)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - line_len), box_color, th)

            # Class badge label
            cls_lower = track.class_name.lower()
            formatted_name = VEHICLE_EMOJIS.get(cls_lower, f"🚗 {track.class_name.upper()}")
            label_text = f"{formatted_name} #{track.track_id}"

            (tw, th_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            ly = max(th_text + 6, y - 6)

            cv2.rectangle(annotated, (x, ly - th_text - 4), (x + tw + 8, ly + 2), (24, 28, 38), -1)
            cv2.rectangle(annotated, (x, ly - th_text - 4), (x + tw + 8, ly + 2), box_color, 1)
            cv2.putText(
                annotated, label_text, (x + 4, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA,
            )

            vehicle_records.append({
                "track_id": track.track_id,
                "class_name": track.class_name,
                "bbox": track.bbox,
                "conf": track.confidence,
            })

        return annotated, vehicle_records, len(self.tracks)
