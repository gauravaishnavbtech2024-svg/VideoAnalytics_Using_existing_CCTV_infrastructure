"""
High-Speed YOLOv8 Multi-Person Human Detector & Tracker Module.
===============================================================

Optimized for 30+ FPS real-time surveillance:
- CUDA FP16 Half-Precision Tensor Core acceleration.
- Optimized 480px inference resolution for instant human bounding box generation.
- Clean spatial IoU tracking with smooth Kalman-inspired box smoothing.
- Clean single-tag badge rendering.
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
DEFAULT_HUMAN_MODEL = PROJECT_ROOT / "models" / "human_detection" / "master_human_model" / "weights" / "best.pt"


@dataclass
class HumanTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float = 0.0
    missing_frames: int = 0


class HumanTracker:
    """High-speed YOLO Human Detection and Tracking Engine."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.40,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.conf_threshold = conf_threshold
        self.device = device
        self.tracks: List[HumanTrack] = []
        self.next_id = 1
        self.use_half = "cuda" in str(device).lower() and torch.cuda.is_available()

        # Load weights
        chosen_path = self._find_best_model_path(model_path)
        print(f"[HumanTracker] Loading YOLO Model on {self.device} (FP16={self.use_half}): {chosen_path.name}")
        self.model = YOLO(str(chosen_path))
        if self.use_half:
            try:
                self.model.to(self.device).half()
            except Exception:
                pass

    def _find_best_model_path(self, model_path: str | Path | None) -> Path:
        if model_path:
            p = Path(model_path)
            if p.is_file():
                return p

        candidates = [
            PROJECT_ROOT / "models" / "human_detection" / "master_human_model" / "weights" / "best.pt",
            PROJECT_ROOT / "models" / "human_detection" / "master_human_model" / "weights" / "last.pt",
            PROJECT_ROOT / "yolov8s.pt",
        ]
        for c in candidates:
            if c.is_file() and c.stat().st_size > 100000:
                return c
        return Path("yolov8s.pt")

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

    def detect_humans(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Fast YOLO inference returning (x, y, w, h, conf)."""
        if frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]
        detections = []

        try:
            results = self.model.predict(
                source=frame_bgr,
                conf=self.conf_threshold,
                classes=[0] if self.model.names and 0 in self.model.names else None,
                device=self.device,
                imgsz=288,
                verbose=False,
            )

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    bw = x2 - x1
                    bh = y2 - y1

                    if bw > 20 and bh > 30:
                        detections.append((x1, y1, bw, bh, float(conf)))
        except Exception:
            pass

        return detections

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        existing_face_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        run_detection: bool = True,
    ) -> Tuple[np.ndarray, List[Dict], int]:
        """
        Detect and track humans across frames with exactly ONE clean tag per human.
        Returns: (annotated_frame_bgr, human_records, count)
        """
        if frame_bgr is None:
            return frame_bgr, [], 0

        annotated = frame_bgr.copy()

        if run_detection:
            detections = self.detect_humans(frame_bgr)

            # Spatial tracking update
            updated_tracks: List[HumanTrack] = []
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
                    nx, ny, nw, nh, conf = detections[best_det_idx]
                    ox, oy, ow, oh = track.bbox
                    track.bbox = (
                        int(0.6 * nx + 0.4 * ox),
                        int(0.6 * ny + 0.4 * oy),
                        int(0.6 * nw + 0.4 * ow),
                        int(0.6 * nh + 0.4 * oh),
                    )
                    track.confidence = conf
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
                    x, y, w, h, conf = det
                    updated_tracks.append(
                        HumanTrack(
                            track_id=self.next_id,
                            bbox=(x, y, w, h),
                            confidence=conf,
                        )
                    )
                    self.next_id += 1

            self.tracks = updated_tracks

        human_records: List[Dict] = []

        # Render Bounding Boxes: Single clean tag per human box
        for track in self.tracks:
            x, y, w, h = track.bbox

            # If an existing face box is already covering this exact head area, skip duplicate tag
            has_nested_face = False
            if existing_face_boxes:
                for fx, fy, fw, fh in existing_face_boxes:
                    # Check if face box is inside or close to the human top area
                    if (x - 20 <= fx <= x + w + 20) and (y - 20 <= fy <= y + int(h * 0.6)):
                        has_nested_face = True
                        break

            # If face is already tagged, only draw human outer outline without conflicting tag
            box_color = (0, 220, 255)  # Cyan
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

            if not has_nested_face:
                label_text = f"🚶 PERSON #{track.track_id}"
                (tw, th_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                ly = max(th_text + 6, y - 6)

                cv2.rectangle(annotated, (x, ly - th_text - 4), (x + tw + 8, ly + 2), (18, 30, 42), -1)
                cv2.rectangle(annotated, (x, ly - th_text - 4), (x + tw + 8, ly + 2), box_color, 1)
                cv2.putText(
                    annotated, label_text, (x + 4, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA,
                )

            human_records.append({
                "track_id": track.track_id,
                "bbox": track.bbox,
                "conf": track.confidence,
            })

        return annotated, human_records, len(self.tracks)
