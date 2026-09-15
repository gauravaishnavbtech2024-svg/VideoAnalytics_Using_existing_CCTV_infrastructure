"""
High-Performance Real-Time Face Detector & Face Recognition Tracker Module.
===========================================================================

Optimized for 30+ FPS real-time surveillance:
1. OpenCV YuNet DNN Deep Learning Face Detector with downscaled fast forward pass.
2. OpenCV SFace Deep Feature Recognition Engine with intelligent embedding caching.
3. Single clean tag per face box:
   - Green Square + '● <NAME>' for Known / Authorized individuals.
   - Red Square + '○ UNKNOWN' for Unknown / Unauthorized individuals.
4. 100% Offline & Local processing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YUNET_MODEL = PROJECT_ROOT / "models" / "face" / "face_detection_yunet_2023mar.onnx"
DEFAULT_SFACE_MODEL = PROJECT_ROOT / "models" / "face" / "face_recognition_sface_2021dec.onnx"
DEFAULT_KNOWN_FACES_DIR = PROJECT_ROOT / "known_faces"


@dataclass
class FaceTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float = 0.0
    name: str = "UNKNOWN"
    is_known: bool = False
    match_score: float = 0.0
    frames_since_sface: int = 0
    missing_frames: int = 0


class FastFaceTracker:
    """Real-time Face Detector, Tracker and SFace Recognition System."""

    def __init__(
        self,
        min_size: Tuple[int, int] = (25, 25),
        conf_threshold: float = 0.48,
        match_threshold: float = 0.38,
        max_alerts_per_person: int = 3,
        yunet_path: str | Path | None = None,
        sface_path: str | Path | None = None,
        known_faces_dir: str | Path | None = None,
        **kwargs,
    ) -> None:
        self.min_size = min_size
        self.conf_threshold = conf_threshold
        self.match_threshold = match_threshold
        self.max_alerts_per_person = max_alerts_per_person
        self.tracks: List[FaceTrack] = []

        self.next_id = 1
        self.frame_count = 0

        self.detector = None
        self.recognizer = None
        self.use_dnn = False

        self.known_faces_dir = Path(known_faces_dir) if known_faces_dir else DEFAULT_KNOWN_FACES_DIR
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        self.known_database: List[Tuple[str, np.ndarray]] = []

        # 1. Initialize YuNet DNN Face Detector
        y_path = Path(yunet_path) if yunet_path else DEFAULT_YUNET_MODEL
        if y_path.is_file():
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    str(y_path),
                    "",
                    (320, 240),
                    self.conf_threshold,
                    0.3,
                    5000,
                )
                self.use_dnn = True
            except Exception as e:
                print(f"[FaceTracker] YuNet load failed: {e}")

        # 2. Initialize SFace Recognition Engine
        s_path = Path(sface_path) if sface_path else DEFAULT_SFACE_MODEL
        if s_path.is_file():
            try:
                self.recognizer = cv2.FaceRecognizerSF.create(str(s_path), "")
                print(f"[FaceTracker] Loaded SFace Recognition Engine: {s_path.name}")
            except Exception as e:
                print(f"[FaceTracker] SFace load failed: {e}")

        if not self.use_dnn:
            self.cascade_frontal = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        self.load_known_faces()

    def load_known_faces(self) -> int:
        """Scan known_faces/ directory and compute reference embeddings for all images."""
        self.known_database.clear()
        if not self.known_faces_dir.is_dir() or self.recognizer is None or self.detector is None:
            return 0

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [p for p in self.known_faces_dir.rglob("*") if p.suffix.lower() in img_exts]

        for img_path in image_files:
            person_name = img_path.stem.upper()
            if img_path.parent != self.known_faces_dir:
                person_name = img_path.parent.name.upper()

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(img_bgr)

            if faces is not None and len(faces) > 0:
                face_data = faces[0]
                aligned = self.recognizer.alignCrop(img_bgr, face_data)
                feat = self.recognizer.feature(aligned)
                self.known_database.append((person_name, feat))

        print(f"[FaceTracker] Enrolled {len(self.known_database)} known face identities from: {self.known_faces_dir.name}/")
        for name, _ in self.known_database:
            print(f"   [+] Verified Identity: {name}")
        return len(self.known_database)

    def _match_face(self, frame_bgr: np.ndarray, raw_face: np.ndarray) -> Tuple[bool, str, float]:
        """Align crop, compute SFace embedding, and match against enrolled database."""
        if self.recognizer is None or not self.known_database:
            return False, "UNKNOWN", 0.0

        try:
            aligned = self.recognizer.alignCrop(frame_bgr, raw_face)
            feat = self.recognizer.feature(aligned)

            best_score = -1.0
            best_name = "UNKNOWN"

            for name, ref_feat in self.known_database:
                score = self.recognizer.match(feat, ref_feat, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score >= self.match_threshold:
                return True, best_name, float(best_score)
            else:
                return False, "UNKNOWN", float(best_score)
        except Exception:
            return False, "UNKNOWN", 0.0

    def detect_raw_faces(self, frame_bgr: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float, np.ndarray]]:
        """Ultra-fast face detection pass using 320x240 input scaling."""
        h, w = frame_bgr.shape[:2]
        results = []

        if self.use_dnn and self.detector is not None:
            det_w, det_h = 320, 240
            scale_x = w / det_w
            scale_y = h / det_h

            small_frame = cv2.resize(frame_bgr, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
            self.detector.setInputSize((det_w, det_h))
            _, faces = self.detector.detect(small_frame)

            if faces is not None:
                for face in faces:
                    # Scale coordinates & landmarks back to original resolution
                    scaled_face = face.copy()
                    x = int(face[0] * scale_x)
                    y = int(face[1] * scale_y)
                    fw = int(face[2] * scale_x)
                    fh = int(face[3] * scale_y)
                    conf = float(face[-1])

                    scaled_face[0] = x
                    scaled_face[1] = y
                    scaled_face[2] = fw
                    scaled_face[3] = fh
                    # Scale landmarks (eyes, nose, mouth corners)
                    for li in range(4, 14, 2):
                        scaled_face[li] = face[li] * scale_x
                        scaled_face[li + 1] = face[li + 1] * scale_y

                    x = max(0, x)
                    y = max(0, y)
                    fw = min(fw, w - x)
                    fh = min(fh, h - y)

                    if fw >= self.min_size[0] and fh >= self.min_size[1]:
                        results.append(((x, y, fw, fh), conf, scaled_face))
        return results

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

    def process_frame(self, frame_bgr: np.ndarray, run_detection: bool = True) -> Tuple[np.ndarray, int]:
        """
        Detect, recognize, and track faces with exactly ONE clear tag per box.
        Returns: (annotated_frame, face_count)
        """
        if frame_bgr is None:
            return frame_bgr, 0

        self.frame_count += 1
        annotated = frame_bgr.copy()

        if run_detection:
            raw_faces = self.detect_raw_faces(frame_bgr)

            # Match with active tracks
            updated_tracks: List[FaceTrack] = []
            matched_detections = set()

            for track in self.tracks:
                best_iou = 0.0
                best_idx = -1
                for idx, (bbox, conf, face_data) in enumerate(raw_faces):
                    if idx in matched_detections:
                        continue
                    iou = self._iou(track.bbox, bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                if best_iou >= 0.25 and best_idx != -1:
                    bbox, conf, face_data = raw_faces[best_idx]
                    nx, ny, nw, nh = bbox
                    ox, oy, ow, oh = track.bbox
                    track.bbox = (
                        int(0.6 * nx + 0.4 * ox),
                        int(0.6 * ny + 0.4 * oy),
                        int(0.6 * nw + 0.4 * ow),
                        int(0.6 * nh + 0.4 * oh),
                    )
                    track.confidence = conf
                    track.missing_frames = 0
                    track.frames_since_sface += 1

                    # Re-verify embedding only periodically or if unknown
                    if not track.is_known or (track.frames_since_sface > 30):
                        is_k, name, score = self._match_face(frame_bgr, face_data)
                        track.is_known = is_k
                        track.name = name
                        track.match_score = score
                        track.frames_since_sface = 0

                    updated_tracks.append(track)
                    matched_detections.add(best_idx)
                else:
                    track.missing_frames += 1
                    if track.missing_frames < 6:
                        updated_tracks.append(track)

            # New faces detected
            for idx, (bbox, conf, face_data) in enumerate(raw_faces):
                if idx not in matched_detections:
                    is_k, name, score = self._match_face(frame_bgr, face_data)
                    updated_tracks.append(
                        FaceTrack(
                            track_id=self.next_id,
                            bbox=bbox,
                            confidence=conf,
                            name=name,
                            is_known=is_k,
                            match_score=score,
                            frames_since_sface=0,
                        )
                    )
                    self.next_id += 1

            self.tracks = updated_tracks

        # Render Bounding Boxes: EXACTLY ONE CLEAN TAG PER BOX
        for track in self.tracks:
            x, y, w, h = track.bbox

            if track.is_known:
                box_color = (0, 255, 120)       # Neon Green
                bg_badge = (12, 45, 20)          # Dark Green
                label_text = f"● {track.name}"
            else:
                box_color = (0, 40, 255)         # Bright Neon Red
                bg_badge = (18, 15, 55)          # Dark Red
                label_text = "○ UNKNOWN"

            # 1. Main Bounding Box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

            # 2. High-Tech Corner Accents
            line_len = max(8, min(w, h) // 4)
            th = 3
            # Top-left
            cv2.line(annotated, (x, y), (x + line_len, y), box_color, th)
            cv2.line(annotated, (x, y), (x, y + line_len), box_color, th)
            # Top-right
            cv2.line(annotated, (x + w, y), (x + w - line_len, y), box_color, th)
            cv2.line(annotated, (x + w, y), (x + w, y + line_len), box_color, th)
            # Bottom-left
            cv2.line(annotated, (x, y + h), (x + line_len, y + h), box_color, th)
            cv2.line(annotated, (x, y + h), (x, y + h - line_len), box_color, th)
            # Bottom-right
            cv2.line(annotated, (x + w, y + h), (x + w - line_len, y + h), box_color, th)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - line_len), box_color, th)

            # 3. Exactly ONE Single Tag Badge
            (tw, th_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            ly = max(th_text + 6, y - 6)

            cv2.rectangle(annotated, (x, ly - th_text - 4), (x + tw + 8, ly + 2), bg_badge, -1)
            cv2.rectangle(annotated, (x, ly - th_text - 4), (x + tw + 8, ly + 2), box_color, 1)
            cv2.putText(
                annotated, label_text, (x + 4, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return annotated, len(self.tracks)

    @property
    def total_unknown_count(self) -> int:
        """Total number of unknown faces currently visible in frame."""
        return sum(1 for t in self.tracks if not t.is_known)

    @property
    def known_count(self) -> int:
        """Number of active verified/authorized faces in frame."""
        return sum(1 for t in self.tracks if t.is_known)
