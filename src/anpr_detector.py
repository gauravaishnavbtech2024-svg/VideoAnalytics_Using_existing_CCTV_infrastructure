"""
High-Speed Automatic Number Plate Recognition (ANPR) & OCR Engine.
==================================================================

Features:
- Specifically loads trained license plate weights (detecting ONLY vehicle number plates, never humans/persons).
- Supports both standard horizontal BBoxes and OBB (Oriented Bounding Boxes).
- Non-blocking Asynchronous Background EasyOCR:
  YOLO detection runs at 30+ FPS while character recognition executes in the background.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ANPRDetector:
    """High-speed YOLO + Asynchronous EasyOCR Number Plate Detector."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.35,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        enable_ocr: bool = True,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.device = device
        self.enable_ocr = enable_ocr
        self.is_obb = False
        self.use_half = "cuda" in str(device).lower() and torch.cuda.is_available()

        # Cached OCR text per spatial plate box
        self.plate_text_cache: Dict[str, str] = {}
        self.ocr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.pending_ocr = set()

        # 1. Load Trained License Plate YOLO Model
        chosen_path = self._find_model_path(model_path)
        print(f"[ANPRDetector] Loading trained Plate model on {self.device}: {chosen_path.name}")
        self.model = YOLO(str(chosen_path))
        if self.use_half:
            try:
                self.model.to(self.device).half()
            except Exception:
                pass

        task_type = getattr(self.model, "task", "")
        if "obb" in str(task_type).lower() or "obb" in str(chosen_path).lower():
            self.is_obb = True

        print(f"[ANPRDetector] Plate Task: {self.model.task} | Classes: {self.model.names}")

        # 2. Initialize EasyOCR
        self.ocr_reader = None
        if self.enable_ocr:
            self._init_easyocr()

    def _find_model_path(self, model_path: str | Path | None) -> Path:
        if model_path:
            p = Path(model_path)
            if p.is_file():
                return p

        candidates = [
            PROJECT_ROOT / "models" / "anpr" / "license_plate_model" / "weights" / "best.pt",
            PROJECT_ROOT / "models" / "anpr" / "run_2_Indian_number_plate.v2i.yolov8-obb" / "weights" / "best.pt",
            PROJECT_ROOT / "models" / "anpr" / "run_1_Indian_number_plate.v1i.yolov8-obb" / "weights" / "best.pt",
            PROJECT_ROOT / "models" / "anpr" / "license_plate_model" / "weights" / "last.pt",
        ]
        for c in candidates:
            if c.is_file() and c.stat().st_size > 100000:
                return c

        # Search any .pt in models/anpr
        anpr_dir = PROJECT_ROOT / "models" / "anpr"
        if anpr_dir.exists():
            for pt in anpr_dir.rglob("*.pt"):
                if pt.is_file() and "best" in pt.stem:
                    return pt

        return PROJECT_ROOT / "yolov8s.pt"

    def _init_easyocr(self) -> None:
        try:
            import easyocr
            use_gpu = ("cuda" in str(self.device).lower() and torch.cuda.is_available())
            self.ocr_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
            print(f"[ANPRDetector] EasyOCR initialized with GPU={use_gpu}")
        except Exception as e:
            print(f"[ANPRDetector] EasyOCR initialization failed: {e}")
            self.ocr_reader = None

    def preprocess_plate_for_ocr(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Enhance plate contrast and sharpness for maximum OCR accuracy."""
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr

        h, w = crop_bgr.shape[:2]
        if h < 50:
            scale = 60.0 / max(1, h)
            crop_bgr = cv2.resize(crop_bgr, (int(w * scale), 60), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 7, 50, 50)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        return clahe.apply(denoised)

    def _sync_read_ocr(self, crop_bgr: np.ndarray, cache_key: str) -> None:
        """Background OCR extraction thread."""
        try:
            if self.ocr_reader is None or crop_bgr is None or crop_bgr.size == 0:
                return

            enhanced = self.preprocess_plate_for_ocr(crop_bgr)
            results = self.ocr_reader.readtext(
                enhanced,
                detail=1,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                batch_size=1,
            )
            if not results:
                results = self.ocr_reader.readtext(
                    crop_bgr,
                    detail=1,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                )

            extracted_parts = []
            for (_, text, conf) in results:
                clean_t = re.sub(r"[^A-Z0-9]", "", text.upper())
                if len(clean_t) >= 2 and conf >= 0.20:
                    extracted_parts.append(clean_t)

            final_text = "".join(extracted_parts)

            # Strict validation: Reject person tags, UI text, or non-plate noise
            blacklist = ["PERSON", "UNKNOWN", "USER", "STUDENT", "STAFF", "EMPLOYEE", "CAMERA", "FEED", "HUMAN", "ALERT", "HEALTH", "CYBER", "CARD", "BADGE", "NAME", "OFFICE"]
            for b in blacklist:
                if b in final_text:
                    return

            has_letter = bool(re.search(r"[A-Z]", final_text))
            has_digit = bool(re.search(r"[0-9]", final_text))

            if len(final_text) >= 5 and has_letter and has_digit:
                self.plate_text_cache[cache_key] = final_text
        except Exception:
            pass
        finally:
            self.pending_ocr.discard(cache_key)

    def trigger_async_ocr(self, crop_bgr: np.ndarray, cache_key: str) -> str:
        """Trigger non-blocking OCR or return cached result instantly."""
        if cache_key in self.plate_text_cache:
            return self.plate_text_cache[cache_key]

        if cache_key not in self.pending_ocr and self.enable_ocr and self.ocr_reader is not None:
            self.pending_ocr.add(cache_key)
            self.ocr_executor.submit(self._sync_read_ocr, crop_bgr.copy(), cache_key)

        return ""

    def _rectify_obb_crop(self, frame_bgr: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
        """Crop and perspective warp an oriented bounding box."""
        try:
            rect = pts.astype("float32")
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB), 60)

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB), 25)

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(frame_bgr, M, (maxWidth, maxHeight))
        except Exception:
            return None

    def _is_inside_person(
        self,
        box: Tuple[int, int, int, int],
        person_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> bool:
        """Check if a detection box overlaps with or is inside a detected human body / person area."""
        if not person_boxes:
            return False

        bx, by, bw, bh = box
        bcx, bcy = bx + bw // 2, by + bh // 2

        for (px, py, pw, ph) in person_boxes:
            # Check center point
            if px <= bcx <= px + pw and py <= bcy <= py + ph:
                return True

            # Check IoU / overlap
            xi1 = max(bx, px)
            yi1 = max(by, py)
            xi2 = min(bx + bw, px + pw)
            yi2 = min(by + bh, py + ph)
            iw = max(0, xi2 - xi1)
            ih = max(0, yi2 - yi1)
            inter = iw * ih
            box_area = bw * bh
            if box_area > 0 and (inter / box_area) > 0.30:
                return True

        return False

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        person_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        run_detection: bool = True,
    ) -> Tuple[np.ndarray, List[Dict], int]:
        """
        Detect ONLY license plates with high FPS and overlay clean bounding boxes.
        Rejects any detections overlapping with human/person bodies (e.g. ID cards, badges).
        Returns: (annotated_frame_bgr, plate_records, count)
        """
        if frame_bgr is None:
            return frame_bgr, [], 0

        annotated = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]

        if run_detection:
            detections: List[Dict] = []
            try:
                results = self.model.predict(
                    source=frame_bgr,
                    conf=self.conf_threshold,
                    device=self.device,
                    imgsz=288,
                    verbose=False,
                )

                if len(results) > 0:
                    res = results[0]

                    # 1. Check OBB results
                    if hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0:
                        obb_boxes = res.obb.xyxyxyxy.cpu().numpy()
                        confs = res.obb.conf.cpu().numpy()

                        for pts, conf in zip(obb_boxes, confs):
                            pts_int = pts.astype(np.int32)
                            min_x, min_y = int(np.min(pts_int[:, 0])), int(np.min(pts_int[:, 1]))
                            max_x, max_y = int(np.max(pts_int[:, 0])), int(np.max(pts_int[:, 1]))
                            obb_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

                            # Reject if on a human/person
                            if self._is_inside_person(obb_bbox, person_boxes):
                                continue

                            crop = self._rectify_obb_crop(frame_bgr, pts)
                            cache_key = f"obb_{pts_int[0][0] // 30}_{pts_int[0][1] // 30}"
                            plate_text = self.trigger_async_ocr(crop, cache_key) if crop is not None else ""

                            detections.append({
                                "text": plate_text,
                                "conf": float(conf),
                                "crop": crop,
                                "pts": pts_int,
                                "bbox": obb_bbox,
                                "is_obb": True,
                            })

                    # 2. Check Standard BBoxes
                    elif hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                        boxes = res.boxes.xyxy.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()
                        cls_ids = res.boxes.cls.cpu().numpy() if res.boxes.cls is not None else [0] * len(boxes)

                        for box, conf, cls_id in zip(boxes, confs, cls_ids):
                            cls_name = self.model.names.get(int(cls_id), "").lower() if self.model.names else ""
                            if self.model.names and len(self.model.names) > 1:
                                if "plate" not in cls_name and "license" not in cls_name:
                                    continue

                            x1, y1, x2, y2 = [int(v) for v in box]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            bw, bh = x2 - x1, y2 - y1

                            # Reject if on a human/person
                            if self._is_inside_person((x1, y1, bw, bh), person_boxes):
                                continue

                            if bw > 25 and bh > 12 and (bw / max(1, bh) >= 1.1):
                                crop = frame_bgr[y1:y2, x1:x2]
                                cache_key = f"{x1 // 30}_{y1 // 30}_{bw // 20}_{bh // 20}"
                                plate_text = self.trigger_async_ocr(crop, cache_key)

                                detections.append({
                                    "text": plate_text,
                                    "conf": float(conf),
                                    "bbox": (x1, y1, bw, bh),
                                    "crop": crop,
                                    "is_obb": False,
                                    "cache_key": cache_key,
                                })

                self.cached_detections = detections
            except Exception:
                pass

        # Filter cached detections if person_boxes updated
        filtered_dets = []
        for det in self.cached_detections:
            bbox = det.get("bbox")
            if bbox and self._is_inside_person(bbox, person_boxes):
                continue
            filtered_dets.append(det)

        # Render cached detections onto annotated frame
        for det in filtered_dets:
            if det.get("is_obb", False):
                pts_int = det.get("pts")
                if pts_int is not None:
                    cv2.polylines(annotated, [pts_int], isClosed=True, color=(0, 255, 140), thickness=2)
                    top_pt = pts_int[np.argmin(pts_int[:, 1])]
                    tx, ty = int(top_pt[0]), max(20, int(top_pt[1]) - 8)
                    plate_text = det.get("text", "")
                    label = f"🚗 {plate_text}" if plate_text else "🚗 NUMBER PLATE"
                    (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)

                    cv2.rectangle(annotated, (tx, ty - th_t - 4), (tx + tw + 8, ty + 2), (16, 20, 28), -1)
                    cv2.rectangle(annotated, (tx, ty - th_t - 4), (tx + tw + 8, ty + 2), (0, 255, 140), 1)
                    cv2.putText(annotated, label, (tx + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                bbox = det.get("bbox")
                if bbox:
                    x1, y1, bw, bh = bbox
                    x2, y2 = x1 + bw, y1 + bh
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 140), 2)

                    # Update plate text if async OCR finished
                    cache_k = det.get("cache_key", "")
                    if cache_k in self.plate_text_cache:
                        det["text"] = self.plate_text_cache[cache_k]

                    plate_text = det.get("text", "")
                    label = f"🚗 {plate_text}" if plate_text else "🚗 NUMBER PLATE"
                    (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
                    ly = max(th_t + 6, y1 - 6)

                    cv2.rectangle(annotated, (x1, ly - th_t - 4), (x1 + tw + 8, ly + 2), (16, 20, 28), -1)
                    cv2.rectangle(annotated, (x1, ly - th_t - 4), (x1 + tw + 8, ly + 2), (0, 255, 140), 1)
                    cv2.putText(annotated, label, (x1 + 4, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated, filtered_dets, len(filtered_dets)
