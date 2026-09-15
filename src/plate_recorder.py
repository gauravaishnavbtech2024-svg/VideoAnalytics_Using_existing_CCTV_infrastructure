"""
Plate Recorder Module for Automatic Number Plate Recognition (ANPR).
====================================================================

Logs recognized license plates into structured CSV records and saves
corresponding image crops with timestamps.

Features:
- Fuzzy String Similarity Matching (prevents re-clicking the same plate with slight OCR variations).
- Single-Capture Guarantee: Logs and snapshots each vehicle/number plate exactly once.
- HSRP Prefix Sanitization (strips 'IND', 'INDIA' blue hologram noise).
- Alphanumeric Structure Validation (rejects non-plate noise like pure numbers).
"""

from __future__ import annotations

import csv
import difflib
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS_DIR = PROJECT_ROOT / "records" / "number_plates"
DEFAULT_CROPS_DIR = DEFAULT_RECORDS_DIR / "crops"
DEFAULT_CSV_PATH = DEFAULT_RECORDS_DIR / "plate_records.csv"


class PlateRecorder:
    """Manages recording and deduplication of detected number plates."""

    def __init__(
        self,
        records_dir: Path | str = DEFAULT_RECORDS_DIR,
        cooldown_seconds: float = 45.0,
        similarity_threshold: float = 0.50,
    ) -> None:
        self.records_dir = Path(records_dir)
        self.crops_dir = self.records_dir / "crops"
        self.csv_path = self.records_dir / "plate_records.csv"
        self.cooldown_seconds = cooldown_seconds
        self.similarity_threshold = similarity_threshold

        # Memory cache: list of (plate_text, logged_timestamp)
        self.recent_logs: List[Tuple[str, float]] = []

        self._init_storage()

    def _init_storage(self) -> None:
        """Create folders and CSV file with headers if not existing."""
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp",
                    "Date",
                    "Time",
                    "PlateNumber",
                    "Camera",
                    "Confidence(%)",
                    "CropImagePath",
                ])

    def clean_plate_text(self, text: str) -> str:
        """Sanitize raw OCR output into clean uppercase alphanumeric license plate."""
        if not text:
            return ""

        # Keep only alphanumeric characters
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()

        # Strip common HSRP country code prefixes (from blue hologram stripe)
        for prefix in ["INDIA", "IND", "IN0", "IN"]:
            if cleaned.startswith(prefix) and len(cleaned) > len(prefix) + 4:
                cleaned = cleaned[len(prefix):]
                break

        return cleaned

    def is_valid_plate(self, cleaned_text: str) -> bool:
        """Check if string matches genuine vehicle license plate structure and reject person/UI noise."""
        if len(cleaned_text) < 5 or len(cleaned_text) > 13:
            return False

        # Reject person / UI / system noise words
        rejected_words = [
            "PERSON", "UNKNOWN", "USER", "STUDENT", "STAFF", "EMPLOYEE",
            "CAMERA", "FEED", "HUMAN", "ALERT", "HEALTH", "CYBER",
            "CARD", "BADGE", "IDENT", "NAME", "OFFICE", "COLLEGE", "SCHOOL"
        ]
        for r in rejected_words:
            if r in cleaned_text:
                return False

        num_letters = len(re.findall(r"[A-Z]", cleaned_text))
        num_digits = len(re.findall(r"[0-9]", cleaned_text))

        # Real vehicle plates must contain at least 2 letters (State/Region) and at least 2 digits (Series/Number)
        if num_letters < 2 or num_digits < 2:
            return False

        return True

    def _is_duplicate(self, candidate_text: str, current_time: float) -> bool:
        """Check if candidate matches a recently logged plate via fuzzy similarity."""
        # Purge entries older than cooldown window
        self.recent_logs = [
            (text, ts) for (text, ts) in self.recent_logs
            if (current_time - ts) < self.cooldown_seconds
        ]

        for prev_text, _ in self.recent_logs:
            # Exact match
            if candidate_text == prev_text:
                return True

            # Fuzzy similarity ratio match (handles slight OCR character shifts across frames)
            sim = difflib.SequenceMatcher(None, candidate_text, prev_text).ratio()
            if sim >= self.similarity_threshold:
                return True

        return False

    def log_plate(
        self,
        plate_text: str,
        confidence: float,
        camera_name: str,
        plate_crop_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str]:
        """
        Record plate detection into CSV and save image crop EXACTLY ONCE per vehicle.

        Returns: (is_newly_recorded, cleaned_plate_text)
        """
        cleaned_text = self.clean_plate_text(plate_text)
        if not self.is_valid_plate(cleaned_text):
            return False, cleaned_text

        now_time = time.time()

        # Deduplication check: if plate (or fuzzy match) was logged recently, skip!
        if self._is_duplicate(cleaned_text, now_time):
            return False, cleaned_text

        # Record new plate
        self.recent_logs.append((cleaned_text, now_time))

        dt = datetime.now()
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M:%S")
        file_ts = dt.strftime("%Y%m%d_%H%M%S")

        # Save cropped plate image ONCE
        crop_rel_path = ""
        if plate_crop_bgr is not None and plate_crop_bgr.size > 0:
            crop_filename = f"{cleaned_text}_{file_ts}.jpg"
            crop_full_path = self.crops_dir / crop_filename
            try:
                cv2.imwrite(str(crop_full_path), plate_crop_bgr)
                crop_rel_path = f"crops/{crop_filename}"
            except Exception as e:
                print(f"[PlateRecorder] Error saving crop: {e}")

        # Append record to CSV
        try:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp_str,
                    date_str,
                    time_str,
                    cleaned_text,
                    camera_name,
                    f"{confidence * 100:.1f}",
                    crop_rel_path,
                ])
            print(f"  [ANPR Captured ONCE] -> PLATE: {cleaned_text} | TIME: {time_str} | CAM: {camera_name}")
            return True, cleaned_text
        except Exception as e:
            print(f"[PlateRecorder] CSV write error: {e}")
            return False, cleaned_text
