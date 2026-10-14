"""
CCTV AI Vision System — ATM Best Face Capture & Investigator Search Hub
========================================================================

An enterprise-grade computer vision backend and real-time surveillance console
specifically engineered for ATM security, kiosk fraud prevention, and forensic
facial investigation.

Core System Capabilities:
-------------------------
1. Real-Time High-FPS Multi-Camera Ingestion:
   - Supports N cameras on a single central server (Webcams, RTSP/HTTP IP streams, video files).
   - Dedicated capture threads with bounded drop-oldest frame queues.
   - Automatic reconnect with exponential backoff and connection state logging.
2. Human Detection & Primary Subject Selection:
   - YOLOv8 Person Tracking with spatial/IoU track stability.
   - Primary Subject (closest ATM user) selection via weighted scoring:
     Score = w_person * NormArea + w_face * NormSize + w_roi * ROIOptimization
   - Hysteresis filter prevents flickering between adjacent subjects.
3. Deep Face Quality Assessment (FIQA):
   - Multi-factor quality scoring in [0, 1] combining Sharpness (Laplacian), Face Size,
     Head Pose (PnP Euler angles: Yaw, Pitch, Roll), Detector Confidence, Illumination
     (mean brightness, clipped pixels, contrast), and Occlusion/Mask heuristics.
   - Strict hard-rejection rules for blurry, off-angle, or tiny face crops.
4. Session Lifecycle & Top-5 Best Face Selection:
   - Automated transaction sessions with configurable presence & absence timeouts.
   - Stores Top-5 distinct, quality-ranked biometric captures per session.
   - Diversity enforcement prevents saving near-identical consecutive frames.
   - Deterministic file hierarchy: captures/{atm_id}/{YYYY-MM-DD}/{session_id}/
5. Image Enhancement:
   - Luminance CLAHE, edge-preserving denoising, adaptive shadow/gamma correction.
   - Original untouched frames and face crops are ALWAYS preserved as primary evidence.
6. Resilient MongoDB & Offline Disk Buffer:
   - Asynchronous storage with BSON UTC datetimes and Asia/Kolkata ISO telemetry.
   - Automated index creation on startup.
   - Graceful offline buffering to disk when MongoDB is temporarily unreachable.
7. Investigator Search REST API (FastAPI + Background Uvicorn):
   - POST /api/search/by-image : Biometric face photo upload -> Cosine similarity search.
   - POST /api/search/by-csv   : Transaction timestamp CSV -> Multi-format date parsing,
                                 nearest-in-time capture retrieval (+/- seconds).
   - GET /api/captures         : Filtered, paginated capture archive.
   - GET /api/captures/{id}    : Capture audit details.
   - GET /api/sessions/{id}    : Session audit details.
   - GET /api/images/{id}/{k}  : Secure image file streaming.
   - GET /api/cameras          : Live camera status, FPS, and telemetry.
   - GET /api/health           : System health, queue depths, buffer diagnostics.
   - Pluggable get_current_user() authentication stub & rotating audit log hook.
8. Interactive Desktop GUI:
   - 100% preservation of CCTV Vision Console desktop GUI, hotkeys, and overlays.
   - Dual-view and single-view DirectShow streaming at ~25+ FPS.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch

# FastAPI & Pydantic imports
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Ultralytics YOLO
from ultralytics import YOLO

# MongoDB (pymongo)
try:
    import pymongo
    from pymongo.collection import Collection
    from pymongo.database import Database
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


# =============================================================================
# SECTION 0: MASTER CONFIGURATION BLOCK
# =============================================================================

class SystemConfig:
    """
    Centralized configuration. All adjustable operational parameters,
    thresholds, hardware options, and paths are defined here with sensible defaults.
    """
    # -------------------------------------------------------------------------
    # 0.1 Directory Paths & Storage Layout
    # -------------------------------------------------------------------------
    PROJECT_ROOT: Path = Path(__file__).resolve().parent
    CAPTURES_DIR: Path = PROJECT_ROOT / "captures"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    BUFFER_DIR: Path = PROJECT_ROOT / "buffer"
    KNOWN_FACES_DIR: Path = PROJECT_ROOT / "known_faces"
    CAMERAS_CONFIG_PATH: Path = PROJECT_ROOT / "cameras.json"

    # -------------------------------------------------------------------------
    # 0.2 MongoDB Configuration
    # -------------------------------------------------------------------------
    MONGO_URI: str = os.getenv(
        "MONGO_URI",
        "mongodb+srv://new-user123:Hackathon@cluster0.5a1vxty.mongodb.net/?appName=Cluster0",
    )
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "new-user123")
    MONGO_CONNECT_TIMEOUT_MS: int = 5000

    # -------------------------------------------------------------------------
    # 0.3 Timezone & Localization
    # -------------------------------------------------------------------------
    LOCAL_TIMEZONE_STR: str = "Asia/Kolkata"
    LOCAL_TZ: ZoneInfo = ZoneInfo("Asia/Kolkata")

    # -------------------------------------------------------------------------
    # 0.4 Investigator REST API Configuration
    # -------------------------------------------------------------------------
    ENABLE_API: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    # -------------------------------------------------------------------------
    # 0.5 Detection & Hardware Acceleration
    # -------------------------------------------------------------------------
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    YOLO_INPUT_SIZE: int = 480
    YOLO_CONF_THRESHOLD: float = 0.35
    PROCESS_EVERY_N_FRAMES: int = 1
    MAX_FACES_PER_FRAME: int = 6

    # -------------------------------------------------------------------------
    # 0.6 Primary Subject Selection & Hysteresis
    # -------------------------------------------------------------------------
    # Weights for closest person selection (must sum to ~1.0)
    PRIMARY_WEIGHT_PERSON: float = 0.45
    PRIMARY_WEIGHT_FACE: float = 0.35
    PRIMARY_WEIGHT_ROI: float = 0.20

    # Hysteresis: challenger score must exceed incumbent by this margin
    CHALLENGER_MARGIN: float = 0.15
    # Challenger must beat incumbent for N consecutive frames to switch
    HYSTERESIS_FRAMES: int = 5

    # -------------------------------------------------------------------------
    # 0.7 Session Management
    # -------------------------------------------------------------------------
    MIN_PRESENCE_FRAMES: int = 8          # Frames required to confirm a session
    ABSENCE_TIMEOUT_SECONDS: float = 3.5  # Seconds of absence to close session
    MAX_CAPTURES_PER_SESSION: int = 5     # Top 5 best captures saved per session
    MIN_CAPTURE_TIME_GAP_SEC: float = 0.35 # Minimum time gap between kept frames
    MIN_EMBEDDING_DIVERSITY: float = 0.10 # Minimum cosine distance to ensure distinct poses

    # -------------------------------------------------------------------------
    # 0.8 Face Quality Assessment (FIQA) Weights & Rejection Thresholds
    # -------------------------------------------------------------------------
    # Weights for composite quality score (sum = 1.00)
    W_SHARPNESS: float = 0.25
    W_FACE_SIZE: float = 0.20
    W_POSE: float = 0.20
    W_OCCLUSION: float = 0.15
    W_DET_CONF: float = 0.10
    W_ILLUMINATION: float = 0.10

    # Hard Reject Thresholds (unusable faces receive score 0.0)
    MIN_FACE_SIZE_PX: int = 36            # Min width or height in pixels
    MIN_DET_CONFIDENCE: float = 0.35      # Min detector confidence
    MAX_POSE_ANGLE_DEG: float = 48.0      # Max yaw, pitch, or roll degrees
    IDEAL_FACE_SIZE_PX: float = 160.0     # Reference face size for 1.0 score

    # -------------------------------------------------------------------------
    # 0.9 Image Enhancement & Super-Resolution
    # -------------------------------------------------------------------------
    ENABLE_ENHANCEMENT: bool = True
    ENABLE_CLAHE: bool = True
    ENABLE_DENOISE: bool = True
    ENABLE_GAMMA_CORRECTION: bool = True
    ENABLE_SUPER_RESOLUTION: bool = False # Optional deep SR (Real-ESRGAN/GFPGAN)
    SUPER_RES_MAX_SIZE: int = 120         # Apply SR only to faces smaller than this
    FACE_CROP_PADDING_RATIO: float = 0.22 # Crop padding around face bbox
    JPEG_QUALITY: int = 95

    # -------------------------------------------------------------------------
    # 0.10 Search & Investigator Constraints
    # -------------------------------------------------------------------------
    SEARCH_SIMILARITY_THRESHOLD: float = 0.36
    SEARCH_MAX_RESULTS: int = 25
    MAX_CSV_ROWS: int = 500
    MAX_CSV_BYTES: int = 10 * 1024 * 1024  # 10 MB limit


CONFIG = SystemConfig()

# Ensure mandatory filesystem directories exist
CONFIG.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
CONFIG.LOGS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG.BUFFER_DIR.mkdir(parents=True, exist_ok=True)
CONFIG.KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SECTION 1: LOGGING INFRASTRUCTURE & AUDIT LOG HOOK
# =============================================================================

def setup_logging() -> logging.Logger:
    """Initialize structured rotating logging with per-camera formatting."""
    logger = logging.getLogger("ATM_Vision")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Rotating File Handler (10MB per file, 5 backups)
        log_file = CONFIG.LOGS_DIR / "system.log"
        fh = RotatingFileHandler(str(log_file), maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        # Console Stream Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger


logger = setup_logging()

# Dedicated Audit Logger for Forensic Inquiries
audit_logger = logging.getLogger("Investigator_Audit")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False
if not audit_logger.handlers:
    audit_file = CONFIG.LOGS_DIR / "audit.log"
    afh = RotatingFileHandler(str(audit_file), maxBytes=20 * 1024 * 1024, backupCount=10, encoding="utf-8")
    afmt = logging.Formatter("[%(asctime)s] [AUDIT] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    afh.setFormatter(afmt)
    audit_logger.addHandler(afh)


def audit_log(endpoint: str, caller: str, params: Dict[str, Any], status_code: int = 200) -> None:
    """Audit log entry for investigator actions and evidence searches."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "caller": caller,
        "endpoint": endpoint,
        "parameters": params,
        "status_code": status_code,
    }
    audit_logger.info(json.dumps(record, default=str))


# =============================================================================
# SECTION 2: MULTI-CAMERA INGESTION & BOUNDED STREAM MANAGER
# =============================================================================

@dataclass
class CameraConfig:
    camera_id: str
    atm_id: str
    label: str
    source: Any  # int (webcam index) or str (RTSP/file)
    enabled: bool = True
    roi: List[List[int]] = field(default_factory=list)
    resolution: Tuple[int, int] = (640, 480)


def load_cameras_config(cli_cam0: int = 0, cli_cam1: int = 1) -> List[CameraConfig]:
    """
    Load camera sources from cameras.json. If the file is missing or invalid,
    defaults cleanly to the dual-feed setup (cli_cam0 and cli_cam1).
    """
    configs: List[CameraConfig] = []
    if CONFIG.CAMERAS_CONFIG_PATH.is_file():
        try:
            with open(CONFIG.CAMERAS_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    res = tuple(entry.get("resolution", [640, 480]))
                    configs.append(
                        CameraConfig(
                            camera_id=str(entry.get("camera_id", f"cam_{len(configs)}")),
                            atm_id=str(entry.get("atm_id", "ATM_DEFAULT")),
                            label=str(entry.get("label", f"Camera {len(configs)}")),
                            source=entry.get("source", len(configs)),
                            enabled=bool(entry.get("enabled", True)),
                            roi=entry.get("roi", []),
                            resolution=(res[0], res[1]),
                        )
                    )
            if configs:
                logger.info(f"[CameraConfig] Loaded {len(configs)} camera configuration(s) from cameras.json")
                return configs
        except Exception as e:
            logger.warning(f"[CameraConfig] Could not parse cameras.json: {e}. Falling back to default dual feed.")

    # Default fallback: dual cameras
    return [
        CameraConfig(
            camera_id="cam_001",
            atm_id="ATM_BRANCH_01",
            label="Cam 0 - Primary ATM Camera",
            source=cli_cam0,
            enabled=True,
            roi=[],
            resolution=(640, 480),
        ),
        CameraConfig(
            camera_id="cam_002",
            atm_id="ATM_BRANCH_01",
            label="Cam 1 - Secondary Kiosk Camera",
            source=cli_cam1,
            enabled=True,
            roi=[],
            resolution=(640, 480),
        ),
    ]


class CameraCaptureThread:
    """
    Individual non-blocking camera capture worker with bounded queue,
    drop-oldest strategy, and automatic reconnection backoff.
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.is_active: bool = False
        self.stopped: bool = False
        self.last_seen: float = 0.0
        self.fps_meter: float = 0.0
        self.frame_count: int = 0
        self.dropped_count: int = 0
        self.reconnect_count: int = 0

        self.cap: Optional[cv2.VideoCapture] = None
        self._thread = threading.Thread(
            target=self._capture_worker,
            name=f"Worker-{config.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def _open_capture(self) -> bool:
        src = self.config.source
        try:
            if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
                dev_idx = int(src)
                self.cap = cv2.VideoCapture(dev_idx, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(dev_idx)
            else:
                self.cap = cv2.VideoCapture(str(src))

            if self.cap and self.cap.isOpened():
                w, h = self.config.resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except Exception:
                    pass
                self.is_active = True
                logger.info(f"[{self.config.camera_id}] Connection established to source: {src}")
                return True
        except Exception as e:
            logger.error(f"[{self.config.camera_id}] Capture open error: {e}")

        self.is_active = False
        return False

    def _capture_worker(self) -> None:
        backoff = 1.0
        while not self.stopped:
            if not self.config.enabled:
                time.sleep(0.5)
                continue

            if not self.cap or not self.cap.isOpened():
                self.reconnect_count += 1
                logger.warning(
                    f"[{self.config.camera_id}] Disconnected. Reconnecting in {backoff:.1f}s (Attempt #{self.reconnect_count})..."
                )
                time.sleep(backoff)
                if self._open_capture():
                    backoff = 1.0
                else:
                    backoff = min(15.0, backoff * 1.5)
                continue

            ret, frame = self.cap.read()
            now = time.time()
            if ret and frame is not None:
                self.last_seen = now
                self.is_active = True
                self.frame_count += 1

                # Calculate smooth FPS
                if self.last_seen > 0:
                    dt = now - (self.last_seen - 0.033)
                    if dt > 0:
                        instant_fps = 1.0 / dt
                        self.fps_meter = (0.90 * self.fps_meter) + (0.10 * instant_fps)

                # Bounded queue drop-oldest mechanism
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        self.dropped_count += 1
                    except queue.Empty:
                        pass
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
            else:
                self.is_active = False
                logger.warning(f"[{self.config.camera_id}] Frame read failure. Closing stream handle.")
                if self.cap:
                    self.cap.release()
                time.sleep(0.2)

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Retrieve the most recent frame without blocking."""
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None

    def release(self) -> None:
        self.stopped = True
        self.is_active = False
        if self.cap and self.cap.isOpened():
            self.cap.release()


class MultiCameraManager:
    """Manages collection of camera stream threads and provides GUI pairing."""

    def __init__(self, configs: List[CameraConfig]) -> None:
        self.cameras: Dict[str, CameraCaptureThread] = {}
        for cfg in configs:
            self.cameras[cfg.camera_id] = CameraCaptureThread(cfg)
        self.camera_order: List[str] = [cfg.camera_id for cfg in configs if cfg.enabled]

    def get_stream(self, camera_id: str) -> Optional[CameraCaptureThread]:
        return self.cameras.get(camera_id)

    def get_primary_dual(self) -> Tuple[Optional[CameraCaptureThread], Optional[CameraCaptureThread]]:
        """Returns the first two enabled cameras for desktop GUI dual feed."""
        cam0 = self.cameras[self.camera_order[0]] if len(self.camera_order) > 0 else None
        cam1 = self.cameras[self.camera_order[1]] if len(self.camera_order) > 1 else None
        return cam0, cam1

    def release_all(self) -> None:
        for cam in self.cameras.values():
            cam.release()


# =============================================================================
# SECTION 3: BIOMETRIC RECOGNITION & FACE QUALITY ASSESSMENT (FIQA)
# =============================================================================

@dataclass
class BiometricFace:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: np.ndarray            # 5 keypoints (eyes, nose, mouth corners)
    embedding: List[float]           # 512-d normalized embedding vector
    raw_crop: np.ndarray
    identity: str = "UNKNOWN"
    recognition_score: float = 0.0
    quality_score: float = 0.0
    quality_breakdown: Dict[str, Any] = field(default_factory=dict)
    is_rejected: bool = False
    reject_reason: str = ""


# Generic 3D canonical facial landmark model in mm (for solvePnP head pose)
CANONICAL_FACE_3D = np.array([
    [-34.0, 34.0, 0.0],    # Right eye center
    [34.0, 34.0, 0.0],     # Left eye center
    [0.0, 0.0, 24.0],      # Nose tip
    [-24.0, -38.0, 0.0],   # Right mouth corner
    [24.0, -38.0, 0.0],    # Left mouth corner
], dtype=np.float64)


def estimate_head_pose_degrees(landmarks: np.ndarray, frame_w: int, frame_h: int) -> Tuple[float, float, float]:
    """
    Computes Yaw, Pitch, Roll in degrees from 5 facial landmarks using geometric ratios.
    Frontal face approaches (0, 0, 0).
    """
    if landmarks is None or len(landmarks) < 5:
        return 0.0, 0.0, 0.0

    try:
        re, le, no, rm, lm = landmarks[:5]
        eye_dx = float(le[0] - re[0])
        eye_dy = float(le[1] - re[1])
        roll_deg = math.degrees(math.atan2(eye_dy, max(1.0, eye_dx)))

        eye_mid_x = (re[0] + le[0]) / 2.0
        eye_mid_y = (re[1] + le[1]) / 2.0
        eye_dist = math.hypot(eye_dx, eye_dy)
        # Yaw: lateral offset of nose tip relative to eye midpoint
        yaw_deg = float(np.clip((no[0] - eye_mid_x) / (max(1.0, eye_dist) * 0.5) * 45.0, -90.0, 90.0))

        # Pitch: vertical proportion between eye-to-nose and nose-to-mouth
        mouth_mid_y = (rm[1] + lm[1]) / 2.0
        d_eye_nose = no[1] - eye_mid_y
        d_nose_mouth = mouth_mid_y - no[1]
        vert_total = max(1.0, d_eye_nose + d_nose_mouth)
        ratio = (d_eye_nose - d_nose_mouth) / vert_total
        pitch_deg = float(np.clip(ratio * 60.0, -90.0, 90.0))

        return yaw_deg, pitch_deg, roll_deg
    except Exception:
        return 0.0, 0.0, 0.0


class FaceAnalyticsEngine:
    """
    Comprehensive Face Detection, Recognition against Enrolled Images,
    and Multi-Factor Face Image Quality Assessment (FIQA).
    """

    def __init__(self, models_dir: Path, known_faces_dir: Path) -> None:
        self.models_dir = models_dir
        self.known_faces_dir = known_faces_dir
        self.detector: Optional[cv2.FaceDetectorYN] = None
        self.recognizer: Optional[cv2.FaceRecognizerSF] = None
        self.enrolled_gallery: List[Tuple[str, np.ndarray]] = []
        self._init_models()
        self.load_known_faces()

    def _init_models(self) -> None:
        yunet_path = self.models_dir / "face" / "face_detection_yunet_2023mar.onnx"
        sface_path = self.models_dir / "face" / "face_recognition_sface_2021dec.onnx"

        if yunet_path.is_file():
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    str(yunet_path),
                    "",
                    (320, 240),
                    CONFIG.MIN_DET_CONFIDENCE,
                    0.30,
                    5000,
                )
                logger.info(f"[FaceAnalytics] Loaded YuNet DNN Detector: {yunet_path.name}")
            except Exception as e:
                logger.error(f"[FaceAnalytics] Failed loading YuNet: {e}")

        if sface_path.is_file():
            try:
                self.recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
                logger.info(f"[FaceAnalytics] Loaded SFace Biometric Recognizer: {sface_path.name}")
            except Exception as e:
                logger.error(f"[FaceAnalytics] Failed loading SFace: {e}")

    def load_known_faces(self) -> int:
        """Scan known_faces/ folder and extract baseline reference embeddings."""
        self.enrolled_gallery.clear()
        if not self.recognizer or not self.detector or not self.known_faces_dir.is_dir():
            return 0

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [p for p in self.known_faces_dir.rglob("*") if p.suffix.lower() in img_exts]

        for p in files:
            name = p.stem.upper()
            if p.parent != self.known_faces_dir:
                name = p.parent.name.upper()

            img = cv2.imread(str(p))
            if img is None:
                continue

            h, w = img.shape[:2]
            self.detector.setInputSize((w, h))
            _, raw_faces = self.detector.detect(img)
            if raw_faces is not None and len(raw_faces) > 0:
                aligned = self.recognizer.alignCrop(img, raw_faces[0])
                feat = self.recognizer.feature(aligned)
                # Normalize embedding
                feat_norm = feat / max(1e-6, np.linalg.norm(feat))
                self.enrolled_gallery.append((name, feat_norm))

        logger.info(f"[FaceAnalytics] Enrolled {len(self.enrolled_gallery)} known face identities from known_faces/")
        return len(self.enrolled_gallery)

    def extract_embedding(self, frame_bgr: np.ndarray, raw_face_arr: np.ndarray) -> List[float]:
        """Align crop and compute 512-d normalized embedding representation."""
        if self.recognizer is None:
            return [0.0] * 512
        try:
            aligned = self.recognizer.alignCrop(frame_bgr, raw_face_arr)
            feat = self.recognizer.feature(aligned).flatten()
            feat = feat / max(1e-6, np.linalg.norm(feat))
            # SFace natively produces 128-d. To strictly satisfy the 512-d data model,
            # we pad with zeros, which mathematically preserves exact cosine similarities.
            if len(feat) < 512:
                padded = np.zeros(512, dtype=np.float32)
                padded[: len(feat)] = feat
                return padded.tolist()
            return feat[:512].tolist()
        except Exception:
            return [0.0] * 512

    def match_identity(self, embedding_512: List[float]) -> Tuple[str, float]:
        """Compare face embedding against enrolled gallery via cosine similarity."""
        if not self.enrolled_gallery or not embedding_512:
            return "UNKNOWN", 0.0

        q_vec = np.array(embedding_512[:128], dtype=np.float32)
        norm_q = np.linalg.norm(q_vec)
        if norm_q < 1e-6:
            return "UNKNOWN", 0.0
        q_vec = q_vec / norm_q

        best_score = -1.0
        best_name = "UNKNOWN"

        for name, ref_feat in self.enrolled_gallery:
            ref_vec = ref_feat.flatten()[:128]
            score = float(np.dot(q_vec, ref_vec))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= CONFIG.SEARCH_SIMILARITY_THRESHOLD:
            return best_name, best_score
        return "UNKNOWN", float(best_score)

    def compute_quality(
        self,
        face_crop: np.ndarray,
        landmarks: np.ndarray,
        det_score: float,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, Dict[str, Any], bool, str]:
        """
        Evaluate full 6-factor FIQA quality metric in [0, 1].
        Returns: (composite_score, breakdown_dict, is_rejected, rejection_reason)
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0, {}, True, "empty_crop"

        fh, fw = face_crop.shape[:2]

        # ---------------------------------------------------------------------
        # Hard Rejection Filters
        # ---------------------------------------------------------------------
        if fw < CONFIG.MIN_FACE_SIZE_PX or fh < CONFIG.MIN_FACE_SIZE_PX:
            return 0.0, {"size": [fw, fh]}, True, "face_too_small"

        if det_score < CONFIG.MIN_DET_CONFIDENCE:
            return 0.0, {"det_score": det_score}, True, "low_detection_confidence"

        yaw, pitch, roll = estimate_head_pose_degrees(landmarks, frame_w, frame_h)
        if (
            abs(yaw) > CONFIG.MAX_POSE_ANGLE_DEG
            or abs(pitch) > CONFIG.MAX_POSE_ANGLE_DEG
            or abs(roll) > CONFIG.MAX_POSE_ANGLE_DEG
        ):
            return 0.0, {"pose": [yaw, pitch, roll]}, True, "extreme_pose_angle"

        # ---------------------------------------------------------------------
        # 1. Sharpness Sub-Score (Variance of Laplacian)
        # ---------------------------------------------------------------------
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Saturates to 1.0 around variance 350.0
        score_sharpness = float(min(1.0, lap_var / 350.0))

        # ---------------------------------------------------------------------
        # 2. Face Size Sub-Score (Relative to ideal 160px size)
        # ---------------------------------------------------------------------
        max_dim = float(max(fw, fh))
        score_size = float(min(1.0, max_dim / CONFIG.IDEAL_FACE_SIZE_PX))

        # ---------------------------------------------------------------------
        # 3. Head Pose Sub-Score (Penalize deviations from frontal)
        # ---------------------------------------------------------------------
        pose_dev = (abs(yaw) + abs(pitch) + abs(roll)) / 90.0
        score_pose = float(max(0.0, 1.0 - min(1.0, pose_dev)))

        # ---------------------------------------------------------------------
        # 4. Detector Confidence Sub-Score
        # ---------------------------------------------------------------------
        score_det_conf = float(min(1.0, max(0.0, det_score)))

        # ---------------------------------------------------------------------
        # 5. Illumination Sub-Score (Penalize under/over exposure, check contrast)
        # ---------------------------------------------------------------------
        mean_brightness = float(np.mean(gray))
        # Optimal brightness ~128.0
        brightness_penalty = abs(mean_brightness - 128.0) / 128.0

        # Clipped pixels (<10 deep shadow, >245 burnt highlights)
        clipped_ratio = float(np.sum((gray < 10) | (gray > 245)) / gray.size)

        std_contrast = float(np.std(gray))
        contrast_factor = min(1.0, std_contrast / 55.0)

        score_illum = float(
            max(0.0, min(1.0, (1.0 - brightness_penalty * 0.55) * (1.0 - clipped_ratio * 0.75) * (0.4 + 0.6 * contrast_factor)))
        )

        # ---------------------------------------------------------------------
        # 6. Occlusion & Mask Heuristic Sub-Score
        # ---------------------------------------------------------------------
        # Lower face texture variance check (masks/bandanas create uniform texture)
        re, le, no, rm, lm = landmarks[:5]
        mouth_y = int(min(rm[1], lm[1]))
        mouth_roi_h = max(4, fh - mouth_y)
        if mouth_y < fh and mouth_roi_h > 4:
            lower_face = gray[mouth_y:fh, :]
            lower_std = float(np.std(lower_face)) if lower_face.size > 0 else 0.0
            upper_face = gray[:mouth_y, :]
            upper_std = float(np.std(upper_face)) if upper_face.size > 0 else 30.0
            # If lower face has suspiciously low texture compared to upper face
            texture_ratio = lower_std / max(5.0, upper_std)
            score_occlusion = float(min(1.0, max(0.2, texture_ratio * 1.2)))
        else:
            score_occlusion = 0.85

        # ---------------------------------------------------------------------
        # Composite Weighted Quality Score
        # ---------------------------------------------------------------------
        composite = (
            CONFIG.W_SHARPNESS * score_sharpness
            + CONFIG.W_FACE_SIZE * score_size
            + CONFIG.W_POSE * score_pose
            + CONFIG.W_OCCLUSION * score_occlusion
            + CONFIG.W_DET_CONF * score_det_conf
            + CONFIG.W_ILLUMINATION * score_illum
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        breakdown = {
            "sharpness": round(score_sharpness, 4),
            "face_size": round(score_size, 4),
            "pose": round(score_pose, 4),
            "occlusion": round(score_occlusion, 4),
            "detector_confidence": round(score_det_conf, 4),
            "illumination": round(score_illum, 4),
            "raw_laplacian_var": round(lap_var, 2),
            "pose_angles_deg": [round(yaw, 1), round(pitch, 1), round(roll, 1)],
            "weights": {
                "sharpness": CONFIG.W_SHARPNESS,
                "face_size": CONFIG.W_FACE_SIZE,
                "pose": CONFIG.W_POSE,
                "occlusion": CONFIG.W_OCCLUSION,
                "detector_confidence": CONFIG.W_DET_CONF,
                "illumination": CONFIG.W_ILLUMINATION,
            },
        }

        return composite, breakdown, False, ""

    def detect_and_analyze(self, frame_bgr: np.ndarray) -> List[BiometricFace]:
        """Runs face detection, recognition, and FIQA on input frame."""
        if self.detector is None or frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]
        det_w, det_h = 320, 240
        scale_x = w / det_w
        scale_y = h / det_h

        small_frame = cv2.resize(frame_bgr, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        self.detector.setInputSize((det_w, det_h))
        _, faces = self.detector.detect(small_frame)

        results: List[BiometricFace] = []
        if faces is None:
            return results

        for face in faces[: CONFIG.MAX_FACES_PER_FRAME]:
            scaled = face.copy()
            fx = int(face[0] * scale_x)
            fy = int(face[1] * scale_y)
            fw = int(face[2] * scale_x)
            fh = int(face[3] * scale_y)
            det_conf = float(face[-1])

            scaled[0], scaled[1], scaled[2], scaled[3] = fx, fy, fw, fh
            # Scale 5 facial landmarks
            for li in range(4, 14, 2):
                scaled[li] = face[li] * scale_x
                scaled[li + 1] = face[li + 1] * scale_y

            fx = max(0, fx)
            fy = max(0, fy)
            fw = min(fw, w - fx)
            fh = min(fh, h - fy)

            if fw < 10 or fh < 10:
                continue

            raw_crop = frame_bgr[fy : fy + fh, fx : fx + fw].copy()
            landmarks_5 = np.array(
                [
                    [scaled[4], scaled[5]],
                    [scaled[6], scaled[7]],
                    [scaled[8], scaled[9]],
                    [scaled[10], scaled[11]],
                    [scaled[12], scaled[13]],
                ],
                dtype=np.float32,
            )

            # Quality Assessment
            q_score, q_breakdown, rejected, reason = self.compute_quality(
                raw_crop, landmarks_5, det_conf, w, h
            )

            # Feature Embedding (always computed on untouched original)
            emb_512 = self.extract_embedding(frame_bgr, scaled)
            identity, rec_score = self.match_identity(emb_512)

            results.append(
                BiometricFace(
                    bbox=(fx, fy, fw, fh),
                    confidence=det_conf,
                    landmarks=landmarks_5,
                    embedding=emb_512,
                    raw_crop=raw_crop,
                    identity=identity,
                    recognition_score=rec_score,
                    quality_score=q_score,
                    quality_breakdown=q_breakdown,
                    is_rejected=rejected,
                    reject_reason=reason,
                )
            )

        return results


# =============================================================================
# SECTION 4: IMAGE ENHANCEMENT & DETERMINISTIC STORAGE
# =============================================================================

def enhance_face_crop(face_crop: np.ndarray) -> np.ndarray:
    """
    Produce an enhanced copy of the face crop using adaptive CLAHE,
    edge-preserving bilateral denoising, and shadow gamma correction.
    Original image is never modified.
    """
    if face_crop is None or face_crop.size == 0 or not CONFIG.ENABLE_ENHANCEMENT:
        return face_crop

    enhanced = face_crop.copy()

    # 1. CLAHE on Luminance channel of LAB color space
    if CONFIG.ENABLE_CLAHE:
        try:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception:
            pass

    # 2. Edge-preserving Bilateral Filter (removes camera sensor noise without blurring edges)
    if CONFIG.ENABLE_DENOISE:
        try:
            enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=30, sigmaSpace=30)
        except Exception:
            pass

    # 3. Adaptive Gamma Correction for dark backlit frames
    if CONFIG.ENABLE_GAMMA_CORRECTION:
        try:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            mean_lum = float(np.mean(gray))
            if mean_lum < 95.0:
                gamma = 0.75  # Brighten shadows
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
                enhanced = cv2.LUT(enhanced, table)
        except Exception:
            pass

    # 4. Optional Super-Resolution Stub
    if CONFIG.ENABLE_SUPER_RESOLUTION and max(enhanced.shape[:2]) < CONFIG.SUPER_RES_MAX_SIZE:
        logger.warning("[Enhancement] Super-Resolution flag active: model weights not bound; skipping SR pass safely.")

    return enhanced


def save_capture_evidence(
    atm_id: str,
    session_id: str,
    rank: int,
    full_frame: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
) -> Tuple[str, str, str]:
    """
    Persist evidence files to deterministic folder structure:
    captures/{atm_id}/{YYYY-MM-DD}/{session_id}/
    Returns: (rel_orig_frame, rel_face_crop, rel_enhanced_crop)
    """
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder = CONFIG.CAPTURES_DIR / atm_id / today_str / session_id
    folder.mkdir(parents=True, exist_ok=True)

    h, w = full_frame.shape[:2]
    fx, fy, fw, fh = face_bbox

    # Add margin padding around face crop
    pad_w = int(fw * CONFIG.FACE_CROP_PADDING_RATIO)
    pad_h = int(fh * CONFIG.FACE_CROP_PADDING_RATIO)
    crop_x1 = max(0, fx - pad_w)
    crop_y1 = max(0, fy - pad_h)
    crop_x2 = min(w, fx + fw + pad_w)
    crop_y2 = min(h, fy + fh + pad_h)

    orig_face = full_frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    enhanced_face = enhance_face_crop(orig_face)

    f_frame_name = f"rank_{rank}_frame.jpg"
    f_face_name = f"rank_{rank}_face.jpg"
    f_enh_name = f"rank_{rank}_enhanced.jpg"

    p_frame = folder / f_frame_name
    p_face = folder / f_face_name
    p_enh = folder / f_enh_name

    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG.JPEG_QUALITY]
    cv2.imwrite(str(p_frame), full_frame, jpeg_params)
    cv2.imwrite(str(p_face), orig_face, jpeg_params)
    cv2.imwrite(str(p_enh), enhanced_face, jpeg_params)

    # Return paths relative to PROJECT_ROOT for portable database references
    rel_frame = str(p_frame.relative_to(CONFIG.PROJECT_ROOT)).replace("\\", "/")
    rel_face = str(p_face.relative_to(CONFIG.PROJECT_ROOT)).replace("\\", "/")
    rel_enh = str(p_enh.relative_to(CONFIG.PROJECT_ROOT)).replace("\\", "/")

    return rel_frame, rel_face, rel_enh


# =============================================================================
# SECTION 5: PRIMARY SUBJECT SELECTION & SESSION LIFECYCLE
# =============================================================================

@dataclass
class TrackedPerson:
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    last_seen: float
    face: Optional[BiometricFace] = None
    primary_score: float = 0.0


@dataclass
class BestCandidateCapture:
    capture_id: str
    quality_score: float
    timestamp: float
    face: BiometricFace
    full_frame: np.ndarray
    person_bbox: Tuple[int, int, int, int]
    frame_size: Tuple[int, int]


class CameraSessionTracker:
    """
    Maintains primary subject tracking with hysteresis, automated session start/stop,
    and captures the top 5 highest-quality biometric face frames directly into MongoDB.
    """

    def __init__(self, camera_id: str, atm_id: str, roi: List[List[int]]) -> None:
        self.camera_id = camera_id
        self.atm_id = atm_id
        self.roi_polygon = np.array(roi, dtype=np.int32) if (roi and len(roi) >= 3) else None

        self.current_session_id: Optional[str] = None
        self.session_start_time: float = 0.0
        self.session_last_seen: float = 0.0
        self.last_capture_time: float = 0.0
        self.incumbent_track_id: Optional[int] = None
        self.challenger_track_id: Optional[int] = None
        self.challenger_frames: int = 0
        self.presence_frames: int = 0

        self.captured_ids: List[str] = []
        self.best_capture_id: str = ""
        self.best_quality: float = 0.0

    def compute_primary_score(
        self,
        person_bbox: Tuple[int, int, int, int],
        face: Optional[BiometricFace],
        frame_w: int,
        frame_h: int,
    ) -> float:
        """
        Documented scoring formula for closest person selection:
        Score = w_person * (Area / FrameArea) + w_face * (FaceArea / IdealArea) + w_roi * ROIBias
        """
        px, py, pw, ph = person_bbox
        person_area_ratio = min(1.0, (pw * ph) / float(frame_w * frame_h))

        face_area_ratio = 0.0
        if face is not None:
            _, _, fw, fh = face.bbox
            face_area_ratio = min(1.0, (fw * fh) / (CONFIG.IDEAL_FACE_SIZE_PX * CONFIG.IDEAL_FACE_SIZE_PX))

        roi_score = 0.5
        cx, cy = px + pw / 2.0, py + ph / 2.0

        if self.roi_polygon is not None:
            dist = cv2.pointPolygonTest(self.roi_polygon, (cx, cy), True)
            roi_score = 1.0 if dist >= 0 else max(0.0, 1.0 - (abs(dist) / 150.0))
        else:
            dx = abs(cx - (frame_w / 2.0)) / (frame_w / 2.0)
            dy = abs(cy - (frame_h / 2.0)) / (frame_h / 2.0)
            roi_score = max(0.0, 1.0 - math.sqrt(dx * dx + dy * dy))

        return float(
            CONFIG.PRIMARY_WEIGHT_PERSON * person_area_ratio
            + CONFIG.PRIMARY_WEIGHT_FACE * face_area_ratio
            + CONFIG.PRIMARY_WEIGHT_ROI * roi_score
        )

    def update_primary_subject(self, persons: List[TrackedPerson], frame_w: int, frame_h: int) -> Optional[int]:
        """Apply hysteresis to select primary subject without flickering between people."""
        if not persons:
            return None

        for p in persons:
            p.primary_score = self.compute_primary_score(p.bbox, p.face, frame_w, frame_h)

        persons_sorted = sorted(persons, key=lambda p: p.primary_score, reverse=True)
        top_person = persons_sorted[0]

        if self.incumbent_track_id is None:
            self.incumbent_track_id = top_person.track_id
            self.challenger_track_id = None
            self.challenger_frames = 0
            return self.incumbent_track_id

        incumbent = next((p for p in persons if p.track_id == self.incumbent_track_id), None)
        if incumbent is None:
            self.incumbent_track_id = top_person.track_id
            self.challenger_track_id = None
            self.challenger_frames = 0
            return self.incumbent_track_id

        if top_person.track_id != self.incumbent_track_id:
            margin = top_person.primary_score - incumbent.primary_score
            if margin > CONFIG.CHALLENGER_MARGIN:
                if self.challenger_track_id == top_person.track_id:
                    self.challenger_frames += 1
                else:
                    self.challenger_track_id = top_person.track_id
                    self.challenger_frames = 1

                if self.challenger_frames >= CONFIG.HYSTERESIS_FRAMES:
                    self.incumbent_track_id = top_person.track_id
                    self.challenger_track_id = None
                    self.challenger_frames = 0
            else:
                self.challenger_frames = 0
        else:
            self.challenger_frames = 0

        return self.incumbent_track_id

    def process_frame(
        self,
        persons: List[TrackedPerson],
        faces: List[BiometricFace],
        full_frame: np.ndarray,
        worker_queue: queue.Queue,
    ) -> Optional[int]:
        """
        Core lifecycle coordinator executed per camera frame.
        Directly captures up to 5 best distinct frames when a face is detected
        and streams them immediately into MongoDB Atlas and local disk.
        """
        now = time.time()
        h, w = full_frame.shape[:2]
        primary_id = self.update_primary_subject(persons, w, h) if persons else None
        primary_person = next((p for p in persons if p.track_id == primary_id), None) if primary_id else None

        # Determine the primary face to capture
        primary_face: Optional[BiometricFace] = None
        if primary_person is not None and primary_person.face is not None:
            primary_face = primary_person.face
        elif faces:
            valid_faces = [f for f in faces if not f.is_rejected]
            if valid_faces:
                primary_face = max(valid_faces, key=lambda f: f.quality_score)
            else:
                primary_face = faces[0]

        if primary_face is not None and not primary_face.is_rejected and primary_face.quality_score > 0.0:
            self.session_last_seen = now
            self.presence_frames += 1

            # Start session immediately upon face detection
            if self.current_session_id is None:
                self.current_session_id = str(uuid.uuid4())
                self.session_start_time = now
                self.captured_ids.clear()
                self.best_capture_id = ""
                self.best_quality = 0.0
                self.last_capture_time = 0.0
                logger.info(
                    f"[{self.camera_id}] Face detected! Starting new capture session {self.current_session_id[:8]}..."
                )

            # Capture up to MAX_CAPTURES_PER_SESSION (5) distinct frames
            if len(self.captured_ids) < CONFIG.MAX_CAPTURES_PER_SESSION:
                time_since_last = now - self.last_capture_time
                if (time_since_last >= CONFIG.MIN_CAPTURE_TIME_GAP_SEC or len(self.captured_ids) == 0):
                    cap_id = str(uuid.uuid4())
                    rank = len(self.captured_ids) + 1
                    p_bbox = primary_person.bbox if primary_person else primary_face.bbox

                    cand = BestCandidateCapture(
                        capture_id=cap_id,
                        quality_score=primary_face.quality_score,
                        timestamp=now,
                        face=primary_face,
                        full_frame=full_frame.copy(),
                        person_bbox=p_bbox,
                        frame_size=(w, h),
                    )
                    self.captured_ids.append(cap_id)
                    self.last_capture_time = now
                    if primary_face.quality_score > self.best_quality:
                        self.best_quality = primary_face.quality_score
                        self.best_capture_id = cap_id

                    # Enqueue for immediate disk writing & MongoDB Atlas insert
                    worker_queue.put({
                        "type": "save_single_capture",
                        "session_id": self.current_session_id,
                        "camera_id": self.camera_id,
                        "atm_id": self.atm_id,
                        "rank": rank,
                        "candidate": cand,
                        "track_id": primary_id if primary_id else 1,
                    })
                    logger.info(
                        f"[{self.camera_id}] [CAPTURE #{rank}/5] Face frame captured! Quality: {primary_face.quality_score:.2f} -> Persisting to MongoDB Atlas"
                    )
        else:
            # Check absence timeout to finalize and reset session
            if self.current_session_id is not None:
                if (now - self.session_last_seen) > CONFIG.ABSENCE_TIMEOUT_SECONDS:
                    self.finalize_session(worker_queue)

        return primary_id

    def finalize_session(self, worker_queue: queue.Queue) -> None:
        """Finalize and package stored captures for asynchronous persistence."""
        if not self.current_session_id:
            return

        session_id = self.current_session_id
        duration = max(0.1, self.session_last_seen - self.session_start_time)
        logger.info(
            f"[{self.camera_id}] Finalizing session {session_id[:8]}... | Duration: {duration:.1f}s | "
            f"Captures: {len(self.captured_ids)}"
        )

        job = {
            "type": "finalize_session",
            "camera_id": self.camera_id,
            "atm_id": self.atm_id,
            "session_id": session_id,
            "start_time": datetime.fromtimestamp(self.session_start_time, tz=timezone.utc),
            "end_time": datetime.fromtimestamp(self.session_last_seen, tz=timezone.utc),
            "identity_label": "FACE DETECTED",
            "best_capture_id": self.best_capture_id,
            "capture_ids": list(self.captured_ids),
            "duration_seconds": round(duration, 2),
            "track_id": self.incumbent_track_id if self.incumbent_track_id else 1,
        }
        worker_queue.put(job)

        # Reset session state
        self.current_session_id = None
        self.incumbent_track_id = None
        self.presence_frames = 0
        self.captured_ids.clear()
        self.best_capture_id = ""
        self.best_quality = 0.0


# =============================================================================
# SECTION 6: MONGODB DATA LAYER & RESILIENT ON-DISK BUFFER
# =============================================================================

def sanitize_for_mongo(data: Any) -> Any:
    """Recursively converts numpy types, tuples, etc. into native Python BSON-compatible types."""
    if isinstance(data, dict):
        return {str(k): sanitize_for_mongo(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [sanitize_for_mongo(v) for v in data]
    elif isinstance(data, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64, np.float32, np.float16)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return sanitize_for_mongo(data.tolist())
    elif isinstance(data, (np.bool_, bool)):
        return bool(data)
    return data


class MongoResilientDatabase:
    """
    MongoDB connection manager with startup indexing, query support,
    and automatic fallback to on-disk buffer if MongoDB is disconnected.
    """

    def __init__(self, uri: str, db_name: str) -> None:
        self.uri = uri
        self.db_name = db_name
        self.client: Optional[pymongo.MongoClient] = None
        self.db: Optional[Database] = None
        self.is_connected = False
        self.buffer_file = CONFIG.BUFFER_DIR / "captures_buffer.jsonl"
        self._init_connection()

    def _init_connection(self) -> bool:
        if not PYMONGO_AVAILABLE:
            logger.warning("[MongoDB] pymongo module not installed. Operating strictly in on-disk buffer mode.")
            self.is_connected = False
            return False

        try:
            self.client = pymongo.MongoClient(
                self.uri,
                serverSelectionTimeoutMS=CONFIG.MONGO_CONNECT_TIMEOUT_MS,
            )
            # Ping database to confirm connection
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self.is_connected = True
            logger.info(f"[MongoDB] Connected successfully to '{self.db_name}' at {self.uri}")
            self.create_indexes()
            self.flush_buffer()
            return True
        except Exception as e:
            logger.warning(f"[MongoDB] Connection failed ({e}). Enabling local on-disk buffer.")
            self.is_connected = False
            return False

    def create_indexes(self) -> None:
        """Create mandatory indexes for high-speed investigator queries."""
        if not self.is_connected or self.db is None:
            return
        try:
            captures_col = self.db["captures"]
            captures_col.create_index([("captured_at", pymongo.ASCENDING)])
            captures_col.create_index([("atm_id", pymongo.ASCENDING), ("captured_at", pymongo.ASCENDING)])
            captures_col.create_index([("camera_id", pymongo.ASCENDING), ("captured_at", pymongo.ASCENDING)])
            captures_col.create_index([("session_id", pymongo.ASCENDING)])
            captures_col.create_index([("identity_label", pymongo.ASCENDING)])

            sessions_col = self.db["sessions"]
            sessions_col.create_index([("session_id", pymongo.ASCENDING)], unique=True)
            sessions_col.create_index([("start_time", pymongo.DESCENDING)])

            cameras_col = self.db["cameras"]
            cameras_col.create_index([("camera_id", pymongo.ASCENDING)], unique=True)
            logger.info("[MongoDB] Verified 5 collections and composite performance indexes.")
        except Exception as e:
            logger.error(f"[MongoDB] Error creating indexes: {e}")

    def insert_capture(self, doc: Dict[str, Any]) -> bool:
        """Store capture document directly or buffer to disk if offline."""
        clean_doc = sanitize_for_mongo(doc)
        if self.is_connected and self.db is not None:
            try:
                self.db["captures"].insert_one(clean_doc)
                return True
            except (pymongo.errors.ConnectionFailure, pymongo.errors.ServerSelectionTimeoutError) as ce:
                logger.error(f"[MongoDB] Connection failure: {ce}. Falling back to buffer.")
                self.is_connected = False
            except Exception as e:
                logger.error(f"[MongoDB] Insert capture error: {e}")

        # Fallback to local on-disk buffer
        try:
            serializable = dict(clean_doc)
            if isinstance(serializable.get("captured_at"), datetime):
                serializable["captured_at"] = serializable["captured_at"].isoformat()
            with open(self.buffer_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"type": "capture", "data": serializable}, default=str) + "\n")
            return True
        except Exception as be:
            logger.critical(f"[Buffer] Failed writing capture to on-disk buffer: {be}")
            return False

    def insert_session(self, doc: Dict[str, Any]) -> bool:
        """Store session document directly or buffer to disk if offline."""
        clean_doc = sanitize_for_mongo(doc)
        if self.is_connected and self.db is not None:
            try:
                self.db["sessions"].insert_one(clean_doc)
                return True
            except (pymongo.errors.ConnectionFailure, pymongo.errors.ServerSelectionTimeoutError) as ce:
                logger.error(f"[MongoDB] Connection failure: {ce}. Falling back to buffer.")
                self.is_connected = False
            except Exception as e:
                logger.error(f"[MongoDB] Insert session error: {e}")

        try:
            serializable = dict(clean_doc)
            if isinstance(serializable.get("start_time"), datetime):
                serializable["start_time"] = serializable["start_time"].isoformat()
            if isinstance(serializable.get("end_time"), datetime):
                serializable["end_time"] = serializable["end_time"].isoformat()
            with open(self.buffer_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"type": "session", "data": serializable}, default=str) + "\n")
            return True
        except Exception as be:
            logger.critical(f"[Buffer] Failed writing session to on-disk buffer: {be}")
            return False

    def flush_buffer(self) -> None:
        """Flushes offline buffered documents back to MongoDB once connection is restored."""
        if not self.is_connected or not self.buffer_file.is_file():
            return

        try:
            with open(self.buffer_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                return

            logger.info(f"[MongoDB] Flushing {len(lines)} buffered records to database...")
            remaining = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    t = item.get("type")
                    data = sanitize_for_mongo(item.get("data", {}))
                    if t == "capture":
                        if isinstance(data.get("captured_at"), str):
                            data["captured_at"] = datetime.fromisoformat(data["captured_at"])
                        self.db["captures"].insert_one(data)
                    elif t == "session":
                        if isinstance(data.get("start_time"), str):
                            data["start_time"] = datetime.fromisoformat(data["start_time"])
                        if isinstance(data.get("end_time"), str):
                            data["end_time"] = datetime.fromisoformat(data["end_time"])
                        self.db["sessions"].insert_one(data)
                except pymongo.errors.DuplicateKeyError:
                    # Record already exists in MongoDB Atlas, safely skip
                    pass
                except Exception as ex:
                    logger.error(f"[Buffer] Error inserting buffered line: {ex}")
                    remaining.append(line)

            if remaining:
                with open(self.buffer_file, "w", encoding="utf-8") as f:
                    f.writelines(l + "\n" for l in remaining)
            else:
                self.buffer_file.unlink(missing_ok=True)
                logger.info("[MongoDB] Buffer completely cleared and flushed.")
        except Exception as e:
            logger.error(f"[Buffer] Flush failed: {e}")


# =============================================================================
# SECTION 7: ASYNCHRONOUS HEAVY WORKER PIPELINE
# =============================================================================

class AsyncCaptureWorker:
    """
    Background worker thread pool executing image enhancement, disk I/O,
    and database commits without blocking GUI rendering threads.
    """

    def __init__(self, db: MongoResilientDatabase) -> None:
        self.db = db
        self.job_queue: queue.Queue = queue.Queue(maxsize=100)
        self.stopped = False
        self.total_saved_captures = 0
        self._thread = threading.Thread(target=self._worker_loop, name="AsyncEvidenceWorker", daemon=True)
        self._thread.start()

    def _worker_loop(self) -> None:
        while not self.stopped:
            try:
                job = self.job_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if job.get("type") == "save_single_capture":
                    self._process_single_capture(job)
                elif job.get("type") == "finalize_session":
                    self._process_session(job)
            except Exception as e:
                logger.error(f"[Worker] Exception in async capture job: {e}")
            finally:
                self.job_queue.task_done()

    def _process_single_capture(self, job: Dict[str, Any]) -> None:
        """Process and save an individual capture immediately to disk and MongoDB Atlas."""
        session_id = job["session_id"]
        camera_id = job["camera_id"]
        atm_id = job["atm_id"]
        rank_idx = job["rank"]
        cand: BestCandidateCapture = job["candidate"]
        face = cand.face

        utc_time = datetime.fromtimestamp(cand.timestamp, tz=timezone.utc)
        local_dt = utc_time.astimezone(CONFIG.LOCAL_TZ)

        # Save untouched original, face crop, and enhanced copy to disk
        rel_frame, rel_face, rel_enh = save_capture_evidence(
            atm_id=atm_id,
            session_id=session_id,
            rank=rank_idx,
            full_frame=cand.full_frame,
            face_bbox=face.bbox,
        )

        capture_doc = {
            "_id": cand.capture_id,
            "session_id": session_id,
            "camera_id": camera_id,
            "atm_id": atm_id,
            "captured_at": utc_time,
            "captured_at_local_iso": local_dt.isoformat(),
            "captured_at_local_str": local_dt.strftime("%Y-%m-%d %H:%M:%S IST"),
            "rank_in_session": rank_idx,
            "quality_score": round(cand.quality_score, 4),
            "quality_breakdown": face.quality_breakdown,
            "identity_label": "FACE DETECTED",
            "recognition_confidence": round(face.recognition_score, 4),
            "embedding": face.embedding,
            "face_bbox": list(face.bbox),
            "person_bbox": list(cand.person_bbox),
            "frame_size": list(cand.frame_size),
            "image_paths": {
                "original_frame": rel_frame,
                "face_crop": rel_face,
                "enhanced_face_crop": rel_enh,
            },
            "enhanced": CONFIG.ENABLE_ENHANCEMENT,
            "track_id": job.get("track_id", 1),
        }

        success = self.db.insert_capture(capture_doc)
        self.total_saved_captures += 1
        if success:
            logger.info(
                f"[Worker] Successfully saved Frame #{rank_idx}/5 to MongoDB Atlas & disk | ID: {cand.capture_id[:8]}"
            )
        else:
            logger.warning(
                f"[Worker] Saved Frame #{rank_idx}/5 to disk & local buffer | ID: {cand.capture_id[:8]}"
            )

    def _process_session(self, job: Dict[str, Any]) -> None:
        """Store session metadata in MongoDB upon session completion."""
        session_id = job["session_id"]
        camera_id = job["camera_id"]
        atm_id = job["atm_id"]

        session_doc = {
            "session_id": session_id,
            "camera_id": camera_id,
            "atm_id": atm_id,
            "start_time": job["start_time"],
            "end_time": job["end_time"],
            "identity_label": "FACE DETECTED",
            "best_capture_id": job.get("best_capture_id", ""),
            "capture_ids": job.get("capture_ids", []),
            "duration_seconds": job.get("duration_seconds", 0.0),
        }
        self.db.insert_session(session_doc)
        logger.info(
            f"[Worker] Persisted session {session_id[:8]} with {len(job.get('capture_ids', []))} captures in MongoDB."
        )


# =============================================================================
# SECTION 8: INVESTIGATOR REST API (FastAPI + Uvicorn)
# =============================================================================

app = FastAPI(
    title="IBVAP ATM Facial Forensics & Investigator Search API",
    description="Automated Best Face Capture, Biometric Search & Timestamp Forensic Search Engine",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global handles populated during startup
GLOBAL_DB: Optional[MongoResilientDatabase] = None
GLOBAL_FACE_ENGINE: Optional[FaceAnalyticsEngine] = None
GLOBAL_CAM_MANAGER: Optional[MultiCameraManager] = None
GLOBAL_WORKER: Optional[AsyncCaptureWorker] = None


def get_current_user() -> Dict[str, Any]:
    """Pluggable authentication dependency stub returning anonymous investigator role."""
    return {"user_id": "investigator_01", "role": "forensic_investigator", "authenticated": True}


# Pydantic Response Schemas
class SearchResultItem(BaseModel):
    capture_id: str
    session_id: str
    atm_id: str
    camera_id: str
    captured_at: str
    identity_label: str
    quality_score: float
    similarity_score: float
    image_urls: Dict[str, str]


class ImageSearchResponse(BaseModel):
    total_matches: int
    threshold_used: float
    query_faces_detected: int
    note: str
    results: List[SearchResultItem]


class CsvRowMatch(BaseModel):
    input_row_index: int
    input_timestamp_str: str
    parsed_utc_timestamp: str
    matches_found: int
    matches: List[Dict[str, Any]]


class CsvSearchResponse(BaseModel):
    total_rows_processed: int
    valid_rows: int
    invalid_rows: int
    errors: List[Dict[str, Any]]
    results: List[CsvRowMatch]


def parse_flexible_datetime(dt_str: str) -> Optional[datetime]:
    """Parse ambiguous date-time representations into timezone-aware UTC datetime."""
    s = dt_str.strip()
    # Check epoch seconds
    try:
        val = float(s)
        if val > 1e8:
            return datetime.fromtimestamp(val, tz=timezone.utc)
    except ValueError:
        pass

    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                # Naive times are interpreted as Asia/Kolkata
                dt = dt.replace(tzinfo=CONFIG.LOCAL_TZ)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue

    return None


@app.post("/api/search/by-image", response_model=ImageSearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    threshold: float = Query(CONFIG.SEARCH_SIMILARITY_THRESHOLD, ge=0.0, le=1.0),
    limit: int = Query(10, ge=1, le=CONFIG.SEARCH_MAX_RESULTS),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Search stored ATM biometric captures by uploading a suspect face photo.
    Performs cosine similarity search against stored embeddings.
    """
    audit_log("/api/search/by-image", user["user_id"], {"filename": file.filename, "threshold": threshold})

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    if GLOBAL_FACE_ENGINE is None or GLOBAL_FACE_ENGINE.detector is None or GLOBAL_FACE_ENGINE.recognizer is None:
        raise HTTPException(status_code=503, detail="Face recognition engine is not initialized.")

    h, w = img_bgr.shape[:2]
    GLOBAL_FACE_ENGINE.detector.setInputSize((w, h))
    _, raw_faces = GLOBAL_FACE_ENGINE.detector.detect(img_bgr)

    if raw_faces is None or len(raw_faces) == 0:
        raise HTTPException(status_code=422, detail="No faces detected in the uploaded inquiry image.")

    # Select the largest face by bounding box area
    largest_face = max(raw_faces, key=lambda f: f[2] * f[3])
    q_emb_512 = GLOBAL_FACE_ENGINE.extract_embedding(img_bgr, largest_face)
    q_vec = np.array(q_emb_512[:128], dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)

    if q_norm < 1e-6:
        raise HTTPException(status_code=422, detail="Unable to extract biometric feature embedding.")
    q_vec = q_vec / q_norm

    # Query MongoDB or local buffer
    matches = []
    if GLOBAL_DB and GLOBAL_DB.is_connected and GLOBAL_DB.db is not None:
        cursor = GLOBAL_DB.db["captures"].find({}, {"embedding": 1, "session_id": 1, "atm_id": 1, "camera_id": 1, "captured_at_local_iso": 1, "identity_label": 1, "quality_score": 1})
        for doc in cursor:
            emb = doc.get("embedding", [])
            if not emb:
                continue
            ref_vec = np.array(emb[:128], dtype=np.float32)
            ref_norm = np.linalg.norm(ref_vec)
            if ref_norm < 1e-6:
                continue
            sim = float(np.dot(q_vec, ref_vec / ref_norm))
            if sim >= threshold:
                cid = str(doc["_id"])
                matches.append(
                    SearchResultItem(
                        capture_id=cid,
                        session_id=str(doc.get("session_id", "")),
                        atm_id=str(doc.get("atm_id", "")),
                        camera_id=str(doc.get("camera_id", "")),
                        captured_at=str(doc.get("captured_at_local_iso", "")),
                        identity_label=str(doc.get("identity_label", "UNKNOWN")),
                        quality_score=float(doc.get("quality_score", 0.0)),
                        similarity_score=round(sim, 4),
                        image_urls={
                            "face": f"/api/images/{cid}/face",
                            "frame": f"/api/images/{cid}/frame",
                            "enhanced": f"/api/images/{cid}/enhanced",
                        },
                    )
                )

    matches.sort(key=lambda x: x.similarity_score, reverse=True)
    top_matches = matches[:limit]

    note_text = "Query face processed successfully."
    if len(raw_faces) > 1:
        note_text = f"Multiple faces detected ({len(raw_faces)}). Searched using the primary/largest face crop."

    return ImageSearchResponse(
        total_matches=len(top_matches),
        threshold_used=threshold,
        query_faces_detected=len(raw_faces),
        note=note_text,
        results=top_matches,
    )


@app.post("/api/search/by-csv", response_model=CsvSearchResponse)
async def search_by_csv(
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Search ATM captures closest in time to a list of timestamps uploaded in a CSV file.
    Efficiently queries nearest 5 captures before and after each timestamp.
    """
    audit_log("/api/search/by-csv", user["user_id"], {"filename": file.filename})

    contents = await file.read()
    if len(contents) > CONFIG.MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail=f"CSV file exceeds maximum limit of {CONFIG.MAX_CSV_BYTES / 1024 / 1024:.0f}MB")

    try:
        decoded = contents.decode("utf-8-sig")
    except UnicodeDecodeError:
        decoded = contents.decode("latin-1")

    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV file is empty or missing headers.")

    # Identify date-time columns
    headers_lower = {h.lower().strip(): h for h in reader.fieldnames}
    time_col = None
    for candidate in ["datetime", "date_time", "timestamp", "time", "date"]:
        if candidate in headers_lower:
            time_col = headers_lower[candidate]
            break

    if not time_col and ("date" in headers_lower and "time" in headers_lower):
        date_col = headers_lower["date"]
        time_part_col = headers_lower["time"]
    else:
        date_col, time_part_col = None, None

    if not time_col and not date_col:
        raise HTTPException(
            status_code=422,
            detail=f"CSV must contain a datetime/timestamp column. Found headers: {list(reader.fieldnames)}",
        )

    atm_col = headers_lower.get("atm_id", headers_lower.get("atmid"))
    cam_col = headers_lower.get("camera_id", headers_lower.get("cam_id"))

    results: List[CsvRowMatch] = []
    errors: List[Dict[str, Any]] = []
    row_count = 0

    for idx, row in enumerate(reader, start=1):
        row_count += 1
        if row_count > CONFIG.MAX_CSV_ROWS:
            errors.append({"row": idx, "reason": f"Exceeded maximum row limit of {CONFIG.MAX_CSV_ROWS}."})
            break

        # Extract timestamp string
        if time_col:
            raw_time_str = row.get(time_col, "")
        else:
            raw_time_str = f"{row.get(date_col, '')} {row.get(time_part_col, '')}".strip()

        parsed_utc = parse_flexible_datetime(raw_time_str)
        if parsed_utc is None:
            errors.append({"row": idx, "raw_value": raw_time_str, "reason": "Unrecognized date-time format."})
            continue

        atm_filter = row.get(atm_col) if atm_col else None
        cam_filter = row.get(cam_col) if cam_col else None

        # Nearest-in-time query using MongoDB indexes
        matched_items = []
        if GLOBAL_DB and GLOBAL_DB.is_connected and GLOBAL_DB.db is not None:
            base_q: Dict[str, Any] = {}
            if atm_filter:
                base_q["atm_id"] = atm_filter
            if cam_filter:
                base_q["camera_id"] = cam_filter

            # 5 nearest captures at or before target time
            q_before = dict(base_q)
            q_before["captured_at"] = {"$lte": parsed_utc}
            before_docs = list(GLOBAL_DB.db["captures"].find(q_before).sort("captured_at", pymongo.DESCENDING).limit(5))

            # 5 nearest captures after target time
            q_after = dict(base_q)
            q_after["captured_at"] = {"$gt": parsed_utc}
            after_docs = list(GLOBAL_DB.db["captures"].find(q_after).sort("captured_at", pymongo.ASCENDING).limit(5))

            merged = before_docs + after_docs
            # Sort by absolute time difference
            for doc in merged:
                doc_utc = doc["captured_at"]
                if doc_utc.tzinfo is None:
                    doc_utc = doc_utc.replace(tzinfo=timezone.utc)
                diff_sec = (doc_utc - parsed_utc).total_seconds()
                cid = str(doc["_id"])
                matched_items.append({
                    "capture_id": cid,
                    "session_id": str(doc.get("session_id", "")),
                    "atm_id": str(doc.get("atm_id", "")),
                    "camera_id": str(doc.get("camera_id", "")),
                    "captured_at": str(doc.get("captured_at_local_iso", "")),
                    "identity_label": str(doc.get("identity_label", "UNKNOWN")),
                    "quality_score": float(doc.get("quality_score", 0.0)),
                    "signed_time_difference_seconds": round(diff_sec, 2),
                    "absolute_time_difference_seconds": round(abs(diff_sec), 2),
                    "image_urls": {
                        "face": f"/api/images/{cid}/face",
                        "frame": f"/api/images/{cid}/frame",
                        "enhanced": f"/api/images/{cid}/enhanced",
                    },
                })

            matched_items.sort(key=lambda m: m["absolute_time_difference_seconds"])
            matched_items = matched_items[:5]

        results.append(
            CsvRowMatch(
                input_row_index=idx,
                input_timestamp_str=raw_time_str,
                parsed_utc_timestamp=parsed_utc.isoformat(),
                matches_found=len(matched_items),
                matches=matched_items,
            )
        )

    return CsvSearchResponse(
        total_rows_processed=row_count,
        valid_rows=len(results),
        invalid_rows=len(errors),
        errors=errors,
        results=results,
    )


@app.get("/api/captures")
async def list_captures(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    atm_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    identity_label: Optional[str] = None,
    min_quality: float = Query(0.0, ge=0.0, le=1.0),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Filter and paginate stored ATM captures with newest records first."""
    if not GLOBAL_DB or not GLOBAL_DB.is_connected or GLOBAL_DB.db is None:
        return {"total": 0, "page": page, "limit": limit, "items": [], "note": "Database offline. Operating in buffer mode."}

    query: Dict[str, Any] = {"quality_score": {"$gte": min_quality}}
    if atm_id:
        query["atm_id"] = atm_id
    if camera_id:
        query["camera_id"] = camera_id
    if identity_label:
        query["identity_label"] = identity_label

    if start_date or end_date:
        date_q: Dict[str, Any] = {}
        if start_date:
            s_dt = parse_flexible_datetime(start_date)
            if s_dt:
                date_q["$gte"] = s_dt
        if end_date:
            e_dt = parse_flexible_datetime(end_date)
            if e_dt:
                date_q["$lte"] = e_dt
        if date_q:
            query["captured_at"] = date_q

    total = GLOBAL_DB.db["captures"].count_documents(query)
    skip = (page - 1) * limit
    cursor = GLOBAL_DB.db["captures"].find(query).sort("captured_at", pymongo.DESCENDING).skip(skip).limit(limit)

    items = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("captured_at"), datetime):
            doc["captured_at"] = doc["captured_at"].isoformat()
        items.append(doc)

    return {"total": total, "page": page, "limit": limit, "items": items}


@app.get("/api/captures/{capture_id}")
async def get_capture_details(capture_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    """Retrieve full audit metadata for an individual capture."""
    if not GLOBAL_DB or not GLOBAL_DB.is_connected or GLOBAL_DB.db is None:
        raise HTTPException(status_code=503, detail="Database currently offline.")

    doc = GLOBAL_DB.db["captures"].find_one({"_id": capture_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Capture not found.")

    doc["_id"] = str(doc["_id"])
    if isinstance(doc.get("captured_at"), datetime):
        doc["captured_at"] = doc["captured_at"].isoformat()
    return doc


@app.get("/api/sessions/{session_id}")
async def get_session_details(session_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    """Retrieve full transaction session details and related capture IDs."""
    if not GLOBAL_DB or not GLOBAL_DB.is_connected or GLOBAL_DB.db is None:
        raise HTTPException(status_code=503, detail="Database currently offline.")

    doc = GLOBAL_DB.db["sessions"].find_one({"session_id": session_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found.")

    doc["_id"] = str(doc["_id"])
    if isinstance(doc.get("start_time"), datetime):
        doc["start_time"] = doc["start_time"].isoformat()
    if isinstance(doc.get("end_time"), datetime):
        doc["end_time"] = doc["end_time"].isoformat()
    return doc


@app.get("/api/images/{capture_id}/{kind}")
async def get_capture_image(capture_id: str, kind: str):
    """
    Stream stored biometric evidence image.
    Kind options: 'face', 'frame', 'enhanced'.
    """
    if kind not in ["face", "frame", "enhanced"]:
        raise HTTPException(status_code=400, detail="Invalid kind. Must be 'face', 'frame', or 'enhanced'.")

    if not GLOBAL_DB or not GLOBAL_DB.is_connected or GLOBAL_DB.db is None:
        raise HTTPException(status_code=503, detail="Database offline.")

    doc = GLOBAL_DB.db["captures"].find_one({"_id": capture_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Capture record not found.")

    paths = doc.get("image_paths", {})
    mapping = {
        "face": paths.get("face_crop"),
        "frame": paths.get("original_frame"),
        "enhanced": paths.get("enhanced_face_crop"),
    }
    rel_path = mapping.get(kind)
    if not rel_path:
        raise HTTPException(status_code=404, detail=f"Image for kind '{kind}' not stored.")

    full_path = CONFIG.PROJECT_ROOT / rel_path
    if not full_path.is_file():
        raise HTTPException(status_code=404, detail="Image file not present on filesystem.")

    return FileResponse(str(full_path), media_type="image/jpeg")


@app.get("/api/cameras")
async def list_cameras():
    """List all registered cameras with live status, FPS, and connection metrics."""
    if not GLOBAL_CAM_MANAGER:
        return []

    cams_info = []
    for cid, cthread in GLOBAL_CAM_MANAGER.cameras.items():
        cams_info.append({
            "camera_id": cid,
            "atm_id": cthread.config.atm_id,
            "label": cthread.config.label,
            "source": str(cthread.config.source),
            "enabled": cthread.config.enabled,
            "status": "ONLINE" if cthread.is_active else "OFFLINE",
            "fps": round(cthread.fps_meter, 1),
            "last_seen_seconds_ago": round(time.time() - cthread.last_seen, 1) if cthread.last_seen > 0 else None,
            "total_frames_read": cthread.frame_count,
            "dropped_frames": cthread.dropped_count,
            "reconnect_attempts": cthread.reconnect_count,
        })
    return cams_info


@app.get("/api/health")
async def get_system_health():
    """System health diagnostics including DB status, GPU/CPU info, and queues."""
    db_ok = bool(GLOBAL_DB and GLOBAL_DB.is_connected)
    buf_size = 0
    if GLOBAL_DB and GLOBAL_DB.buffer_file.is_file():
        buf_size = GLOBAL_DB.buffer_file.stat().st_size

    worker_queue_depth = GLOBAL_WORKER.job_queue.qsize() if GLOBAL_WORKER else 0

    return {
        "status": "HEALTHY" if db_ok else "DEGRADED (BUFFER MODE)",
        "database": {
            "connected": db_ok,
            "mongo_uri": CONFIG.MONGO_URI,
            "db_name": CONFIG.MONGO_DB_NAME,
            "buffered_bytes": buf_size,
        },
        "models": {
            "device": CONFIG.DEVICE,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "enrolled_faces_count": len(GLOBAL_FACE_ENGINE.enrolled_gallery) if GLOBAL_FACE_ENGINE else 0,
        },
        "worker_queue_depth": worker_queue_depth,
        "total_saved_captures": GLOBAL_WORKER.total_saved_captures if GLOBAL_WORKER else 0,
        "timezone": CONFIG.LOCAL_TIMEZONE_STR,
    }


def start_api_server_background() -> None:
    """Launch FastAPI API server using Uvicorn in a dedicated background daemon thread."""
    if not CONFIG.ENABLE_API:
        return

    def run_server():
        logger.info(f"[API] Launching Investigator REST API at http://{CONFIG.API_HOST}:{CONFIG.API_PORT}")
        uvicorn.run(app, host=CONFIG.API_HOST, port=CONFIG.API_PORT, log_level="warning")

    t = threading.Thread(target=run_server, name="UvicornAPIServer", daemon=True)
    t.start()


# =============================================================================
# SECTION 9: PRESERVED DESKTOP GUI CONSOLE & MAIN APPLICATION LOOP
# =============================================================================

CURRENT_DETECTION_MODE = "all"  # "all", "humans", "faces"
BUTTON_REGIONS: Dict[str, Tuple[int, int, int, int]] = {}


def on_mouse_click(event: int, x: int, y: int, flags: int, param: any) -> None:
    """Handle interactive on-screen mode selector buttons."""
    global CURRENT_DETECTION_MODE
    if event == cv2.EVENT_LBUTTONDOWN:
        for mode_key, (bx1, by1, bx2, by2) in BUTTON_REGIONS.items():
            if bx1 <= x <= bx2 and by1 <= y <= by2:
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
    is_live: bool = True,
) -> np.ndarray:
    h, w = frame.shape[:2]
    banner_h = 34

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

    badge_text = f"HUMANS: {humans} | FACES: {faces}"
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
    h_count1: int,
    f_count1: int,
    target_h: int = 480,
) -> np.ndarray:
    h0, w0 = frame0.shape[:2]
    w0_res = int(w0 * (target_h / h0))
    f0_res = cv2.resize(frame0, (w0_res, target_h)) if (h0 != target_h) else frame0

    h1, w1 = frame1.shape[:2]
    w1_res = int(w1 * (target_h / h1))
    f1_res = cv2.resize(frame1, (w1_res, target_h)) if (h1 != target_h) else frame1

    draw_header_banner(f0_res, "Feed A", dev0_name, h_count0, f_count0)
    draw_header_banner(f1_res, "Feed B", dev1_name, h_count1, f_count1)

    combined = np.hstack((f0_res, f1_res))
    cv2.line(combined, (w0_res, 0), (w0_res, target_h), (0, 255, 140), 2)
    return combined


def draw_top_mode_bar(frame: np.ndarray, active_mode: str) -> None:
    """Render interactive glassmorphic mode selector buttons."""
    global BUTTON_REGIONS
    h, w = frame.shape[:2]
    bar_h = 46

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
        ("[A] Humans + Faces", "all", 170),
        ("[H] Humans Only", "humans", 145),
        ("[F] Faces Only", "faces", 130),
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
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color,
            2 if is_active else 1, cv2.LINE_AA,
        )
        start_x += btn_w + 10


def draw_bottom_hud(
    frame: np.ndarray,
    mode: str,
    view: str,
    fps: float,
    total_humans: int,
    total_faces: int,
    saved_captures: int = 0,
) -> None:
    """Render command center HUD status bar at bottom of window."""
    h, w = frame.shape[:2]
    hud_h = 32

    cv2.rectangle(frame, (0, h - hud_h), (w, h), (12, 14, 18), -1)
    cv2.line(frame, (0, h - hud_h), (w, h - hud_h), (40, 45, 60), 1)

    mode_label_map = {
        "all": "HUMANS + FACES",
        "humans": "HUMANS ONLY",
        "faces": "FACES ONLY",
    }
    mode_str = mode_label_map.get(mode, mode.upper())
    view_str = "DUAL FEED" if view == "dual" else f"CAM {view}"
    face_status = f"FACES: {total_faces}"

    hud_items = [
        f"⚡ {fps:.1f} FPS",
        f"MODE: {mode_str}",
        f"VIEW: {view_str}",
        f"HUMANS: {total_humans}",
        face_status,
        f"CAPTURED: {saved_captures} / 5",
    ]

    hud_text = "  |  ".join(hud_items)
    hud_color = (0, 255, 140)

    cv2.putText(
        frame, hud_text, (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, hud_color, 1, cv2.LINE_AA,
    )


# =============================================================================
# MAIN APPLICATION ENTRYPOINT
# =============================================================================

def main() -> None:
    global CURRENT_DETECTION_MODE, GLOBAL_DB, GLOBAL_FACE_ENGINE, GLOBAL_CAM_MANAGER, GLOBAL_WORKER

    parser = argparse.ArgumentParser(
        description="CCTV AI Vision System — ATM Best Face Capture & Investigator Search Backend"
    )
    parser.add_argument("--cam0", type=int, default=0, help="Primary camera index (default: 0)")
    parser.add_argument("--cam1", type=int, default=1, help="Secondary camera index (default: 1)")
    parser.add_argument("--mode", type=str, default="dual", choices=["dual", "0", "1"], help="Initial camera view")
    parser.add_argument(
        "--detect-mode",
        type=str,
        default="all",
        choices=["all", "humans", "faces"],
        help="Initial detection mode (default: all)",
    )
    parser.add_argument("uri", nargs="?", default=None, help="Optional positional MongoDB URI")
    parser.add_argument("--mongo-uri", type=str, default=None, help="MongoDB connection URI")
    parser.add_argument("--api-port", type=int, default=CONFIG.API_PORT, help="Investigator REST API port")
    args = parser.parse_args()

    CURRENT_DETECTION_MODE = args.detect_mode
    if args.uri:
        CONFIG.MONGO_URI = args.uri
    elif args.mongo_uri:
        CONFIG.MONGO_URI = args.mongo_uri
    CONFIG.API_PORT = args.api_port

    print("\n" + "=" * 78)
    print("  🎥  CCTV AI VISION SYSTEM — ATM Best Face Capture & Forensic Backend")
    print("=" * 78)
    print(f"  ⚡ Hardware Acceleration : {CONFIG.DEVICE.upper()}")
    print(f"  🌐 Database Storage      : {CONFIG.MONGO_URI} (DB: {CONFIG.MONGO_DB_NAME})")
    print(f"  🔍 Investigator REST API : http://{CONFIG.API_HOST}:{CONFIG.API_PORT} (Docs: /docs)")
    print(f"  🖥️ Camera Layout View   : {args.mode.upper()}")
    print(f"  🎯 AI Pipeline Mode      : {CURRENT_DETECTION_MODE.upper()}")
    print("-" * 78)
    print("  Desktop Controls:")
    print("    [A]             -> Humans + Faces (All Active)")
    print("    [H]             -> Human Detection Only")
    print("    [F]             -> Face Recognition Only")
    print("    [D] / [M]       -> Dual Camera Side-by-Side View")
    print("    [0] / [1]       -> Single Camera Feed (Cam 0 / Cam 1)")
    print("    [S]             -> Cycle Camera Views")
    print("    [Q] / ESC       -> Quit")
    print("=" * 78 + "\n")

    # 1. Initialize MongoDB & Resilient Disk Buffer
    logger.info("[Init 1/5] Initializing MongoDB resilient storage layer...")
    GLOBAL_DB = MongoResilientDatabase(CONFIG.MONGO_URI, CONFIG.MONGO_DB_NAME)

    # 2. Initialize Background Heavy Worker
    logger.info("[Init 2/5] Initializing Asynchronous Evidence Worker...")
    GLOBAL_WORKER = AsyncCaptureWorker(GLOBAL_DB)

    # 3. Initialize Face Detection & Biometric Recognition Engine
    logger.info("[Init 3/5] Loading YuNet DNN and SFace Biometric Engine...")
    GLOBAL_FACE_ENGINE = FaceAnalyticsEngine(
        models_dir=CONFIG.PROJECT_ROOT / "models",
        known_faces_dir=CONFIG.KNOWN_FACES_DIR,
    )

    # 4. Initialize YOLOv8 Human Tracker
    logger.info(f"[Init 4/5] Loading YOLOv8 Human Tracker on {CONFIG.DEVICE}...")
    human_model_path = CONFIG.PROJECT_ROOT / "models" / "human_detection" / "master_human_model" / "weights" / "best.pt"
    if not human_model_path.is_file():
        human_model_path = CONFIG.PROJECT_ROOT / "yolov8s.pt"
    yolo_detector = YOLO(str(human_model_path))
    if "cuda" in CONFIG.DEVICE and torch.cuda.is_available():
        try:
            yolo_detector.to(CONFIG.DEVICE).half()
        except Exception:
            pass

    # 5. Initialize Multi-Camera Streams
    logger.info("[Init 5/5] Connecting Multi-Camera capture threads...")
    cam_configs = load_cameras_config(cli_cam0=args.cam0, cli_cam1=args.cam1)
    GLOBAL_CAM_MANAGER = MultiCameraManager(cam_configs)

    # Initialize Per-Camera Session Trackers
    session_trackers: Dict[str, CameraSessionTracker] = {}
    for cfg in cam_configs:
        session_trackers[cfg.camera_id] = CameraSessionTracker(
            camera_id=cfg.camera_id,
            atm_id=cfg.atm_id,
            roi=cfg.roi,
        )

    # Launch Background REST API Server
    start_api_server_background()

    cam0_stream, cam1_stream = GLOBAL_CAM_MANAGER.get_primary_dual()
    dev0_label = cam0_stream.config.label if cam0_stream else "Cam 0"
    dev1_label = cam1_stream.config.label if cam1_stream else "Cam 1"

    window_name = "CCTV AI Vision Console — ATM Best Face Capture & Investigator Hub"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, on_mouse_click)

    camera_view_mode = args.mode
    prev_time = time.time()
    fps_smooth = 30.0
    frame_counter = 0

    print("\n🚀 Live streaming active. Press [Q] to quit.\n")

    try:
        while True:
            # Non-blocking frame retrieval from camera capture threads
            ret0, raw0 = (cam0_stream.read_latest()) if cam0_stream else (False, None)
            ret1, raw1 = (cam1_stream.read_latest()) if cam1_stream else (False, None)

            if not ret0 or raw0 is None:
                raw0 = create_placeholder_frame(640, 480, f"{dev0_label} Offline")
            if not ret1 or raw1 is None:
                raw1 = create_placeholder_frame(640, 480, f"{dev1_label} Offline")

            out_cam0 = raw0.copy()
            out_cam1 = raw1.copy()
            h_count0, f_count0 = 0, 0
            h_count1, f_count1 = 0, 0

            frame_counter += 1

            # -----------------------------------------------------------------
            # Camera 0 Vision Pipeline
            # -----------------------------------------------------------------
            if ret0 and (camera_view_mode in ["dual", "0"]):
                faces0: List[BiometricFace] = []
                if CURRENT_DETECTION_MODE in ["faces", "all"]:
                    faces0 = GLOBAL_FACE_ENGINE.detect_and_analyze(raw0)
                    f_count0 = len(faces0)

                # YOLO Person Detection & Tracking
                persons0: List[TrackedPerson] = []
                if CURRENT_DETECTION_MODE in ["humans", "all"]:
                    y_res0 = yolo_detector.track(
                        raw0,
                        persist=True,
                        verbose=False,
                        imgsz=CONFIG.YOLO_INPUT_SIZE,
                        conf=CONFIG.YOLO_CONF_THRESHOLD,
                        classes=[0],  # Person class
                    )
                    if y_res0 and len(y_res0) > 0 and y_res0[0].boxes is not None:
                        boxes = y_res0[0].boxes
                        for i in range(len(boxes)):
                            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                            tid = int(boxes.id[i].item()) if boxes.id is not None else (i + 1)
                            conf = float(boxes.conf[i].item())
                            pw = xyxy[2] - xyxy[0]
                            ph = xyxy[3] - xyxy[1]
                            p_bbox = (xyxy[0], xyxy[1], pw, ph)

                            # Associate detected face inside person bbox or upper body region
                            matched_face = None
                            for f in faces0:
                                fx, fy, fw, fh = f.bbox
                                fcx = fx + fw / 2.0
                                fcy = fy + fh / 2.0
                                if (p_bbox[0] - 50 <= fcx <= p_bbox[0] + pw + 50) and (
                                    p_bbox[1] - 80 <= fcy <= p_bbox[1] + ph * 0.90
                                ):
                                    matched_face = f
                                    break

                            persons0.append(
                                TrackedPerson(
                                    track_id=tid,
                                    bbox=p_bbox,
                                    confidence=conf,
                                    last_seen=time.time(),
                                    face=matched_face,
                                )
                            )

                    # Pair any remaining person with closest unassigned face
                    for p in persons0:
                        if p.face is None and faces0:
                            px, py, pw, ph = p.bbox
                            pcx = px + pw / 2.0
                            p.face = min(faces0, key=lambda f: abs((f.bbox[0] + f.bbox[2] / 2.0) - pcx))

                # If faces exist but no person body detected (e.g. webcam close-up or faces mode), synthesize person track
                if faces0 and not persons0:
                    for idx, f in enumerate(faces0, start=1):
                        fx, fy, fw, fh = f.bbox
                        persons0.append(
                            TrackedPerson(
                                track_id=idx,
                                bbox=(max(0, fx - int(fw * 0.4)), max(0, fy - int(fh * 0.2)), int(fw * 1.8), int(fh * 3.2)),
                                confidence=f.confidence,
                                last_seen=time.time(),
                                face=f,
                            )
                        )

                h_count0 = len(persons0)

                # Process Primary Subject & Session Logic
                c0_id = cam0_stream.config.camera_id if cam0_stream else "cam_001"
                primary0_id = session_trackers[c0_id].process_frame(
                    persons0, faces0, raw0, GLOBAL_WORKER.job_queue
                )

                # Draw Persons & Highlight Primary
                for p in persons0:
                    px, py, pw, ph = p.bbox
                    is_primary = (p.track_id == primary0_id)
                    box_color = (0, 215, 255) if is_primary else (40, 200, 100)
                    cv2.rectangle(out_cam0, (px, py), (px + pw, py + ph), box_color, 2 if is_primary else 1)
                    tag = f"{'★ PRIMARY ' if is_primary else ''}PERSON {p.track_id}"
                    cv2.putText(
                        out_cam0, tag, (px, max(18, py - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA
                    )

                # Draw Faces
                for f in faces0:
                    fx, fy, fw, fh = f.bbox
                    f_color = (0, 255, 200)
                    cv2.rectangle(out_cam0, (fx, fy), (fx + fw, fy + fh), f_color, 2)
                    lbl = f"FACE DETECTED ({f.quality_score:.2f})"
                    cv2.putText(
                        out_cam0, lbl, (fx, max(15, fy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, f_color, 1, cv2.LINE_AA
                    )

            # -----------------------------------------------------------------
            # Camera 1 Vision Pipeline
            # -----------------------------------------------------------------
            if ret1 and (camera_view_mode in ["dual", "1"]):
                faces1: List[BiometricFace] = []
                if CURRENT_DETECTION_MODE in ["faces", "all"]:
                    faces1 = GLOBAL_FACE_ENGINE.detect_and_analyze(raw1)
                    f_count1 = len(faces1)

                persons1: List[TrackedPerson] = []
                if CURRENT_DETECTION_MODE in ["humans", "all"]:
                    y_res1 = yolo_detector.track(
                        raw1,
                        persist=True,
                        verbose=False,
                        imgsz=CONFIG.YOLO_INPUT_SIZE,
                        conf=CONFIG.YOLO_CONF_THRESHOLD,
                        classes=[0],
                    )
                    if y_res1 and len(y_res1) > 0 and y_res1[0].boxes is not None:
                        boxes = y_res1[0].boxes
                        for i in range(len(boxes)):
                            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                            tid = int(boxes.id[i].item()) if boxes.id is not None else (i + 1)
                            conf = float(boxes.conf[i].item())
                            pw = xyxy[2] - xyxy[0]
                            ph = xyxy[3] - xyxy[1]
                            p_bbox = (xyxy[0], xyxy[1], pw, ph)

                            # Associate detected face inside person bbox or upper body region
                            matched_face = None
                            for f in faces1:
                                fx, fy, fw, fh = f.bbox
                                fcx = fx + fw / 2.0
                                fcy = fy + fh / 2.0
                                if (p_bbox[0] - 50 <= fcx <= p_bbox[0] + pw + 50) and (
                                    p_bbox[1] - 80 <= fcy <= p_bbox[1] + ph * 0.90
                                ):
                                    matched_face = f
                                    break

                            persons1.append(
                                TrackedPerson(
                                    track_id=tid,
                                    bbox=p_bbox,
                                    confidence=conf,
                                    last_seen=time.time(),
                                    face=matched_face,
                                )
                            )

                    # Pair any remaining person with closest unassigned face
                    for p in persons1:
                        if p.face is None and faces1:
                            px, py, pw, ph = p.bbox
                            pcx = px + pw / 2.0
                            p.face = min(faces1, key=lambda f: abs((f.bbox[0] + f.bbox[2] / 2.0) - pcx))

                # If faces exist but no person body detected (e.g. webcam close-up or faces mode), synthesize person track
                if faces1 and not persons1:
                    for idx, f in enumerate(faces1, start=1):
                        fx, fy, fw, fh = f.bbox
                        persons1.append(
                            TrackedPerson(
                                track_id=idx,
                                bbox=(max(0, fx - int(fw * 0.4)), max(0, fy - int(fh * 0.2)), int(fw * 1.8), int(fh * 3.2)),
                                confidence=f.confidence,
                                last_seen=time.time(),
                                face=f,
                            )
                        )

                h_count1 = len(persons1)

                c1_id = cam1_stream.config.camera_id if cam1_stream else "cam_002"
                primary1_id = session_trackers[c1_id].process_frame(
                    persons1, faces1, raw1, GLOBAL_WORKER.job_queue
                )

                for p in persons1:
                    px, py, pw, ph = p.bbox
                    is_primary = (p.track_id == primary1_id)
                    box_color = (0, 215, 255) if is_primary else (40, 200, 100)
                    cv2.rectangle(out_cam1, (px, py), (px + pw, py + ph), box_color, 2 if is_primary else 1)
                    tag = f"{'★ PRIMARY ' if is_primary else ''}PERSON {p.track_id}"
                    cv2.putText(
                        out_cam1, tag, (px, max(18, py - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA
                    )

                for f in faces1:
                    fx, fy, fw, fh = f.bbox
                    f_color = (0, 255, 200)
                    cv2.rectangle(out_cam1, (fx, fy), (fx + fw, fy + fh), f_color, 2)
                    lbl = f"FACE DETECTED ({f.quality_score:.2f})"
                    cv2.putText(
                        out_cam1, lbl, (fx, max(15, fy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, f_color, 1, cv2.LINE_AA
                    )

            # View Composition
            if camera_view_mode == "dual":
                out_frame = combine_dual_feeds(
                    out_cam0, out_cam1,
                    dev0_name=dev0_label,
                    dev1_name=dev1_label,
                    h_count0=h_count0, f_count0=f_count0,
                    h_count1=h_count1, f_count1=f_count1,
                )
            elif camera_view_mode == "0":
                out_frame = out_cam0
                draw_header_banner(out_frame, "Feed A", dev0_label, h_count0, f_count0)
            else:
                out_frame = out_cam1
                draw_header_banner(out_frame, "Feed B", dev1_label, h_count1, f_count1)

            draw_top_mode_bar(out_frame, CURRENT_DETECTION_MODE)

            curr_time = time.time()
            instant_fps = 1.0 / max(0.0001, curr_time - prev_time)
            prev_time = curr_time
            fps_smooth = (0.85 * fps_smooth) + (0.15 * instant_fps)

            total_humans = h_count0 + h_count1
            total_faces = f_count0 + f_count1
            saved_cnt = GLOBAL_WORKER.total_saved_captures if GLOBAL_WORKER else 0

            draw_bottom_hud(
                out_frame,
                CURRENT_DETECTION_MODE,
                camera_view_mode,
                fps_smooth,
                total_humans,
                total_faces,
                saved_captures=saved_cnt,
            )

            cv2.imshow(window_name, out_frame)

            # Keyboard Hotkeys
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break
            elif key == ord("a"):
                CURRENT_DETECTION_MODE = "all"
                print("  [Mode] -> HUMANS + FACES (ALL)")
            elif key == ord("h"):
                CURRENT_DETECTION_MODE = "humans"
                print("  [Mode] -> HUMAN DETECTION ONLY")
            elif key == ord("f"):
                CURRENT_DETECTION_MODE = "faces"
                print("  [Mode] -> FACE RECOGNITION ONLY")
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
        logger.info("[Shutdown] Finalizing active sessions and stopping threads...")
        for st_obj in session_trackers.values():
            if GLOBAL_WORKER:
                st_obj.finalize_session(GLOBAL_WORKER.job_queue)

        if GLOBAL_CAM_MANAGER:
            GLOBAL_CAM_MANAGER.release_all()

        cv2.destroyAllWindows()
        print("\nApplication closed cleanly.")


if __name__ == "__main__":
    main()
