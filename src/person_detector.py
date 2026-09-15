"""
Hybrid Detection + SiamFC Tracking Pipeline.

Phase 1 (Detection): Uses Faster R-CNN with a VERY high confidence threshold (0.80+)
         to initially find persons in a frame. Runs only every N frames to save GPU.
Phase 2 (Tracking):  Uses YOUR TRAINED SiamFC model (best.pt) to track detected persons
         across frames via cross-correlation. This is precise and does NOT confuse objects.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)

from src.face_matcher import FaceMatcher
from src.sot_model import SiamFC, load_sot_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACES_DIR = PROJECT_ROOT / "captured_faces"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "sot" / "best.pt"


class TrackedPerson:
    """State for one actively tracked person."""

    def __init__(self, person_id: int, bbox: tuple[int, int, int, int], template: np.ndarray) -> None:
        self.person_id = person_id
        self.bbox = bbox           # (x, y, w, h)
        self.template = template   # RGB crop used as SiamFC template (127x127)
        self.lost_frames = 0       # consecutive frames where tracking score was low
        self.unmatched_cycles = 0  # consecutive detection cycles without detector confirmation
        self.is_authorized = False
        self.person_name = "Unknown"
        self.match_score = 0.0

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2


class AutomaticPersonTracker:
    """Hybrid Detect + SiamFC Track pipeline that uses YOUR trained best.pt model."""

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        tracking_score_threshold: float = 0.45,
        detect_every_n_frames: int = 3,
        save_faces_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        known_dir: str | Path | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.tracking_score_threshold = tracking_score_threshold
        self.detect_every_n_frames = detect_every_n_frames
        self.device = torch.device(device)
        self.save_faces_dir = Path(save_faces_dir) if save_faces_dir else DEFAULT_FACES_DIR
        self.save_faces_dir.mkdir(parents=True, exist_ok=True)

        self.frame_count = 0
        self.next_person_id = 1
        self.tracked_persons: list[TrackedPerson] = []
        self.saved_person_ids: set[int] = set()

        # ---- Load Face Verification Matcher ----
        self.face_matcher = FaceMatcher(known_dir=known_dir, device=device)

        # ---- Load YOUR trained SiamFC model (best.pt) ----
        ckpt = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self.has_siamfc = False
        if ckpt.is_file():
            try:
                self.siamfc = load_sot_model(str(ckpt), device=self.device)
                self.siamfc.eval()
                self.has_siamfc = True
                print(f"[SiamFC] Loaded trained model from: {ckpt}")
            except Exception as e:
                print(f"[SiamFC] Could not load checkpoint: {e}")
        else:
            print(f"[SiamFC] Checkpoint not found at {ckpt}, tracking will use detection only.")

        # ---- Load Faster R-CNN for initial person detection ----
        self.has_detector = False
        try:
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            self.detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
            self.detector.to(self.device)
            self.detector.eval()
            self.has_detector = True
            print(f"[Detector] Faster R-CNN loaded on {self.device}")
        except Exception as e:
            print(f"[Detector] Could not load: {e}")

    # ------------------------------------------------------------------ #
    #                       DETECTION (Phase 1)                           #
    # ------------------------------------------------------------------ #

    def _detect_persons(self, frame_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Run Faster R-CNN to detect persons. Filter out small background patches."""
        if not self.has_detector:
            return []

        h, w = frame_rgb.shape[:2]
        min_bw = max(45, int(w * 0.08))  # Minimum realistic person box width (~50px)
        min_bh = max(65, int(h * 0.12))  # Minimum realistic person box height (~75px)

        tensor = (
            torch.from_numpy(frame_rgb)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self.device)
            / 255.0
        )

        with torch.no_grad():
            preds = self.detector(tensor)[0]

        boxes = preds["boxes"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()

        person_boxes = []
        for box, label, score in zip(boxes, labels, scores):
            if label == 1 and score >= self.confidence_threshold:
                x1, y1, x2, y2 = [int(v) for v in box]
                bx, by = max(0, x1), max(0, y1)
                bw = min(w - bx, x2 - x1)
                bh = min(h - by, y2 - y1)
                # Only keep realistic size person boxes
                if bw >= min_bw and bh >= min_bh:
                    person_boxes.append((bx, by, bw, bh))

        return person_boxes

    # ------------------------------------------------------------------ #
    #                       TRACKING (Phase 2 — SiamFC best.pt)          #
    # ------------------------------------------------------------------ #

    def _crop_template(self, frame_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Extract 127x127 template crop centered on bbox for SiamFC."""
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        fh, fw = frame_rgb.shape[:2]

        # Context padding (SiamFC standard: half-perimeter)
        context = (w + h) / 2
        sz = int(np.sqrt((w + context) * (h + context)))

        x1 = max(0, cx - sz // 2)
        y1 = max(0, cy - sz // 2)
        x2 = min(fw, cx + sz // 2)
        y2 = min(fh, cy + sz // 2)

        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame_rgb[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]

        return cv2.resize(crop, (127, 127))

    def _crop_search(self, frame_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, int, int, int]:
        """Extract 255x255 search region around last known bbox for SiamFC."""
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        fh, fw = frame_rgb.shape[:2]

        context = (w + h) / 2
        sz = int(np.sqrt((w + context) * (h + context)) * 2)

        x1 = max(0, cx - sz // 2)
        y1 = max(0, cy - sz // 2)
        x2 = min(fw, cx + sz // 2)
        y2 = min(fh, cy + sz // 2)

        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame_rgb[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]

        search_img = cv2.resize(crop, (255, 255))
        return search_img, x1, y1, sz

    def _siamfc_track(
        self, frame_rgb: np.ndarray, person: TrackedPerson
    ) -> tuple[tuple[int, int, int, int], float]:
        """Use trained SiamFC model to track a person across a frame. Returns updated bbox and score."""
        template_rgb = person.template
        search_rgb, search_x1, search_y1, search_sz = self._crop_search(frame_rgb, person.bbox)

        # Convert to tensors
        z = torch.from_numpy(template_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        x = torch.from_numpy(search_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

        with torch.no_grad():
            score_map = self.siamfc(z, x)  # (1, 1, 17, 17)

        score_map = score_map.squeeze().cpu().numpy()  # (17, 17)
        max_score = float(score_map.max())

        # Find peak location
        peak_y, peak_x = np.unravel_index(score_map.argmax(), score_map.shape)

        # Map peak back to frame coordinates
        fh, fw = frame_rgb.shape[:2]
        _, _, old_w, old_h = person.bbox

        # Peak relative to search center (17x17 grid)
        offset_x = (peak_x - 8) / 17.0 * search_sz
        offset_y = (peak_y - 8) / 17.0 * search_sz

        old_cx, old_cy = person.center
        new_cx = int(np.clip(old_cx + offset_x, old_w // 2, fw - old_w // 2))
        new_cy = int(np.clip(old_cy + offset_y, old_h // 2, fh - old_h // 2))

        new_x = max(0, new_cx - old_w // 2)
        new_y = max(0, new_cy - old_h // 2)
        new_w = min(old_w, fw - new_x)
        new_h = min(old_h, fh - new_y)

        return (new_x, new_y, new_w, new_h), max_score

    # ------------------------------------------------------------------ #
    #                       MATCHING & LIFECYCLE                          #
    # ------------------------------------------------------------------ #

    def _iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        """Compute IoU between two (x, y, w, h) boxes."""
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / max(1, union)

    def _save_snapshot(self, frame_rgb: np.ndarray, person_id: int, box: tuple[int, int, int, int]) -> str | None:
        """Automatically crop and save detected person image."""
        x, y, w, h = box
        crop = frame_rgb[y : y + h, x : x + w]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"person_ID{person_id}_{timestamp}.jpg"
        save_path = self.save_faces_dir / filename

        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), crop_bgr)
        return str(save_path)

    # ------------------------------------------------------------------ #
    #                       MAIN FRAME PROCESSING                        #
    # ------------------------------------------------------------------ #

    def process_frame(
        self, frame_rgb: np.ndarray
    ) -> tuple[np.ndarray, list[dict], int, bool]:
        """
        Strict detection-confirmed hybrid pipeline with live Face Verification:
        - Matches against known_faces/ folder:
            - Green Box for Authorized Matched Persons
            - Red Box + Security Alert for Unknown/Unauthorized Persons
        """
        self.frame_count += 1

        # ------ Phase 2: SiamFC tracking on existing tracked persons ------
        if self.has_siamfc and self.tracked_persons:
            alive = []
            for person in self.tracked_persons:
                updated_bbox, track_score = self._siamfc_track(frame_rgb, person)

                if track_score >= self.tracking_score_threshold:
                    person.bbox = updated_bbox
                    person.lost_frames = 0
                    alive.append(person)
                else:
                    # Drop immediately if tracking score is low
                    person.lost_frames += 1

            self.tracked_persons = alive

        # ------ Phase 1: Run detector periodically to discover & validate tracks ------
        run_detection = (
            self.frame_count % self.detect_every_n_frames == 1
            or len(self.tracked_persons) == 0
        )

        if run_detection:
            det_boxes = self._detect_persons(frame_rgb)
            matched_track_ids = set()

            for det_box in det_boxes:
                # Check if this detection overlaps an already tracked person
                best_match = None
                best_iou = 0.25  # minimum IoU threshold
                for p in self.tracked_persons:
                    iou_val = self._iou(det_box, p.bbox)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_match = p

                if best_match is not None:
                    # Update existing track with fresh bounding box & fresh template crop
                    best_match.bbox = det_box
                    best_match.template = self._crop_template(frame_rgb, det_box)
                    best_match.lost_frames = 0
                    best_match.unmatched_cycles = 0
                    matched_track_ids.add(best_match.person_id)
                else:
                    # New person detected! Create a tracked person with SiamFC template
                    template = self._crop_template(frame_rgb, det_box)
                    pid = self.next_person_id
                    self.next_person_id += 1
                    person = TrackedPerson(pid, det_box, template)
                    self.tracked_persons.append(person)
                    matched_track_ids.add(pid)

            # Purge any track that was NOT matched by the detector on this detection cycle
            surviving_tracks = []
            for p in self.tracked_persons:
                if p.person_id in matched_track_ids:
                    p.unmatched_cycles = 0
                    surviving_tracks.append(p)
                else:
                    p.unmatched_cycles += 1
                    # Instantly purge if unconfirmed by detector
                    if p.unmatched_cycles < 1:
                        surviving_tracks.append(p)

            self.tracked_persons = surviving_tracks

        # ------ Face Verification & Render Bounding Boxes ------
        annotated_img = frame_rgb.copy()
        updated_tracks = []
        has_unauthorized = False
        fh, fw = frame_rgb.shape[:2]

        for person in self.tracked_persons:
            if person.lost_frames > 0 or person.unmatched_cycles > 0:
                continue

            x, y, w, h = person.bbox
            cx, cy = person.center
            pid = person.person_id

            # Crop face/person for verification
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(fw, x + w), min(fh, y + h)
            crop = frame_rgb[y1:y2, x1:x2]

            # Verify against known_faces/
            if crop.size > 0:
                is_auth, name, score = self.face_matcher.verify_crop(crop)
                person.is_authorized = is_auth
                person.person_name = name
                person.match_score = score

            # Box Color: Green (0, 255, 0) if Authorized | Red (255, 0, 0) in RGB if Unauthorized
            if person.is_authorized:
                color_rgb = (0, 255, 0)    # Green
                accent_rgb = (0, 255, 255)  # Cyan
                label_str = f"MATCH: {person.person_name}"
            else:
                color_rgb = (255, 0, 0)    # Red in RGB
                accent_rgb = (255, 165, 0) # Orange
                label_str = "UNAUTHORIZED / UNKNOWN"
                has_unauthorized = True

            # Main bounding box
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color_rgb, 3)

            # Corner accents
            line_len = min(w, h) // 4
            for (cx1, cy1), (cx2, cy2) in [
                ((x, y), (x + line_len, y)), ((x, y), (x, y + line_len)),
                ((x + w, y), (x + w - line_len, y)), ((x + w, y), (x + w, y + line_len)),
                ((x, y + h), (x + line_len, y + h)), ((x, y + h), (x, y + h - line_len)),
                ((x + w, y + h), (x + w - line_len, y + h)), ((x + w, y + h), (x + w, y + h - line_len)),
            ]:
                cv2.line(annotated_img, (cx1, cy1), (cx2, cy2), accent_rgb, 3)

            # Center dot
            cv2.circle(annotated_img, (cx, cy), 5, (255, 255, 0), -1)

            # Status label above box
            cv2.putText(
                annotated_img, label_str, (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rgb, 2,
            )

            updated_tracks.append({
                "id": pid,
                "box": [x, y, w, h],
                "center": (cx, cy),
                "authorized": person.is_authorized,
                "name": person.person_name,
            })

        person_count = len(updated_tracks)
        return annotated_img, updated_tracks, person_count, has_unauthorized
