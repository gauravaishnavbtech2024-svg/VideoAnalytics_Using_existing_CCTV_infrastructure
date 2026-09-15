"""
Face Matcher & Security Verification Module.

Loads reference images from known_faces/ folder and compares live face crops
using 128-d feature embeddings.

Returns:
- is_authorized: True (Authorized/Matched) or False (Unknown/Unauthorized)
- person_name: Name of matched person or 'Unknown'
- match_score: Cosine similarity score
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.face_model import FaceDetectorNet, load_face_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWN_DIR = PROJECT_ROOT / "known_faces"
DEFAULT_FACE_MODEL = PROJECT_ROOT / "models" / "face" / "best.pt"


class FaceMatcher:
    """Matches live face crops against a folder of known person images."""

    def __init__(
        self,
        known_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        similarity_threshold: float = 0.82,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.known_dir = Path(known_dir) if known_dir else DEFAULT_KNOWN_DIR
        self.known_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.device = torch.device(device)

        # OpenCV Face Detector for tight face crops
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.face_model: FaceDetectorNet | None = None
        self.known_embeddings: list[torch.Tensor] = []
        self.known_names: list[str] = []

        # Load FaceDetectorNet model
        m_path = Path(model_path) if model_path else DEFAULT_FACE_MODEL
        if m_path.is_file():
            try:
                self.face_model = load_face_model(str(m_path), device=self.device)
                print(f"[FaceMatcher] Loaded trained face model: {m_path}")
            except Exception as e:
                print(f"[FaceMatcher] Could not load face checkpoint {m_path}: {e}")

        if self.face_model is None:
            self.face_model = FaceDetectorNet().to(self.device)
            self.face_model.eval()

        self._last_known_files: dict[str, float] = {}
        self.reload_known_faces()

    def check_and_auto_reload(self) -> bool:
        """Automatically re-scans known_faces/ folder if files are added, modified, or removed."""
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        current_files = {
            p.name: p.stat().st_mtime
            for p in self.known_dir.glob("*")
            if p.suffix.lower() in valid_exts
        }
        if current_files != self._last_known_files:
            print(f"\n[FaceMatcher] 🔄 Detected file changes in '{self.known_dir.name}/' -> Updating database...")
            self.reload_known_faces()
            return True
        return False

    def reload_known_faces(self) -> int:
        """
        Scan known_faces/ folder and generate robust multi-view reference feature vectors
        (angles, flips, lighting variations) for each registered face.
        """
        self.known_embeddings = []
        self.known_names = []

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = [p for p in self.known_dir.glob("*") if p.suffix.lower() in valid_exts]

        # Update cache
        self._last_known_files = {
            p.name: p.stat().st_mtime for p in image_paths
        }

        for img_p in image_paths:
            try:
                img_bgr = cv2.imread(str(img_p))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                name = img_p.stem.replace("_", " ").title()

                # Extract base embedding
                emb = self.extract_embedding(img_rgb)
                if emb is not None:
                    self.known_embeddings.append(emb)
                    self.known_names.append(name)

                # Generate multi-view variations (slight rotations & flip) for robust live video matching
                h, w = img_rgb.shape[:2]
                for angle in [-6, -3, 3, 6]:
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                    rot = cv2.warpAffine(img_rgb, M, (w, h))
                    for flip in [False, True]:
                        var_img = cv2.flip(rot, 1) if flip else rot
                        var_emb = self.extract_embedding(var_img)
                        if var_emb is not None:
                            self.known_embeddings.append(var_emb)
                            self.known_names.append(name)

                print(f"  [Registered Known Face]: {name} (with multi-view templates)")
            except Exception as e:
                print(f"[FaceMatcher] Error reading {img_p.name}: {e}")

        unique_count = len(set(self.known_names))
        print(f"[FaceMatcher] Total authorized identities registered: {unique_count}")
        return unique_count

    def _crop_tight_face(self, image_rgb: np.ndarray) -> np.ndarray:
        """Locate exact facial region using OpenCV Haar Cascade or fallback to image."""
        if image_rgb.size == 0:
            return image_rgb

        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        # Run face cascade
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(35, 35)
        )

        if len(faces) > 0:
            # Pick largest detected face
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            # Add 10% margin around face
            margin_x = int(fw * 0.10)
            margin_y = int(fh * 0.10)
            x1 = max(0, fx - margin_x)
            y1 = max(0, fy - margin_y)
            x2 = min(w, fx + fw + margin_x)
            y2 = min(h, fy + fh + margin_y)
            return image_rgb[y1:y2, x1:x2]

        # If already a face crop, use it directly
        return image_rgb

    def extract_embedding(self, crop_rgb: np.ndarray) -> torch.Tensor | None:
        """Extract normalized 128-d face embedding from tight face crop."""
        if self.face_model is None or crop_rgb.size == 0:
            return None

        try:
            face_crop = self._crop_tight_face(crop_rgb)
            if face_crop.size == 0:
                face_crop = crop_rgb

            crop_resized = cv2.resize(face_crop, (112, 112))
            tensor = (
                torch.from_numpy(crop_resized)
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                .to(self.device)
                / 255.0
            )
            tensor = (tensor - 0.5) / 0.5
            with torch.no_grad():
                emb = self.face_model.get_embedding(tensor)
            return emb[0]
        except Exception:
            return None

    def verify_crop(self, crop_rgb: np.ndarray) -> tuple[bool, str, float]:
        """
        Verify live crop against known person database.
        Returns: (is_authorized, person_name, match_distance)
        """
        if not self.known_embeddings:
            return False, "Unknown", 99.0

        emb = self.extract_embedding(crop_rgb)
        if emb is None:
            return False, "Unknown", 99.0

        known_matrix = torch.stack(self.known_embeddings)  # (N, 128)
        # Compute L2 Euclidean Distance across 128-d embeddings
        distances = torch.norm(known_matrix - emb, dim=1)  # (N,)

        min_dist, best_idx = torch.min(distances, dim=0)
        best_dist = float(min_dist)

        # Balanced distance threshold:
        # Same Person with pose/lighting variations: dist ~ 0.20 - 0.55
        # Different Person: dist ~ 1.10 - 1.50
        max_l2_threshold = 0.70
        if best_dist <= max_l2_threshold:
            matched_name = self.known_names[int(best_idx)]
            return True, matched_name, best_dist

        return False, "Unknown", best_dist
