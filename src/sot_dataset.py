"""
VisDrone2019-SOT (Single Object Tracking) PyTorch Dataset.

Extracts Template (Exemplar $127x127$) and Search ($255x255$) image pairs
from VisDrone SOT image sequences and bounding box annotations [x, y, w, h].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VisDroneSOTDataset(Dataset):
    """
    Dataset loader for VisDrone2019 Single Object Tracking dataset.

    Structure expected:
        root_dir/
            sequences/
                uav0000003_00000_s/
                    img0000001.jpg
                    ...
            annotations/
                uav0000003_00000_s.txt
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        examplar_size: int = 127,
        search_size: int = 255,
        score_size: int = 17,
        max_frame_dist: int = 100,
        samples_per_epoch: int = 5000,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.exemplar_size = examplar_size
        self.search_size = search_size
        self.score_size = score_size
        self.max_frame_dist = max_frame_dist
        self.samples_per_epoch = samples_per_epoch

        self.seq_dirs = self._find_sequences()
        if not self.seq_dirs:
            raise FileNotFoundError(
                f"No sequence directories found in {self.dataset_dir / 'sequences'}"
            )

        self.sequences = self._load_all_annotations()

    def _find_sequences(self) -> list[Path]:
        seq_base = self.dataset_dir / "sequences"
        if not seq_base.is_dir():
            return []
        return [p for p in sorted(seq_base.iterdir()) if p.is_dir()]

    def _load_all_annotations(self) -> list[dict]:
        seq_data = []
        annot_base = self.dataset_dir / "annotations"

        for seq_dir in self.seq_dirs:
            seq_name = seq_dir.name
            annot_file = annot_base / f"{seq_name}.txt"
            if not annot_file.is_file():
                continue

            img_files = sorted(seq_dir.glob("*.jpg"))
            if not img_files:
                continue

            boxes = []
            with annot_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [float(p) for p in line.replace(",", " ").split()[:4]]
                    if len(parts) == 4:
                        boxes.append(parts)

            # Match frames with valid bounding boxes (non-zero width & height)
            valid_pairs = []
            for i, (img_p, box) in enumerate(zip(img_files, boxes)):
                if box[2] > 5 and box[3] > 5:
                    valid_pairs.append((img_p, box))

            if len(valid_pairs) > 1:
                seq_data.append(
                    {
                        "name": seq_name,
                        "pairs": valid_pairs,
                    }
                )

        return seq_data

    def _crop_and_resize(
        self, image: np.ndarray, bbox: list[float], out_size: int
    ) -> np.ndarray:
        """Crop search/template region around bbox center and resize to out_size."""
        x, y, w, h = bbox
        cx, cy = x + w / 2.0, y + h / 2.0

        # Context padding factor
        context = 0.5 * (w + h)
        crop_sz = math.sqrt((w + context) * (h + context))

        h_img, w_img = image.shape[:2]

        x1 = int(round(cx - crop_sz / 2.0))
        y1 = int(round(cy - crop_sz / 2.0))
        x2 = int(round(cx + crop_sz / 2.0))
        y2 = int(round(cy + crop_sz / 2.0))

        # Padding boundaries
        pad_x1 = max(0, -x1)
        pad_y1 = max(0, -y1)
        pad_x2 = max(0, x2 - w_img)
        pad_y2 = max(0, y2 - h_img)

        x1_c = max(0, x1)
        y1_c = max(0, y1)
        x2_c = min(w_img, x2)
        y2_c = min(h_img, y2)

        crop = image[y1_c:y2_c, x1_c:x2_c]
        if pad_x1 > 0 or pad_y1 > 0 or pad_x2 > 0 or pad_y2 > 0:
            avg_color = image.mean(axis=(0, 1)).astype(np.uint8)
            crop = cv2.copyMakeBorder(
                crop,
                pad_y1,
                pad_y2,
                pad_x1,
                pad_x2,
                cv2.BORDER_CONSTANT,
                value=avg_color.tolist(),
            )

        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((out_size, out_size, 3), dtype=np.uint8)

        resized = cv2.resize(crop, (out_size, out_size))
        return resized

    def _create_label_map(self) -> torch.Tensor:
        """Create 2D Gaussian response label map (17x17)."""
        r = self.score_size // 2
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        dist = np.sqrt(x**2 + y**2)
        labels = (dist <= 2.0).astype(np.float32)
        return torch.from_numpy(labels).unsqueeze(0)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Pick random sequence
        seq_idx = np.random.randint(0, len(self.sequences))
        seq = self.sequences[seq_idx]
        n_frames = len(seq["pairs"])

        # Pick template frame and search frame
        idx_z = np.random.randint(0, n_frames)
        idx_x_max = min(n_frames - 1, idx_z + self.max_frame_dist)
        idx_x_min = max(0, idx_z - self.max_frame_dist)
        idx_x = np.random.randint(idx_x_min, idx_x_max + 1)

        img_z_path, box_z = seq["pairs"][idx_z]
        img_x_path, box_x = seq["pairs"][idx_x]

        img_z_bgr = cv2.imread(str(img_z_path))
        img_x_bgr = cv2.imread(str(img_x_path))

        if img_z_bgr is None or img_x_bgr is None:
            # Fallback zero tensor
            z_crop = np.zeros((self.exemplar_size, self.exemplar_size, 3), dtype=np.uint8)
            x_crop = np.zeros((self.search_size, self.search_size, 3), dtype=np.uint8)
        else:
            img_z_rgb = cv2.cvtColor(img_z_bgr, cv2.COLOR_BGR2RGB)
            img_x_rgb = cv2.cvtColor(img_x_bgr, cv2.COLOR_BGR2RGB)

            z_crop = self._crop_and_resize(img_z_rgb, box_z, self.exemplar_size)
            x_crop = self._crop_and_resize(img_x_rgb, box_x, self.search_size)

        z_tensor = torch.from_numpy(z_crop).permute(2, 0, 1).float() / 255.0
        x_tensor = torch.from_numpy(x_crop).permute(2, 0, 1).float() / 255.0
        label_map = self._create_label_map()

        return {
            "template": z_tensor,
            "search": x_tensor,
            "label": label_map,
        }
