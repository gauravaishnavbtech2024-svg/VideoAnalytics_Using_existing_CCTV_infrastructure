"""
Fast PyTorch Dataset for extracted WebFace images (folder format).

Reads from datasets/faces/extracted/ where images are organized as:
    extracted/
      0/       ← Identity 0
        img_00000.jpg
        img_00001.jpg
      1/       ← Identity 1
        img_00000.jpg
      ...
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


EXTRACTED_DIR = Path(__file__).resolve().parents[1] / "datasets" / "faces" / "extracted"


class WebFaceFolderDataset(Dataset):
    """
    Fast PyTorch Dataset supporting single or multiple face dataset folders
    (e.g., CASIA-WebFace + MS1M-RetinaFace-t1).

    Returns:
    - face_image: (3, 112, 112) normalized tensor
    - label: integer identity label (mapped across all datasets)
    """

    def __init__(
        self,
        roots: str | Path | list[str | Path] | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        if roots is None:
            roots = [EXTRACTED_DIR]
        elif isinstance(roots, (str, Path)):
            roots = [Path(roots)]
        else:
            roots = [Path(r) for r in roots]

        self.roots = [r for r in roots if r.is_dir()]

        if not self.roots:
            raise FileNotFoundError(
                f"No valid face dataset directories found in: {roots}.\n"
                f"Extract dataset images using 'python scripts/extract_faces.py' first."
            )

        self.samples: list[tuple[Path, int]] = []
        global_class_idx = 0
        name_to_class: dict[str, int] = {}

        # Scan each dataset root and build unique global class mappings
        for root in self.roots:
            # Check for subfolder-based identity structure (e.g. Original Images/Person Name/...)
            sub_dirs = [d for d in root.rglob("*") if d.is_dir() and len(list(d.glob("*.jpg")) + list(d.glob("*.png"))) > 0]
            
            if sub_dirs:
                for d in sorted(sub_dirs, key=lambda p: p.name):
                    cname = d.name
                    if cname not in name_to_class:
                        name_to_class[cname] = global_class_idx
                        global_class_idx += 1
                    c_idx = name_to_class[cname]
                    for img_p in sorted(list(d.glob("*.jpg")) + list(d.glob("*.png"))):
                        self.samples.append((img_p, c_idx))
            else:
                # Flat files with Name_0.jpg naming (e.g., Faces/Faces/Person Name_0.jpg)
                flat_images = sorted(list(root.rglob("*.jpg")) + list(root.rglob("*.png")))
                for img_p in flat_images:
                    stem = img_p.stem
                    cname = stem.rsplit("_", 1)[0] if "_" in stem else stem
                    if cname not in name_to_class:
                        name_to_class[cname] = global_class_idx
                        global_class_idx += 1
                    c_idx = name_to_class[cname]
                    self.samples.append((img_p, c_idx))

        self.num_classes = max(1, global_class_idx)

        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        print(f"[MultiFaceDataset] Loaded {len(self.samples):,} images across {self.num_classes:,} identities from {len(self.roots)} dataset sources.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if img.size != (112, 112):
            img = img.resize((112, 112), Image.BILINEAR)

        return self.transform(img), label

