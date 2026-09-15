"""
Merge 5 Roboflow Human Detection Datasets and Train YOLOv8 Human Tracker.
========================================================================

Fixes:
1. Label Normalization: Maps all human labels/classes to standard class 0.
2. Polygon/Segment Sanitization: Converts any non-standard polygons into clean bounding boxes.
3. Stable Optimizer: Uses standard 'SGD' optimizer instead of experimental Muon to prevent PyTorch tensor view errors.
4. Cleans stale *.cache files to prevent Windows file locking issues.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_BASE = Path(r"C:\Users\asus\OneDrive\Desktop\datasetsss\human detection dataset")
TARGET_DIR = PROJECT_ROOT / "datasets" / "human_detection_master"


def clean_label_line(line: str) -> str | None:
    """Normalize label line to standard YOLO bounding box: '0 xc yc w h'."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        # Standard bounding box format: class xc yc w h
        if len(parts) == 5:
            xc, yc, w, h = [float(v) for v in parts[1:5]]
            if 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
                return f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
        else:
            # Polygon coordinates: class x1 y1 x2 y2 x3 y3 ...
            coords = [float(v) for v in parts[1:]]
            xs = coords[0::2]
            ys = coords[1::2]
            if xs and ys:
                min_x = max(0.0, min(xs))
                max_x = min(1.0, max(xs))
                min_y = max(0.0, min(ys))
                max_y = min(1.0, max(ys))
                bw = max_x - min_x
                bh = max_y - min_y
                xc = min_x + bw / 2.0
                yc = min_y + bh / 2.0
                if bw > 0.005 and bh > 0.005:
                    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
    except Exception:
        return None
    return None


def merge_datasets(source_base: Path, target_dir: Path) -> Path:
    """Merge and clean 5 datasets into unified master dataset."""
    target_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]
    for split in splits:
        (target_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (target_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Remove any stale cache files
        for cache_f in (target_dir / split).glob("*.cache"):
            try:
                cache_f.unlink()
            except Exception:
                pass

    dataset_dirs = [d for d in source_base.iterdir() if d.is_dir()]
    print("=" * 65)
    print("🚀 Merging & Normalizing 5 Human Datasets into Master Dataset")
    print("=" * 65)
    print(f"  Source Root  : {source_base}")
    print(f"  Target Root  : {target_dir}")
    print(f"  Datasets     : {len(dataset_dirs)}")
    print("-" * 65)

    stats = {s: 0 for s in splits}

    for idx, d_path in enumerate(dataset_dirs, start=1):
        prefix = f"ds{idx}_"
        print(f"\n[{idx}/{len(dataset_dirs)}] Processing: {d_path.name}")

        for split in splits:
            img_src = d_path / split / "images"
            lbl_src = d_path / split / "labels"

            if not img_src.is_dir():
                continue

            img_dst = target_dir / split / "images"
            lbl_dst = target_dir / split / "labels"

            images = list(img_src.glob("*.*"))
            for img_file in tqdm(images, desc=f"  Copying & Normalizing {split}"):
                ext = img_file.suffix.lower()
                if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    continue

                new_stem = f"{prefix}{img_file.stem}"
                new_img_name = f"{new_stem}{ext}"
                shutil.copy2(img_file, img_dst / new_img_name)

                lbl_file = lbl_src / f"{img_file.stem}.txt"
                if lbl_file.is_file():
                    with open(lbl_file, "r", encoding="utf-8") as rf:
                        lines = rf.readlines()

                    clean_lines = []
                    for line in lines:
                        cl = clean_label_line(line)
                        if cl:
                            clean_lines.append(cl)

                    if clean_lines:
                        with open(lbl_dst / f"{new_stem}.txt", "w", encoding="utf-8") as wf:
                            wf.writelines(clean_lines)

                stats[split] += 1

    # Write master data.yaml
    master_yaml = target_dir / "data.yaml"
    with open(master_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {target_dir.resolve().as_posix()}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n\n")
        f.write("nc: 1\n")
        f.write("names: ['Human']\n")

    print("\n" + "=" * 65)
    print("✅ Master Dataset Ready & Normalized!")
    print(f"  Train Images : {stats['train']:,}")
    print(f"  Valid Images : {stats['valid']:,}")
    print(f"  Test Images  : {stats['test']:,}")
    print(f"  Data Config  : {master_yaml}")
    print("=" * 65)

    return master_yaml


def train_yolo(data_yaml: Path, epochs: int = 35, batch: int = 8, device: str = "0", model_name: str = "yolov8s.pt") -> None:
    """Train YOLOv8 on master dataset with stable SGD optimizer and AMP."""
    import torch
    from ultralytics import YOLO

    print("\n" + "=" * 65)
    print("🚀 Starting Human Detection Training on Master Dataset")
    print("=" * 65)
    print(f"  Model Backbone : {model_name}")
    print(f"  Dataset Config : {data_yaml}")
    print(f"  Epochs         : {epochs}")
    print(f"  Batch Size     : {batch}")
    print(f"  Optimizer      : SGD (Stable Standard)")
    print(f"  Device         : {device} (CUDA: {torch.cuda.is_available()})")
    print("-" * 65)

    model = YOLO(model_name)
    output_project = PROJECT_ROOT / "models" / "human_detection"

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=640,
        device=device,
        optimizer="SGD",
        lr0=0.01,
        workers=4,
        project=str(output_project),
        name="master_human_model",
        exist_ok=True,
        verbose=True,
    )

    best_pt = output_project / "master_human_model" / "weights" / "best.pt"
    if best_pt.is_file():
        dest = output_project / "best.pt"
        shutil.copy2(best_pt, dest)
        print("\n" + "=" * 65)
        print("🎉 Training Complete!")
        print(f"  Best Weights Saved To: {dest}")
        print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge 5 Human Datasets & Train YOLOv8")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="GPU index or cpu")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Pretrained YOLO model (yolov8s.pt, yolov8n.pt)")
    args = parser.parse_args()

    yaml_file = merge_datasets(SOURCE_BASE, TARGET_DIR)
    train_yolo(yaml_file, epochs=args.epochs, batch=args.batch, device=args.device, model_name=args.model)
