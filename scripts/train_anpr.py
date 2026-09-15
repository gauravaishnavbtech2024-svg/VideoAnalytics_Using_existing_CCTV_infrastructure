"""
Automatic Number Plate Recognition (ANPR) Model Training & Evaluation Script.
=============================================================================

Trains high-precision YOLOv8 on the 'License Plate Detection' dataset
and pairs it with EasyOCR text recognition engine.

Usage:
    python scripts/train_anpr.py --epochs 35 --batch 8 --device 0 --model yolov8s.pt
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "anpr" / "License Plate Detection" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "models" / "anpr"


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Automatic Number Plate Recognition (ANPR)")
    parser.add_argument("--data-yaml", type=str, default=str(DEFAULT_DATASET), help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Base model backbone or checkpoint path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="GPU index or cpu")
    parser.add_argument("--resume", action="store_true", help="Resume exact training from last.pt checkpoint")
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml)
    if not data_yaml.is_file():
        print(f"❌ Error: data.yaml not found at {data_yaml}")
        return

    # Clean old cache files
    dataset_folder = data_yaml.parent
    for c_file in dataset_folder.rglob("*.cache"):
        try:
            c_file.unlink()
        except Exception:
            pass

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for last.pt if user requested resume or if default last.pt exists
    last_ckpt = output_dir / "license_plate_model" / "weights" / "last.pt"
    chosen_model = args.model
    if args.resume and last_ckpt.is_file():
        chosen_model = str(last_ckpt)
    elif args.model == "yolov8s.pt" and last_ckpt.is_file():
        print(f"[ANPR Train] Found previous checkpoint: {last_ckpt.name} (resuming/fine-tuning from previous training)")
        chosen_model = str(last_ckpt)

    print("=" * 65)
    print("Starting YOLO ANPR License Plate Training")
    print("=" * 65)
    print(f"  Dataset        : {dataset_folder.name}")
    print(f"  Config         : {data_yaml}")
    print(f"  Initial Model  : {chosen_model}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Batch Size     : {args.batch}")
    print(f"  Device         : {args.device}")
    print(f"  Resume Exact   : {args.resume}")
    print("-" * 65)

    from ultralytics import YOLO

    model = YOLO(chosen_model)

    if args.resume:
        model.train(resume=True)
    else:
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            optimizer="SGD",
            lr0=0.01,
            workers=4,
            project=str(output_dir),
            name="license_plate_model",
            exist_ok=True,
            verbose=True,
        )

    best_pt = output_dir / "license_plate_model" / "weights" / "best.pt"
    if best_pt.is_file():
        dest = output_dir / "best.pt"
        shutil.copy2(best_pt, dest)
        print("\n" + "=" * 65)
        print("[ANPR Train] ANPR Plate Model Training Complete!")
        print(f"  Best Weights Saved To: {dest}")
        print("=" * 65)


if __name__ == "__main__":
    main()
