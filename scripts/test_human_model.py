"""
Test and Evaluate Trained Human Detection Model on Test Datasets.
================================================================

Features:
1. Full Benchmark Metric Evaluation (mAP50, mAP50-95, Precision, Recall) on test/ split.
2. Visual Inference on sample test images (saves annotated visual predictions to runs/test_predictions/).

Usage:
    python scripts/test_human_model.py --weights models/human_detection/best.pt --num-samples 10
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "human_detection" / "best.pt"
DEFAULT_DATA_YAML = PROJECT_ROOT / "datasets" / "human_detection_master" / "data.yaml"
OUTPUT_PRED_DIR = PROJECT_ROOT / "runs" / "test_predictions"


def main():
    parser = argparse.ArgumentParser(description="Test and Evaluate Trained Human Detection Model")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Path to trained best.pt")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML), help="Path to data.yaml")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of visual test predictions to save")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    data_path = Path(args.data)

    if not weights_path.is_file():
        print(f"❌ Error: Weights file not found at: {weights_path}")
        print("Please wait for training to finish or provide a valid checkpoint.")
        return

    print("=" * 65)
    print("🎯 Evaluating Trained Human Detection Model on Test Dataset")
    print("=" * 65)
    print(f"  Weights Checkpoint : {weights_path}")
    print(f"  Dataset Config     : {data_path}")
    print(f"  Device             : {args.device}")
    print("-" * 65)

    model = YOLO(str(weights_path))

    # 1. Run Benchmark Validation on the Test Split
    print("\n[1/2] Computing Quantitative Metrics on Test Split...")
    metrics = model.val(
        data=str(data_path),
        split="test",
        device=args.device,
        verbose=True,
    )

    print("\n" + "-" * 65)
    print("📊 Test Benchmark Results:")
    print(f"  Precision (P)   : {metrics.box.p[0] * 100:.1f}%")
    print(f"  Recall (R)      : {metrics.box.r[0] * 100:.1f}%")
    print(f"  mAP @ 0.50      : {metrics.box.map50 * 100:.1f}%")
    print(f"  mAP @ 0.50:0.95 : {metrics.box.map * 100:.1f}%")
    print("-" * 65)

    # 2. Visual Inference on Test Images
    print("\n[2/2] Generating Visual Annotated Predictions...")
    test_img_dir = data_path.parent / "test" / "images"
    if test_img_dir.is_dir():
        img_files = list(test_img_dir.glob("*.*"))
        if img_files:
            OUTPUT_PRED_DIR.mkdir(parents=True, exist_ok=True)
            samples = random.sample(img_files, min(args.num_samples, len(img_files)))

            for idx, img_p in enumerate(samples, start=1):
                results = model.predict(
                    source=str(img_p),
                    conf=0.35,
                    device=args.device,
                    verbose=False,
                )
                if results and len(results) > 0:
                    annotated_bgr = results[0].plot()
                    out_path = OUTPUT_PRED_DIR / f"test_pred_{idx}_{img_p.name}"
                    cv2.imwrite(str(out_path), annotated_bgr)

            print(f"  ✅ Saved {len(samples)} annotated test images to: {OUTPUT_PRED_DIR}")

    print("\n" + "=" * 65)
    print("🎉 Test Evaluation Complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
