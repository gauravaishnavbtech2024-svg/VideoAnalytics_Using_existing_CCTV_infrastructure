"""
Extract face images from InsightFace RecordIO format to folder structure.

Converts train.rec + train.idx → folders of 112x112 .jpg images organized by identity.
This makes training MUCH faster since PyTorch ImageFolder can read files directly
instead of doing random seeks into a 2.7GB binary.

Usage:
    python scripts/extract_faces.py
    python scripts/extract_faces.py --max-samples 10000
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import time
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets" / "faces" / "faces_webface_112x112"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "faces" / "extracted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract faces from RecordIO to folder structure")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR), help="Path to RecordIO dataset directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of extracted images")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-resolve path if typos in hyphens vs underscores (e.g., ms1m_retinaface_t1 -> ms1m-retinaface-t1)
    if not dataset_dir.is_dir():
        alt_name = dataset_dir.name.replace("_", "-") if "_" in dataset_dir.name else dataset_dir.name.replace("-", "_")
        alt_path = dataset_dir.parent / alt_name
        if alt_path.is_dir():
            dataset_dir = alt_path

    rec_path = dataset_dir / "train.rec"
    idx_path = dataset_dir / "train.idx"
    prop_path = dataset_dir / "property"

    if not rec_path.is_file():
        print(f"ERROR: train.rec not found at {rec_path}")
        print(f"Hint: Check if your dataset folder name matches correctly under datasets/faces/.")
        return

    # Read property
    num_classes = 10572
    if prop_path.is_file():
        num_classes = int(prop_path.read_text().strip().split(",")[0])

    # Read index file
    print("[1/3] Reading index file...")
    offsets = []
    with open(idx_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                offsets.append(int(parts[1]))

    total_records = len(offsets)
    # Skip header records (first num_classes+1 entries are metadata)
    start_idx = num_classes + 1
    end_idx = total_records
    if args.max_samples:
        end_idx = min(total_records, start_idx + args.max_samples)

    num_to_extract = end_idx - start_idx
    print(f"[2/3] Extracting {num_to_extract} face images from RecordIO...")
    print(f"  Source  : {rec_path}")
    print(f"  Output  : {output_dir}")
    print(f"  Classes : {num_classes}")

    extracted = 0
    errors = 0
    t0 = time.time()

    # Use in-memory dictionary for image counting to avoid disk globbing bottleneck
    img_counters: dict[int, int] = {}

    with open(rec_path, "rb") as rec_file:
        for i in range(start_idx, end_idx):
            offset = offsets[i]
            rec_file.seek(offset)

            # Read RecordIO header
            header = rec_file.read(8)
            if len(header) < 8:
                errors += 1
                continue

            magic_and_lflag = struct.unpack("<I", header[:4])[0]
            cflag = magic_and_lflag >> 29
            length = magic_and_lflag & ((1 << 29) - 1)

            if cflag == 0:
                data = rec_file.read(length)
            elif cflag == 1:
                data = rec_file.read(length)
                while True:
                    next_header = rec_file.read(8)
                    if len(next_header) < 8:
                        break
                    next_magic = struct.unpack("<I", next_header[:4])[0]
                    next_cflag = next_magic >> 29
                    next_length = next_magic & ((1 << 29) - 1)
                    data += rec_file.read(next_length)
                    if next_cflag == 2:
                        break
            else:
                data = rec_file.read(length)

            if len(data) < 24:
                errors += 1
                continue

            # Parse InsightFace header
            header_flag = struct.unpack("<f", data[0:4])[0]
            header_label = struct.unpack("<f", data[4:8])[0]

            if header_flag == 2.0:  # metadata record
                continue

            label = int(header_label)
            img_data = data[24:]

            if len(img_data) < 100:
                errors += 1
                continue

            try:
                # Fast direct binary write if raw bytes are valid JPEG, or PIL fallback
                label_dir = output_dir / str(label)
                label_dir.mkdir(exist_ok=True)

                count = img_counters.get(label, 0)
                img_counters[label] = count + 1
                img_path = label_dir / f"img_{count:05d}.jpg"

                if img_data.startswith(b"\xff\xd8"):  # Valid JPEG header
                    with open(img_path, "wb") as img_f:
                        img_f.write(img_data)
                else:
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    img.save(img_path, "JPEG", quality=95)

                extracted += 1

                if extracted % 10000 == 0:
                    elapsed = time.time() - t0
                    rate = extracted / max(0.01, elapsed)
                    print(f"  Progress: {extracted:,}/{num_to_extract:,} ({100*extracted/num_to_extract:.1f}%) | {rate:.0f} img/s")

            except Exception:
                errors += 1
                continue

    elapsed = time.time() - t0
    print(f"\n[3/3] Extraction complete!")
    print(f"  Extracted : {extracted} images")
    print(f"  Errors    : {errors}")
    print(f"  Time      : {elapsed:.1f}s")
    print(f"  Output    : {output_dir}")


if __name__ == "__main__":
    main()
