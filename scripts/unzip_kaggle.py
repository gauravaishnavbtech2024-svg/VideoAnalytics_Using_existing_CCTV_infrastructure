"""
Helper script to extract Kaggle face dataset zip files into the project dataset directory.

Usage:
    python scripts/unzip_kaggle.py --zip-path "C:/path/to/archive.zip"
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "faces" / "kaggle_faces"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unzip Kaggle Face Dataset")
    parser.add_argument("--zip-path", type=str, required=True, help="Path to downloaded archive.zip file")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Destination extraction folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path)
    output_dir = Path(args.output_dir)

    if not zip_path.is_file():
        print(f"[ERROR] Zip file not found at: {zip_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📦 Extracting {zip_path.name} to {output_dir}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    # Count extracted images and classes
    all_images = list(output_dir.rglob("*.jpg")) + list(output_dir.rglob("*.png")) + list(output_dir.rglob("*.jpeg"))
    identity_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

    print("\n" + "=" * 60)
    print("✅ Unzip Complete!")
    print(f"  Extracted Folder: {output_dir}")
    print(f"  Total Images    : {len(all_images):,}")
    print(f"  Identities      : {len(identity_dirs):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
