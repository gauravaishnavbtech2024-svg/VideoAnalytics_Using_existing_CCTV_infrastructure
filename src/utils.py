"""Utility functions for crowd counting."""

from __future__ import annotations

import os
from pathlib import Path


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create directory if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(folder: str | os.PathLike) -> list[Path]:
    """Return sorted list of image paths in a folder (including nested sequence folders)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    files = [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images found in: {folder}")
    return files
