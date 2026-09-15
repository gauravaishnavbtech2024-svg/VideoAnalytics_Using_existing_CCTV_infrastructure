"""
Standalone Interactive Real-Time Person Detector & Tracker with Camera Switcher & Face Snapshot Saver.

Features:
- Camera index selector dropdown/prompt (USB Webcam Camera 0, 1, 2).
- Automatic face/person snapshot capture saved to `captured_faces/`.

Run from project root:
    python scripts/track_webcam.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

import os

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.person_detector import AutomaticPersonTracker


def find_available_cameras(max_tested: int = 3) -> list[int]:
    """Return list of active camera indices available on the system."""
    available = []
    for i in range(max_tested):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available.append(i)
                        cap.release()
                        break
                    cap.release()
            except Exception:
                continue
    return available if available else [0]


def open_webcam(camera_id: int = 0) -> cv2.VideoCapture:
    """Try opening specified camera with DirectShow, MSMF, or default backends."""
    backends = [
        (camera_id, cv2.CAP_DSHOW),
        (camera_id, cv2.CAP_MSMF),
        (camera_id, cv2.CAP_ANY),
    ]
    for idx, backend in backends:
        try:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap
                cap.release()
        except Exception:
            continue

    return cv2.VideoCapture(camera_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-Time Person Tracking & Snapshot Saver")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index (e.g. 0 or 1 for USB Webcam)")
    parser.add_argument("--save-dir", type=str, default=str(PROJECT_ROOT / "captured_faces"), help="Folder to save detected faces")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available_cams = find_available_cameras()

    print("=" * 60)
    print("[Automatic Real-Time Person Tracker & Face Snapshot Saver]")
    print("=" * 60)
    print(f"Available Cameras Found : {available_cams}")
    print(f"Active Camera Selected  : Camera {args.camera_id}")
    print(f"Face Snapshots Folder   : {args.save_dir}")
    print("-" * 60)
    print("1. Automatic detection & tracking is active.")
    print("2. Newly detected face/person images are automatically saved to captured_faces/")
    print("3. Press 'q' anytime to exit.\n")

    cap = open_webcam(args.camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}. Available cameras: {available_cams}")
        return

    tracker = AutomaticPersonTracker(save_faces_dir=args.save_dir)
    window_name = f"Automatic Person Tracking - Camera {args.camera_id} (Press Q to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        annotated_rgb, tracks, count, new_saved_paths = tracker.process_frame(frame_rgb, auto_save_snapshots=True)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        for saved_path in new_saved_paths:
            print(f"📸 [Snapshot Saved]: {saved_path}")

        curr_time = time.time()
        fps = 1.0 / max(0.001, curr_time - prev_time)
        prev_time = curr_time

        # Render status overlay
        cv2.putText(
            annotated_bgr,
            f"Cam: {args.camera_id} | FPS: {fps:.1f} | Persons: {count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow(window_name, annotated_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Tracking stopped.")


if __name__ == "__main__":
    main()
