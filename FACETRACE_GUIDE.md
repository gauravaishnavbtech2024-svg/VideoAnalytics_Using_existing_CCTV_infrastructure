# 🛡️ FaceTrace — ATM Face Finder (Live Multi-Camera Forensic Console)

A state-of-the-art forensic computer vision desktop application built with **Kivy** and **OpenCV**, running **100% real-time video analytics** across all connected hardware cameras with **STRICTLY ZERO MOCK / DEMO DATA**.

---

## ⚡ 100% Real-Time Architecture & Zero Mock Data Guarantee

| Capability | Implementation in `facetrace_app.py` |
| :--- | :--- |
| **Pre-Trained Deep Learning Models** | Uses your pre-trained **YuNet DNN Face Detector** (`models/face/face_detection_yunet_2023mar.onnx`) and **SFace Biometric Recognizer** (`models/face/face_recognition_sface_2021dec.onnx`) with 5-point facial landmark geometry and 128-d cosine matching. |
| **All Connected Cameras** | Concurrently streams from **Camera 0 (Laptop Integrated Webcam)** and **Camera 1 (External USB ATM Camera)** with dynamic multi-camera DirectShow threads. |
| **Dual & Split-Screen View** | Side-by-side live CCTV console displaying both cameras simultaneously with real-time HUD timestamps, device beacons, and face reticles. |
| **Real Biometric FIQA Scoring** | Computes Sharpness (Laplacian variance), Lighting (Luminance mean/std), Face Angle (Landmark-derived Yaw & Roll), and Resolution directly from the pixel buffer in real time. |
| **Dynamic Multi-Person Tracking** | Identifies and tracks all subjects in view. Automatically links **Person 1 (Active User at keypad)** and tags **Person 2 (Bystander in vestibule)** with proximity alerts. |
| **Live Watchlist Matcher** | Real-time cross-referencing against enrolled suspect photos in [`known_faces/`](known_faces/) (e.g. `gaurav (2).jpg`). Shows "Verified Citizen" if legitimate or triggers a red alert with side-by-side evidence if similarity $\ge 75\%$. |
| **Automated 5-Second Best-5 Capture** | **Self-Triggering Biometric Evaluator**: Whenever a new person appears, waits 5 seconds while evaluating all candidate frames, isolates the **5 highest-FIQA frames**, commits evidence to disk & database automatically, and refreshes the UI without requiring any manual clicks. |
| **Empty-State Compliance** | If no previous transaction data has been saved to disk, the application cleanly displays **0 cases found**. Strictly NO mock transactions or cartoon placeholders. |
| **Evidence Persistence** | "📸 Capture Best Frame Now" automatically isolates the highest-FIQA candidate frame and persists court-admissible evidence to `captures/ATM_LIVE/{date}/{session_id}/`. |
| **1-Click Database Reset** | "🗑️ Clear All Captures" allows clearing all previous disk captures at any time to verify the 100% clean empty state. |
| **Certified Forensic Reports** | "Generate Forensic Report" creates court-ready JSON dossiers with SHA-256 cryptographic hashes of the raw image evidence. |

---

## 🚀 How to Run FaceTrace

### Method 1: Double-Click Launcher
Double-click [`run_facetrace.bat`](run_facetrace.bat) in the root directory.

### Method 2: Command Line (PowerShell)
```powershell
cd e:\cctvvvv
.venv\Scripts\activate
python facetrace_app.py
```

---

## 🎯 Live Controls & Interface Reference

1. **Top Navigation Bar**:
   - `🎥 Live Cameras`: Toggles direct real-time video feeds from all connected cameras.
   - `Case search`: Switches between the live cameras and stored forensic cases.
   - `Camera Dropdown`: Switch seamlessly between `Dual Camera View (Cam 0 + Cam 1)`, `Camera 0 (Laptop)`, and `Camera 1 (USB)`.
   - `🔔 Alert Badge`: Real-time alarm counter that turns red when an active camera detects a watchlist match.
2. **Hero Bar**:
   - `📸 Capture Best Frame Now`: Instantly evaluates candidate frames and records a real transaction session to disk.
   - `🔄 Rescan Disks`: Refreshes disk records and suspect watchlist from the filesystem.
   - `🗑️ Clear All Captures`: Wipes test captures to reset the system to an empty database (0 cases).
3. **Left Panel (Cases List)**:
   - Displays real captured cases or the clean empty state ("0 Cases Recorded").
4. **Center Viewer**:
   - 30 FPS video canvas with orange target reticles `[ ]` and live composite quality score badge.
5. **Right Panel (Forensics Breakdown)**:
   - Real-time Watchlist Status (Verified Citizen vs Red Alert).
   - Real-time Subject Attribution (Person 1 at Keypad vs Bystanders in queue).
   - Live Rolling Candidate Frames Strip (`[72] [78] [85 ★ Best]`).
   - Live Quality Radar (Sharpness, Lighting, Face Angle, Face Visible, Resolution).

---

## ⏱️ How the Automated 5-Second Best-5 Engine Operates

1. **New Person Detection**:
   - The application continuously monitors both cameras with **YuNet DNN**.
   - When a person arrives (or when a new face embedding with SFace cosine similarity $< 0.363$ appears), the state machine switches from `IDLE` to `MONITORING`.
2. **5-Second Real-Time Countdown**:
   - An animated progress bar and countdown banner appear on the live video stream:
     `[ ⏳ AUTO-ANALYZING WITHDRAWAL: X.Xs REMAINING | SELECTING BEST 5 FRAMES... ]`
   - Every 150ms (~33 samples across 5 seconds), candidate face crops and full frames are buffered with full FIQA metrics (Roll, Yaw, Sharpness, Lighting, Resolution, Detector Confidence).
3. **Auto-Selection of Top 5 Best Frames**:
   - At exactly 5.0 seconds, candidate frames are sorted in descending order of composite FIQA score.
   - The **top 5 highest-quality frames** are saved to disk under `captures/ATM_LIVE/{date}/{session_id}/` as:
     - `rank_1_face.jpg` & `rank_1_frame.jpg` (Best Frame)
     - `rank_2_face.jpg` & `rank_2_frame.jpg`
     - `rank_3_face.jpg` & `rank_3_frame.jpg`
     - `rank_4_face.jpg` & `rank_4_frame.jpg`
     - `rank_5_face.jpg` & `rank_5_frame.jpg`
     - `metadata.json` (SHA-256 cryptographic hash, FIQA scores, suspect match result, model info).
4. **Instant Database & UI Refresh**:
   - The case list on the left panel updates immediately (e.g. `1 cases found`).
   - The new transaction card appears at the top.
   - The 5 candidate buttons `[95 ★ Best] [94] [94] [94] [94]` appear on the right panel. Clicking any button previews that exact frame.
5. **No Duplicate Captures**:
   - While the same person remains at the ATM, the system stays in `CAPTURED` state to prevent duplicate spam.
6. **Automatic Reset for Next Customer**:
   - When the person walks away (0 faces detected for $\ge 2.0$ seconds), the system automatically resets to `IDLE` state, ready for the next person.
7. **Forensic Portrait Framing & Center Viewer Toggle**:
   - **Spacious Portrait Framing**: Captures include generous 50% horizontal and 60% vertical margins, capturing the complete head, hair, ears, chin, neck, and upper shoulders rather than an ultra-zoomed face mask.
   - **Dual Center Viewer Modes**: Easily switch between `👤 Portrait` (unzoomed head-and-shoulders biometric dossier) and `📹 Full CCTV` (entire surveillance camera scene) using the buttons at the top of the viewer.

---

## 🚨 1. Automatic Real-Time Suspect Alert System

Whenever any connected camera detects an enrolled suspect from `known_faces/` (e.g., `gaurav (2).jpg`) with biometric similarity $\ge 75\%$:
1. **Non-Intrusive Bottom-Right Corner Notification (`CornerAlertPopup`)**:
   - Automatically docks directly in the **bottom-right corner** of the screen (`pos_hint={'right': 0.99, 'y': 0.02}`).
   - Transparent overlay with zero screen dimming so it **never obstructs the center workspace** or interrupts video analysis.
   - Touch events outside the corner alert pass freely through to the rest of the application.
2. **Audible System Alarm Chime**:
   - Triggers an asynchronous audio chime (`winsound.MessageBeep` / `winsound.Beep`) through the workstation speakers.
3. **Live Surveillance Flashing Banner**:
   - Displays a bright red emergency banner across the live camera feed:
     `CRITICAL ALERT: SUSPECT [NAME] ([CONFIDENCE]%)`
4. **Side-by-Side Biometric Comparison**:
   - Left: Live CCTV face crop captured from the ATM.
   - Center: SFace cosine similarity percentage and confidence badge.
   - Right: Enrolled police watchlist photograph.
5. **Instant Threat Response Actions**:
   - `🚨 Dispatch Cyber Cell`: Logs urgent police dispatch notification and secures the ATM terminal.
   - `🔕 Acknowledge`: Safely acknowledges and dismisses the alert dialog.
6. **Intelligent 12-Second Debounce**:
   - Ensures smooth surveillance workflow without spamming popups while the suspect stays in view.

---

## 🎞️ 2. Forensic Video Clip Biometric Analysis Studio

A dedicated forensic laboratory tool for offline evidence video scanning:

### How to Launch:
- Click **`🎞️ Video Analysis`** in the Top Nav Bar, or
- Click **`🎞️ Scan Video Clip`** in the Hero Bar.

### Studio Features & Workflow:
1. **Target Suspect Selection**:
   - Select any enrolled profile from the dropdown or click **`👤 Browse...`** to pick any image from your computer using the native Windows file dialog.
   - Shows a live thumbnail preview of the target suspect.
2. **Video Clip Source**:
   - Select sample footage or click **`📂 Browse...`** to load any `.mp4`, `.avi`, `.mov`, or `.mkv` surveillance clip.
3. **Configurable Biometric Thresholds**:
   - **Match Threshold**: Default 75% cosine confidence.
   - **Frame Stride**: Every 1 frame (deep frame-by-frame) or every 2/4 frames (high-speed triage).
4. **Instant Match Alert During Scanning**:
   - The system scans frame-by-frame using your pre-trained **YuNet DNN** detector and **SFace** recognizer.
   - **The exact moment a face matches the target suspect**:
     - An **audible alarm chime** rings out.
     - A red alert banner flashes above the video player:
       `🚨 SUSPECT MATCH FOUND AT [MM:SS.S] (XX.X% CONFIDENCE)!`
     - The video canvas draws a red bounding box with the calibrated match percentage.
     - A new card is instantly added to the **Detected Matches Timeline** on the right with the timestamp, frame number, face thumbnail, and a `👁 View` button.
5. **Interactive Evidence Timeline**:
   - Click `👁 View` on any match card in the timeline to immediately jump the video player canvas to that specific matched frame.
6. **Court-Admissible Dossier Export**:
   - Click **`📋 Export Forensic Dossier`** to save an encrypted, timestamped `metadata.json` along with all cropped face and full evidence frames under `captures/VIDEO_ANALYSIS/<session_id>/`.
