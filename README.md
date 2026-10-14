# IBVAP — ATM Best Face Capture & Investigator Search Platform

An intelligent computer vision platform and forensic search backend for ATM surveillance and banking kiosk security. It automatically tracks individuals at an ATM, isolates the primary customer using a spatial-biometric heuristic with hysteresis, assesses face quality across 6 physical factors, selects and enhances the Top 5 best distinct face captures per session, persists forensic metadata to MongoDB, and serves a FastAPI REST engine for investigator search (by face photo and by timestamp CSV).

---

## 🚀 Key Features

1. **Multi-Camera Bounded Ingestion**:
   - Supports $N$ cameras concurrently (USB webcams, RTSP IP camera streams, local video files) configured via `cameras.json`.
   - Dedicated threaded workers with bounded drop-oldest queues prevent camera latency contention.
   - Automatic reconnect with exponential backoff on dropped streams.

2. **Primary Subject Selection & Hysteresis**:
   - Scores each person in frame:
     $$\text{Score} = w_{\text{person}} \cdot \text{Area}_{\text{norm}} + w_{\text{face}} \cdot \text{Face}_{\text{norm}} + w_{\text{roi}} \cdot \text{ROI}_{\text{proximity}}$$
   - Hysteresis filter ensures a challenger must surpass the incumbent by `CHALLENGER_MARGIN` for `HYSTERESIS_FRAMES` consecutive frames, eliminating flickering between adjacent individuals.

3. **Multi-Factor Face Image Quality Assessment (FIQA)**:
   - Computes a composite quality score in $[0, 1]$:
     - **Sharpness**: Variance of Laplacian with saturating curve.
     - **Face Size**: Normalized relative to an ideal $160\text{ px}$ target.
     - **Head Pose**: Perspective-n-Point (solvePnP) Euler angles (Yaw, Pitch, Roll).
     - **Detector Confidence**: Deep learning detection probability.
     - **Illumination**: Brightness balance, contrast, and clipped-pixel ratio.
     - **Occlusion & Masking**: Landmark geometry and lower-face texture variance analysis.
   - Strict hard-rejections for blurry, tiny, or extreme-pose crops.

4. **Session Management & Top 5 Selection**:
   - Starts upon person presence (`MIN_PRESENCE_FRAMES`) and closes on departure (`ABSENCE_TIMEOUT_SECONDS`).
   - Maintains a diversity-enforced min-heap of the top 5 highest-quality face captures.
   - Saves original untouched frames, padded original face crops, and CLAHE/denoised enhanced copies in:
     `captures/{atm_id}/{YYYY-MM-DD}/{session_id}/`

5. **Resilient MongoDB & Offline Buffer**:
   - Asynchronous storage with BSON UTC datetimes and Asia/Kolkata ISO telemetry.
   - Automatic indexing on `captured_at`, `(atm_id, captured_at)`, `(camera_id, captured_at)`, `session_id`, `identity_label`.
   - If MongoDB is unreachable, captures automatically buffer to `buffer/captures_buffer.jsonl` and flush upon reconnection.

6. **Investigator REST Search API (FastAPI)**:
   - `POST /api/search/by-image`: Multipart face image upload -> Cosine similarity search against biometric embeddings -> Top $K$ matches.
   - `POST /api/search/by-csv`: Multipart transaction timestamp CSV upload -> Multi-format date parsing -> Nearest-in-time capture retrieval (+/- seconds).
   - `GET /api/captures`: Filtered, paginated capture archive.
   - `GET /api/images/{capture_id}/{kind}`: Image file streaming (`face`, `frame`, `enhanced`).
   - `GET /api/cameras`: Live camera feed status and telemetry.
   - `GET /api/health`: Database state, models loaded, CPU/GPU, queue depths.
   - Pluggable `get_current_user()` dependency stub and rotating `logs/audit.log` hook.

---

## 🛠️ Setup & Run Instructions

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Configure Cameras (Optional)
Edit `cameras.json` in the root folder to configure camera sources, ATM IDs, and standing zone ROIs:
```json
[
  {
    "camera_id": "cam_001",
    "atm_id": "ATM_BRANCH_01",
    "label": "Cam 0 - ATM Vestibule Main",
    "source": 0,
    "enabled": true,
    "roi": [[120, 80], [520, 80], [580, 470], [60, 470]],
    "resolution": [640, 480]
  }
]
```

### 3. Launch System
```powershell
python main.py --cam0 0 --cam1 1 --mode dual --detect-mode all
```

Options:
- `--cam0 <int>`: Primary camera index (default: 0).
- `--cam1 <int>`: Secondary camera index (default: 1).
- `--mode {dual, 0, 1}`: Initial camera view layout.
- `--detect-mode {all, humans, faces}`: Initial AI detection mode.
- `--mongo-uri <str>`: MongoDB connection URI (default: `mongodb://localhost:27017`).
- `--api-port <int>`: REST API port (default: 8000).

---

## ⌨️ Desktop GUI Hotkeys

| Key | Action |
|:---|:---|
| **[A]** | All AI active (Humans + Faces) |
| **[H]** | Human Detection & Tracking only |
| **[F]** | Face Detection & Recognition only |
| **[D] / [M]** | Dual camera side-by-side view |
| **[0] / [1]** | Single camera feed (Cam 0 / Cam 1) |
| **[S]** | Cycle through camera views |
| **[Q] / ESC** | Safe application shutdown |

---

## 📖 API Documentation

Interactive Swagger documentation is available at:
```
http://localhost:8000/docs
```
OpenAPI specification is available at:
```
http://localhost:8000/openapi.json
```
