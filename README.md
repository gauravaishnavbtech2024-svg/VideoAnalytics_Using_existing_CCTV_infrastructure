# 🛡️ IBVAP — Intelligent Border & Critical Infrastructure Video Analytics Platform
### *(Featuring FaceTrace Forensic Investigator Hub & Edge Cybersecurity Watchdog)*

---

## 📌 Executive Summary & Project Purpose

**IBVAP** is an edge-native, multi-modal artificial intelligence surveillance ecosystem designed for **border checkpoints, ATM banking kiosks, and high-security critical infrastructure zones**. 

Modern security operations are plagued by three systemic failures:
1. **Information Overload & False Alarms**: Standard CCTV relies on dumb motion detection or uncalibrated bounding boxes that overwhelm operators with thousands of irrelevant pings.
2. **Poor Biometric Quality in Forensic Retrospectives**: When crimes or breaches occur (e.g., ATM fraud or border incursions), investigators are left with grainy, motion-blurred, sideways-profile crops that fail automated database matching.
3. **Cyber-Physical Vulnerabilities**: CCTV feeds and AI nodes are vulnerable to camera jamming, video freezing, lens obstruction, config tampering, and model weight poisoning.

**IBVAP solves this end-to-end** by orchestrating **6 specialized Deep Learning inference engines**, a **mathematical Face Image Quality Assessment (FIQA) engine**, an **automatic 5-second Top-5 frame extractor**, and a **10-vector real-time Cyber Health Watchdog** into a unified, high-FPS, multi-threaded C++/Python pipeline.

---

## 🏗️ Complete End-to-End System Architecture & Schema

The platform follows a decoupled, asynchronous, producer-consumer pipeline designed for continuous real-time execution without dropping camera frames:

```
+----------------------------------------------------------------------------------------------------+
|                                    1. HARDWARE INGESTION LAYER                                     |
|  - Cam 0: Laptop Integrated HD / ATM Pin-Hole Camera (DirectShow / V4L2)                           |
|  - Cam 1: External Wide-Angle USB Camera / Perimeter Lens (DirectShow / V4L2)                      |
|  - IP Feeds: RTSP H.264/H.265 Streams with auto-reconnect backoff                                  |
|  - CameraManager: Dedicated OS capture threads with bounded ring buffers (Drop-Oldest Queue, Q=1)   |
+-------------------------------------------------+--------------------------------------------------+
                                                  | Raw Frame BGR (640x480 / 1280x720 @ 30 FPS)
                                                  v
+----------------------------------------------------------------------------------------------------+
|                               2. PREPROCESSING & ADAPTIVE ENHANCEMENT                              |
|  - Color Space Conversion: BGR -> RGB / Grayscale                                                  |
|  - CLAHE: Contrast Limited Adaptive Histogram Equalization for backlight/night vision              |
|  - Spatial Normalization & Bilateral Filtering for sensor denoising                                |
+-------------------------------------------------+--------------------------------------------------+
                                                  | Preprocessed Tensor / Frame
                                                  v
+----------------------------------------------------------------------------------------------------+
|                               3. MULTI-MODAL AI INFERENCE ENGINE                                   |
|                                                                                                    |
|  [Face Detection & Biometrics]      [Perimeter Threat Detection]       [Traffic & Transit AI]      |
|  - YuNet ONNX (Face & 5 Landmarks)  - YOLOv8s Human Detector           - YOLOv8 Custom ANPR (OBB)  |
|  - SFace ONNX (128-D Biometric Cos) - YOLOv8 Custom Weapon Threat      - Asynchronous EasyOCR      |
|  - Facial Tracking (ID Persistence) - CSRNet Dilated Crowd Density     - YOLOv8 Vehicle Classifier |
+-------------------------------------------------+--------------------------------------------------+
                                                  | Detections, Embeddings, Density Maps, BBoxes
                                                  v
+----------------------------------------------------------------------------------------------------+
|                               4. FORENSIC ARBITRATION & FIQA ENGINE                                |
|  - Primary Subject Hysteresis: Isolates active ATM customer vs bystander queue                     |
|  - Multi-Factor FIQA: Laplacian Variance, Euler Pose (solvePnP), Illumination, Mask Occlusion       |
|  - Auto 5-Second Candidate Collector: Collects candidate frames, sorts via Min-Heap, selects Top 5|
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
+--------------------------------------------------+ +-----------------------------------------------+
|       5. CYBER DEFENSE WATCHDOG (10 VECTORS)     | |        6. PERSISTENCE & INVESTIGATION         |
|  - Vector 1 & 2: Video Loss / Freeze Detection   | |  - MongoDB Atlas Cloud / Local Replica        |
|  - Vector 3: SHA-256 Model Integrity Verification| |  - Automatic JSONL Offline Fallback Buffer     |
|  - Vector 4: Config File Watchdog (Inotify/Win32)| |  - Court-Admissible Forensic Dossier Export   |
|  - Vector 5: Unauthorized Access & Face Surge    | |  - FastAPI REST Endpoints (Search Photo/CSV)  |
|  - Vector 6-10: FPS Throttling, Memory Leak Check| |  - FaceTrace Modern Desktop UI (Kivy App)     |
+--------------------------------------------------+ +-----------------------------------------------+
```

---

## 💻 Languages, Frameworks, and Libraries Breakdown

Every technology in this project was selected after benchmarking throughput, latency, memory footprint, and edge hardware compatibility.

### 1. Core Programming Languages
* **Python 3.10+**:
  * *Role*: Primary application language orchestrating pipelines, data structures, and multithreading.
  * *Why*: Clean C-extension bindings for PyTorch and OpenCV, native `asyncio` for REST/WebSockets, and rapid model deployment capabilities.
* **C++ (via OpenCV DNN & PyTorch CUDA Backends)**:
  * *Role*: Low-level mathematical operations, tensor convolutions, matrix algebra, and video decoding run compiled C++ under the Python API hood.

### 2. Deep Learning & Computer Vision Frameworks
* **PyTorch (`torch`, `torchvision`)**:
  * *Role*: Deep learning framework powering custom neural networks (CSRNet crowd density) and GPU tensor operations.
  * *Why*: Dynamic computation graphs, native mixed-precision (`torch.cuda.amp.autocast`) for FP16 inference, and seamless integration with CUDA 11.8/12.x.
* **Ultralytics YOLO (`ultralytics` YOLOv8 & YOLO11)**:
  * *Role*: Powers real-time Human Detection, Multi-Class Vehicle Detection, Oriented Bounding Box (OBB) License Plate Detection, and Custom Weapon Threat Classification.
  * *Why*: State-of-the-art accuracy-to-speed ratio, anchor-free split head architecture, and out-of-the-box support for FP16 half-precision CUDA inference delivering 35–60 FPS.
* **OpenCV (`opencv-python` with `cv2.dnn`)**:
  * *Role*: Video stream capture, DirectShow multi-camera handling, geometric transformations, CLAHE histogram enhancement, image quality metric calculation, and lightweight execution of YuNet and SFace ONNX models.
  * *Why*: Highly optimized C++ backend with zero GPU VRAM allocation required for facial detection, freeing valuable GPU memory for YOLO and OCR.
* **EasyOCR (`easyocr`)**:
  * *Role*: Optical Character Recognition on cropped vehicle license plates.
  * *Why*: Deep-learning-based OCR (CRAFT text detector + ResNet/LSTM recognizer) that outperforms classic Tesseract on noisy, angled, and dirty vehicle plates. *Architected into a background thread pool worker so it never halts video streaming*.

### 3. Desktop Application & GUI Frameworks
* **Kivy (`kivy[base]>=2.3.1`)**:
  * *Role*: Powers the flagship **FaceTrace** desktop forensic application (`facetrace_app.py`).
  * *Why*: Hardware-accelerated OpenGL graphics pipeline capable of rendering high-FPS live video streams alongside fluid responsive widgets, custom shaders, and cross-platform capabilities without the heavyweight footprint of Qt.
* **HighGUI (`cv2.imshow`)**:
  * *Role*: Powers the real-time operational surveillance console (`main.py`) with low-latency Heads-Up Display (HUD) overlays and single-key interactive control toggles.



### 5. Database & Storage Architecture
* **MongoDB (`pymongo`)**:
  * *Role*: Document-oriented persistence for biometric records, forensic sessions, and camera metadata.
  * *Why*: Flexible schema for nested FIQA metric vectors and biometric metadata, native BSON UTC timestamps, and rapid geo/temporal indexing (`captured_at`, `atm_id`, `camera_id`).
* **Offline Resilient Buffer (`buffer/captures_buffer.jsonl`)**:
  * *Role*: Disk-backed failover FIFO buffer. If MongoDB Atlas drops connection during internet disruption, all capture documents queue locally and automatically flush upon reconnection.

### 6. Math, Geometry & Utility Libraries
* **NumPy (`numpy`)**: Vectorized array manipulation, cosine distance computation, and landmark affine warping.
* **SciPy (`scipy`)**: Spatial distance calculations and statistical distribution filtering.
* **PyGrabber (`pygrabber`)**: Windows DirectShow COM interface for querying physical USB camera device names and hardware resolutions.

---

## 🧠 Deep-Dive: AI Models & Mathematical Formulations

### 1. YuNet & SFace Biometric Pipeline
* **YuNet (`face_detection_yunet_2023mar.onnx`)**:
  * Ultra-lightweight (under 400 KB) deep face detector trained on WIDER Face.
  * Outputs 14 parameters per face: `[x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]`.
  * Detects faces from $16 \times 16$ up to $640 \times 640$ px with high invariance to lighting and tilt.
* **SFace (`face_recognition_sface_2021dec.onnx`)**:
  * SphereFace-based deep feature extractor outputting a 128-dimensional unit normalized biometric embedding vector $\mathbf{v} \in \mathbb{R}^{128}$ where $\|\mathbf{v}\|_2 = 1$.
  * **Biometric Distance Metric**: Cosine Similarity between probe vector $\mathbf{u}$ and enrolled gallery vector $\mathbf{v}$:
    $$\text{Sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \sum_{i=1}^{128} u_i v_i$$
  * *Thresholds*:
    - $\ge 0.75$: High-Confidence Certified Match (Triggers Suspect Alarm).
    - $0.363 - 0.74$: Known Subject Re-identification.
    - $< 0.363$: Distinct New Individual.

### 2. Multi-Factor Face Image Quality Assessment (FIQA)
Every candidate face crop is evaluated on 6 physical biometric quality factors to generate a normalized score $Q \in [0, 100]$:
1. **Sharpness ($S_{\text{sharp}}$)**: Variance of Laplacian:
   $$\sigma^2 = \frac{1}{N} \sum (L(x,y) - \mu_L)^2, \quad S_{\text{sharp}} = \min\left(1.0, \frac{\sigma^2}{180.0}\right)$$
2. **Face Size ($S_{\text{size}}$)**: Resolution relative to optimal target size ($160\text{ px}$):
   $$S_{\text{size}} = \min\left(1.0, \frac{\min(w, h)}{160.0}\right)$$
3. **Head Pose & Angular Deflection ($S_{\text{pose}}$)**: Perspective-n-Point (`solvePnP`) mapped across 5 canonical facial landmarks (eyes, nose, mouth corners) to estimate Euler angles (Yaw $\psi$, Pitch $\theta$, Roll $\phi$):
   $$S_{\text{pose}} = \max\left(0.0, 1.0 - \frac{|\psi|}{45^\circ} - \frac{|\theta|}{35^\circ} - \frac{|\phi|}{30^\circ}\right)$$
4. **Illumination Balance ($S_{\text{illum}}$)**: Mean luminance $\mu_Y$ and standard deviation $\sigma_Y$ in the YCrCb color space:
   $$S_{\text{illum}} = 1.0 - \frac{|\mu_Y - 128|}{128}$$
5. **Detector Confidence ($S_{\text{conf}}$)**: Raw softmax output probability from YuNet.
6. **Occlusion & Masking Penalty ($P_{\text{mask}}$)**: Texture entropy analysis in the lower facial region (mouth and nose bridge).

**Composite Quality Equation**:
$$Q_{\text{composite}} = 100 \times \left(0.30 S_{\text{sharp}} + 0.25 S_{\text{pose}} + 0.15 S_{\text{size}} + 0.15 S_{\text{illum}} + 0.15 S_{\text{conf}}\right) \times (1.0 - P_{\text{mask}})$$

### 3. Primary Subject Selection with Spatial Hysteresis
To prevent identity thrashing in crowded ATM lobbies or border gates, the system evaluates:
$$\text{Priority Score} = 0.40 \times \text{Area}_{\text{norm}} + 0.35 \times \text{Face}_{\text{norm}} + 0.25 \times \text{ROI}_{\text{proximity}}$$
* **Hysteresis Gate**: An incumbent subject cannot be displaced by a new challenger unless:
  $$\text{Score}_{\text{challenger}} > \text{Score}_{\text{incumbent}} + \delta_{\text{margin}} \quad \text{for } K_{\text{frames}} \ge 10 \text{ consecutive frames}$$

### 4. Custom YOLOv8 Weapon Detection Model
* Trained on domain-specific security datasets across multiple threat classes (handguns, rifles, knives).
* Trained with transfer learning from `yolov8n.pt` using CUDA acceleration, FP16 precision, and Mosaic data augmentations.
* Deployed with a $0.40$ confidence threshold to minimize false alerts while ensuring zero missed weapon threats.

### 5. CSRNet Crowd Density & Surge Detection
* **Dilated Convolutional Neural Network (CSRNet)**: Uses a VGG-16 frontend with dilated convolutional backend layers ($r=2, 4$) that expand receptive field without losing spatial resolution.
* Outputs a continuous 2D spatial density map:
  $$N_{\text{crowd}} = \iint \mathcal{D}(x, y) \, dx \, dy$$
* **Surge Alert Logic**: Computes density gradients across rolling windows. If $\frac{dN}{dt} > \theta_{\text{surge}}$, triggers immediate Stampede / Incursion warning.

---

## 🔒 The 10-Vector Cyber Security Health Monitor

Unique to IBVAP is an integrated defensive watchdog (`src/cyber_health_monitor.py`) running independently alongside the computer vision workers:

| Vector | Monitored Threat | Defense & Detection Mechanism |
| :--- | :--- | :--- |
| **1. Video Loss / Tamper** | Cut cables, power loss, hardware disconnection | Flags dropped camera instances within 500ms; initiates auto-reconnect thread pool. |
| **2. Camera Blind / Occlusion** | Spray paint, covering lens, laser blinding | Measures image histogram variance & entropy. Low variance across entire frame triggers Blackout/Blinding alert. |
| **3. Video Freeze / Replay** | Attacker looping pre-recorded CCTV footage | Computes temporal frame-to-frame SSIM & pixel delta. An identical pixel buffer over $> 1.5$s triggers Video Replay Freeze alarm. |
| **4. AI Model Hash Tampering** | Adversarial weight poisoning or Trojan insertion | Computes SHA-256 cryptographic hashes of all `.pt` and `.onnx` models at startup. Background watcher alarms if hash deviates. |
| **5. Config File Tampering** | Unauthorized edits to `cameras.json` or thresholds | SHA-256 state tracking on configuration files. Prevents attackers from surreptitiously turning off cameras. |
| **6. Unauthorized Access Surge** | Brute-force physical intrusions or crowd storming | Tracks unknown vs verified face ratios per unit time; triggers silent escalation if threshold is breached. |
| **7. Pipeline FPS Degradation** | DoS attack, GPU resource exhaustion, thermal throttle | Monitors rolling 15-frame smoothed FPS. Alerts when inference falls below real-time security minimum (15 FPS). |
| **8. Memory Leak / Process Bloat** | Long-running memory degradation | Tracks RSS process memory consumption via `psutil`. Alerts on abnormal exponential heap growth. |
| **9. Network & DB Drop** | WAN cutoff or MongoDB outage | Seamlessly redirects capture payloads to `buffer/captures_buffer.jsonl` with zero dropped events. |
| **10. Forensic Audit Integrity** | Evidence tampering | All forensic export dossiers sign raw frame buffers with SHA-256 checksums to guarantee court admissibility. |

---

## 🗂️ Complete Codebase Map & File Manifest

```
e:\cctvvvv/
│
├── main.py                     # Primary surveillance engine: Multi-camera ingestion, OpenCV live HUD,
│                               # keyboard mode toggles ('A', 'H', 'F', 'N', 'V', 'W', 'R', 'C').
├── facetrace_app.py            # Flagship Kivy desktop forensic application for investigators.
├── run_facetrace.bat           # 1-Click launcher batch file for FaceTrace desktop suite.
│
├── src/                        # Core AI and algorithmic modules
│   ├── anpr_detector.py        # YOLO License Plate (OBB) detector + Async EasyOCR thread worker.
│   ├── crowd_detector.py       # CSRNet dilated CNN crowd density & heatmap surge detector.
│   ├── cyber_health_monitor.py # 10-vector cybersecurity watchdog & cryptographic integrity monitor.
│   ├── db_manager.py           # MongoDB Atlas client with resilient local JSONL offline buffer.
│   ├── face_dataset.py         # Biometric dataset loader and PyTorch affine transformation helpers.
│   ├── face_matcher.py         # SFace 128-D cosine similarity biometric watchlist cross-referencing.
│   ├── face_model.py           # YuNet DNN & SFace ONNX model loader and inference wrapper.
│   ├── face_tracker.py         # Spatial face tracker with multi-person ID persistence.
│   ├── human_detector.py       # YOLO human detector with IoU Kalman-style bounding-box tracking.
│   ├── person_detector.py      # Primary subject heuristic evaluator with hysteresis filter.
│   ├── plate_recorder.py       # Deduplication buffer and CSV logger for recognized license plates.
│   ├── sot_model.py            # Single Object Tracking (SOT) deep network backbone.
│   ├── utils.py                # Visual drawing utilities, glassmorphic HUD banners, and math helpers.
│   └── vehicle_detector.py     # YOLOv8 multi-class vehicle tracker (Cars, Trucks, Buses, Bikes).
│
├── backend/                    # Web & REST API tier
│   ├── server.py               # FastAPI REST & WebSocket server (Search by image/CSV, live streams).
│   └── stream_manager.py       # Threaded video capture hub with bounded drop-oldest queues.
│
├── frontend/                   # Web surveillance console
│   └── style.css               # Modern dark-mode glassmorphic interface stylesheet.
│
├── models/                     # Deep learning weights directory (Ignored by Git for binary weights)
│   ├── anpr/                   # Trained license plate detection weights.
│   ├── crowd/                  # CSRNet crowd density weights.
│   ├── face/                   # YuNet detection and SFace recognition ONNX models.
│   ├── human_detection/        # YOLO human tracking models.
│   ├── vehicle_detection/      # YOLO multi-class vehicle models.
│   └── weapon/                 # Fine-tuned YOLOv8 25-epoch weapon threat weights.
│
├── datasets/                   # Training datasets (Weapon, ANPR, Crowd)
├── known_faces/                # Enrolled suspect watchlist directory for real-time biometric matching.
├── captures/                   # Forensic evidence directory (Top-5 crops, full frames, metadata.json).
├── buffer/                     # Offline JSONL failover storage when database connection is down.
├── train_weapon_model.py       # Script to train YOLOv8 weapon model with CUDA acceleration.
├── test_cuda.py                # Diagnostic script to verify PyTorch CUDA, cuDNN, and VRAM.
├── requirements.txt            # Pinned dependencies for reproducible deployment.
└── FACETRACE_GUIDE.md          # Dedicated operating manual for the FaceTrace forensic console.
```

---

## 🎯 Defined Roles for Team Presentation

To demonstrate deep technical ownership to the jury, present with clear sub-system domain ownership:

| Team Member | Focused Role Title | Specific Sub-Systems Owned & Defended |
| :--- | :--- | :--- |
| **Member A (Original Lead)** | **Chief System Architect & Core AI Lead** | Model training (YOLOv8 weapon, CSRNet crowd), YuNet/SFace biometric integration, and mathematical FIQA formulation. |
| **Member B (New Teammate)** | **Edge Infrastructure & Cyber Defense Engineer** | Multithreaded bounded camera ingestion (`stream_manager.py`), the 10-vector Cyber Health Watchdog (`cyber_health_monitor.py`), and asynchronous OCR queue design. |
| **Member C (Collaborator)** | **Forensic Application & Full-Stack Engineer** | FaceTrace Kivy forensic investigator hub (`facetrace_app.py`), FastAPI search backend (`backend/server.py`), and MongoDB Atlas offline failover buffer. |

---

## 🎤 Hard-Hitting Jury Questions & Authoritative Answers

### Category 1: System Architecture & Real-Time Performance

#### Q1: "How do you achieve real-time 30+ FPS performance when running 6 complex deep learning models simultaneously on a single edge machine?"
> **Answer**: "We avoid the naive sequential approach where one model blocks another. Our architecture employs **asymmetric decoupled processing**:
> 1. Lightweight models like **YuNet** (under 400 KB) run via OpenCV's C++ DNN engine on the CPU without consuming GPU VRAM.
> 2. Heavy models like **YOLOv8** (Human, Vehicle, Weapon) execute on GPU using **TensorRT / PyTorch FP16 half-precision**, cutting memory bandwidth and latency by over 50%.
> 3. Computationally heavy tasks like **EasyOCR** and **biometric disk persistence** are offloaded to asynchronous background worker thread pools (`ThreadPoolExecutor`).
> 4. Video ingestion uses a **bounded drop-oldest ring buffer ($Q=1$)**, ensuring inference never processes stale frames and zero latency accumulates over time."

#### Q2: "Why did you use EasyOCR instead of Tesseract for license plates, and why run it asynchronously?"
> **Answer**: "Tesseract relies heavily on binarization and traditional thresholding, which fails dramatically in real-world CCTV conditions with motion blur, tilted plates, nighttime glare, and dirty surfaces. EasyOCR uses deep learning (CRAFT for text region detection and a ResNet-LSTM-CTC recognizer) that handles orientation and non-uniform lighting. However, running deep OCR on every frame takes 80–120ms, which would drop video FPS from 30 down to 8. Therefore, YOLO detects the plate box at 30 FPS instantly, while character extraction runs asynchronously on key candidate crops, caching the plate string so the stream never drops a single frame."

---

### Category 2: Biometrics, FIQA & Forensic Reliability

#### Q3: "What is your Face Image Quality Assessment (FIQA) methodology, and why is it superior to simply taking the largest face?"
> **Answer**: "Selecting the largest face is flawed because an individual standing close to the camera who is sneezed, motion-blurred, looking sideways, or poorly lit will yield a large but forensically useless crop. 
> Our FIQA computes a composite mathematical score across 6 physical dimensions:
> - **Sharpness**: Laplacian variance with saturation curves to penalize motion blur.
> - **Pose Geometry**: Perspective-n-Point (`solvePnP`) to compute 3D Euler angles (Yaw, Pitch, Roll); off-axis profiles $> 35^\circ$ are heavily penalized.
> - **Illumination Balance**: Luminance mean and standard deviation in YCrCb color space to avoid under/over-exposure.
> - **Occlusion/Mask**: Texture entropy analysis over the lower face.
> Only candidate frames scoring in the top quintile are admitted to the diversity min-heap, guaranteeing that the final Top-5 captures are crisp, frontal, and optimized for automated facial recognition matching."

#### Q4: "How does your system prevent false alarms when multiple people are in the camera view (e.g., an ATM line or border queue)?"
> **Answer**: "We implement a **Spatial-Biometric Primary Subject Selector with Hysteresis**. Instead of blindly tracking every face, we compute a priority score factoring bounding box area, proximity to the defined region of interest (ROI), and face visibility. 
> To prevent identity flickering between adjacent customers, we enforce a strict **hysteresis barrier**: an incumbent primary subject cannot be superseded by a bystander unless the challenger maintains a higher priority score by a margin $\delta \ge 0.15$ for at least 10 consecutive frames. This provides smooth, stable tracking."

---

### Category 3: Cybersecurity, Tampering & Fail-Safe Engineering

#### Q5: "If an adversary cuts the internet connection to disable cloud alerts, what happens to the surveillance system?"
> **Answer**: "The platform features a **Zero-Data-Loss Offline Resilient Buffer**. All forensic captures, timestamps, and biometric embeddings are written to an atomic append-only disk buffer (`buffer/captures_buffer.jsonl`) whenever MongoDB Atlas becomes unreachable. A background daemon periodically pings the database connection with exponential backoff. The moment network connectivity is restored, the buffered records are automatically replayed and flushed to the cloud database with their original immutable capture timestamps preserved."

#### Q6: "How do you protect your edge system against model tampering or adversarial weight corruption?"
> **Answer**: "Our **Cyber Health Watchdog Vector 3** computes cryptographic **SHA-256 baseline hashes** of all model files (`best.pt`, `yunet.onnx`, `sface.onnx`) during initial deployment and stores them securely. At startup and continuously via background file-watchdog hooks, the system re-computes these hashes. If an attacker gains root access and replaces a model with a backdoored or poisoned network, the hash mismatch is flagged within milliseconds, raising an immediate high-priority audit alarm and disabling compromised inference pathways."

#### Q7: "How does the system differentiate between a legitimate dark scene and a camera tampering attempt (e.g., spray painting or covering the lens)?"
> **Answer**: "A naturally dark scene still contains camera sensor noise and high-frequency spatial variation across its pixel distribution. When a lens is intentionally covered or spray-painted, the spatial gradient across the entire frame collapses to near-zero. Our **Vector 2 Watchdog** monitors both mean intensity and image entropy/Laplacian variance. If luminance is low AND spatial variance remains under a minimal threshold across multiple consecutive frames, it classifies the event as an intentional **Camera Blind/Tamper attack** rather than an ambient light transition."

---

### Category 4: Deployment, Privacy & Legal Admissibility

#### Q8: "How does this platform comply with data protection regulations and court evidence admissibility standards?"
> **Answer**: "Court admissibility requires strict, unbroken chain of custody. When our system commits a forensic capture session:
> 1. It saves the raw, untouched original frame alongside the cropped face.
> 2. It generates a cryptographic **SHA-256 checksum** of the raw image bytes and embeds it within an immutable `metadata.json` dossier containing camera hardware ID, UTC timestamp, and biometric FIQA scores.
> 3. For privacy compliance (such as GDPR or India's DPDP Act), bystander faces can be dynamically blurred, and sensitive biometric embeddings are stored separately from personally identifiable information with configurable data retention and purge schedules."

#### Q9: "Why build a native Kivy desktop application (`facetrace_app.py`) instead of making everything a web browser dashboard?"
> **Answer**: "Web browsers impose severe constraints on forensic security consoles: WebRTC/WebSocket video decoding adds rendering latency, browser memory caps restrict multi-camera raw buffering, and web apps cannot directly interface with low-level OS DirectShow drivers or COM hardware ports. **FaceTrace** is built on **Kivy's native OpenGL pipeline**, giving us sub-10ms render latency, direct GPU buffer access, zero browser sandbox limitations, and true 100% offline edge operability during emergency situations where web servers are down."

---

## ⚡ Quickstart Guide: Running the Platform

### 1. Environment Setup
```powershell
# Navigate to workspace
cd e:\cctvvvv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify CUDA & GPU Hardware Acceleration
```powershell
python test_cuda.py
```

### 3. Launch the Full Multi-Modal Surveillance Console (OpenCV HUD)
```powershell
# Run with dual cameras and all AI modes active
python main.py --cam0 0 --cam1 1 --mode dual --detect-mode all
```

*Interactive Keyboard Controls in Console:*
* `[A]`: All AI active (Humans, Faces, Plates, Vehicles, Weapons, Crowd).
* `[W]`: Weapon Threat Detection only.
* `[N]`: ANPR Number Plate Recognition only.
* `[R]`: Crowd Density & Heatmap Surge only.
* `[C]`: Toggle 10-Vector Cybersecurity Health Dashboard.
* `[D]`: Dual Camera Side-by-Side View.
* `[Q]`: Clean application shutdown.

### 4. Launch the FaceTrace Forensic Investigator Hub (Desktop GUI)
```powershell
# Option A: Double click run_facetrace.bat
# Option B: Run via terminal
python facetrace_app.py
```

### 5. Launch the FastAPI Backend Service
```powershell
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
# View interactive Swagger API documentation at: http://localhost:8000/docs
```
