# IBVAP — End-to-End System Architecture

> **Intelligent Biometric & Video Analytics Platform (IBVAP)**
> A Fully Offline, Multi-Camera, Multi-AI CCTV Surveillance System

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture Diagram](#2-high-level-architecture-diagram)
3. [Data Flow Pipeline](#3-data-flow-pipeline)
4. [Module-by-Module Architecture](#4-module-by-module-architecture)
5. [AI Model Architectures (Deep Dive)](#5-ai-model-architectures-deep-dive)
6. [Cybersecurity Architecture](#6-cybersecurity-architecture)
7. [Camera & Input Management](#7-camera--input-management)
8. [Data Storage Architecture](#8-data-storage-architecture)
9. [UI Architecture (Streamlit Console)](#9-ui-architecture-streamlit-console)
10. [Offline-First Design Philosophy](#10-offline-first-design-philosophy)
11. [Glossary of Difficult Terms](#11-glossary-of-difficult-terms)

---

## 1. System Overview

IBVAP is a **completely offline** surveillance system that processes live video feeds from multiple cameras simultaneously using **six parallel AI pipelines**:

| Pipeline | What It Does |
|----------|-------------|
| **Human Detection & Tracking** | Finds and tracks all humans in the frame using YOLOv8 |
| **Face Detection** | Locates faces in real-time using OpenCV YuNet DNN |
| **Face Recognition** | Matches detected faces against a registered database using SFace embeddings |
| **License Plate Detection** | Detects vehicle number plates using fine-tuned YOLOv8 (+ OBB) |
| **Plate OCR** | Reads plate text using EasyOCR in background threads |
| **Vehicle Classification** | Classifies vehicles into 8 categories (Bus, Car, Truck, etc.) |

All six pipelines run **simultaneously on the same video frame** with a target throughput of **15–30+ FPS**.

---

## 2. High-Level Architecture Diagram

```mermaid
graph TB
    subgraph Input Sources
        CAM0["USB Camera 0"]
        CAM1["USB Camera 1"]
        VID["Video File Upload"]
        RTSP["RTSP / IP Stream"]
    end

    subgraph Camera Manager
        DCM["DualCameraManager<br/>Synchronized grab/retrieve"]
    end

    subgraph AI Processing Pipelines
        HP["Human Detection Pipeline<br/>(YOLOv8 + IoU Tracking)"]
        FP["Face Detection Pipeline<br/>(YuNet DNN + SFace Recognition)"]
        AP["ANPR Pipeline<br/>(YOLOv8-OBB + Async EasyOCR)"]
        VP["Vehicle Classification<br/>(YOLOv8 8-Class)"]
        PP["Person Detector + SiamFC<br/>(Faster R-CNN + SiamFC Tracker)"]
    end

    subgraph Security Engine
        CSM["CyberSecurityHealthMonitor<br/>10-Vector Watchdog"]
    end

    subgraph Data Storage
        CSV["plate_records.csv<br/>(Plate Logs)"]
        CROPS["crops/ Directory<br/>(Plate Images)"]
        FACES["captured_faces/<br/>(Person Snapshots)"]
        KNOWN["known_faces/<br/>(Registered Identities)"]
    end

    subgraph UI Layer
        ST["Streamlit Console<br/>(Live Feed + Metrics + Records + CyberSec)"]
    end

    CAM0 --> DCM
    CAM1 --> DCM
    VID --> DCM
    RTSP --> DCM

    DCM --> HP
    DCM --> FP
    DCM --> AP
    DCM --> VP

    HP --> ST
    FP --> ST
    AP --> ST
    VP --> ST

    AP --> CSV
    AP --> CROPS
    PP --> FACES
    FP --> KNOWN

    CSM --> ST
    CSM -.->|"SHA-256 Hash Verification"| HP
    CSM -.->|"Watchdog Thread"| FP
    CSM -.->|"Database Integrity Check"| CSV
```

---

## 3. Data Flow Pipeline

The system processes each video frame through the following steps:

```mermaid
sequenceDiagram
    participant CAM as Camera Hardware
    participant MGR as DualCameraManager
    participant YOLO_H as Human Tracker (YOLO)
    participant YUNET as Face Detector (YuNet)
    participant SFACE as Face Recognizer (SFace)
    participant YOLO_P as Plate Detector (YOLO-OBB)
    participant OCR as EasyOCR (Background Thread)
    participant REC as PlateRecorder
    participant UI as Streamlit UI

    CAM->>MGR: Raw BGR Frame (640×480)
    MGR->>MGR: Synchronized grab() + retrieve()

    par Human Detection
        MGR->>YOLO_H: frame_bgr
        YOLO_H->>YOLO_H: YOLO predict (imgsz=288, FP16)
        YOLO_H->>YOLO_H: IoU tracking + box smoothing
        YOLO_H->>UI: Annotated frame + human count
    and Face Detection + Recognition
        MGR->>YUNET: frame_bgr (downscaled to 320×240)
        YUNET->>YUNET: DNN face detection
        YUNET->>SFACE: Detected face + landmarks
        SFACE->>SFACE: alignCrop() → feature() → cosine match
        SFACE->>UI: Known/Unknown label + bounding box
    and License Plate Detection
        MGR->>YOLO_P: frame_bgr
        YOLO_P->>YOLO_P: YOLO-OBB predict (imgsz=288)
        YOLO_P->>OCR: Plate crop (async submit)
        OCR-->>YOLO_P: Cached OCR text (when ready)
        YOLO_P->>REC: Plate text + confidence + camera
        REC->>REC: Deduplication (fuzzy match + cooldown)
        REC->>REC: CSV append + crop save
        YOLO_P->>UI: Annotated frame + plate text overlay
    end

    UI->>UI: Combine dual feeds → display RGB frame
    UI->>UI: Update metric cards (FPS, Humans, Faces, Plates)
```

### 3.1 Frame Processing Timing

| Stage | Time Budget | Technique |
|-------|------------|-----------|
| Camera grab + retrieve | ~5 ms | Synchronized dual grab → retrieve (avoids frame skew) |
| YOLO Human Detection | ~8–15 ms | FP16 half-precision, 288px input |
| YuNet Face Detection | ~3–6 ms | 320×240 downscale, DNN forward pass |
| SFace Embedding | ~4–8 ms | Pre-computed known database, cosine match |
| YOLO Plate Detection | ~8–15 ms | OBB model, 288px input |
| EasyOCR | ~80–200 ms | **Async background thread** (does not block main loop) |
| Frame rendering + display | ~2–5 ms | OpenCV draw + Streamlit `st.image()` |
| **Total per frame** | **~30–50 ms** | **≈ 20–33 FPS** |

---

## 4. Module-by-Module Architecture

### 4.1 Human Detection Module (`human_detector.py`)

```mermaid
graph LR
    A["Input Frame<br/>(BGR)"] --> B["YOLOv8 Predict<br/>(288px, FP16)"]
    B --> C["Filter class=0<br/>(person only)"]
    C --> D["IoU Matching<br/>vs Active Tracks"]
    D --> E{"Match Found?<br/>(IoU ≥ 0.25)"}
    E -->|Yes| F["Smooth Update<br/>(60% new + 40% old)"]
    E -->|No New| G["missing_frames++<br/>Drop if ≥ 6"]
    E -->|No Existing| H["Create New Track<br/>(assign ID)"]
    F --> I["Render Badge<br/>🚶 PERSON #N"]
    H --> I
```

**Key Design Decisions:**
- **Box Smoothing**: New detections are blended with previous positions using a **60/40 weighted average** (`0.6 * new + 0.4 * old`). This prevents jittery bounding boxes that would occur from raw per-frame detections.
- **Track Persistence**: A track survives up to 6 frames without detection before being dropped. This handles momentary occlusions (when a person walks behind an object briefly).
- **FP16 Inference**: On NVIDIA GPUs, the model runs in half-precision floating point (16-bit instead of 32-bit), which approximately **doubles** inference speed.

---

### 4.2 Face Detection & Recognition Module (`face_tracker.py`)

```mermaid
graph TB
    A["Input Frame (BGR)"] --> B["Resize to 320×240"]
    B --> C["YuNet DNN Detect"]
    C --> D["Scale coordinates<br/>back to original resolution"]
    D --> E["For each detected face:"]
    E --> F["IoU match vs<br/>existing FaceTracks"]
    F --> G{"Matched?"}
    G -->|Yes + Unknown or 30 frames| H["SFace alignCrop()"]
    G -->|Yes + Already Known| I["Keep existing label"]
    G -->|No Match| J["New FaceTrack"]
    H --> K["SFace feature()"]
    K --> L["Cosine similarity<br/>vs known_database"]
    L --> M{"Score ≥ 0.38?"}
    M -->|Yes| N["● KNOWN NAME<br/>(Green Box)"]
    M -->|No| O["○ UNKNOWN<br/>(Red Box)"]
    J --> H
```

**Two-Stage Face Recognition:**

1. **Stage 1 — YuNet Detection**: OpenCV's YuNet is a DNN-based face detector that outputs face bounding boxes **plus 5 facial landmarks** (2 eyes, nose tip, 2 mouth corners). These landmarks enable precise face alignment.

2. **Stage 2 — SFace Recognition**: The SFace engine takes the aligned face crop and produces a high-dimensional **embedding vector** (a list of numbers that represents the "identity" of a face). The system then compares this embedding against all registered faces using **cosine similarity** (a mathematical measure of how similar two vectors are — 1.0 = identical, 0.0 = completely different).

**Embedding Caching**: The system only re-computes the SFace embedding every 30 frames for already-known faces. This saves approximately 70% of face recognition compute.

---

### 4.3 ANPR Detection Module (`anpr_detector.py`)

```mermaid
graph TB
    A["Input Frame (BGR)"] --> B["YOLOv8 Predict"]
    B --> C{"OBB Results<br/>Available?"}
    C -->|Yes| D["xyxyxyxy polygon<br/>4-corner oriented box"]
    C -->|No| E["Standard xyxy<br/>bounding box"]
    D --> F["Perspective Warp<br/>(rectify tilted plate)"]
    E --> G["Direct Crop"]
    F --> H["Aspect Ratio Check<br/>(width/height ≥ 1.1)"]
    G --> H
    H --> I{"Inside a<br/>Person Box?"}
    I -->|Yes| J["REJECT<br/>(ID card/badge noise)"]
    I -->|No| K["Submit to<br/>EasyOCR Thread Pool"]
    K --> L["Background OCR:<br/>Preprocess → Read → Clean"]
    L --> M["Plate Text Cache<br/>(spatial key)"]
    M --> N["PlateRecorder<br/>Deduplication + CSV Log"]
```

**Why Async OCR?**
EasyOCR is computationally expensive (~100-200ms per plate). If OCR ran on the main thread, the video feed would freeze every time a plate was detected. Instead, the system uses a **ThreadPoolExecutor** with 2 worker threads. The main detection loop submits OCR jobs and immediately continues processing the next frame. When OCR finishes, the result is written to a **cache dictionary** that the main loop reads on subsequent frames.

**Person Rejection Logic**: The system checks if a detected "plate" bounding box overlaps with any human bounding box (IoU > 30% or center point inside). This prevents ID cards, employee badges, and printed text on clothing from being falsely logged as license plates.

---

### 4.4 Hybrid Person Detector + SiamFC Tracker (`person_detector.py`)

```mermaid
graph TB
    A["Input Frame (RGB)"] --> B{"Frame # divisible<br/>by N (every 3 frames)?"}
    B -->|Yes| C["Faster R-CNN<br/>Person Detection<br/>(conf ≥ 0.75)"]
    B -->|No| D["SiamFC Tracking<br/>Only"]

    C --> E["Match detections<br/>vs tracked persons<br/>(IoU ≥ 0.25)"]
    E --> F{"Matched?"}
    F -->|Yes| G["Update bbox +<br/>Refresh template crop"]
    F -->|No| H["New TrackedPerson<br/>(127×127 template)"]

    D --> I["For each tracked person:"]
    I --> J["Crop 255×255<br/>search region"]
    J --> K["SiamFC forward()<br/>z(template) × x(search)"]
    K --> L["17×17 Score Map"]
    L --> M{"Max score ≥ 0.45?"}
    M -->|Yes| N["Update bbox to<br/>peak location"]
    M -->|No| O["Drop track"]

    G --> P["Face Verification<br/>(FaceMatcher)"]
    H --> P
    N --> P
    P --> Q{"Authorized?"}
    Q -->|Yes| R["GREEN box<br/>MATCH: Name"]
    Q -->|No| S["RED box<br/>UNAUTHORIZED"]
```

**The Detect-Then-Track Paradigm:**

This is the most sophisticated module in the system. It combines **two fundamentally different approaches**:

1. **Detection (Faster R-CNN)**: A heavy but accurate neural network that scans the entire frame for persons. This runs only every 3 frames to save GPU compute. It acts as the "ground truth" that discovers new people and validates existing tracks.

2. **Tracking (SiamFC)**: A lightweight Siamese network that tracks a specific person across frames by **cross-correlating** a small template image (127×127 pixels of the person) against a search region (255×255 pixels around their last known position). The output is a 17×17 score map where the peak indicates the person's new position.

**Why This Hybrid?**
- Detection-only: Accurate but slow (~50ms per frame). Cannot maintain identity across frames.
- Tracking-only: Fast but drifts over time. Cannot discover new people.
- **Hybrid**: Detection runs every 3rd frame to anchor the tracks. SiamFC fills in the 2 frames in between. This gives both **accuracy and speed**.

---

### 4.5 Vehicle Detection Module (`vehicle_detector.py`)

The vehicle detector follows the same architecture as the human detector but classifies into 8 categories:

| Class ID | Vehicle Type | Emoji Badge |
|----------|-------------|-------------|
| 0 | Bus | 🚌 BUS |
| 1 | Car | 🚗 CAR |
| 2 | E-Bike | 🚲 E-BIKE |
| 3 | Jeep | 🚙 JEEP |
| 4 | Motorcycle | 🏍️ MOTORCYCLE |
| 5 | Tricycle | 🛺 TRICYCLE |
| 6 | Truck | 🚚 TRUCK |
| 7 | Van | 🚐 VAN |

---

### 4.6 Plate Recorder Module (`plate_recorder.py`)

```mermaid
graph LR
    A["Raw OCR Text"] --> B["clean_plate_text()<br/>Remove non-alphanumeric<br/>Strip HSRP prefix"]
    B --> C["is_valid_plate()<br/>Length 5-13<br/>Has letters AND digits<br/>Not a blacklisted word"]
    C --> D{"Valid?"}
    D -->|No| E["Reject"]
    D -->|Yes| F["_is_duplicate()<br/>Fuzzy match vs<br/>recent_logs"]
    F --> G{"Duplicate<br/>(sim ≥ 0.50)?"}
    G -->|Yes| H["Skip<br/>(already logged)"]
    G -->|No| I["Log to CSV<br/>+ Save crop image"]
```

**Fuzzy String Matching**: OCR is imperfect. The same plate viewed 2 seconds apart might be read as `MH12AB1234` and `MH12A81234` (the 'B' misread as '8'). The system uses **SequenceMatcher** to compute a similarity ratio between the candidate and all recently logged plates. If the similarity exceeds 50%, the system considers it a duplicate and skips logging. This prevents the same vehicle from generating dozens of records.

---

## 5. AI Model Architectures (Deep Dive)

### 5.1 SiamFC (Siamese Fully-Convolutional Network)

```mermaid
graph LR
    subgraph Template Branch
        Z["Template<br/>(3×127×127)"] --> ZB["SiamFCBackbone<br/>5 Conv Layers"]
        ZB --> ZF["Feature Map<br/>(256×6×6)"]
    end

    subgraph Search Branch
        X["Search Region<br/>(3×255×255)"] --> XB["SiamFCBackbone<br/>(shared weights)"]
        XB --> XF["Feature Map<br/>(256×22×22)"]
    end

    ZF --> CC["Cross-Correlation<br/>(depthwise conv2d)"]
    XF --> CC
    CC --> SM["Score Map<br/>(1×17×17)"]
    SM --> UP["Upsample<br/>(bilinear)"]
    UP --> PEAK["argmax → Peak<br/>= Target Location"]
```

**How Cross-Correlation Works**: Think of the template as a "stamp." The system slides this stamp over every position in the search region and measures how well they match. The position with the highest match score is where the tracked object has moved to.

**Architecture Details:**
- Conv1: 3→96 channels, 11×11 kernel, stride 2
- Conv2: 96→256, 5×5 kernel, grouped convolution
- Conv3: 256→384, 3×3 kernel
- Conv4: 384→384, 3×3 kernel, grouped
- Conv5: 384→256, 3×3 kernel, grouped (no activation — raw features)

---

### 5.2 FaceDetectorNet (MobileFaceNet-Inspired)

```mermaid
graph TB
    A["Input Face<br/>(3×112×112)"] --> B["Conv2d 3→64<br/>stride=2 → 56×56"]
    B --> C["DepthwiseSeparable<br/>64→64 → 64→128<br/>stride=2 → 28×28"]
    C --> D["DepthwiseSeparable<br/>128→128 → 128→256<br/>stride=2 → 14×14"]
    D --> E["DepthwiseSeparable<br/>256→256 → 256→512<br/>stride=2 → 7×7"]
    E --> F["DepthwiseSeparable<br/>512→512"]
    F --> G["AdaptiveAvgPool<br/>→ 512×1×1"]
    G --> H["Linear 512→128<br/>+ BatchNorm"]
    H --> I["L2 Normalize"]
    I --> J["128-d Embedding<br/>(Face Identity Vector)"]
```

**What is a Depthwise Separable Convolution?**
A regular convolution with C_in input channels and C_out output channels using a K×K kernel needs `C_in × C_out × K × K` parameters. A depthwise separable convolution splits this into two steps:
1. **Depthwise**: Apply one K×K filter per channel independently → `C_in × K × K` parameters
2. **Pointwise**: Mix channels using 1×1 convolutions → `C_in × C_out` parameters

This reduces parameters by ~8-9× while maintaining similar accuracy. This is crucial for real-time face recognition on limited hardware.

---

## 6. Cybersecurity Architecture

```mermaid
graph TB
    subgraph 10-Vector Security Audit
        V1["1. Camera Status<br/>Stream liveness, FPS"]
        V2["2. Network Status<br/>Gateway ping, DNS"]
        V3["3. CPU/GPU Status<br/>Load %, VRAM"]
        V4["4. Disk Status<br/>Free space, I/O health"]
        V5["5. Model Integrity<br/>SHA-256 checksums"]
        V6["6. Database Integrity<br/>CSV structure validation"]
        V7["7. Unauthorized Access<br/>Unknown face count"]
        V8["8. Config Changes<br/>File mtime watcher"]
        V9["9. System Clock<br/>Drift/spoofing detection"]
        V10["10. Service Health<br/>Overall security grade"]
    end

    subgraph Execution Model
        INIT["Startup:<br/>Compute baseline hashes<br/>Cache config mtimes<br/>Run full audit"]
        DAEMON["Background Daemon Thread<br/>(every 3 seconds)"]
        FULL["On-Demand Full Audit<br/>(via Streamlit tab)"]
    end

    INIT --> V1 & V2 & V3 & V4 & V5 & V6 & V7 & V8 & V9 & V10
    DAEMON --> V2 & V3 & V4 & V6 & V9 & V10
    FULL --> V1 & V2 & V3 & V4 & V5 & V6 & V7 & V8 & V9 & V10

    subgraph Outputs
        METRICS["HealthMetric objects<br/>(name, status, value, details)"]
        LOGS["Security Event Audit Log<br/>(timestamp, type, severity, message)"]
        HASHES["Baseline SHA-256<br/>Hash Table"]
    end

    V5 --> HASHES
    V7 --> LOGS
    V10 --> METRICS
```

### Security Grading System

| Grade | Condition | Meaning |
|-------|-----------|---------|
| **OPTIMAL** | All 10 vectors are "OK" | System fully secure and operational |
| **WARNING** | Any vector is "WARNING" | Minor issues (high CPU, single camera offline, etc.) |
| **CRITICAL** | Any vector is "CRITICAL" | Security violation (model tampering, DB corruption, disk full) |

### Model Integrity Verification (SHA-256)

At startup, the system computes a **SHA-256 cryptographic hash** of every AI model weight file (`.pt` and `.onnx`). SHA-256 produces a unique 64-character hexadecimal fingerprint for any file. If even a single byte of the model is changed, the hash will be completely different. The system periodically re-computes these hashes and compares against the baseline. If a mismatch is found, a **CRITICAL: MODEL TAMPERING DETECTED** alert is raised.

---

## 7. Camera & Input Management

### DualCameraManager Architecture

```mermaid
graph TB
    A["init_cameras()"] --> B["Enumerate devices<br/>(pygrabber/DirectShow)"]
    B --> C["Open Camera 0<br/>(cv2.CAP_DSHOW)"]
    C --> D["Set resolution 640×480<br/>Buffer size = 1"]
    D --> E["Warm-up: Read 5 frames"]
    E --> F["Open Camera 1<br/>(150ms delay)"]
    F --> G["Same config"]

    H["read_sync()"] --> I["cap0.grab()"]
    I --> J["cap1.grab()"]
    J --> K["cap0.retrieve()"]
    K --> L["cap1.retrieve()"]
    L --> M["Return (ret0, frame0,<br/>ret1, frame1)"]
```

**Why Grab-Then-Retrieve?**
USB cameras have internal buffers. If you call `cap.read()` sequentially on two cameras, Camera 1's frame might be captured 30-50ms after Camera 0's frame — causing visible temporal desync in the dual view. By calling `grab()` on both cameras first (which takes the snapshot but doesn't decode it), then `retrieve()` on both (which decodes the already-captured snapshot), both frames are captured as close in time as possible.

**Buffer Size = 1**: Setting `CAP_PROP_BUFFERSIZE = 1` ensures the system always gets the **latest** frame, not an old buffered frame. Without this, there could be a 3-5 frame lag.

---

## 8. Data Storage Architecture

```mermaid
graph LR
    subgraph records/number_plates/
        CSV["plate_records.csv<br/>Timestamp | Date | Time |<br/>PlateNumber | Camera |<br/>Confidence(%) | CropImagePath"]
        CROPS["crops/<br/>MH12AB1234_20260908_141530.jpg<br/>KA01XY5678_20260908_141545.jpg"]
    end

    subgraph known_faces/
        KF1["john_doe.jpg"]
        KF2["jane_smith.jpg"]
        KF3["guard_ravi.png"]
    end

    subgraph captured_faces/
        CF1["person_ID1_20260908_143000.jpg"]
        CF2["person_ID2_20260908_143005.jpg"]
    end

    subgraph models/
        M1["face/ (YuNet + SFace ONNX + best.pt)"]
        M2["anpr/ (license_plate YOLO best.pt)"]
        M3["human_detection/ (human YOLO best.pt)"]
        M4["vehicle_detection/ (vehicle YOLO best.pt)"]
    end
```

### CSV Record Schema

| Column | Type | Example |
|--------|------|---------|
| Timestamp | datetime | 2026-09-08 14:15:30 |
| Date | date | 2026-09-08 |
| Time | time | 14:15:30 |
| PlateNumber | string | MH12AB1234 |
| Camera | string | Cam 0 (Integrated Webcam) |
| Confidence(%) | float | 87.3 |
| CropImagePath | path | crops/MH12AB1234_20260908_141530.jpg |

---

## 9. UI Architecture (Streamlit Console)

```mermaid
graph TB
    subgraph Header
        LOGO["🛡️ CCTV AI Surveillance & Cybersecurity Hub"]
        STATUS["● LIVE STREAMING / ○ STREAM IDLE"]
    end

    subgraph Sidebar
        SRC["Input Source Mode<br/>(Webcams / Video / RTSP)"]
        CAM_SEL["Camera Selection<br/>(Primary + Secondary)"]
        LAYOUT["Viewport Layout<br/>(Dual / Cam 0 / Cam 1)"]
        AI_MODE["AI Pipeline Mode<br/>(All-In-One / Humans / Faces / Plates / Dual)"]
        SLIDERS["Sensitivity Sliders<br/>(Confidence thresholds)"]
    end

    subgraph Main Content Area
        subgraph Tab 1 - Live Feed
            VID["Live Video Feed<br/>(dual side-by-side or single)"]
            ALERTS["Real-Time ANPR Alerts<br/>(plate cards with timestamps)"]
        end
        subgraph Tab 2 - Records
            SEARCH["Search & Filter"]
            TABLE["Plate Records DataTable<br/>(sortable, downloadable CSV)"]
        end
        subgraph Tab 3 - CyberSecurity
            CARDS["10 Security Vector Cards<br/>(OK / WARNING / CRITICAL)"]
            HASH_TABLE["SHA-256 Hash Table"]
            AUDIT_LOG["Security Event Audit Log"]
        end
    end

    subgraph Metrics Bar
        M1["Live FPS"]
        M2["Humans Count"]
        M3["Faces Count"]
        M4["Plates Count"]
        M5["Logged Count"]
        M6["Cyber Health Grade"]
    end
```

---

## 10. Offline-First Design Philosophy

| Aspect | Online Alternative | IBVAP Offline Approach |
|--------|-------------------|----------------------|
| Face Recognition | AWS Rekognition, Azure Face API | Local SFace + custom FaceDetectorNet |
| OCR | Google Vision API | Local EasyOCR engine |
| Object Detection | Cloud ML endpoints | Local YOLOv8 with CUDA |
| Data Storage | Cloud database | Local CSV + filesystem |
| Model Hosting | Model-as-a-Service | Local `.pt` / `.onnx` files |
| UI | Cloud-hosted dashboard | Local Streamlit (`localhost:8501`) |
| Security Monitoring | SIEM cloud services | Local CyberSecurityHealthMonitor |

> [!IMPORTANT]
> **Zero data leaves the machine.** All video frames, face embeddings, plate text, and recognition results stay on the local disk. The system can operate in a completely **air-gapped** environment (no internet cable, no Wi-Fi).

---

## 11. Glossary of Difficult Terms

### AI & Machine Learning Terms

| Term | Simple Explanation |
|------|-------------------|
| **Inference** | Running a trained model on new data to get predictions (as opposed to "training" which is teaching the model) |
| **Embedding** | A fixed-length list of numbers (e.g., 128 numbers) that captures the "essence" of something. Two face images of the same person will produce similar embeddings, while different people produce very different embeddings |
| **Cosine Similarity** | A measure of how similar two lists of numbers are. It computes the cosine of the angle between two vectors — 1.0 means identical direction (same person), 0.0 means perpendicular (unrelated), -1.0 means opposite |
| **L2 Distance (Euclidean)** | The straight-line distance between two points in high-dimensional space. Smaller distance = more similar. Used as an alternative to cosine similarity for face matching |
| **Cross-Correlation** | Sliding one small image (template) over a larger image (search region) and measuring similarity at each position. The peak in the resulting score map indicates where the template appears in the search region |
| **FP16 / Half Precision** | Using 16-bit floating-point numbers instead of 32-bit. This doubles the amount of data that fits in GPU memory and typically doubles processing speed, with minimal accuracy loss |
| **Confidence Threshold** | The minimum "certainty score" (0.0 to 1.0) a model must produce for a detection to be accepted. Lower threshold = more detections but more false alarms. Higher threshold = fewer but more reliable detections |
| **Transfer Learning** | Starting with a model that was already trained on a large dataset (like ImageNet with millions of images) and then fine-tuning it on your specific data (like Indian license plates). This requires far less data and training time than starting from scratch |

### Computer Vision Terms

| Term | Simple Explanation |
|------|-------------------|
| **Bounding Box** | A rectangle drawn around a detected object, defined by its top-left corner coordinates and width/height — e.g., `(x=100, y=200, w=50, h=80)` |
| **IoU (Intersection over Union)** | A measure of how much two bounding boxes overlap. IoU = Area of Overlap / Area of Union. An IoU of 1.0 means perfect overlap; 0.0 means no overlap. Used to match detections across frames (tracking) |
| **OBB (Oriented Bounding Box)** | A rotated rectangle that can be tilted at any angle, unlike a standard axis-aligned bounding box. Used for license plates that appear at an angle in the image |
| **Perspective Warp** | Mathematically "unrotating" an angled image to make it appear flat and straight (head-on). Used to rectify tilted license plates before OCR |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization — an image processing technique that improves contrast in different regions of an image independently, making text on plates more readable in varying lighting |
| **DNN Module** | OpenCV's Deep Neural Network module — allows running pre-trained neural networks (like YuNet, SFace) without needing PyTorch or TensorFlow |
| **Haar Cascade** | A classical (non-deep-learning) face detection algorithm. Fast but less accurate than DNN-based detectors. Used as a fallback when the YuNet model is unavailable |
| **BGR vs RGB** | Two ways to store color pixels. OpenCV uses BGR (Blue-Green-Red) order; most other libraries use RGB (Red-Green-Blue). Converting between them (`cvtColor`) is essential to avoid color-inverted images |

### Neural Network Architecture Terms

| Term | Simple Explanation |
|------|-------------------|
| **Siamese Network** | A neural network architecture with two identical sub-networks that share the same weights. Each sub-network processes one input, and their outputs are compared. Named after Siamese twins because of the twin structure |
| **Depthwise Separable Convolution** | A computationally efficient alternative to standard convolutions. It processes each color channel independently (depthwise), then combines them (pointwise). Uses ~8× fewer calculations |
| **Backbone** | The core feature extraction portion of a neural network that processes raw pixels into meaningful features. The "backbone" is reused across different tasks (detection, tracking, classification) |
| **Score Map** | A 2D grid of numbers produced by a tracking network. Each cell in the grid represents how likely the tracked object is at that position. The cell with the highest number is where the object most likely is |
| **Adaptive Average Pooling** | A layer that takes any size input and produces a fixed-size output by averaging groups of values. `AdaptiveAvgPool2d(1)` collapses a feature map of any spatial size to 1×1, creating a single summary number per channel |

### Tracking & Detection Terms

| Term | Simple Explanation |
|------|-------------------|
| **Detection vs Tracking** | Detection finds objects in a single frame (doesn't know it's the same person frame-to-frame). Tracking maintains identity across frames (knows Person #3 in frame 10 is the same Person #3 in frame 11) |
| **Track ID** | A unique number assigned to each tracked object. Even if a person leaves the frame and returns, they get a new Track ID (re-identification is not performed) |
| **Box Smoothing** | Blending the new detection position with the previous position to create a smooth trajectory. Without smoothing, bounding boxes would "jitter" due to slight frame-to-frame variations |
| **Template (127×127)** | A small cropped image of the tracked object used as a reference. The SiamFC tracker compares this template against the search region in each new frame |
| **Search Region (255×255)** | A larger crop centered on the object's last known position. The tracker searches within this region to find where the object has moved |

### ANPR & OCR Terms

| Term | Simple Explanation |
|------|-------------------|
| **ANPR** | Automatic Number Plate Recognition — the complete pipeline of detecting a plate, reading its text, and logging it |
| **OCR** | Optical Character Recognition — converting an image of text into machine-readable text strings |
| **HSRP** | High Security Registration Plate — India's standardized license plate format with a blue holographic stripe containing "IND" text. The system strips this prefix to get the actual plate number |
| **Fuzzy Matching** | Comparing two text strings and computing a similarity score even when they're not exactly identical. "MH12AB1234" vs "MH12A81234" might have 90% similarity, indicating they're probably the same plate with a single OCR error |
| **Deduplication** | Preventing the same plate from being logged multiple times. Uses a cooldown timer (default 45 seconds) plus fuzzy matching to ensure each vehicle is recorded exactly once |

### Cybersecurity Terms

| Term | Simple Explanation |
|------|-------------------|
| **SHA-256** | A cryptographic hash function that converts any file into a unique 64-character hexadecimal string. Even changing a single byte in the file produces a completely different hash. Used to verify that model weights haven't been tampered with |
| **Model Tampering** | Maliciously modifying AI model weights to change their behavior (e.g., making the face recognizer always say "authorized"). SHA-256 checksums detect this |
| **Watchdog Thread** | A background thread that continuously monitors system health metrics every 3 seconds, independent of the main video processing loop |
| **Air-Gapped** | A computer that has no network connection whatsoever — physically disconnected from the internet. IBVAP is designed to work in air-gapped environments |
| **Monotonic Clock** | A clock that only moves forward and cannot be set backwards. The system uses `time.monotonic()` to detect if someone attempts to change the system clock to replay old timestamps |

### UI & Framework Terms

| Term | Simple Explanation |
|------|-------------------|
| **Streamlit** | A Python framework that creates web applications with minimal code. It turns Python scripts into interactive web pages that run locally on `localhost:8501` |
| **Session State** | Streamlit's mechanism for storing data that persists across page reruns. Used to track whether streaming is active, which plates have been detected, etc. |
| **Glassmorphism** | A UI design trend featuring semi-transparent backgrounds with backdrop blur, creating a "frosted glass" effect. Used in the metric cards and security vector cards |
| **DirectShow** | A Windows multimedia framework for accessing cameras and video devices. The `pygrabber` library uses DirectShow to enumerate USB camera names |
