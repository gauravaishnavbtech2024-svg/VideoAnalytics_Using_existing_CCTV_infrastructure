"""
FastAPI Server for IBVAP Surveillance Backend.
==============================================
Provides REST and WebSocket endpoints for live video streaming, AI telemetry,
biometric enrollment, snapshot scanning, and 10-vector cybersecurity health.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.stream_manager import stream_manager
from src.cyber_health_monitor import cyber_monitor
from src.db_manager import db_manager

KNOWN_FACES_DIR = PROJECT_ROOT / "known_faces"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background camera streaming on server boot, cleanup on shutdown."""
    print("\n[IBVAP Server] Starting Surveillance Stream Manager...")
    stream_manager.start()
    yield
    print("\n[IBVAP Server] Shutting down Stream Manager...")
    stream_manager.stop()


app = FastAPI(
    title="IBVAP Surveillance Console API",
    description="Edge-Optimized Multi-AI CCTV Pipeline & Cybersecurity Watchdog",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for external PWA / web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 1. Video Streaming & Telemetry Endpoints
# =============================================================================

@app.get("/api/stream", summary="Live MJPEG Camera Stream")
def get_video_stream(feed: str = "composite"):
    """Stream live annotated camera frames in multipart/x-mixed-replace format."""
    return StreamingResponse(
        stream_manager.mjpeg_generator(which=feed),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/cameras", summary="Get Connected Camera Devices")
def get_cameras_info():
    """Return available cameras, current selection, and layout mode."""
    return {
        "cameras": stream_manager.get_available_cameras(),
        "active": {
            "cam0": stream_manager.cam0_id,
            "cam1": stream_manager.cam1_id,
            "cam0_name": stream_manager.dev0_name,
            "cam1_name": stream_manager.dev1_name,
            "layout": stream_manager.layout_mode,
        }
    }


@app.post("/api/cameras/layout", summary="Switch Multi-Cam View Layout")
async def set_camera_layout(data: Dict[str, Any]):
    layout = str(data.get("layout", "dual"))
    stream_manager.set_layout(layout)
    return {"status": "success", "layout": stream_manager.layout_mode}


@app.post("/api/cameras/select", summary="Change Active Camera Hardware")
async def select_cameras(data: Dict[str, Any]):
    cam0 = int(data.get("cam0", stream_manager.cam0_id))
    cam1 = int(data.get("cam1", stream_manager.cam1_id))
    stream_manager.set_camera_ids(cam0, cam1)
    return {
        "status": "success",
        "cam0": stream_manager.cam0_id,
        "cam1": stream_manager.cam1_id,
        "dev0_name": stream_manager.dev0_name,
        "dev1_name": stream_manager.dev1_name,
    }


@app.get("/api/stats", summary="Current Telemetry Metrics")
def get_current_stats():
    """Return instant snapshot of pipeline detection counters and FPS."""
    return JSONResponse(stream_manager.stats)


@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """
    Bidirectional WebSocket connection:
    - Streams live detection metrics & FPS to client at 5Hz.
    - Listens for pipeline toggle commands from frontend.
    """
    await websocket.accept()
    print("[WebSocket] Client connected to live telemetry stream.")
    try:
        while True:
            # Check for incoming client messages (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                data = json.loads(msg)
                stream_manager.update_toggles(data)
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass

            # Send telemetry update to client
            stats_payload = dict(stream_manager.stats)
            await websocket.send_text(json.dumps(stats_payload))
            await asyncio.sleep(0.2)  # 5Hz update cadence

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected from telemetry.")
    except Exception as e:
        print(f"[WebSocket] Exception: {e}")


# =============================================================================
# 2. Pipeline Controls (Toggles)
# =============================================================================

@app.get("/api/toggles", summary="Get Pipeline Toggle Status")
def get_toggles():
    return {
        "humans": stream_manager.enable_humans,
        "faces": stream_manager.enable_faces,
        "plates": stream_manager.enable_plates,
        "vehicles": stream_manager.enable_vehicles,
        "crowd": stream_manager.enable_crowd,
        "conf_threshold": stream_manager.conf_threshold,
    }


@app.post("/api/toggles", summary="Update Pipeline Toggles")
async def set_toggles(data: Dict[str, Any]):
    stream_manager.update_toggles(data)
    return {"status": "success", "updated": data}


# =============================================================================
# 3. Snapshot Scanning (Single Image Multi-AI Inspection)
# =============================================================================

@app.post("/api/scan-image", summary="Multi-AI Snapshot Inspector")
async def scan_single_image(file: UploadFile = File(...)):
    """Upload a snapshot and run all 5 AI detectors, returning annotated image + JSON."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    annotated = img_bgr.copy()
    detections: List[Dict[str, Any]] = []

    # 1. Faces
    face_boxes = []
    if stream_manager.face_tracker:
        annotated, f_cnt = stream_manager.face_tracker.process_frame(annotated, run_detection=True)
        for t in stream_manager.face_tracker.tracks:
            face_boxes.append(t.bbox)
            detections.append({
                "type": "face",
                "label": t.name,
                "confidence": round(float(t.match_score if t.is_known else t.confidence), 2),
                "bbox": list(t.bbox),
                "is_authorized": t.is_known,
            })

    # 2. Humans
    human_boxes = []
    if stream_manager.human_tracker:
        annotated, _, h_cnt = stream_manager.human_tracker.process_frame(
            annotated, existing_face_boxes=face_boxes, run_detection=True
        )
        for t in stream_manager.human_tracker.tracks:
            human_boxes.append(t.bbox)
            detections.append({
                "type": "human",
                "label": f"Person #{t.track_id}",
                "confidence": round(float(t.confidence), 2),
                "bbox": list(t.bbox),
            })

    # 3. Vehicles
    if stream_manager.vehicle_tracker:
        annotated, _, v_cnt = stream_manager.vehicle_tracker.process_frame(annotated, run_detection=True)
        for t in stream_manager.vehicle_tracker.tracks:
            detections.append({
                "type": "vehicle",
                "label": t.class_name.upper(),
                "confidence": round(float(t.confidence), 2),
                "bbox": list(t.bbox),
            })

    # 4. Plates
    if stream_manager.anpr_detector:
        person_boxes = human_boxes + face_boxes
        annotated, p_dets, _ = stream_manager.anpr_detector.process_frame(
            annotated, person_boxes=person_boxes, run_detection=True
        )
        for det in p_dets:
            detections.append({
                "type": "plate",
                "label": det.get("text", "PLATE"),
                "confidence": round(float(det.get("conf", 0.0)), 2),
                "bbox": list(det.get("bbox", (0, 0, 0, 0))),
            })

    # 5. Crowd Density
    crowd_res = {"count": 0, "level": "NORMAL"}
    if stream_manager.crowd_detector:
        boxes_for_crowd = human_boxes if human_boxes else face_boxes
        annotated, c_res = stream_manager.crowd_detector.process_frame(
            annotated, person_boxes=boxes_for_crowd, generate_overlay=True
        )
        crowd_res = {"count": c_res.estimated_count, "level": c_res.density_level}

    # Encode annotated image to Base64 for instant web preview
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_img = base64.b64encode(buf).decode("utf-8")

    return {
        "status": "success",
        "detections": detections,
        "crowd": crowd_res,
        "annotated_image_b64": f"data:image/jpeg;base64,{b64_img}",
    }


# =============================================================================
# 4. Biometric Face Registry Management
# =============================================================================

@app.get("/api/known-faces", summary="List Registered Authorized Personnel")
def list_known_faces():
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    faces = []
    for f in KNOWN_FACES_DIR.iterdir():
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            clean_name = f.stem.split("(")[0].strip().replace("_", " ").title()
            faces.append({"name": clean_name, "filename": f.name})
    return {"total": len(faces), "faces": faces}


@app.post("/api/enroll-face", summary="Enroll New Identity into Biometric DB")
async def enroll_face(file: UploadFile = File(...), name: str = Form(...)):
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    clean_name = name.strip().replace(" ", "_")
    target_path = KNOWN_FACES_DIR / f"{clean_name}.jpg"

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid facial image file.")

    cv2.imwrite(str(target_path), img_bgr)

    # Force face tracker to re-cache known embeddings
    if stream_manager.face_tracker:
        stream_manager.face_tracker._load_known_faces()

    return {"status": "success", "message": f"Enrolled {name.strip()} successfully."}


@app.delete("/api/known-faces/{name}", summary="Delete Enrolled Identity")
def delete_known_face(name: str):
    clean_name = name.strip().lower()
    for f in KNOWN_FACES_DIR.iterdir():
        if f.stem.lower() == clean_name or f.stem.replace("_", " ").lower() == clean_name:
            f.unlink()
            if stream_manager.face_tracker:
                stream_manager.face_tracker._load_known_faces()
            return {"status": "success", "message": f"Deleted {name}."}

    raise HTTPException(status_code=404, detail=f"Identity '{name}' not found.")


# =============================================================================
# 5. Cybersecurity Watchdog & System Health
# =============================================================================

@app.get("/api/health", summary="10-Vector Cybersecurity Health Telemetry")
def get_cybersecurity_health():
    cyber_monitor.run_full_audit()
    metrics_list = []
    for key, m in cyber_monitor.metrics.items():
        metrics_list.append({
            "vector": m.name,
            "status": m.status,
            "value": m.value,
            "details": m.details,
            "timestamp": m.last_checked,
        })
    return {
        "status": "active",
        "service_health": cyber_monitor.metrics.get("service", {}).status if "service" in cyber_monitor.metrics else "OPTIMAL",
        "vectors": metrics_list,
    }


# =============================================================================
# 6. Forensic Database Logs
# =============================================================================

@app.get("/api/logs/{table_name}", summary="Query Surveillance Database Events")
def get_database_logs(table_name: str, limit: int = 50):
    valid_tables = ["anpr_records", "face_logs", "vehicle_records", "crowd_events", "system_alerts"]
    if table_name not in valid_tables:
        raise HTTPException(status_code=400, detail=f"Invalid table name. Choose from {valid_tables}")

    df = db_manager.export_table_to_df(table_name, limit=limit)
    return {
        "table": table_name,
        "count": len(df),
        "records": df.to_dict(orient="records"),
    }


# =============================================================================
# 7. Static Files (PWA Frontend Mounting)
# =============================================================================

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 75)
    print("  🚀 Launching IBVAP Surveillance Backend API")
    print(f"  ⚡ Accelerator Device: {DEVICE.upper()}")
    print("  🌐 Server running at http://127.0.0.1:8000")
    print("=" * 75 + "\n")
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)
