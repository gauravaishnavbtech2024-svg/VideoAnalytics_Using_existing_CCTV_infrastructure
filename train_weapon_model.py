import os
import sys
import time
import shutil
import torch
from ultralytics import YOLO

def main():
    print("=" * 70)
    print("🔥 YOLOv8 WEAPON DETECTION MODEL TRAINING (25 EPOCHS)")
    print("=" * 70)
    
    # 1. Hardware Check
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    device_id = 0 if cuda_avail else "cpu"
    print(f"[*] Training Device: {device_name} (ID: {device_id})")
    if cuda_avail:
        print(f"[*] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"[*] CUDA Version: {torch.version.cuda}")

    # 2. Paths
    dataset_yaml = os.path.abspath(r"datasets\weapon detection dataset\dataset_merged\data.yaml")
    project_dir = os.path.abspath(r"models\weapon")
    run_name = "run_25epochs"
    weights_dir = os.path.join(project_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # Check dataset
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"data.yaml not found at {dataset_yaml}")
        
    print(f"[*] Dataset Config: {dataset_yaml}")
    print(f"[*] Output Directory: {project_dir}")

    # 3. Model Selection
    # Fine-tune starting from our previous 5-epoch best weights or baseline yolov8n
    prev_best = os.path.join(weights_dir, "best.pt")
    if os.path.exists(prev_best):
        print(f"[*] Starting from pre-trained weights: {prev_best}")
        model = YOLO(prev_best)
    else:
        print("[*] Starting from base model: yolov8n.pt")
        model = YOLO("yolov8n.pt")

    # 4. Training
    start_time = time.time()
    batch_size = 16 if cuda_avail else 8
    img_size = 320 # Fast real-time CCTV inference resolution
    
    print(f"[*] Initiating 25 Epochs (Batch: {batch_size}, ImgSz: {img_size}, Device: {device_id})...")
    
    results = model.train(
        data=dataset_yaml,
        epochs=25,
        imgsz=img_size,
        batch=batch_size,
        device=device_id,
        workers=2,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        plots=True,
        save=True,
        val=True,
        verbose=True
    )
    
    train_duration = time.time() - start_time
    print(f"\n[+] Training completed in {train_duration / 60:.2f} minutes!")

    # 5. Copy New Best Weights
    trained_best_pt = os.path.join(project_dir, run_name, "weights", "best.pt")
    final_best_pt = os.path.join(weights_dir, "best.pt")
    if os.path.exists(trained_best_pt):
        shutil.copy2(trained_best_pt, final_best_pt)
        print(f"[+] Saved updated PyTorch weights: {final_best_pt} ({os.path.getsize(final_best_pt)/(1024**2):.2f} MB)")

    # 6. Export to ONNX
    print("[*] Exporting model to ONNX format...")
    try:
        final_model = YOLO(final_best_pt)
        onnx_path = final_model.export(format="onnx", imgsz=img_size, dynamic=False)
        final_onnx = os.path.join(weights_dir, "best.onnx")
        if onnx_path and os.path.exists(onnx_path) and onnx_path != final_onnx:
            shutil.copy2(onnx_path, final_onnx)
        print(f"[+] Exported ONNX model: {final_onnx}")
    except Exception as ex:
        print(f"[-] Note on ONNX export: {ex}")

    # 7. Evaluate on Test Set
    print("\n[*] Running final validation benchmark...")
    eval_model = YOLO(final_best_pt)
    val_res = eval_model.val(data=dataset_yaml, split="test", imgsz=img_size, device=device_id)
    print("\n" + "=" * 70)
    print(f"🎯 BENCHMARK RESULTS (25 EPOCHS):")
    print(f"   mAP50:      {val_res.box.map50 * 100:.2f}%")
    print(f"   mAP50-95:   {val_res.box.map * 100:.2f}%")
    print(f"   Precision:  {val_res.box.mp * 100:.2f}%")
    print(f"   Recall:     {val_res.box.mr * 100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
