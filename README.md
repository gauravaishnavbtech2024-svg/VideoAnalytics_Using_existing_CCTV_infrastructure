# CCTV Vision System — Phase 1: Crowd Counting

This project will grow into a full Python vision system:

1. Crowd counting (this phase)
2. Object detection in images
3. Object detection in videos
4. Single-object tracking (USB webcam)
5. Multi-object tracking

---

## Folder layout

```
cctvvvv/
├── datasets/crowd/
│   ├── train/
│   │   ├── images/          ← put training images here
│   │   └── ground_truth/    ← put training labels here
│   └── test/
│       ├── images/          ← put test/val images here
│       └── ground_truth/    ← put test/val labels here
├── models/crowd/              ← trained weights saved here
├── src/                       ← core Python modules
├── scripts/train_crowd.py     ← training script
├── apps/app_crowd.py          ← Streamlit demo
└── requirements.txt
```

---

## Step 1 — Put your dataset in place

### Supported label formats

| Format | File example | Contents |
|--------|--------------|----------|
| ShanghaiTech `.mat` | `GT_IMG_1.mat` | point annotations in `image_info` |
| UCF-QNRF `.mat` | `IMG_1.mat` | `annPoints` field |
| Plain text `.txt` | `IMG_1.txt` | one `x y` head point per line |
| NumPy `.npy` | `IMG_1.npy` | array shape `(N, 2)` |

### Naming rules

For an image named `IMG_1.jpg`, the label file should be one of:

- `ground_truth/GT_IMG_1.mat`
- `ground_truth/IMG_1.mat`
- `ground_truth/IMG_1.txt`
- `ground_truth/IMG_1.npy`

### Example custom `.txt` label

```
120.5 340.2
455.0 210.8
300.1 400.0
```

Each line is one person/head location in pixel coordinates.

---

## Step 2 — Install dependencies

Open PowerShell in the project folder:

```powershell
cd C:\Users\asus\OneDrive\Desktop\cctvvvv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Use Python 3.10+ with an NVIDIA GPU + CUDA for faster training.

---

## Step 3 — Train the model

Default paths assume ShanghaiTech-style layout under `datasets/crowd/`.

```powershell
python scripts/train_crowd.py
```

Useful options:

```powershell
python scripts/train_crowd.py --epochs 100 --batch-size 2 --lr 1e-5
python scripts/train_crowd.py --device cpu
python scripts/train_crowd.py --sigma-mode fixed --fixed-sigma 4.0
```

Outputs:

- `models/crowd/best.pt` — best validation MAE
- `models/crowd/last.pt` — latest epoch
- `models/crowd/history.json` — loss/MAE per epoch

---

## Step 4 — Test in Streamlit

```powershell
streamlit run apps/app_crowd.py
```

Then:

1. Upload a crowd image, or
2. Use the **Webcam** tab for a live capture

The app shows estimated count and an optional density heatmap.

---

## What to do / what not to do

**Do**

- Keep train and test folders separate
- Match every image to a label file
- Start with `--batch-size 2` or `4` if GPU memory is limited
- Use pretrained VGG weights (default) for faster convergence

**Do not**

- Commit large datasets or `.pt` weights to git
- Train without labels — the script will fail early if none are found
- Expect accurate counts before the model has trained for several epochs

---

## Next phase

After crowd counting works on your data, tell me your object-detection dataset format (YOLO/COCO/etc.) and we will add Phase 2 files in the same structure.
