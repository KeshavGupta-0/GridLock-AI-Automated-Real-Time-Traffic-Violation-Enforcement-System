# 🚦 Traffic Violation Detection System

**AI-Powered Automated Traffic Image Analysis — Gridlock Hackathon 2.0, Problem Statement 3**

A computer vision-based solution that automatically processes traffic images, detects vehicles and road users, identifies 7 types of traffic violations, classifies them with confidence scores, extracts license plate numbers via OCR, and generates annotated evidence for enforcement.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                              │
│   ┌──────────┐    ┌──────────────┐    ┌───────────────┐         │
│   │  DETECT   │    │ VIOLATION LOG│    │  ANALYTICS    │         │
│   │  (Upload) │    │ (SQLite DB)  │    │ (Charts/Metrics)│       │
│   └─────┬─────┘    └──────────────┘    └───────────────┘         │
│         │                                                        │
├─────────▼────────────────────────────────────────────────────────┤
│                     ORCHESTRATOR                                  │
│         run_pipeline(image) → unified JSON result                │
│                                                                   │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │                 DETECTION LAYER                          │    │
│   │                                                          │    │
│   │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐           │    │
│   │  │Helmet  │ │Seatbelt│ │Red     │ │Wrong   │           │    │
│   │  │YOLOv8n │ │API/YOLO│ │Light   │ │Side    │           │    │
│   │  │(local) │ │        │ │(API)   │ │(API)   │           │    │
│   │  └───┬────┘ └────────┘ └───┬────┘ └────────┘           │    │
│   │      │                     │                             │    │
│   │  ┌───▼────┐           ┌───▼────┐  ┌────────┐           │    │
│   │  │Triple  │           │Stop    │  │Illegal │           │    │
│   │  │Riding  │           │Line    │  │Parking │           │    │
│   │  │(logic) │           │(geom.) │  │(API)   │           │    │
│   │  └────────┘           └────────┘  └────────┘           │    │
│   └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│   ┌───────────────┐  ┌───────────────┐  ┌──────────────────┐    │
│   │ PREPROCESSING │  │  PLATE OCR    │  │ EVIDENCE GEN     │    │
│   │ CLAHE+Denoise │  │ Indian_LPR /  │  │ Annotate + Save  │    │
│   │               │  │ EasyOCR       │  │ to SQLite        │    │
│   └───────────────┘  └───────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7 Violation Types

| # | Violation | Implementation | Training Required? |
|---|-----------|----------------|-------------------|
| 1 | Helmet Non-Compliance | YOLOv8n fine-tuned | ✅ Yes (Colab) |
| 2 | Triple Riding | Logic layer on #1 | ❌ No |
| 3 | Seatbelt Non-Compliance | Roboflow API / YOLOv8n | Optional |
| 4 | Red-Light Violation | Roboflow Hosted API | ❌ No |
| 5 | Wrong-Side Driving | Roboflow Hosted API | ❌ No |
| 6 | Stop-Line Violation | Geometry rule on #4 | ❌ No |
| 7 | Illegal Parking | Roboflow Hosted API | ❌ No |

### Architecture Decisions

- **Violations 2 & 6 are logic layers, not models.** Triple riding counts persons per motorcycle bounding box from the helmet model's detections — a separate model would be redundant. Stop-line violation checks whether a vehicle's bounding box crosses a configurable line while a red light is active — this is a geometry problem, not a classification problem.

- **Violations 4, 5, 7 use hosted APIs.** For a 3-day prototype timeline, using Roboflow's pretrained hosted models eliminates training time entirely for these violation types, letting us focus engineering effort on the two models that need it (helmet + seatbelt).

- **YOLOv8n (Nano), not YOLOv8s/m.** The nano variant provides the best speed/accuracy tradeoff for real-time enforcement: ~5ms inference on GPU vs ~20ms for the small variant, with only a marginal mAP decrease. For traffic camera footage (typically well-lit, fixed angle), YOLOv8n's accuracy is sufficient.

---

## Setup Instructions

### 1. Clone and install dependencies

```bash
cd "Gridlock 2.0/Round 2"
pip install -r requirements.txt
```

### 2. Set up Roboflow API key

Get a free API key from [roboflow.com](https://roboflow.com), then either:

```bash
# Option A: Environment variable
export ROBOFLOW_API_KEY="your_key_here"

# Option B: Enter in the Streamlit sidebar when the app starts
```

### 3. Set up License Plate OCR (optional)

If you have a GPU, the `Indian_LPR` OCR engine is already pre-configured in the `models/lpr` folder.
If it fails to load due to environment differences, the system automatically and seamlessly falls back to `EasyOCR` (which is pre-installed via requirements.txt) — absolutely no manual action needed.

### 4. Train models on Google Colab

Upload these scripts to Colab (with T4 GPU runtime):

```bash
# Helmet model (~1 hour training)
python train/train_helmet.py

# Seatbelt model (~1 hour training, optional if using API)
python train/train_seatbelt.py
```

After training, download the `best.pt` files and place them:
- `models/helmet_model/weights/best.pt`
- `models/seatbelt_model/weights/best.pt`

### 5. Run the app

```bash
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
├── models/
│   ├── helmet_model/weights/      # YOLOv8n weights (after Colab training)
│   ├── seatbelt_model/weights/    # YOLOv8n weights (after Colab training)
│   └── lpr/                       # Indian_LPR clone (after setup_lpr.py)
├── pipeline/
│   ├── __init__.py
│   ├── preprocess.py              # CLAHE + denoising for low-light images
│   ├── detectors.py               # 7 violation detection functions
│   ├── ocr.py                     # License plate detection + OCR
│   ├── orchestrator.py            # run_pipeline() — merges all detectors
│   └── evidence.py                # Image annotation + SQLite storage
├── app/
│   └── streamlit_app.py           # 3-tab dashboard (Detect, Log, Analytics)
├── data/
│   └── samples/                   # Test images
├── eval/
│   ├── evaluate.py                # Model validation + pipeline timing
│   └── results.md                 # Evaluation metrics
├── train/
│   ├── train_helmet.py            # Colab training script for helmet model
│   └── train_seatbelt.py          # Colab training script for seatbelt model
├── output/
│   ├── annotated/                 # Annotated evidence images
│   └── metadata/                  # Violation metadata
├── config.py                      # Central configuration
├── requirements.txt               # Python dependencies
├── violations.db                  # SQLite database (auto-created)
└── README.md                      # This file
```

---

## Evaluation

Run after training models:

```bash
python eval/evaluate.py
```

This generates `eval/results.md` with:
- mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1 for each trained model
- Average inference time per image
- Computational efficiency notes

Headline metrics are also displayed in the Analytics tab of the dashboard.

---

## Performance Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| mAP@0.5 | >0.70 | Based on published helmet detection benchmarks |
| Precision | >0.75 | Minimize false positives for enforcement credibility |
| F1 Score | >0.70 | Balance precision/recall |
| Inference Time | <3s/image | On CPU; <100ms on T4 GPU |

---

## Future Work

- **Video stream support** — Process RTSP camera feeds in real-time
- **Edge deployment** — TensorRT optimization for NVIDIA Jetson
- **Enhanced seatbelt detection** — Fine-tune on larger dataset for Indian vehicles
- **Multi-camera correlation** — Track violations across multiple intersection cameras
- **Automated challan generation** — Integration with traffic enforcement databases
- **Night vision enhancement** — Specialized preprocessing for IR camera images

---

## Tech Stack

- **Detection:** Ultralytics YOLOv8, Roboflow Inference API
- **OCR:** Indian_LPR / EasyOCR
- **Preprocessing:** OpenCV (CLAHE, denoising)
- **Dashboard:** Streamlit
- **Database:** SQLite
- **Visualization:** Plotly
- **Training:** Google Colab (T4 GPU)

---

*Built for Gridlock Hackathon 2.0 — Problem Statement 3*
