"""
Training Script — Helmet Non-Compliance Detection (YOLOv8n)

Usage:
    Run on Google Colab with T4 GPU for fastest training (~1 hour):
    
    1. Upload this file to Colab
    2. Install dependencies:
       !pip install ultralytics roboflow
    3. Set your Roboflow API key:
       import os
       os.environ["ROBOFLOW_API_KEY"] = "your_key_here"
    4. Run:
       !python train_helmet.py

    After training, download the best weights:
       models/helmet_model/weights/best.pt
    and place them in the same path in your local project.
"""

import os
import sys
from pathlib import Path

# ── Configuration ──
API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
WORKSPACE = "keshavs-workspace-e8zcy"
PROJECT = "bike-helmet-detection-2vdjo-5lfzy"
VERSION = 1
FORMAT = "yolov8"

EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 10
MODEL_BASE = "yolov8n.pt"  # Nano model — fastest, good enough for prototype

# Output directory
OUTPUT_DIR = Path("models/helmet_model")


def main():
    print("=" * 60)
    print("  Helmet Detection — YOLOv8n Fine-Tuning")
    print("=" * 60)

    # ── Step 1: Download dataset from Roboflow ──
    print("\n[1/4] Downloading dataset from Roboflow...")

    if not API_KEY:
        print("ERROR: ROBOFLOW_API_KEY not set.")
        print("Set it via: export ROBOFLOW_API_KEY='your_key'")
        print("Or in Python: os.environ['ROBOFLOW_API_KEY'] = 'your_key'")
        sys.exit(1)

    from roboflow import Roboflow

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    dataset = version.download(FORMAT)

    data_yaml = os.path.join(dataset.location, "data.yaml")
    print(f"  Dataset downloaded to: {dataset.location}")
    print(f"  data.yaml path: {data_yaml}")

    # ── Step 2: Fine-tune YOLOv8n ──
    print(f"\n[2/4] Training YOLOv8n for {EPOCHS} epochs...")

    from ultralytics import YOLO

    model = YOLO(MODEL_BASE)

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        save=True,
        project=str(OUTPUT_DIR),
        name="train",
        exist_ok=True,
        verbose=True,
    )

    # ── Step 3: Locate best weights ──
    print("\n[3/4] Locating best weights...")

    best_weights = OUTPUT_DIR / "train" / "weights" / "best.pt"
    target_weights = OUTPUT_DIR / "weights" / "best.pt"

    if best_weights.exists():
        # Copy to expected location
        target_weights.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(best_weights), str(target_weights))
        print(f"  Best weights saved to: {target_weights}")
    else:
        print(f"  WARNING: best.pt not found at {best_weights}")
        print("  Check the training output directory for weights.")

    # ── Step 4: Print metrics ──
    print("\n[4/4] Training Results:")
    print("-" * 40)

    try:
        metrics = model.val()
        print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"  Precision:     {metrics.box.mp:.4f}")
        print(f"  Recall:        {metrics.box.mr:.4f}")

        # Compute F1
        p = metrics.box.mp
        r = metrics.box.mr
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"  F1 Score:      {f1:.4f}")

        # Save metrics to a file for eval/results.md
        metrics_file = OUTPUT_DIR / "metrics.txt"
        with open(metrics_file, "w") as f:
            f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
            f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
            f.write(f"Precision: {metrics.box.mp:.4f}\n")
            f.write(f"Recall: {metrics.box.mr:.4f}\n")
            f.write(f"F1: {f1:.4f}\n")
        print(f"\n  Metrics saved to: {metrics_file}")

    except Exception as e:
        print(f"  Could not compute validation metrics: {e}")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Weights: {target_weights}")
    print("  Copy this file to your local project's")
    print("  models/helmet_model/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
