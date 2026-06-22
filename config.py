"""
Central configuration for the Traffic Violation Detection System.
All paths, model IDs, thresholds, and settings are defined here.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
METADATA_DIR = OUTPUT_DIR / "metadata"
DATA_DIR = PROJECT_ROOT / "data" / "samples"
DB_PATH = PROJECT_ROOT / "violations.db"

# Model weight paths
HELMET_MODEL_PATH = MODELS_DIR / "helmet_model" / "weights" / "best.pt"
SEATBELT_MODEL_PATH = MODELS_DIR / "seatbelt_model" / "weights" / "best.pt"
LPR_DIR = MODELS_DIR / "lpr"
LPR_OD_WEIGHTS = LPR_DIR / "best_od.pth"
LPR_OCR_WEIGHTS = LPR_DIR / "best_lprnet.pth"

# ──────────────────────────────────────────────
# Roboflow Configuration
# ──────────────────────────────────────────────
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_API_URL = "https://detect.roboflow.com"

# Roboflow model IDs for API-based detectors
ROBOFLOW_MODELS = {
    "red_light": "red-light-violation-detect-u07bz/1",
    "wrong_side": "wrong-way-driving-detection-no2km/1",
    "illegal_parking": "illegal-parking-ltiwv-wzewa/1",
    "seatbelt": "seatbelt-smjqq-cibzd/1",
}

# Roboflow dataset identifiers for training
ROBOFLOW_DATASETS = {
    "helmet": {
        "workspace": "keshavs-workspace-e8zcy",
        "project": "bike-helmet-detection-2vdjo-5lfzy",
        "version": 1,
        "format": "yolov8",
    },
    "seatbelt": {
        "workspace": "keshavs-workspace-e8zcy",
        "project": "seatbelt-smjqq-cibzd",
        "version": 1,
        "format": "yolov8",
    },
}

# ──────────────────────────────────────────────
# Detection Thresholds
# ──────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
STOP_LINE_Y_FRACTION = 0.6  # Default: stop line at 60% of image height

# Triple riding: max pixel distance from motorcycle bbox edge for a person to count
TRIPLE_RIDING_PROXIMITY_PX = 30

# ──────────────────────────────────────────────
# YOLOv8 Training Defaults
# ──────────────────────────────────────────────
TRAIN_EPOCHS = 50
TRAIN_IMGSZ = 640
TRAIN_BATCH = 16
TRAIN_PATIENCE = 10

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
DENOISE_H = 10
DENOISE_H_COLOR = 10

# ──────────────────────────────────────────────
# Violation Type Constants
# ──────────────────────────────────────────────
VIOLATION_TYPES = [
    "helmet_violation",
    "triple_riding",
    "seatbelt_violation",
    "red_light_violation",
    "wrong_side_violation",
    "stop_line_violation",
    "illegal_parking",
]

VIOLATION_LABELS = {
    "helmet_violation": "Helmet Non-Compliance",
    "triple_riding": "Triple Riding",
    "seatbelt_violation": "Seatbelt Non-Compliance",
    "red_light_violation": "Red Light Violation",
    "wrong_side_violation": "Wrong-Side Driving",
    "stop_line_violation": "Stop-Line Violation",
    "illegal_parking": "Illegal Parking",
}

# Colors for bounding boxes (BGR format for OpenCV)
VIOLATION_COLOR = (0, 0, 255)    # Red
PLATE_COLOR = (255, 100, 0)      # Blue
STOP_LINE_COLOR = (0, 0, 255)    # Red
BANNER_COLOR = (0, 0, 0)         # Black
BANNER_TEXT_COLOR = (255, 255, 255)  # White

# ──────────────────────────────────────────────
# Indian_LPR Repository
# ──────────────────────────────────────────────
INDIAN_LPR_REPO = "https://github.com/sanchit2843/Indian_LPR.git"
