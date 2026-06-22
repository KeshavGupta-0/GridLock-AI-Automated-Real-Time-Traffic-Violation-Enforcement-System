"""
Detection Functions — One per violation type.

Every function returns a standardized dict:
{
    "violation_type": str,
    "detected": bool,
    "confidence": float,
    "bbox": list | None,       # [x1, y1, x2, y2]
    "extra": dict              # violation-specific metadata
}

Every function is wrapped in try/except — a live demo must never crash.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ROBOFLOW_API_URL,
    ROBOFLOW_MODELS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    TRIPLE_RIDING_PROXIMITY_PX,
    STOP_LINE_Y_FRACTION,
)


def _safe_result(violation_type: str, error: str = "") -> dict:
    """Return a safe empty result dict (no detection, no crash)."""
    result = {
        "violation_type": violation_type,
        "detected": False,
        "confidence": 0.0,
        "bbox": None,
        "extra": {},
    }
    if error:
        result["extra"]["error"] = error
    return result


def _roboflow_infer(image: np.ndarray, model_id: str, api_key: str) -> list:
    """
    Call Roboflow Hosted Inference API via HTTP requests and return list of predictions.
    Each prediction has: class, confidence, x, y, width, height.
    """
    import requests
    import base64

    # Encode image to base64
    _, img_encoded = cv2.imencode('.jpg', image)
    img_b64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

    # Construct the API URL
    url = f"{ROBOFLOW_API_URL}/{model_id}?api_key={api_key}"

    # Make the POST request
    response = requests.post(
        url,
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    if response.status_code == 200:
        result = response.json()
        if isinstance(result, dict):
            return result.get("predictions", [])
    else:
        raise Exception(f"Roboflow API error {response.status_code}: {response.text}")

    return []


def _bbox_from_roboflow_pred(pred: dict) -> list:
    """Convert Roboflow center-format bbox to [x1, y1, x2, y2]."""
    cx = pred.get("x", 0)
    cy = pred.get("y", 0)
    w = pred.get("width", 0)
    h = pred.get("height", 0)
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    return [x1, y1, x2, y2]


# ═══════════════════════════════════════════════
# Violation 1 — Helmet Non-Compliance
# ═══════════════════════════════════════════════

def detect_helmet(
    image: np.ndarray,
    model=None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Run fine-tuned YOLOv8 helmet model.

    Returns violation record with extra['all_boxes'] containing
    all detected boxes (needed by detect_triple_riding).
    """
    vtype = "helmet_violation"
    try:
        if model is None:
            return _safe_result(vtype, error="Helmet model not loaded")

        results = model(image, conf=confidence_threshold, verbose=False)

        all_boxes = []
        violation_detected = False
        max_conf = 0.0
        violation_bbox = None

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]

                box_info = {
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                }
                all_boxes.append(box_info)

                # Flag violation if "no-helmet" or "no_helmet" or "without helmet" detected
                no_helmet_labels = ["no-helmet", "no_helmet", "without helmet", "without_helmet", "no helmet"]
                if cls_name.lower() in no_helmet_labels and conf > max_conf:
                    violation_detected = True
                    max_conf = conf
                    violation_bbox = [x1, y1, x2, y2]

        return {
            "violation_type": vtype,
            "detected": violation_detected,
            "confidence": round(max_conf, 4),
            "bbox": violation_bbox,
            "extra": {
                "all_boxes": all_boxes,
                "total_detections": len(all_boxes),
            },
        }

    except Exception as e:
        return _safe_result(vtype, error=str(e))


# ═══════════════════════════════════════════════
# Violation 2 — Triple Riding (Logic Layer)
# ═══════════════════════════════════════════════

def detect_triple_riding(
    helmet_result: dict,
    proximity_px: int = TRIPLE_RIDING_PROXIMITY_PX,
) -> dict:
    """
    Pure logic on top of helmet detection results.

    Since the helmet model only detects heads ("With Helmet" / "Without Helmet"),
    we group heads that are physically close to each other.
    If 3 or more heads form a tight cluster, we flag it as triple riding.
    """
    vtype = "triple_riding"
    try:
        all_boxes = helmet_result.get("extra", {}).get("all_boxes", [])

        # Filter out anything that isn't a head (just in case)
        head_labels = ["with helmet", "without helmet", "helmet", "no-helmet", "no_helmet"]
        heads = [b for b in all_boxes if b.get("class", "").lower() in head_labels]

        if len(heads) < 3:
            return _safe_result(vtype)

        import math
        n = len(heads)
        adj = {i: [] for i in range(n)}
        
        # Connect heads that are within proximity_px
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, x2, y2 = heads[i]["bbox"]
                cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
                
                x3, y3, x4, y4 = heads[j]["bbox"]
                cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                
                # Check distance between centers
                dist = math.hypot(cx1 - cx2, cy1 - cy2)
                
                # Dynamic threshold: heads on the same bike are typically within 
                # 2.5 to 3 times the width/height of their own head bounding boxes.
                # This works for both high-res and low-res images automatically!
                h1_width = x2 - x1
                h2_width = x4 - x3
                dynamic_threshold = max(h1_width, h2_width) * 3.5
                
                if dist < dynamic_threshold:
                    adj[i].append(j)
                    adj[j].append(i)

        # Find connected components (groups of heads on the same bike)
        visited = set()
        max_group_size = 0
        violation_bbox = None

        for i in range(n):
            if i not in visited:
                group = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    group.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                if len(group) >= 3:
                    if len(group) > max_group_size:
                        max_group_size = len(group)
                        # Create a bounding box covering all heads in the triple-riding group
                        gx1 = min(heads[idx]["bbox"][0] for idx in group)
                        gy1 = min(heads[idx]["bbox"][1] for idx in group)
                        gx2 = max(heads[idx]["bbox"][2] for idx in group)
                        gy2 = max(heads[idx]["bbox"][3] for idx in group)
                        
                        # Add a small margin
                        margin = 20
                        violation_bbox = [
                            max(0, gx1 - margin), 
                            max(0, gy1 - margin), 
                            gx2 + margin, 
                            gy2 + margin
                        ]

        violation_detected = (max_group_size >= 3)

        return {
            "violation_type": vtype,
            "detected": violation_detected,
            "confidence": 0.9 if violation_detected else 0.0,
            "bbox": violation_bbox,
            "extra": {
                "max_riders_on_single_bike": max_group_size,
                "total_heads_found": len(heads),
            },
        }

    except Exception as e:
        return _safe_result(vtype, error=str(e))


# ═══════════════════════════════════════════════
# Violation 3 — Seatbelt Non-Compliance
# ═══════════════════════════════════════════════

def detect_seatbelt(
    image: np.ndarray,
    api_key: str = "",
    model=None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Detect seatbelt violations.
    Primary: Roboflow hosted API.
    Fallback: local YOLOv8 model (if trained).
    """
    vtype = "seatbelt_violation"
    try:
        # Try local model first if available
        if model is not None:
            results = model(image, conf=confidence_threshold, verbose=False)
            violation_detected = False
            max_conf = 0.0
            violation_bbox = None

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = r.names[cls_id]
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()

                    no_seatbelt_labels = [
                        "person-noseatbelt", "no-seatbelt", "no_seatbelt",
                        "without seatbelt", "noseatbelt",
                    ]
                    if cls_name.lower() in no_seatbelt_labels and conf > max_conf:
                        violation_detected = True
                        max_conf = conf
                        violation_bbox = [int(v) for v in xyxy]

            return {
                "violation_type": vtype,
                "detected": violation_detected,
                "confidence": round(max_conf, 4),
                "bbox": violation_bbox,
                "extra": {"source": "local_model"},
            }

        # Fallback: Roboflow API
        if not api_key:
            return _safe_result(vtype, error="No API key and no local model")

        preds = _roboflow_infer(image, ROBOFLOW_MODELS["seatbelt"], api_key)

        violation_detected = False
        max_conf = 0.0
        violation_bbox = None

        for pred in preds:
            cls_name = pred.get("class", "").lower()
            conf = pred.get("confidence", 0.0)

            no_seatbelt_labels = [
                "person-noseatbelt", "no-seatbelt", "noseatbelt",
            ]
            if cls_name in no_seatbelt_labels and conf >= confidence_threshold:
                if conf > max_conf:
                    violation_detected = True
                    max_conf = conf
                    violation_bbox = _bbox_from_roboflow_pred(pred)

        return {
            "violation_type": vtype,
            "detected": violation_detected,
            "confidence": round(max_conf, 4),
            "bbox": violation_bbox,
            "extra": {"source": "roboflow_api", "predictions": len(preds)},
        }

    except Exception as e:
        return _safe_result(vtype, error=str(e))


# ═══════════════════════════════════════════════
# Violation 4 — Red-Light Violation
# ═══════════════════════════════════════════════

def detect_red_light(
    image: np.ndarray,
    api_key: str = "",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Call Roboflow hosted API for red-light violation detection.
    """
    vtype = "red_light_violation"
    try:
        if not api_key:
            return _safe_result(vtype, error="No Roboflow API key provided")

        preds = _roboflow_infer(image, ROBOFLOW_MODELS["red_light"], api_key)

        violation_detected = False
        max_conf = 0.0
        violation_bbox = None
        all_pred_info = []

        for pred in preds:
            cls_name = pred.get("class", "")
            conf = pred.get("confidence", 0.0)
            bbox = _bbox_from_roboflow_pred(pred)

            all_pred_info.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": bbox,
            })

            # Flag any detection above threshold as a violation
            if conf >= confidence_threshold and conf > max_conf:
                violation_detected = True
                max_conf = conf
                violation_bbox = bbox

        return {
            "violation_type": vtype,
            "detected": violation_detected,
            "confidence": round(max_conf, 4),
            "bbox": violation_bbox,
            "extra": {
                "source": "roboflow_api",
                "all_predictions": all_pred_info,
            },
        }

    except Exception as e:
        return _safe_result(vtype, error=str(e))


# ═══════════════════════════════════════════════
# Violation 5 — Wrong-Side Driving
# ═══════════════════════════════════════════════

def detect_wrong_side(
    image: np.ndarray,
    api_key: str = "",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Call Roboflow hosted API for wrong-way driving detection.
    """
    vtype = "wrong_side_violation"
    try:
        if not api_key:
            return _safe_result(vtype, error="No Roboflow API key provided")

        preds = _roboflow_infer(image, ROBOFLOW_MODELS["wrong_side"], api_key)

        violation_detected = False
        max_conf = 0.0
        violation_bbox = None

        for pred in preds:
            cls_name = pred.get("class", "").lower()
            conf = pred.get("confidence", 0.0)

            wrong_labels = ["wrong-way", "wrong_way", "wrong-side", "wrong_side"]
            if cls_name in wrong_labels and conf >= confidence_threshold:
                if conf > max_conf:
                    violation_detected = True
                    max_conf = conf
                    violation_bbox = _bbox_from_roboflow_pred(pred)

        return {
            "violation_type": vtype,
            "detected": violation_detected,
            "confidence": round(max_conf, 4),
            "bbox": violation_bbox,
            "extra": {"source": "roboflow_api", "predictions": len(preds)},
        }

    except Exception as e:
        return _safe_result(vtype, error=str(e))


# ═══════════════════════════════════════════════
# Violation 6 — Stop-Line Violation (Geometry Rule)
# ═══════════════════════════════════════════════

def detect_stop_line(
    image: np.ndarray,
    red_light_result: dict,
    stop_line_y: float = STOP_LINE_Y_FRACTION,
) -> dict:
    """
    Geometry rule layered on top of red-light detection.

    If a red-light violation is detected AND any vehicle bbox's
    bottom edge (y2) is below the stop line, flag stop-line violation.

    stop_line_y is a fraction of image height (0.0 = top, 1.0 = bottom).
    """
    vtype = "stop_line_violation"
    try:
        img_height = image.shape[0]
        stop_line_px = int(img_height * stop_line_y)

        # Only check if red-light violation was detected
        if not red_light_result.get("detected", False):
            return {
                "violation_type": vtype,
                "detected": False,
                "confidence": 0.0,
                "bbox": None,
                "extra": {
                    "stop_line_y_px": stop_line_px,
                    "reason": "No red-light violation detected",
                },
            }

        # Check if any vehicle bbox crosses the stop line
        all_preds = red_light_result.get("extra", {}).get("all_predictions", [])
        violation_detected = False
        max_conf = 0.0
        violation_bbox = None

        for pred in all_preds:
            bbox = pred.get("bbox", [0, 0, 0, 0])
            conf = pred.get("confidence", 0.0)
            if len(bbox) >= 4:
                y2 = bbox[3]  # Bottom edge of vehicle bbox
                if y2 > stop_line_px and conf > max_conf:
                    violation_detected = True
                    max_conf = conf
                    violation_bbox = bbox

        # If no specific vehicle bbox found, use the red-light bbox
        if not violation_detected and red_light_result.get("bbox"):
            rl_bbox = red_light_result["bbox"]
            if len(rl_bbox) >= 4 and rl_bbox[3] > stop_line_px:
                violation_detected = True
                max_conf = red_light_result.get("confidence", 0.0)
                violation_bbox = rl_bbox

        return {
            "violation_type": vtype,
            "detected": violation_detected,
            "confidence": round(max_conf, 4),
            "bbox": violation_bbox,
            "extra": {
                "stop_line_y_px": stop_line_px,
                "stop_line_y_fraction": stop_line_y,
            },
        }

    except Exception as e:
        return _safe_result(vtype, error=str(e))


# ═══════════════════════════════════════════════
# Violation 7 — Illegal Parking
# ═══════════════════════════════════════════════

def detect_illegal_parking(
    image: np.ndarray,
    api_key: str = "",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Call Roboflow hosted API for illegal parking detection.
    """
    vtype = "illegal_parking"
    try:
        if not api_key:
            return _safe_result(vtype, error="No Roboflow API key provided")

        preds = _roboflow_infer(
            image, ROBOFLOW_MODELS["illegal_parking"], api_key
        )

        violation_detected = False
        max_conf = 0.0
        violation_bbox = None

        for pred in preds:
            cls_name = pred.get("class", "").lower()
            conf = pred.get("confidence", 0.0)

            illegal_labels = [
                "illegal", "illegal-parking", "illegal_parking",
                "illegally parked", "no parking",
            ]
            if cls_name in illegal_labels and conf >= confidence_threshold:
                if conf > max_conf:
                    violation_detected = True
                    max_conf = conf
                    violation_bbox = _bbox_from_roboflow_pred(pred)

        return {
            "violation_type": vtype,
            "detected": violation_detected,
            "confidence": round(max_conf, 4),
            "bbox": violation_bbox,
            "extra": {"source": "roboflow_api", "predictions": len(preds)},
        }

    except Exception as e:
        return _safe_result(vtype, error=str(e))
