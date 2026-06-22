"""
Pipeline Orchestrator.

Runs all 7 detectors + license plate OCR on an image,
merges results into a single unified record.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEFAULT_CONFIDENCE_THRESHOLD, STOP_LINE_Y_FRACTION

from pipeline.preprocess import load_image, preprocess
from pipeline.detectors import (
    detect_helmet,
    detect_triple_riding,
    detect_seatbelt,
    detect_red_light,
    detect_wrong_side,
    detect_stop_line,
    detect_illegal_parking,
)
from pipeline.ocr import detect_plate


def _empty_result(image_path: str = "", error: str = "") -> dict:
    """Return a safe empty pipeline result."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_path": str(image_path),
        "violations_detected": [],
        "violation_count": 0,
        "plate_number": "unknown",
        "plate_confidence": 0.0,
        "inference_time_ms": 0.0,
        "results": {
            "helmet": {"violation_type": "helmet_violation", "detected": False,
                       "confidence": 0.0, "bbox": None, "extra": {"error": error}},
            "triple_riding": {"violation_type": "triple_riding", "detected": False,
                              "confidence": 0.0, "bbox": None, "extra": {"error": error}},
            "seatbelt": {"violation_type": "seatbelt_violation", "detected": False,
                         "confidence": 0.0, "bbox": None, "extra": {"error": error}},
            "red_light": {"violation_type": "red_light_violation", "detected": False,
                          "confidence": 0.0, "bbox": None, "extra": {"error": error}},
            "wrong_side": {"violation_type": "wrong_side_violation", "detected": False,
                           "confidence": 0.0, "bbox": None, "extra": {"error": error}},
            "stop_line": {"violation_type": "stop_line_violation", "detected": False,
                          "confidence": 0.0, "bbox": None, "extra": {"error": error}},
            "illegal_parking": {"violation_type": "illegal_parking", "detected": False,
                                "confidence": 0.0, "bbox": None, "extra": {"error": error}},
        },
    }


def run_pipeline(
    image_source,
    models: dict = None,
    config: dict = None,
) -> dict:
    """
    Run the full detection pipeline on a single image.

    Args:
        image_source: File path, PIL Image, numpy array, bytes, or UploadedFile
        models: dict with optional keys:
            - 'helmet': loaded YOLOv8 model for helmet detection
            - 'seatbelt': loaded YOLOv8 model for seatbelt detection
        config: dict with optional keys:
            - 'api_key': Roboflow API key
            - 'confidence_threshold': float (default 0.4)
            - 'stop_line_y': float (default 0.6)

    Returns:
        Unified pipeline result dict with all violation results + plate OCR.
    """
    if models is None:
        models = {}
    if config is None:
        config = {}

    api_key = config.get("api_key", "")
    conf_thresh = config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    stop_y = config.get("stop_line_y", STOP_LINE_Y_FRACTION)
    image_path = str(image_source) if isinstance(image_source, (str, Path)) else "uploaded_image"

    try:
        # ── Step 1: Load and preprocess ──
        start_time = time.time()
        raw_image = load_image(image_source)
        processed_image = preprocess(raw_image)

        # ── Step 2: Run all detectors ──

        # Violation 1: Helmet (must run first — triple riding depends on it)
        helmet_result = detect_helmet(
            processed_image,
            model=models.get("helmet"),
            confidence_threshold=conf_thresh,
        )

        # Violation 2: Triple riding (logic on top of helmet)
        triple_result = detect_triple_riding(helmet_result)

        # Violation 3: Seatbelt
        seatbelt_result = detect_seatbelt(
            processed_image,
            api_key=api_key,
            model=models.get("seatbelt"),
            confidence_threshold=conf_thresh,
        )

        # Violation 4: Red light (must run before stop-line)
        red_light_result = detect_red_light(
            processed_image,
            api_key=api_key,
            confidence_threshold=conf_thresh,
        )

        # Violation 5: Wrong-side driving
        wrong_side_result = detect_wrong_side(
            processed_image,
            api_key=api_key,
            confidence_threshold=conf_thresh,
        )

        # Violation 6: Stop-line (geometry rule on top of red-light)
        stop_line_result = detect_stop_line(
            processed_image,
            red_light_result=red_light_result,
            stop_line_y=stop_y,
        )

        # Violation 7: Illegal parking
        parking_result = detect_illegal_parking(
            processed_image,
            api_key=api_key,
            confidence_threshold=conf_thresh,
        )

        # ── Step 3: License plate OCR ──
        # Pass the raw_image to OCR because resizing heavily degrades small text readability
        plate_result = detect_plate(raw_image)

        # ── Step 4: Merge results ──
        all_results = {
            "helmet": helmet_result,
            "triple_riding": triple_result,
            "seatbelt": seatbelt_result,
            "red_light": red_light_result,
            "wrong_side": wrong_side_result,
            "stop_line": stop_line_result,
            "illegal_parking": parking_result,
        }

        # Collect detected violations
        violations_detected = [
            key for key, val in all_results.items()
            if val.get("detected", False)
        ]

        inference_time = (time.time() - start_time) * 1000  # ms

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_path": image_path,
            "violations_detected": violations_detected,
            "violation_count": len(violations_detected),
            "plate_number": plate_result.get("plate_text", "unknown"),
            "plate_confidence": plate_result.get("plate_confidence", 0.0),
            "plate_bbox": plate_result.get("plate_bbox"),
            "inference_time_ms": round(inference_time, 2),
            "results": all_results,
            "preprocessed_image": processed_image,  # for annotation
            "raw_image": raw_image,                  # original for display
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        result = _empty_result(image_path, error=str(e))
        result["inference_time_ms"] = 0.0
        return result
