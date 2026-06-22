"""
License Plate Detection + OCR Module.

Primary backend: Indian_LPR (pretrained weights)
Fallback backend: EasyOCR

Returns plate text, bounding box, and confidence for every image.
"""

import numpy as np
from pathlib import Path
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LPR_DIR, LPR_OD_WEIGHTS, LPR_OCR_WEIGHTS


# Global state for lazy loading
_easyocr_reader = None
_lpr_available = None


def _check_lpr_available() -> bool:
    """Check if Indian_LPR repo and weights are set up."""
    global _lpr_available
    if _lpr_available is not None:
        return _lpr_available
    _lpr_available = (
        LPR_DIR.exists()
        and LPR_OD_WEIGHTS.exists()
        and LPR_OCR_WEIGHTS.exists()
    )
    return _lpr_available


def _get_easyocr_reader():
    """Lazy-load EasyOCR reader (heavy import)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


def _clean_plate_text(raw_text: str) -> str:
    """
    Clean OCR output to look like an Indian license plate.
    Remove special characters, normalize spacing.
    """
    # Remove non-alphanumeric characters except spaces
    cleaned = re.sub(r"[^A-Za-z0-9\s]", "", raw_text)
    # Remove extra whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned.upper().strip()


def _detect_plate_easyocr(image: np.ndarray) -> dict:
    """
    Use EasyOCR to find and read text in the image.
    Returns the best plate-like detection.
    """
    reader = _get_easyocr_reader()
    # Read standard text restricting to alphanumerics to prevent symbol hallucinations
    results = reader.readtext(
        image, 
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
    )

    if not results:
        return {
            "plate_text": "unknown",
            "plate_bbox": None,
            "plate_confidence": 0.0,
        }

    best_plate = None
    best_conf = 0.0
    best_bbox = None

    for bbox_pts, text, conf in results:
        cleaned = _clean_plate_text(text)
        cleaned_no_space = cleaned.replace(" ", "")
        
        letters = sum(c.isalpha() for c in cleaned_no_space)
        digits = sum(c.isdigit() for c in cleaned_no_space)
        
        # A valid Indian plate will almost always have at least 2 letters and 3 numbers
        # We allow a slightly wider length margin (6 to 14) to catch partially read plates
        if letters >= 2 and digits >= 3 and 6 <= len(cleaned_no_space) <= 14:
            if conf > best_conf:
                best_conf = conf
                best_plate = cleaned
                pts = np.array(bbox_pts)
                x1, y1 = pts.min(axis=0).astype(int)
                x2, y2 = pts.max(axis=0).astype(int)
                best_bbox = [int(x1), int(y1), int(x2), int(y2)]

    if best_plate:
        return {
            "plate_text": best_plate,
            "plate_bbox": best_bbox,
            "plate_confidence": round(best_conf, 4),
        }

    return {
        "plate_text": "unknown",
        "plate_bbox": None,
        "plate_confidence": 0.0,
    }


def _detect_plate_indian_lpr(image: np.ndarray) -> dict:
    """
    Use Indian_LPR pipeline for plate detection + OCR.
    Requires cloned repo and pretrained weights.
    """
    # Add Indian_LPR to path
    sys.path.insert(0, str(LPR_DIR))


    # Load detection model
    # The Indian_LPR repo structure varies; try common patterns
    try:
        # Try loading via their inference script pattern
        from detect import detect_plate as _lpr_detect
        from lprnet import recognize_plate as _lpr_recognize

        # Run detection
        plate_bbox, plate_conf = _lpr_detect(image, str(LPR_OD_WEIGHTS))
        if plate_bbox is None:
            return {
                "plate_text": "unknown",
                "plate_bbox": None,
                "plate_confidence": 0.0,
            }

        # Crop plate and run OCR
        x1, y1, x2, y2 = plate_bbox
        plate_crop = image[y1:y2, x1:x2]
        plate_text = _lpr_recognize(plate_crop, str(LPR_OCR_WEIGHTS))

        return {
            "plate_text": _clean_plate_text(plate_text),
            "plate_bbox": [int(x1), int(y1), int(x2), int(y2)],
            "plate_confidence": round(float(plate_conf), 4),
        }

    except ImportError:
        # If the repo's module structure is different, fall back
        raise ImportError("Indian_LPR module structure not as expected")


def detect_plate(image: np.ndarray) -> dict:
    """
    Main entry point for license plate detection + OCR.

    1. Try Indian_LPR pipeline (if available)
    2. Fall back to EasyOCR
    3. Never crash — returns safe defaults on failure

    Returns:
        {
            "plate_text": str,
            "plate_bbox": list | None,   # [x1, y1, x2, y2]
            "plate_confidence": float,
        }
    """
    safe_default = {
        "plate_text": "unknown",
        "plate_bbox": None,
        "plate_confidence": 0.0,
    }

    # Try Indian_LPR first
    if _check_lpr_available():
        try:
            result = _detect_plate_indian_lpr(image)
            if result["plate_text"] != "unknown":
                result["extra_source"] = "indian_lpr"
                return result
        except Exception:
            pass  # Fall through to EasyOCR

    # Fallback: EasyOCR
    try:
        result = _detect_plate_easyocr(image)
        result["extra_source"] = "easyocr"
        return result
    except Exception as e:
        safe_default["extra_source"] = f"error: {str(e)}"
        return safe_default
