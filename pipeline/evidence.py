"""
Evidence Generation Module.

- Annotate images with bounding boxes, labels, and violation banners
- Store records in SQLite database
- Query/filter records for the dashboard
"""

import cv2
import json
import sqlite3
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DB_PATH,
    ANNOTATED_DIR,
    VIOLATION_COLOR,
    PLATE_COLOR,
    STOP_LINE_COLOR,
    BANNER_COLOR,
    BANNER_TEXT_COLOR,
    VIOLATION_LABELS,
)


# ═══════════════════════════════════════════════
# Image Annotation
# ═══════════════════════════════════════════════

def annotate_image(image: np.ndarray, pipeline_result: dict) -> np.ndarray:
    """
    Draw bounding boxes, labels, and a summary banner on the image.

    - RED boxes for violations
    - BLUE box for license plate
    - RED horizontal line for stop-line
    - Black semi-transparent banner at top with violation summary
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    results = pipeline_result.get("results", {})
    detected_labels = []

    # Draw violation bounding boxes
    for key, result in results.items():
        if not result.get("detected", False):
            continue

        vtype = result.get("violation_type", key)
        label = VIOLATION_LABELS.get(vtype, vtype)
        conf = result.get("confidence", 0.0)
        bbox = result.get("bbox")

        detected_labels.append(f"{label} ({conf:.0%})")

        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), VIOLATION_COLOR, 2)

            # Draw label background
            label_text = f"VIOLATION: {label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            label_y = max(y1 - 5, th + 5)
            cv2.rectangle(
                annotated,
                (x1, label_y - th - 4),
                (x1 + tw + 4, label_y + 2),
                VIOLATION_COLOR,
                -1,
            )
            cv2.putText(
                annotated, label_text,
                (x1 + 2, label_y - 2),
                font, font_scale, BANNER_TEXT_COLOR, thickness,
            )

    # Draw stop-line if applicable
    stop_result = results.get("stop_line", {})
    stop_line_y = stop_result.get("extra", {}).get("stop_line_y_px")
    if stop_line_y is not None:
        line_color = STOP_LINE_COLOR if stop_result.get("detected") else (0, 255, 255)  # Yellow if no violation
        cv2.line(annotated, (0, stop_line_y), (w, stop_line_y), line_color, 2)
        cv2.putText(
            annotated, "STOP LINE",
            (10, stop_line_y - 8),
            font, 0.4, line_color, 1,
        )

    # Draw license plate bounding box
    plate_bbox = pipeline_result.get("plate_bbox")
    plate_text = pipeline_result.get("plate_number", "unknown")
    if plate_bbox and len(plate_bbox) == 4:
        px1, py1, px2, py2 = [int(v) for v in plate_bbox]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w, px2), min(h, py2)

        cv2.rectangle(annotated, (px1, py1), (px2, py2), PLATE_COLOR, 1)
        plate_label = f"PLATE: {plate_text}"
        (tw, th), _ = cv2.getTextSize(plate_label, font, font_scale, thickness)
        cv2.rectangle(
            annotated,
            (px1, py2 + 2),
            (px1 + tw + 4, py2 + th + 8),
            PLATE_COLOR,
            -1,
        )
        cv2.putText(
            annotated, plate_label,
            (px1 + 2, py2 + th + 4),
            font, font_scale, BANNER_TEXT_COLOR, thickness,
        )

    # Draw summary banner at top
    if detected_labels:
        banner_text = "VIOLATIONS: " + " | ".join(detected_labels)
    else:
        banner_text = "NO VIOLATIONS DETECTED"

    banner_h = 35
    # Semi-transparent black banner
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), BANNER_COLOR, -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    cv2.putText(
        annotated, banner_text,
        (10, 24),
        font, 0.55, BANNER_TEXT_COLOR, 1,
    )

    return annotated


def save_annotated_image(
    annotated: np.ndarray,
    image_name: str = None,
) -> str:
    """Save annotated image to output directory. Returns the file path."""
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    if image_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_name = f"annotated_{timestamp}.jpg"
    output_path = ANNOTATED_DIR / image_name
    cv2.imwrite(str(output_path), annotated)
    return str(output_path)


# ═══════════════════════════════════════════════
# SQLite Database
# ═══════════════════════════════════════════════

def init_db(db_path: str = None):
    """Create the violations table if it doesn't exist."""
    if db_path is None:
        db_path = str(DB_PATH)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT,
            annotated_image_path TEXT,
            violations_detected TEXT,
            violation_count INTEGER,
            plate_number TEXT,
            plate_confidence REAL,
            inference_time_ms REAL,
            raw_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(
    pipeline_result: dict,
    annotated_image_path: str = "",
    db_path: str = None,
):
    """Write one violation record to the database."""
    if db_path is None:
        db_path = str(DB_PATH)

    init_db(db_path)

    # Prepare a serializable copy (remove numpy arrays)
    result_copy = {
        k: v for k, v in pipeline_result.items()
        if k not in ("preprocessed_image", "raw_image")
    }

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO violations
            (timestamp, image_path, annotated_image_path,
             violations_detected, violation_count,
             plate_number, plate_confidence, inference_time_ms, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pipeline_result.get("timestamp", datetime.now(timezone.utc).isoformat()),
            pipeline_result.get("image_path", ""),
            annotated_image_path,
            json.dumps(pipeline_result.get("violations_detected", [])),
            pipeline_result.get("violation_count", 0),
            pipeline_result.get("plate_number", "unknown"),
            pipeline_result.get("plate_confidence", 0.0),
            pipeline_result.get("inference_time_ms", 0.0),
            json.dumps(result_copy, default=str),
        ),
    )
    conn.commit()
    conn.close()


def get_records(
    db_path: str = None,
    violation_type: str = None,
    date_from: str = None,
    date_to: str = None,
    plate_number: str = None,
    limit: int = 500,
) -> list:
    """
    Query violation records with optional filters.
    Returns list of dicts.
    """
    if db_path is None:
        db_path = str(DB_PATH)

    init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM violations WHERE 1=1"
    params = []

    if violation_type:
        query += " AND violations_detected LIKE ?"
        params.append(f"%{violation_type}%")

    if date_from:
        query += " AND timestamp >= ?"
        params.append(date_from)

    if date_to:
        query += " AND timestamp <= ?"
        params.append(date_to)

    if plate_number:
        query += " AND plate_number LIKE ?"
        params.append(f"%{plate_number}%")

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_analytics(db_path: str = None) -> dict:
    """
    Compute aggregate analytics from the violations database.
    Returns stats for the dashboard.
    """
    if db_path is None:
        db_path = str(DB_PATH)

    init_db(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Total images processed
    cursor.execute("SELECT COUNT(*) FROM violations")
    total_images = cursor.fetchone()[0]

    # Total violations found
    cursor.execute("SELECT SUM(violation_count) FROM violations")
    total_violations = cursor.fetchone()[0] or 0

    # Average violations per image
    avg_per_image = round(total_violations / max(total_images, 1), 2)

    # Average inference time
    cursor.execute("SELECT AVG(inference_time_ms) FROM violations")
    avg_inference_ms = round(cursor.fetchone()[0] or 0, 2)

    # Violations by type (parse JSON arrays)
    cursor.execute("SELECT violations_detected FROM violations")
    type_counts = {}
    for row in cursor.fetchall():
        try:
            vtypes = json.loads(row[0])
            for vt in vtypes:
                type_counts[vt] = type_counts.get(vt, 0) + 1
        except (json.JSONDecodeError, TypeError):
            pass

    most_common = max(type_counts, key=type_counts.get) if type_counts else "N/A"

    # Violations per day
    cursor.execute("""
        SELECT DATE(timestamp) as day, SUM(violation_count)
        FROM violations
        GROUP BY DATE(timestamp)
        ORDER BY day
    """)
    per_day = [{"date": row[0], "count": row[1]} for row in cursor.fetchall()]

    conn.close()

    return {
        "total_images": total_images,
        "total_violations": total_violations,
        "avg_per_image": avg_per_image,
        "avg_inference_ms": avg_inference_ms,
        "type_counts": type_counts,
        "most_common": most_common,
        "per_day": per_day,
    }
