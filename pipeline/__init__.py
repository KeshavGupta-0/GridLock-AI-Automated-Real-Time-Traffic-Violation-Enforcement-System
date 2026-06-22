"""
Traffic Violation Detection Pipeline.

Modules:
    preprocess  — Image loading and enhancement (CLAHE, denoising)
    detectors   — One function per violation type (7 total)
    ocr         — License plate detection + OCR (Indian_LPR / EasyOCR)
    orchestrator — run_pipeline() merges all detectors into unified result
    evidence    — Annotate images, store records in SQLite
"""
