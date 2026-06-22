"""
Image Preprocessing Module.

Handles image loading from multiple sources and enhancement
for low-light, noisy, or blurred traffic camera images.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, DENOISE_H, DENOISE_H_COLOR


def load_image(source) -> np.ndarray:
    """
    Accept file path (str/Path), PIL Image, Streamlit UploadedFile,
    bytes, or numpy array. Returns a BGR numpy array.

    Raises ValueError if the source cannot be loaded.
    """
    try:
        # Already a numpy array
        if isinstance(source, np.ndarray):
            if len(source.shape) == 2:
                return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            if source.shape[2] == 4:
                return cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
            return source

        # PIL Image
        if isinstance(source, Image.Image):
            img_rgb = np.array(source.convert("RGB"))
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Bytes or Streamlit UploadedFile (has .read())
        if hasattr(source, "read"):
            file_bytes = np.frombuffer(source.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from uploaded file.")
            return img

        if isinstance(source, bytes):
            file_bytes = np.frombuffer(source, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from bytes.")
            return img

        # File path (str or Path)
        if isinstance(source, (str, Path)):
            path = str(source)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to load image from path: {path}")
            return img

        raise ValueError(f"Unsupported image source type: {type(source)}")

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Enhance a traffic camera image for better detection accuracy.

    Steps:
      1. Convert to LAB color space
      2. Apply CLAHE to L channel for low-light enhancement
      3. Convert back to BGR
      4. Apply denoising for motion blur / rain artifacts

    Returns the preprocessed BGR image.
    """
    # Step 0: Downscale massive images (e.g., from phones) to avoid extreme CPU lag
    max_dim = 1280
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Step 1-3: CLAHE on luminance channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Step 4: Denoising
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr_enhanced,
        None,
        h=DENOISE_H,
        hColor=DENOISE_H_COLOR,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    return denoised
