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
        def _enforce_max_size(img_array: np.ndarray, max_dim: int = 2560) -> np.ndarray:
            h, w = img_array.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                return cv2.resize(img_array, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            return img_array

        # Already a numpy array
        if isinstance(source, np.ndarray):
            if len(source.shape) == 2:
                img = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            elif source.shape[2] == 4:
                img = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
            else:
                img = source
            return _enforce_max_size(img)

        # PIL Image
        if isinstance(source, Image.Image):
            source.thumbnail((2560, 2560), Image.Resampling.LANCZOS)
            img_rgb = np.array(source.convert("RGB"))
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Bytes or Streamlit UploadedFile (has .read())
        if hasattr(source, "read") or isinstance(source, bytes):
            import io
            file_bytes = source.read() if hasattr(source, "read") else source
            try:
                # Use PIL for memory-efficient loading of massive JPEGs
                pil_img = Image.open(io.BytesIO(file_bytes))
                from PIL import ImageOps
                pil_img = ImageOps.exif_transpose(pil_img) # Fix orientation
                pil_img.thumbnail((2560, 2560), Image.Resampling.LANCZOS)
                img_rgb = np.array(pil_img.convert("RGB"))
                return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                # Fallback to cv2
                np_bytes = np.frombuffer(file_bytes, dtype=np.uint8)
                img = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Failed to decode image from bytes.")
                return _enforce_max_size(img)

        # File path (str or Path)
        if isinstance(source, (str, Path)):
            path = str(source)
            try:
                pil_img = Image.open(path)
                from PIL import ImageOps
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_img.thumbnail((2560, 2560), Image.Resampling.LANCZOS)
                img_rgb = np.array(pil_img.convert("RGB"))
                return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to load image from path: {path}")
                return _enforce_max_size(img)

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
