"""
server/inference/preprocessor.py
─────────────────────────────────────────────────────────────────────────────
Responsibility : Turn a raw MQTT payload (Base64-encoded JPEG) into a clean
                 BGR NumPy array that YOLOv8 can consume without surprises.

Pipeline (in order)
    1. Validate  – reject None / empty / non-string input early
    2. Decode    – Base64 → bytes → BGR NumPy array via cv2.imdecode
    3. Sanitise  – force 3-channel BGR (handles RGBA, grayscale, palette PNGs)
    4. CLAHE     – per-channel contrast equalisation on the L channel (LAB space)
                   improves detection confidence at night / dusk
    5. Denoise   – fast bilateral filter (edges preserved, noise removed)
    6. Resize    – letterbox to exactly 640 × 640 (no squash distortion)

Returns
    PreprocessResult  – a small dataclass carrying the frame and diagnostic info
    raises PreprocessError on unrecoverable input

Author : Oussama  (server side)
"""

from __future__ import annotations

import base64
import binascii
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TARGET_SIZE: int = 640          # YOLO input dimension (square)
CLAHE_CLIP: float = 2.0         # Clip limit – higher = more aggressive enhancement
CLAHE_TILE: tuple[int, int] = (8, 8)   # Tile grid size for CLAHE

# Bilateral filter parameters  (sigmaColor/sigmaSpace trade-off: speed vs quality)
BILATERAL_D: int = 5            # Neighbourhood diameter (5 = fast, 9 = thorough)
BILATERAL_SIGMA_COLOR: float = 50.0
BILATERAL_SIGMA_SPACE: float = 50.0

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────────────

class PreprocessError(ValueError):
    """Raised when the incoming payload cannot be turned into a usable frame."""


@dataclass
class PreprocessResult:
    """Carries the processed frame and lightweight diagnostics."""
    frame: np.ndarray                        # BGR uint8, shape (640, 640, 3)
    original_shape: tuple[int, int, int]     # (H, W, C) before any processing
    was_resized: bool                        # True if the image was not already 640×640
    was_converted: bool                      # True if channel count was corrected
    letterbox_pad: tuple[int, int] = field(default=(0, 0))  # (top+bottom, left+right) padding px


# ──────────────────────────────────────────────────────────────────────────────
# CLAHE helper  (operates in LAB colour space so colour balance is untouched)
# ──────────────────────────────────────────────────────────────────────────────

_clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)


def _apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """
    Enhance contrast without blowing out colours.
    Converts BGR → LAB, equalises L channel only, converts back.
    Safe against non-uint8 input: will cast and warn.
    """
    if bgr.dtype != np.uint8:
        log.warning("CLAHE received non-uint8 frame (dtype=%s); casting.", bgr.dtype)
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = _clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ──────────────────────────────────────────────────────────────────────────────
# Letterbox resize  (maintains aspect ratio → no squash distortion for YOLO)
# ──────────────────────────────────────────────────────────────────────────────

def _letterbox(
    img: np.ndarray,
    target: int = TARGET_SIZE,
    pad_colour: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Resize img to (target × target) with grey padding, preserving aspect ratio.

    Returns
        (padded_image, (vertical_pad_total, horizontal_pad_total))
    """
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top    = (target - new_h) // 2
    pad_bottom =  target - new_h - pad_top
    pad_left   = (target - new_w) // 2
    pad_right  =  target - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_colour,
    )
    return padded, (pad_top + pad_bottom, pad_left + pad_right)


# ──────────────────────────────────────────────────────────────────────────────
# Main public entry point
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(payload: Optional[str]) -> PreprocessResult:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    payload : str | None
        Base64-encoded JPEG string as received from the MQTT broker.

    Returns
    -------
    PreprocessResult
        Processed frame + diagnostics.

    Raises
    ------
    PreprocessError
        On any unrecoverable input problem (empty, corrupt, non-image bytes).
    """

    # ── Stage 1 : Input validation ────────────────────────────────────────────
    if payload is None:
        raise PreprocessError("Payload is None – nothing to process.")

    if not isinstance(payload, (str, bytes)):
        raise PreprocessError(
            f"Expected str or bytes payload, got {type(payload).__name__}."
        )

    if len(payload) == 0:
        raise PreprocessError("Payload is empty (zero-length string).")

    # ── Stage 2 : Base64 decode ───────────────────────────────────────────────
    try:
        # Strip whitespace / newlines that can sneak in over MQTT
        if isinstance(payload, str):
            payload = payload.strip()
        raw_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PreprocessError(f"Base64 decode failed: {exc}") from exc

    if len(raw_bytes) == 0:
        raise PreprocessError("Base64 decoded to zero bytes – payload was padding-only.")

    # ── Stage 3 : Bytes → BGR NumPy array ────────────────────────────────────
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)

    # Use IMREAD_UNCHANGED to preserve the *real* channel count so we can
    # detect and log grayscale / RGBA inputs before normalising them.
    # IMREAD_COLOR would silently flatten everything to BGR and we'd never know.
    frame = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if frame is None:
        # cv2.imdecode returns None for non-image bytes (random data, truncated JPEG…)
        raise PreprocessError(
            f"cv2.imdecode returned None – bytes ({len(raw_bytes)} B) are not a "
            "valid image (corrupted, truncated, or unsupported format)."
        )

    # Record original shape before any conversion (may be 2D for true grayscale)
    original_shape: tuple = frame.shape  # (H, W) or (H, W, C)

    # ── Stage 4 : Channel sanitisation ───────────────────────────────────────
    was_converted = False

    if frame.ndim == 2:
        # Pure grayscale – promote to 3-channel BGR
        log.warning("Received grayscale image; converting to BGR.")
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        was_converted = True

    elif frame.shape[2] == 4:
        # BGRA (e.g. PNG with transparency) – drop alpha channel
        log.warning("Received 4-channel image (BGRA); dropping alpha.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        was_converted = True

    elif frame.shape[2] == 1:
        # Single-channel stored as 3D array – promote to BGR
        log.warning("Received single-channel 3D image; converting to BGR.")
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        was_converted = True

    elif frame.shape[2] != 3:
        raise PreprocessError(
            f"Unexpected channel count: {frame.shape[2]}. Expected 1, 3, or 4."
        )

    # ── Stage 5 : CLAHE contrast enhancement ─────────────────────────────────
    frame = _apply_clahe(frame)

    # ── Stage 6 : Bilateral denoise ───────────────────────────────────────────
    frame = cv2.bilateralFilter(
        frame,
        BILATERAL_D,
        BILATERAL_SIGMA_COLOR,
        BILATERAL_SIGMA_SPACE,
    )

    # ── Stage 7 : Letterbox resize to 640 × 640 ──────────────────────────────
    h, w = frame.shape[:2]
    was_resized = not (h == TARGET_SIZE and w == TARGET_SIZE)
    letterbox_pad = (0, 0)

    if was_resized:
        frame, letterbox_pad = _letterbox(frame, TARGET_SIZE)

    # ── Final sanity check ────────────────────────────────────────────────────
    assert frame.shape == (TARGET_SIZE, TARGET_SIZE, 3), (
        f"Post-processing shape mismatch: {frame.shape}"
    )
    assert frame.dtype == np.uint8, f"Post-processing dtype mismatch: {frame.dtype}"

    log.debug(
        "Preprocessed OK | original=%s → resized=%s | "
        "channels_converted=%s | letterbox_pad=%s",
        original_shape,
        frame.shape,
        was_converted,
        letterbox_pad,
    )

    return PreprocessResult(
        frame=frame,
        original_shape=original_shape,
        was_resized=was_resized,
        was_converted=was_converted,
        letterbox_pad=letterbox_pad,
    )