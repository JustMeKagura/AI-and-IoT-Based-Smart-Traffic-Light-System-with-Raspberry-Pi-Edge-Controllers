# =============================================================================
# ATCS - Adaptive Traffic Control System
# edge/hardware/camera.py | USB Camera capture & frame encoding
#
# Responsibilities:
#   - Open and configure the USB camera via OpenCV
#   - Capture raw frames and resize to configured dimensions
#   - JPEG-encode and Base64-encode frames ready for FramePayload
#   - Provide a clean release() method for graceful shutdown
#
# Rules:
#   - NO MQTT here. This module only captures and encodes. Publishing
#     is handled by the Reporter thread in edge/main.py.
#   - NO AI inference. Frames are shipped raw to the server.
#   - Must run on ARM (Raspberry Pi 5). Only opencv-python-headless allowed.
#   - All config values come from ATCSConfig — no hardcoded numbers.
# =============================================================================

import base64
import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from shared.config_loader import ATCSConfig

logger = logging.getLogger(__name__)


class Camera:
    """
    Manages the USB camera lifecycle and frame encoding pipeline.

    Pipeline per frame:
        capture → validate → resize → JPEG encode → Base64 encode → return str

    Parameters
    ----------
    config : ATCSConfig
        Loaded system config. Camera settings read from config.edge.
    """

    def __init__(self, config: ATCSConfig) -> None:
        self._cfg = config.edge
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count: int = 0
        self._last_capture_time: float = 0.0

        # Derived interval from configured FPS
        self._capture_interval: float = 1.0 / max(self._cfg.capture_fps, 0.1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """
        Open the USB camera and apply resolution settings.

        Raises
        ------
        RuntimeError
            If the camera device cannot be opened (wrong index, not connected).
        """
        logger.info(
            "Opening camera at index %d (%dx%d @ %.1f FPS)...",
            self._cfg.camera_index,
            self._cfg.frame_width,
            self._cfg.frame_height,
            self._cfg.capture_fps,
        )

        self._cap = cv2.VideoCapture(self._cfg.camera_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera at index {self._cfg.camera_index}. "
                f"Check that the USB camera is connected and not in use."
            )

        # Apply resolution hints (hardware may round to nearest supported size)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.frame_height)

        # Verify actual resolution the device settled on
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Camera opened. Actual resolution: %dx%d (requested: %dx%d).",
            actual_w, actual_h,
            self._cfg.frame_width, self._cfg.frame_height,
        )

        # Warm-up: discard the first few frames (some cameras return
        # a black or corrupted frame on first read)
        self._warmup()

    def _warmup(self, discard_frames: int = 3) -> None:
        """Discard initial frames to let the camera sensor stabilise."""
        logger.debug("Camera warm-up: discarding %d initial frames.", discard_frames)
        for _ in range(discard_frames):
            if self._cap and self._cap.isOpened():
                self._cap.read()
        logger.debug("Camera warm-up complete.")

    def release(self) -> None:
        """Release the camera device. Call on shutdown or in a finally block."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Camera released.")
        self._cap = None

    @property
    def is_open(self) -> bool:
        """True if the camera device is currently open."""
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Frame Capture
    # ------------------------------------------------------------------

    def capture_frame_b64(self) -> Tuple[str, int, int]:
        """
        Capture one frame, resize, JPEG-encode, and Base64-encode it.

        Returns
        -------
        frame_b64 : str
            Base64-encoded JPEG string, ready to set as FramePayload.frame_b64.
        width : int
            Actual frame width after resize (pixels).
        height : int
            Actual frame height after resize (pixels).

        Raises
        ------
        RuntimeError
            If the camera is not open or a frame cannot be read.
        """
        if not self.is_open:
            raise RuntimeError(
                "Camera is not open. Call camera.open() before capturing."
            )

        ret, frame = self._cap.read()  # type: ignore[union-attr]

        if not ret or frame is None:
            raise RuntimeError(
                "Camera read failed. Device may be disconnected or blocked."
            )

        # Resize to configured target dimensions
        frame = self._resize(frame)
        height, width = frame.shape[:2]

        # JPEG encode
        jpeg_bytes = self._encode_jpeg(frame)

        # Base64 encode → UTF-8 string
        frame_b64 = base64.b64encode(jpeg_bytes).decode("utf-8")

        self._frame_count += 1
        self._last_capture_time = time.time()

        logger.debug(
            "Frame #%d captured: %dx%d, JPEG=%d bytes, B64=%d chars.",
            self._frame_count, width, height,
            len(jpeg_bytes), len(frame_b64),
        )

        return frame_b64, width, height

    def should_capture(self) -> bool:
        """
        Rate-limiter: returns True if enough time has passed since the
        last capture, based on capture_fps in config.

        The Reporter thread calls this in its loop to avoid busy-waiting
        and to honour the configured frame rate.
        """
        return (time.time() - self._last_capture_time) >= self._capture_interval

    def wait_until_ready(self) -> None:
        """
        Blocking sleep until the next capture window opens.
        Prevents the Reporter thread from spinning at 100% CPU.
        """
        elapsed = time.time() - self._last_capture_time
        sleep_time = self._capture_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to configured dimensions if needed.
        Uses INTER_AREA for downscaling (best quality for image compression).
        """
        target_w = self._cfg.frame_width
        target_h = self._cfg.frame_height
        h, w = frame.shape[:2]

        if w == target_w and h == target_h:
            return frame  # Already correct size, skip resize

        resized = cv2.resize(
            frame,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )
        logger.debug("Frame resized: %dx%d → %dx%d.", w, h, target_w, target_h)
        return resized

    def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        """
        Encode a BGR numpy array as JPEG bytes.

        Raises
        ------
        RuntimeError
            If OpenCV fails to encode the frame.
        """
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._cfg.jpeg_quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)

        if not success or buffer is None:
            raise RuntimeError("cv2.imencode failed to JPEG-encode the frame.")

        return buffer.tobytes()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Total frames successfully captured since open() was called."""
        return self._frame_count

    def __repr__(self) -> str:
        status = "open" if self.is_open else "closed"
        return (
            f"Camera(index={self._cfg.camera_index}, "
            f"status={status}, "
            f"frames={self._frame_count})"
        )


# =============================================================================
# USAGE EXAMPLE
# Run from the edge/ directory:
#   python hardware/camera.py
# Requires: config.yaml at project root, USB camera connected.
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Allow running from edge/ directory
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from shared.config_loader import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg)

    camera = Camera(cfg)

    try:
        camera.open()
        print(f"Camera ready: {camera}\n")

        print("Capturing 3 test frames...\n")
        for i in range(1, 4):
            camera.wait_until_ready()
            frame_b64, w, h = camera.capture_frame_b64()

            print(f"  Frame {i}: {w}x{h}, B64 length={len(frame_b64)} chars")

            # Decode and verify round-trip
            decoded = base64.b64decode(frame_b64)
            np_arr  = np.frombuffer(decoded, dtype=np.uint8)
            img     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            assert img is not None, "Round-trip decode failed!"
            print(f"           Round-trip decode: ✅ ({img.shape})")

        print(f"\n✅ Camera test passed. Total frames: {camera.frame_count}")

    except RuntimeError as e:
        print(f"❌ Camera error: {e}")
        sys.exit(1)

    finally:
        camera.release()
        print("Camera released.")
