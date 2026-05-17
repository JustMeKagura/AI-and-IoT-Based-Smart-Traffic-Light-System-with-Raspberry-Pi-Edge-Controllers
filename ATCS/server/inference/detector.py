"""
server/inference/detector.py
─────────────────────────────────────────────────────────────────────────────
Responsibility : Run YOLOv8s inference on a preprocessed BGR frame and return
                 raw vehicle detections for counter.py to aggregate.

Design
    - Detector is a class; instantiated once by main.py and reused every frame.
    - Model loading is lazy (first call to load()) and guarded against repeat calls.
    - Inference is stateless: same input → same output, no side effects.
    - This module does NOT count or filter per lane — that is counter.py's job.

Output contract
    A list of Detection namedtuples:
        [ Detection(class_id, confidence, bbox), ... ]
    where bbox = (x1, y1, x2, y2) in absolute pixel coords of the 640×640 frame.
    Empty list = valid result (no vehicles in frame).

COCO vehicle class IDs (filtered here, documented in shared/constants.py)
    2  → car
    3  → motorcycle
    5  → bus
    7  → truck

Author : Oussama (server side)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Module-level import so tests can patch 'server.inference.detector.YOLO'.
# Guarded so the module remains importable without ultralytics installed
# (CI, edge device, unit tests). load() raises a clear DetectorError if
# ultralytics is genuinely absent at runtime.
try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

VEHICLE_CLASS_IDS: frozenset[int] = frozenset([2, 3, 5, 7])  # car, motorcycle, bus, truck
DEFAULT_CONF: float = 0.15          # From README spec
DEFAULT_IMGSZ: int = 640            # Must match preprocessor TARGET_SIZE
EXPECTED_SHAPE: tuple[int, int, int] = (640, 640, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────────────

class DetectorError(RuntimeError):
    """Raised for unrecoverable detector failures (bad model, bad input, etc.)."""


class DetectorNotLoadedError(DetectorError):
    """Raised when inference is attempted before load() has been called."""


@dataclass(frozen=True)
class Detection:
    """A single vehicle detection from one inference pass."""
    class_id: int                        # COCO class ID (2, 3, 5, or 7)
    confidence: float                    # Model confidence [0.0 – 1.0]
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) absolute pixels

    @property
    def label(self) -> str:
        return {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}.get(
            self.class_id, f"unknown({self.class_id})"
        )

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class InferenceResult:
    """Full result of one inference call, including diagnostics."""
    detections: list[Detection]          # Filtered vehicle detections only
    raw_count: int                       # Total YOLO detections before class filter
    vehicle_count: int                   # Detections after COCO vehicle filter
    inference_ms: float                  # Wall-clock time for YOLO forward pass
    frame_shape: tuple[int, ...]         # Shape of the input frame


# ──────────────────────────────────────────────────────────────────────────────
# Detector class
# ──────────────────────────────────────────────────────────────────────────────

class Detector:
    """
    YOLOv8s vehicle detector.

    Usage
    -----
        detector = Detector(model_path="weights/yolov8s.pt")
        detector.load()                         # once, at startup
        result = detector.predict(frame)        # every frame
        detector.unload()                       # optional, frees GPU memory
    """

    def __init__(
        self,
        model_path: str | Path,
        conf: float = DEFAULT_CONF,
        imgsz: int = DEFAULT_IMGSZ,
        device: str = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        model_path : str | Path
            Path to the YOLOv8s weights file (.pt).
        conf : float
            Confidence threshold for detections (default 0.15 per spec).
        imgsz : int
            Inference image size — must match preprocessor output (640).
        device : str
            'cpu', 'cuda', or 'cuda:0' etc.
        """
        self._model_path = Path(model_path)
        self._conf = conf
        self._imgsz = imgsz
        self._device = device
        self._model = None          # set by load()
        self._loaded: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load the YOLOv8s model from disk. Call once at startup.

        Raises
        ------
        DetectorError
            If the model file is missing, unreadable, or fails to load.
        """
        if self._loaded:
            log.warning("Detector.load() called again — already loaded, skipping.")
            return

        if not self._model_path.exists():
            raise DetectorError(
                f"Model file not found: '{self._model_path}'. "
                "Check model_path in config.yaml."
            )

        if not self._model_path.is_file():
            raise DetectorError(
                f"Model path exists but is not a file: '{self._model_path}'."
            )

        try:
            if YOLO is None:
                raise ImportError("ultralytics could not be imported at startup.")

            log.info("Loading YOLOv8s from '%s' on device='%s'…", self._model_path, self._device)
            self._model = YOLO(str(self._model_path))
            # Warm-up: run a dummy frame so the first real frame isn't slow
            self._warmup()
            self._loaded = True
            log.info("Detector ready (conf=%.2f, imgsz=%d).", self._conf, self._imgsz)

        except ImportError as exc:
            raise DetectorError(
                "ultralytics package is not installed. "
                "Run: pip install ultralytics"
            ) from exc

        except Exception as exc:
            raise DetectorError(
                f"Failed to load model from '{self._model_path}': {exc}"
            ) from exc

    def unload(self) -> None:
        """Release the model and free GPU memory (if applicable)."""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        log.info("Detector unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> InferenceResult:
        """
        Run YOLOv8s on a preprocessed frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 array of shape (640, 640, 3) — output of preprocessor.py.

        Returns
        -------
        InferenceResult
            Filtered vehicle detections + diagnostics.
            Empty detections list is a valid result (no vehicles in frame).

        Raises
        ------
        DetectorNotLoadedError
            If load() has not been called yet.
        DetectorError
            If the frame is invalid or inference fails unexpectedly.
        """
        if not self._loaded:
            raise DetectorNotLoadedError(
                "Detector.predict() called before load(). Call detector.load() at startup."
            )

        self._validate_frame(frame)

        try:
            t0 = time.perf_counter()
            results = self._model.predict(
                source=frame,
                conf=self._conf,
                imgsz=self._imgsz,
                device=self._device,
                verbose=False,      # suppress per-frame ultralytics stdout spam
            )
            inference_ms = (time.perf_counter() - t0) * 1000

        except Exception as exc:
            raise DetectorError(f"YOLO inference failed: {exc}") from exc

        detections = self._parse_results(results)

        raw_count = sum(
            len(r.boxes) for r in results if r.boxes is not None
        )
        vehicle_count = len(detections)

        log.debug(
            "Inference done | %.1f ms | raw=%d | vehicles=%d",
            inference_ms, raw_count, vehicle_count,
        )

        return InferenceResult(
            detections=detections,
            raw_count=raw_count,
            vehicle_count=vehicle_count,
            inference_ms=inference_ms,
            frame_shape=frame.shape,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _warmup(self) -> None:
        """Run one dummy inference to initialise CUDA kernels / JIT compilation."""
        try:
            dummy = np.zeros(EXPECTED_SHAPE, dtype=np.uint8)
            self._model.predict(
                source=dummy,
                conf=self._conf,
                imgsz=self._imgsz,
                device=self._device,
                verbose=False,
            )
            log.debug("Warm-up inference complete.")
        except Exception as exc:
            # Warm-up failure is non-fatal — log and continue
            log.warning("Warm-up inference failed (non-fatal): %s", exc)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """
        Validate that the frame meets the contract from preprocessor.py.

        Raises DetectorError on any violation.
        """
        if frame is None:
            raise DetectorError("Frame is None — did preprocessor.preprocess() succeed?")

        if not isinstance(frame, np.ndarray):
            raise DetectorError(
                f"Frame must be a numpy ndarray, got {type(frame).__name__}."
            )

        if frame.dtype != np.uint8:
            raise DetectorError(
                f"Frame dtype must be uint8, got {frame.dtype}. "
                "Ensure preprocessor returns uint8."
            )

        if frame.shape != EXPECTED_SHAPE:
            raise DetectorError(
                f"Frame shape must be {EXPECTED_SHAPE}, got {frame.shape}. "
                "Ensure preprocessor resizes to 640×640×3."
            )

        if frame.size == 0:
            raise DetectorError("Frame has zero elements (empty array).")

    @staticmethod
    def _parse_results(results: list) -> list[Detection]:
        """
        Extract and filter detections from YOLO results.

        Only returns detections whose class_id is in VEHICLE_CLASS_IDS.
        Non-vehicle classes (people, animals, signs, etc.) are silently dropped.
        """
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                try:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    if class_id not in VEHICLE_CLASS_IDS:
                        continue  # pedestrian, dog, bicycle, etc.

                    detections.append(Detection(
                        class_id=class_id,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    ))

                except (IndexError, AttributeError, ValueError) as exc:
                    # A malformed box should not crash the whole frame
                    log.warning("Skipping malformed detection box: %s", exc)
                    continue

        return detections