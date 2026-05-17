"""
server/logic/smoother.py
─────────────────────────────────────────────────────────────────────────────
Responsibility : Stabilise per-lane vehicle counts over a rolling window to
                 eliminate single-frame glitches (a truck occluding the camera,
                 a detection dropout, a noisy frame at dusk).

Design
    - Smoother is a class; instantiated once by main.py and reused every frame.
    - Per-lane deques of fixed maxlen — each lane's history is independent.
    - New lanes are welcomed mid-run; disappeared lanes retain their history
      silently (stale but harmless — timing_algo will only see what it asks for).
    - Window not yet full (startup) is handled naturally: deque average over
      however many frames have arrived so far.
    - Output is always integer counts (floor of mean) — timing_algo works with
      whole vehicles.

Input contract  (from counter.py)
    LaneCountResult  — per-lane integer counts for one frame

Output contract  (to timing_algo.py)
    SmoothedResult   — {lane_id: smoothed_int_count} + diagnostics

Author : Oussama (server side)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from server.inference.counter import LaneCountResult

log = logging.getLogger(__name__)

DEFAULT_WINDOW: int = 3     # README spec: "Moving average (last 3 frames)"


# ──────────────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────────────

class SmootherError(ValueError):
    """Raised for bad configuration or input type errors."""


@dataclass
class SmoothedResult:
    """
    Stabilised per-lane counts and diagnostics for one frame.

    Attributes
    ----------
    counts : dict[str, int]
        {lane_id: smoothed_vehicle_count} — integer floor of the rolling mean.
    total : int
        Sum of all per-lane smoothed counts.
    window_sizes : dict[str, int]
        How many frames are in each lane's window right now.
        < window_size at startup; equals window_size once warmed up.
    raw_counts : dict[str, int]
        The unsmoothed counts from this frame (for logging / diagnostics).
    frame_index : int
        How many frames have been processed so far (1-based).
    """
    counts: dict[str, int]
    total: int
    window_sizes: dict[str, int]
    raw_counts: dict[str, int]
    frame_index: int


# ──────────────────────────────────────────────────────────────────────────────
# Smoother class
# ──────────────────────────────────────────────────────────────────────────────

class Smoother:
    """
    Rolling-average stabiliser for per-lane vehicle counts.

    Usage
    -----
        smoother = Smoother(window=3)
        result   = smoother.update(lane_count_result)
        result   = smoother.update(lane_count_result)   # window fills up
        smoother.reset()                                # flush history
    """

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        """
        Parameters
        ----------
        window : int
            Number of frames to average over. Must be >= 1.
            window=1 is valid — passthrough mode (no smoothing).

        Raises
        ------
        SmootherError
            If window < 1 or is not an integer.
        """
        if not isinstance(window, int):
            raise SmootherError(
                f"window must be an int, got {type(window).__name__}."
            )
        if window < 1:
            raise SmootherError(
                f"window must be >= 1, got {window}."
            )

        self._window: int = window
        # {lane_id: deque([count_t-2, count_t-1, count_t], maxlen=window)}
        self._history: dict[str, deque[int]] = {}
        self._frame_index: int = 0

        log.info("Smoother initialised (window=%d).", self._window)

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, result: LaneCountResult) -> SmoothedResult:
        """
        Push one frame's counts into the rolling window and return smoothed values.

        Parameters
        ----------
        result : LaneCountResult
            Output of counter.count() for the current frame.

        Returns
        -------
        SmoothedResult
            Smoothed integer counts per lane + diagnostics.

        Raises
        ------
        SmootherError
            If result is None, wrong type, or has malformed counts.
        """
        self._validate_input(result)

        self._frame_index += 1
        raw_counts = dict(result.counts)   # snapshot before we touch anything

        # Push current frame counts into each lane's deque
        for lane_id, count in result.counts.items():
            if lane_id not in self._history:
                # New lane seen for the first time — create its deque
                log.debug("Smoother: new lane '%s' added to history.", lane_id)
                self._history[lane_id] = deque(maxlen=self._window)
            self._history[lane_id].append(count)

        # Compute smoothed counts — only for lanes in this frame's result
        smoothed: dict[str, int] = {}
        window_sizes: dict[str, int] = {}

        for lane_id in result.counts:
            hist = self._history[lane_id]
            smoothed[lane_id] = math.floor(sum(hist) / len(hist))
            window_sizes[lane_id] = len(hist)

        total = sum(smoothed.values())

        log.debug(
            "Smoother frame=%d | raw=%s | smoothed=%s | windows=%s",
            self._frame_index, raw_counts, smoothed, window_sizes,
        )

        return SmoothedResult(
            counts=smoothed,
            total=total,
            window_sizes=window_sizes,
            raw_counts=raw_counts,
            frame_index=self._frame_index,
        )

    def reset(self) -> None:
        """
        Flush all lane history and reset the frame counter.
        Call when the system resumes after a pause or MQTT reconnect.
        """
        self._history.clear()
        self._frame_index = 0
        log.info("Smoother reset — history flushed.")

    @property
    def window(self) -> int:
        """Configured window size."""
        return self._window

    @property
    def frame_index(self) -> int:
        """Number of frames processed since last reset (0 = no frames yet)."""
        return self._frame_index

    @property
    def is_warmed_up(self) -> bool:
        """
        True once every tracked lane has seen at least `window` frames.
        False at startup while the window is still filling.
        """
        if not self._history:
            return False
        return all(len(h) >= self._window for h in self._history.values())

    def get_history(self, lane_id: str) -> list[int]:
        """
        Return a copy of the raw count history for a lane.
        Returns an empty list if the lane has never been seen.
        """
        return list(self._history.get(lane_id, []))

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _validate_input(result: object) -> None:
        """
        Validate the LaneCountResult before processing.

        Raises SmootherError on any violation.
        """
        if result is None:
            raise SmootherError(
                "LaneCountResult is None — did counter.count() succeed?"
            )

        if not isinstance(result, LaneCountResult):
            raise SmootherError(
                f"Expected LaneCountResult, got {type(result).__name__}."
            )

        if not isinstance(result.counts, dict):
            raise SmootherError(
                f"LaneCountResult.counts must be a dict, got {type(result.counts).__name__}."
            )

        for lane_id, count in result.counts.items():
            if not isinstance(count, int):
                raise SmootherError(
                    f"counts['{lane_id}'] must be an int, got {type(count).__name__} ({count!r})."
                )
            if count < 0:
                raise SmootherError(
                    f"counts['{lane_id}'] is negative ({count}). Vehicle counts must be >= 0."
                )