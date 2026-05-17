"""
server/logic/smoother.py
─────────────────────────────────────────────────────────────────────────────
Responsibility : Stabilise noisy per-lane vehicle counts from counter.py
                 using a rolling window average, preventing single-frame
                 glitches from triggering bad timing decisions.

Design
    - Smoother is a class; one instance lives for the lifetime of the server
      process, called once per frame.
    - Per-lane history is stored in a deque of fixed maxlen (window_size).
    - Output counts are floored integers — timing_algo.py works in whole
      vehicles, never fractions.
    - Lane set changes mid-run (config reload) are handled gracefully:
        · New lane  → history starts fresh for that lane.
        · Gone lane → history entry is silently pruned.
    - Negative raw counts are clamped to zero with a warning (defensive;
      counter.py should never produce them).

Input contract  (from counter.py)
    LaneCountResult — specifically result.counts: dict[str, int]

Output contract  (to timing_algo.py)
    SmoothedResult  — smoothed per-lane int counts + window diagnostics

Author : Oussama (server side)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

from server.inference.counter import LaneCountResult

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_WINDOW_SIZE: int = 3    # From README spec


# ──────────────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────────────

class SmootherError(ValueError):
    """Raised for unrecoverable smoother configuration or input errors."""


@dataclass
class SmoothedResult:
    """
    Stabilised per-lane vehicle counts ready for timing_algo.py.

    Attributes
    ----------
    counts : dict[str, int]
        {lane_id: smoothed_vehicle_count} — floored integer means.
    window_size : int
        Configured maximum window length.
    samples_used : dict[str, int]
        {lane_id: n} — how many frames are in the window for each lane.
        Less than window_size during warm-up or after a lane config change.
    raw_counts : dict[str, int]
        The unsmoothed counts from this frame (passthrough for logging).
    """
    counts: dict[str, int]
    window_size: int
    samples_used: dict[str, int]
    raw_counts: dict[str, int]

    @property
    def is_warm(self) -> bool:
        """True once every lane has a full window of samples."""
        if not self.samples_used:
            return False
        return all(n >= self.window_size for n in self.samples_used.values())


# ──────────────────────────────────────────────────────────────────────────────
# Smoother class
# ──────────────────────────────────────────────────────────────────────────────

class Smoother:
    """
    Rolling-window mean smoother for per-lane vehicle counts.

    Usage
    -----
        smoother = Smoother(window_size=3)
        result   = smoother.update(lane_count_result)   # every frame
        smoother.reset()                                 # optional: clear history
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        """
        Parameters
        ----------
        window_size : int
            Number of consecutive frames to average over (default 3).
            Must be >= 1.

        Raises
        ------
        SmootherError
            If window_size is not a positive integer.
        """
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise SmootherError(
                f"window_size must be an int, got {type(window_size).__name__}."
            )
        if window_size < 1:
            raise SmootherError(
                f"window_size must be >= 1, got {window_size}."
            )

        self._window_size = window_size
        # {lane_id: deque[int]} — deque enforces maxlen automatically
        self._history: dict[str, deque[int]] = {}
        log.info("Smoother initialised (window_size=%d).", window_size)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, result: LaneCountResult) -> SmoothedResult:
        """
        Ingest one frame's lane counts and return smoothed values.

        Parameters
        ----------
        result : LaneCountResult
            Output of counter.count(). Uses result.counts.

        Returns
        -------
        SmoothedResult
            Smoothed per-lane counts + window diagnostics.

        Raises
        ------
        SmootherError
            If result is None, wrong type, or contains non-integer counts.
        """
        if result is None:
            raise SmootherError("result is None — pass a LaneCountResult.")

        if not isinstance(result, LaneCountResult):
            raise SmootherError(
                f"Expected LaneCountResult, got {type(result).__name__}."
            )

        raw_counts: dict[str, int] = result.counts
        current_lanes = set(raw_counts.keys())

        # ── Prune lanes that have disappeared from the current count ──────────
        gone = set(self._history.keys()) - current_lanes
        for lane_id in gone:
            log.warning("Lane '%s' disappeared from counts — pruning history.", lane_id)
            del self._history[lane_id]

        # ── Update history for each current lane ──────────────────────────────
        for lane_id, raw_count in raw_counts.items():

            # Validate count value
            if not isinstance(raw_count, int):
                raise SmootherError(
                    f"counts['{lane_id}'] must be an int, got "
                    f"{type(raw_count).__name__} (value={raw_count!r})."
                )

            # Clamp negatives (defensive — counter.py should not produce them)
            if raw_count < 0:
                log.warning(
                    "Negative count %d for lane '%s' clamped to 0.", raw_count, lane_id
                )
                raw_count = 0

            # Create history deque on first appearance of this lane
            if lane_id not in self._history:
                log.debug("New lane '%s' seen — starting fresh history.", lane_id)
                self._history[lane_id] = deque(maxlen=self._window_size)

            self._history[lane_id].append(raw_count)

        # ── Compute smoothed values ───────────────────────────────────────────
        smoothed_counts: dict[str, int] = {}
        samples_used:   dict[str, int] = {}

        for lane_id, buf in self._history.items():
            n = len(buf)
            mean = sum(buf) / n
            smoothed_counts[lane_id] = math.floor(mean)
            samples_used[lane_id]    = n

        log.debug(
            "Smoother update | raw=%s → smoothed=%s | warm=%s",
            raw_counts,
            smoothed_counts,
            all(n >= self._window_size for n in samples_used.values()),
        )

        return SmoothedResult(
            counts=smoothed_counts,
            window_size=self._window_size,
            samples_used=samples_used,
            raw_counts=dict(raw_counts),
        )

    def reset(self) -> None:
        """Clear all lane history. Call after a config reload or scene change."""
        self._history.clear()
        log.info("Smoother history reset.")

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def is_warm(self) -> bool:
        """True once every tracked lane has a full window of samples."""
        if not self._history:
            return False
        return all(len(buf) >= self._window_size for buf in self._history.values())

    @property
    def tracked_lanes(self) -> list[str]:
        """Lane IDs currently tracked in history."""
        return list(self._history.keys())