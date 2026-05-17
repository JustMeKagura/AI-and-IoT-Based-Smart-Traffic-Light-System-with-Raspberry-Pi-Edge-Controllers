"""
server/inference/counter.py
─────────────────────────────────────────────────────────────────────────────
Responsibility : Aggregate Detection objects (from detector.py) into per-lane
                 vehicle counts that timing_algo.py can act on.

Design
    - Lane geometry is injected at construction time (loaded from config.yaml
      by the caller). counter.py owns no I/O and holds no global state.
    - A detection is assigned to the first lane whose rectangle contains the
      detection's bbox centre. First-match = deterministic, no double-counting.
    - Detections whose centre falls outside every lane are recorded in an
      'unassigned' bucket — never silently dropped, so a misconfigured lane
      map is immediately visible in the diagnostics.
    - Lane regions with zero area or inverted coordinates are rejected at
      construction time with a clear error.

Input contract  (from detector.py)
    list[Detection]  — already filtered to COCO vehicle classes [2, 3, 5, 7]
    bboxes are absolute pixel coordinates on the 640 × 640 frame.

Output contract  (to smoother.py)
    LaneCountResult  — per-lane counts dict + diagnostics

Lane config format  (loaded from config.yaml by the caller)
    {
        "north": (x1, y1, x2, y2),   # pixel coords, top-left to bottom-right
        "south": (x1, y1, x2, y2),
        ...
    }

Author : Oussama (server side)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from server.inference.detector import Detection

log = logging.getLogger(__name__)

# Type alias for a lane region rectangle
LaneRegion = tuple[float, float, float, float]   # (x1, y1, x2, y2)
LaneMap    = dict[str, LaneRegion]               # {lane_id: region}


# ──────────────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────────────

class CounterError(ValueError):
    """Raised for unrecoverable counter configuration or input errors."""


@dataclass
class LaneCountResult:
    """
    Per-lane vehicle counts and diagnostics for one inference frame.

    Attributes
    ----------
    counts : dict[str, int]
        {lane_id: vehicle_count} for every configured lane.
        Lanes with zero detections are still present (value = 0).
    total : int
        Sum of all per-lane counts (excludes unassigned).
    unassigned : int
        Detections whose centre fell outside every configured lane.
        Non-zero value indicates a lane map misconfiguration.
    detections_processed : int
        Total number of Detection objects received (for logging/debugging).
    per_class : dict[str, dict[int, int]]
        {lane_id: {class_id: count}} — breakdown by vehicle type per lane.
    """
    counts: dict[str, int]
    total: int
    unassigned: int
    detections_processed: int
    per_class: dict[str, dict[int, int]] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Counter class
# ──────────────────────────────────────────────────────────────────────────────

class Counter:
    """
    Aggregates vehicle detections into per-lane counts.

    Usage
    -----
        lanes = {"north": (0, 0, 320, 640), "south": (320, 0, 640, 640)}
        counter = Counter(lanes)
        result  = counter.count(detections)
    """

    def __init__(self, lanes: LaneMap) -> None:
        """
        Parameters
        ----------
        lanes : LaneMap
            Dict mapping lane IDs to pixel-space rectangles (x1, y1, x2, y2).
            Validated eagerly — raises CounterError on bad geometry.

        Raises
        ------
        CounterError
            If lanes is None, empty, or contains invalid regions.
        """
        if lanes is None:
            raise CounterError("lanes map is None — pass a dict of lane regions.")

        if not isinstance(lanes, dict):
            raise CounterError(
                f"lanes must be a dict, got {type(lanes).__name__}."
            )

        if len(lanes) == 0:
            raise CounterError(
                "lanes map is empty. Define at least one lane region in config.yaml."
            )

        self._lanes: LaneMap = {}
        for lane_id, region in lanes.items():
            self._validate_region(lane_id, region)
            self._lanes[lane_id] = tuple(float(v) for v in region)  # type: ignore

        log.info("Counter initialised with %d lane(s): %s", len(self._lanes), list(self._lanes))

    # ── Public API ────────────────────────────────────────────────────────────

    def count(self, detections: Optional[list[Detection]]) -> LaneCountResult:
        """
        Assign each detection to a lane and return per-lane counts.

        Parameters
        ----------
        detections : list[Detection] | None
            Output of detector.predict().detections.
            None is treated as an empty list (frame with no vehicles).

        Returns
        -------
        LaneCountResult
            Per-lane counts + diagnostics. Always returns a result, never raises
            for empty input or zero detections.

        Raises
        ------
        CounterError
            If detections is the wrong type, or contains a non-Detection element.
        """
        if detections is None:
            log.debug("count() received None — treating as empty detection list.")
            detections = []

        if not isinstance(detections, list):
            raise CounterError(
                f"detections must be a list, got {type(detections).__name__}."
            )

        # Validate elements before processing anything
        for i, det in enumerate(detections):
            if not isinstance(det, Detection):
                raise CounterError(
                    f"detections[{i}] is {type(det).__name__}, expected Detection."
                )

        # Initialise per-lane accumulators
        counts:    dict[str, int]              = {lid: 0 for lid in self._lanes}
        per_class: dict[str, dict[int, int]]   = {lid: {} for lid in self._lanes}
        unassigned = 0

        for det in detections:
            cx, cy = self._bbox_centre(det.bbox)

            # Skip detections with non-finite centre coordinates
            if not (math.isfinite(cx) and math.isfinite(cy)):
                log.warning(
                    "Detection has non-finite bbox centre (%.2f, %.2f) — skipping.",
                    cx, cy,
                )
                unassigned += 1
                continue

            assigned = False
            for lane_id, (x1, y1, x2, y2) in self._lanes.items():
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    counts[lane_id] += 1
                    per_class[lane_id][det.class_id] = (
                        per_class[lane_id].get(det.class_id, 0) + 1
                    )
                    assigned = True
                    break   # first-match: no double-counting

            if not assigned:
                log.debug(
                    "Detection centre (%.1f, %.1f) class=%d fell outside all lanes.",
                    cx, cy, det.class_id,
                )
                unassigned += 1

        if unassigned > 0:
            log.warning(
                "%d detection(s) fell outside all configured lane regions. "
                "Check lane geometry in config.yaml.",
                unassigned,
            )

        total = sum(counts.values())

        log.debug(
            "Count result | total=%d | unassigned=%d | per_lane=%s",
            total, unassigned, counts,
        )

        return LaneCountResult(
            counts=counts,
            total=total,
            unassigned=unassigned,
            detections_processed=len(detections),
            per_class=per_class,
        )

    @property
    def lane_ids(self) -> list[str]:
        """Ordered list of configured lane identifiers."""
        return list(self._lanes.keys())

    @property
    def lane_regions(self) -> LaneMap:
        """Read-only copy of the lane geometry map."""
        return dict(self._lanes)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _bbox_centre(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """Return the (cx, cy) centre of a bounding box."""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _validate_region(lane_id: str, region: object) -> None:
        """
        Validate a single lane region.

        Rules:
          - Must be a sequence of exactly 4 numbers
          - All values must be finite
          - x1 < x2 and y1 < y2 (non-zero area, correct orientation)

        Raises CounterError on any violation.
        """
        try:
            values = tuple(region)  # type: ignore
        except TypeError:
            raise CounterError(
                f"Lane '{lane_id}': region must be an iterable of 4 numbers, "
                f"got {type(region).__name__}."
            )

        if len(values) != 4:
            raise CounterError(
                f"Lane '{lane_id}': region must have exactly 4 values "
                f"(x1, y1, x2, y2), got {len(values)}."
            )

        try:
            x1, y1, x2, y2 = (float(v) for v in values)
        except (TypeError, ValueError) as exc:
            raise CounterError(
                f"Lane '{lane_id}': region values must be numeric: {exc}"
            ) from exc

        for name, val in [("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)]:
            if not math.isfinite(val):
                raise CounterError(
                    f"Lane '{lane_id}': {name}={val} is not finite."
                )

        if x1 >= x2:
            raise CounterError(
                f"Lane '{lane_id}': x1 ({x1}) must be strictly less than x2 ({x2}). "
                "Region has zero width or is inverted."
            )

        if y1 >= y2:
            raise CounterError(
                f"Lane '{lane_id}': y1 ({y1}) must be strictly less than y2 ({y2}). "
                "Region has zero height or is inverted."
            )