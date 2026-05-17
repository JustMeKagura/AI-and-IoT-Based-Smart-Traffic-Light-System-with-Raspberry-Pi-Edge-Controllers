"""
server/logic/timing_algo.py
─────────────────────────────────────────────────────────────────────────────
Responsibility : Translate stable per-lane vehicle counts (from smoother.py)
                 into green phase durations that get sent back to the Pi.

Algorithms
    WEBSTER  – Classic Webster (1958) optimal cycle formula.
               Best for near-saturated intersections with predictable flow.
               Degrades gracefully to max_green on oversaturation (Y ≥ 1).

    LINEAR   – Simple linear interpolation between min_green and max_green.
               Proportional to count / saturation_count.
               Robust, predictable, good for low-traffic deployments.

Safety contract (from spec / PDF)
    - Every output is clamped to [min_green, max_green].
    - Default min = 10 s, max = 60 s.
    - Oversaturation (Y ≥ 1) → max_green + warning log (never a crash).

Output contract (to mqtt_handler / main.py)
    list[PhaseDecision]  — one per lane in the SmoothedResult

Webster parameters
    saturation_flow  : vehicles/second that can pass through when green
                       (default 0.5 veh/s ≈ 1800 veh/h — standard urban value)
    lost_time        : seconds lost per phase to start-up + clearance
                       (default 2.0 s — standard single-phase assumption)

Author : Oussama (server side)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from server.logic.smoother import SmoothedResult

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants / defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MIN_GREEN: float     = 10.0   # seconds (spec)
DEFAULT_MAX_GREEN: float     = 60.0   # seconds (spec)
DEFAULT_SAT_FLOW:  float     = 0.5    # vehicles / second (≈ 1800 veh/h)
DEFAULT_LOST_TIME: float     = 2.0    # seconds / phase
OVERSATURATION_THRESHOLD: float = 0.95  # warn before Y hits 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────────────

class TimingError(ValueError):
    """Raised for bad configuration or unrecoverable input errors."""


class Algorithm(str, Enum):
    WEBSTER = "webster"
    LINEAR  = "linear"


@dataclass(frozen=True)
class PhaseDecision:
    """
    Green phase duration decision for one lane.

    Attributes
    ----------
    lane_id        : str    – lane this decision applies to
    green_duration : float  – final clamped green time in seconds
    raw_duration   : float  – computed duration before safety clamping
    vehicle_count  : int    – smoothed count that drove this decision
    algorithm      : Algorithm – which formula was used
    was_clamped    : bool   – True if raw_duration != green_duration
    clamp_reason   : str    – 'min', 'max', 'oversaturated', or ''
    """
    lane_id:        str
    green_duration: float
    raw_duration:   float
    vehicle_count:  int
    algorithm:      Algorithm
    was_clamped:    bool
    clamp_reason:   str = ""


@dataclass
class TimingResult:
    """
    Full output of one timing computation pass.

    Attributes
    ----------
    decisions      : list[PhaseDecision]  – one per lane
    cycle_length   : float  – total cycle length in seconds (sum of all greens
                              + lost times); informational only
    any_oversaturated : bool – True if at least one lane hit oversaturation
    frame_index    : int    – forwarded from SmoothedResult for traceability
    """
    decisions:          list[PhaseDecision]
    cycle_length:       float
    any_oversaturated:  bool
    frame_index:        int

    def for_lane(self, lane_id: str) -> Optional[PhaseDecision]:
        """Return the PhaseDecision for a specific lane, or None if not found."""
        return next((d for d in self.decisions if d.lane_id == lane_id), None)


# ──────────────────────────────────────────────────────────────────────────────
# TimingAlgo class
# ──────────────────────────────────────────────────────────────────────────────

class TimingAlgo:
    """
    Computes green phase durations from smoothed vehicle counts.

    Usage
    -----
        algo = TimingAlgo(algorithm=Algorithm.WEBSTER)
        result = algo.compute(smoothed_result)
    """

    def __init__(
        self,
        algorithm:   Algorithm = Algorithm.WEBSTER,
        min_green:   float = DEFAULT_MIN_GREEN,
        max_green:   float = DEFAULT_MAX_GREEN,
        sat_flow:    float = DEFAULT_SAT_FLOW,
        lost_time:   float = DEFAULT_LOST_TIME,
    ) -> None:
        """
        Parameters
        ----------
        algorithm  : Algorithm  – WEBSTER or LINEAR
        min_green  : float      – minimum green duration in seconds (>= 1)
        max_green  : float      – maximum green duration in seconds
        sat_flow   : float      – saturation flow in vehicles/second (> 0)
        lost_time  : float      – lost time per phase in seconds (>= 0)

        Raises
        ------
        TimingError  on any invalid configuration value.
        """
        self._validate_config(algorithm, min_green, max_green, sat_flow, lost_time)

        self._algorithm  = Algorithm(algorithm)
        self._min_green  = float(min_green)
        self._max_green  = float(max_green)
        self._sat_flow   = float(sat_flow)
        self._lost_time  = float(lost_time)

        log.info(
            "TimingAlgo ready | algo=%s | green=[%.1fs, %.1fs] | "
            "sat_flow=%.2f veh/s | lost_time=%.1fs",
            self._algorithm.value, self._min_green, self._max_green,
            self._sat_flow, self._lost_time,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(self, smoothed: SmoothedResult) -> TimingResult:
        """
        Compute green phase durations for all lanes in a SmoothedResult.

        Parameters
        ----------
        smoothed : SmoothedResult
            Output of smoother.update() for the current frame.

        Returns
        -------
        TimingResult
            Per-lane PhaseDecisions + aggregate diagnostics.

        Raises
        ------
        TimingError
            If smoothed is None or wrong type.
        """
        self._validate_input(smoothed)

        decisions: list[PhaseDecision] = []
        any_oversaturated = False

        for lane_id, count in smoothed.counts.items():
            decision, oversaturated = self._decide(lane_id, count)
            decisions.append(decision)
            if oversaturated:
                any_oversaturated = True

        cycle_length = sum(d.green_duration for d in decisions) + (
            self._lost_time * len(decisions)
        )

        log.debug(
            "TimingAlgo frame=%d | decisions=%s | cycle=%.1fs | oversaturated=%s",
            smoothed.frame_index,
            {d.lane_id: d.green_duration for d in decisions},
            cycle_length,
            any_oversaturated,
        )

        return TimingResult(
            decisions=decisions,
            cycle_length=cycle_length,
            any_oversaturated=any_oversaturated,
            frame_index=smoothed.frame_index,
        )

    @property
    def algorithm(self) -> Algorithm:
        return self._algorithm

    @property
    def min_green(self) -> float:
        return self._min_green

    @property
    def max_green(self) -> float:
        return self._max_green

    # ── Private: dispatch ──────────────────────────────────────────────────────

    def _decide(self, lane_id: str, count: int) -> tuple[PhaseDecision, bool]:
        """
        Compute a PhaseDecision for one lane.
        Returns (decision, oversaturated_flag).
        """
        if self._algorithm == Algorithm.WEBSTER:
            return self._webster(lane_id, count)
        else:
            return self._linear(lane_id, count)

    # ── Webster formula ────────────────────────────────────────────────────────

    def _webster(self, lane_id: str, count: int) -> tuple[PhaseDecision, bool]:
        """
        Webster (1958) optimal green time for a single phase.

        Flow ratio y = count / saturation_flow
        Optimal cycle C = (1.5L + 5) / (1 - Y)   where Y = y (single phase)
        Green time g = (C - L) * y / Y            simplifies to (C - L) when Y=y

        For a single-phase approach (one lane at a time):
            g = C - L

        On oversaturation (Y >= 1.0): fall back to max_green.
        """
        oversaturated = False
        y = count / self._sat_flow   # flow ratio for this lane
        Y = y                        # single-phase: Y = y

        # Guard: oversaturation
        if Y >= 1.0:
            log.warning(
                "Lane '%s' is oversaturated (Y=%.3f >= 1.0, count=%d). "
                "Falling back to max_green=%.1fs.",
                lane_id, Y, count, self._max_green,
            )
            oversaturated = True
            return PhaseDecision(
                lane_id=lane_id,
                green_duration=self._max_green,
                raw_duration=float("inf"),
                vehicle_count=count,
                algorithm=Algorithm.WEBSTER,
                was_clamped=True,
                clamp_reason="oversaturated",
            ), oversaturated

        # Warn when approaching saturation
        if Y >= OVERSATURATION_THRESHOLD:
            log.warning(
                "Lane '%s' approaching saturation (Y=%.3f). "
                "Webster result may be unreliable.",
                lane_id, Y,
            )

        # Zero traffic → minimum green (avoids C = (1.5L+5)/1 but g → 0)
        if count == 0:
            return self._clamped_decision(
                lane_id, raw=0.0, count=0, algo=Algorithm.WEBSTER
            ), oversaturated

        L = self._lost_time
        C = (1.5 * L + 5.0) / (1.0 - Y)   # optimal cycle length
        g = C - L                            # effective green (single-phase)

        return self._clamped_decision(lane_id, raw=g, count=count, algo=Algorithm.WEBSTER), oversaturated

    # ── Linear formula ─────────────────────────────────────────────────────────

    def _linear(self, lane_id: str, count: int) -> tuple[PhaseDecision, bool]:
        """
        Linear interpolation:
            g = min_green + (count / sat_count) * (max_green - min_green)

        where sat_count = sat_flow * max_green (vehicles that fill the max window).
        Clamped to [min_green, max_green].
        """
        sat_count = self._sat_flow * self._max_green
        ratio = min(count / sat_count, 1.0) if sat_count > 0 else 0.0
        g = self._min_green + ratio * (self._max_green - self._min_green)
        return self._clamped_decision(lane_id, raw=g, count=count, algo=Algorithm.LINEAR), False

    # ── Clamping helper ────────────────────────────────────────────────────────

    def _clamped_decision(
        self,
        lane_id: str,
        raw: float,
        count: int,
        algo: Algorithm,
    ) -> PhaseDecision:
        """Apply [min_green, max_green] safety clamp and build PhaseDecision."""
        if math.isnan(raw) or math.isinf(raw):
            log.warning(
                "Lane '%s': raw duration is %s — clamping to max_green.", lane_id, raw
            )
            clamped = self._max_green
            reason = "max"
        elif raw < self._min_green:
            clamped = self._min_green
            reason = "min"
        elif raw > self._max_green:
            clamped = self._max_green
            reason = "max"
        else:
            clamped = raw
            reason = ""

        was_clamped = clamped != raw

        return PhaseDecision(
            lane_id=lane_id,
            green_duration=round(clamped, 2),
            raw_duration=round(raw, 4) if math.isfinite(raw) else raw,
            vehicle_count=count,
            algorithm=algo,
            was_clamped=was_clamped,
            clamp_reason=reason,
        )

    # ── Validation ─────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_config(
        algorithm, min_green, max_green, sat_flow, lost_time
    ) -> None:
        try:
            Algorithm(algorithm)
        except ValueError:
            raise TimingError(
                f"Unknown algorithm '{algorithm}'. "
                f"Valid options: {[a.value for a in Algorithm]}."
            )

        for name, val, minimum in [
            ("min_green",  min_green,  1.0),
            ("max_green",  max_green,  1.0),
            ("sat_flow",   sat_flow,   1e-6),
            ("lost_time",  lost_time,  0.0),
        ]:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TimingError(f"{name} must be a number, got {type(val).__name__}.")
            if not math.isfinite(val):
                raise TimingError(f"{name} must be finite, got {val}.")
            if val < minimum:
                raise TimingError(f"{name} must be >= {minimum}, got {val}.")

        if min_green >= max_green:
            raise TimingError(
                f"min_green ({min_green}) must be strictly less than "
                f"max_green ({max_green})."
            )

    @staticmethod
    def _validate_input(smoothed: object) -> None:
        if smoothed is None:
            raise TimingError(
                "SmoothedResult is None — did smoother.update() succeed?"
            )
        if not isinstance(smoothed, SmoothedResult):
            raise TimingError(
                f"Expected SmoothedResult, got {type(smoothed).__name__}."
            )
        for lane_id, count in smoothed.counts.items():
            if not isinstance(count, int) or count < 0:
                raise TimingError(
                    f"counts['{lane_id}'] = {count!r} — must be a non-negative int."
                )