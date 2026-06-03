"""
server/logic/timing_algo.py
─────────────────────────────────────────────────────────────────────────────
Responsibility : Translate stable per-lane vehicle counts (from smoother.py)
                 into green phase durations, then expose a conversion method
                 to the shared wire format (shared.schemas.PhaseDecision)
                 for the MQTT layer.

Two types live here
    GreenDecision  – internal computation result (pure logic, no wire concerns)
    TimingResult   – wraps a list of GreenDecision + diagnostics

Wire conversion
    GreenDecision.to_wire_decision(intersection_id, sequence)
        → shared.schemas.PhaseDecision   (ready for ServerMQTTClient)
    TimingResult.to_wire_decisions(intersection_id, sequence)
        → list[shared.schemas.PhaseDecision]  (convenience wrapper for main.py)

Algorithms
    LINEAR   – green = clamp(count / vehicles_per_second, min, max)
               Matches the formula documented in config.yaml.
               Default algorithm.

    WEBSTER  – Classic Webster (1958) optimal cycle formula.
               Best for near-saturated intersections with predictable flow.
               Degrades gracefully to max_green on oversaturation (Y ≥ 1).

Safety contract
    - Every output is clamped to [min_green, max_green].
    - Hard safety floors come from shared/constants.py.
    - Oversaturation (Y ≥ 1) in Webster → max_green + warning (never a crash).

Author : Oussama (server side)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from server.logic.smoother import SmoothedResult

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants / defaults  (real values come from ATCSConfig at runtime)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MIN_GREEN:        float = 10.0   # seconds
DEFAULT_MAX_GREEN:        float = 60.0   # seconds
DEFAULT_VEHICLES_PER_SEC: float = 2.0    # config.yaml: vehicles_per_second
DEFAULT_SAT_FLOW:         float = 0.5    # veh/s — Webster only (≈ 1800 veh/h)
DEFAULT_LOST_TIME:        float = 2.0    # seconds/phase — Webster only
OVERSATURATION_THRESHOLD: float = 0.95   # warn before Y hits 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────────────

class TimingError(ValueError):
    """Raised for bad configuration or unrecoverable input errors."""


class Algorithm(str, Enum):
    LINEAR  = "linear"    # config.yaml formula: green = count / vehicles_per_second
    WEBSTER = "webster"   # Webster (1958) optimal cycle


@dataclass(frozen=True)
class GreenDecision:
    """
    Internal computation result for one lane's green phase duration.

    This is NOT the wire format. Call to_wire_decision() to get the
    shared.schemas.PhaseDecision object that ServerMQTTClient publishes.

    Attributes
    ----------
    lane_id        : str       – lane this decision applies to
    green_duration : float     – final clamped green time in seconds
    raw_duration   : float     – computed duration before safety clamping
    vehicle_count  : int       – smoothed count that drove this decision
    algorithm      : Algorithm – which formula was used
    was_clamped    : bool      – True if raw_duration != green_duration
    clamp_reason   : str       – 'min', 'max', 'oversaturated', or ''
    """
    lane_id:        str
    green_duration: float
    raw_duration:   float
    vehicle_count:  int
    algorithm:      Algorithm
    was_clamped:    bool
    clamp_reason:   str = ""

    def to_wire_decision(
        self,
        intersection_id: str,
        sequence:        int = 0,
    ):
        """
        Convert this GreenDecision to the shared wire format.

        Imports are lazy so timing_algo.py stays importable without the
        shared package (useful in isolated unit tests).

        Parameters
        ----------
        intersection_id : str
            ATCSConfig.intersection.id — must match the Pi's config exactly.
        sequence : int
            FramePayload.sequence that triggered this decision.
            Enables end-to-end latency tracing.

        Returns
        -------
        shared.schemas.PhaseDecision
            Ready for ServerMQTTClient.publish_decision().
        """
        from shared.schemas import PhaseDecision as WireDecision  # type: ignore
        from shared.constants import Phase                         # type: ignore

        return WireDecision(
            intersection_id  = intersection_id,
            phase            = Phase.GREEN,
            duration_seconds = self.green_duration,
            vehicle_count    = self.vehicle_count,
            timestamp        = time.time(),
            sequence         = sequence,
        )


@dataclass
class TimingResult:
    """
    Full output of one timing computation pass.

    Attributes
    ----------
    decisions         : list[GreenDecision] – one per lane
    cycle_length      : float  – sum of greens + lost times (informational)
    any_oversaturated : bool   – True if at least one lane hit oversaturation
    frame_index       : int    – forwarded from SmoothedResult for traceability
    """
    decisions:          list[GreenDecision]
    cycle_length:       float
    any_oversaturated:  bool
    frame_index:        int

    def for_lane(self, lane_id: str) -> Optional[GreenDecision]:
        """Return the GreenDecision for a specific lane, or None if not found."""
        return next((d for d in self.decisions if d.lane_id == lane_id), None)

    def to_wire_decisions(
        self,
        intersection_id: str,
        sequence:        int = 0,
    ) -> list:
        """
        Convert all GreenDecisions to shared.schemas.PhaseDecision objects.

        Convenience wrapper used by main.py:
            wires = timing.to_wire_decisions(cfg.intersection.id, frame.sequence)
            for d in wires:
                mqtt_client.publish_decision(d)

        Returns
        -------
        list[shared.schemas.PhaseDecision]
        """
        return [
            d.to_wire_decision(
                intersection_id = intersection_id,
                sequence        = sequence,
            )
            for d in self.decisions
        ]


# ──────────────────────────────────────────────────────────────────────────────
# TimingAlgo class
# ──────────────────────────────────────────────────────────────────────────────

class TimingAlgo:
    """
    Computes green phase durations from smoothed vehicle counts.

    Usage
    -----
        algo   = TimingAlgo(
                     algorithm        = Algorithm.LINEAR,
                     min_green        = cfg.server.min_green_duration,
                     max_green        = cfg.server.max_green_duration,
                     vehicles_per_sec = cfg.server.vehicles_per_second,
                 )
        result = algo.compute(smoothed_result)
        wires  = result.to_wire_decisions(cfg.intersection.id, frame.sequence)
    """

    def __init__(
        self,
        algorithm:        Algorithm = Algorithm.LINEAR,
        min_green:        float = DEFAULT_MIN_GREEN,
        max_green:        float = DEFAULT_MAX_GREEN,
        vehicles_per_sec: float = DEFAULT_VEHICLES_PER_SEC,
        sat_flow:         float = DEFAULT_SAT_FLOW,
        lost_time:        float = DEFAULT_LOST_TIME,
    ) -> None:
        """
        Parameters
        ----------
        algorithm        : Algorithm – LINEAR (default) or WEBSTER
        min_green        : float     – minimum green duration in seconds
        max_green        : float     – maximum green duration in seconds
        vehicles_per_sec : float     – LINEAR: vehicles discharged per second
                                       of green (from config.yaml)
        sat_flow         : float     – WEBSTER only: saturation flow (veh/s)
        lost_time        : float     – WEBSTER only: lost time per phase (s)

        Raises
        ------
        TimingError on any invalid configuration value.
        """
        self._validate_config(
            algorithm, min_green, max_green, vehicles_per_sec, sat_flow, lost_time
        )

        self._algorithm        = Algorithm(algorithm)
        self._min_green        = float(min_green)
        self._max_green        = float(max_green)
        self._vehicles_per_sec = float(vehicles_per_sec)
        self._sat_flow         = float(sat_flow)
        self._lost_time        = float(lost_time)

        log.info(
            "TimingAlgo ready | algo=%s | green=[%.1fs, %.1fs] | "
            "veh/s=%.2f | sat_flow=%.2f | lost_time=%.1fs",
            self._algorithm.value, self._min_green, self._max_green,
            self._vehicles_per_sec, self._sat_flow, self._lost_time,
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
            Per-lane GreenDecisions + aggregate diagnostics.

        Raises
        ------
        TimingError
            If smoothed is None or wrong type.
        """
        self._validate_input(smoothed)

        decisions:        list[GreenDecision] = []
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
            decisions         = decisions,
            cycle_length      = cycle_length,
            any_oversaturated = any_oversaturated,
            frame_index       = smoothed.frame_index,
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

    def _decide(self, lane_id: str, count: int) -> tuple[GreenDecision, bool]:
        if self._algorithm == Algorithm.WEBSTER:
            return self._webster(lane_id, count)
        return self._linear(lane_id, count)

    # ── Linear formula ─────────────────────────────────────────────────────────

    def _linear(self, lane_id: str, count: int) -> tuple[GreenDecision, bool]:
        """
        green = clamp(count / vehicles_per_second, min_green, max_green)

        Implements config.yaml spec:
            vehicles_per_second: 2  →  green = count / 2
        """
        raw = (
            count / self._vehicles_per_sec
            if self._vehicles_per_sec > 0
            else 0.0
        )
        return self._clamped_decision(
            lane_id, raw=raw, count=count, algo=Algorithm.LINEAR
        ), False

    # ── Webster formula ────────────────────────────────────────────────────────

    def _webster(self, lane_id: str, count: int) -> tuple[GreenDecision, bool]:
        """
        Webster (1958) optimal green time for a single phase.
            y = count / sat_flow
            C = (1.5L + 5) / (1 - Y)
            g = C - L
        On oversaturation (Y >= 1.0): fall back to max_green.
        """
        oversaturated = False
        Y = count / self._sat_flow

        if Y >= 1.0:
            log.warning(
                "Lane '%s' oversaturated (Y=%.3f >= 1.0, count=%d). "
                "Falling back to max_green=%.1fs.",
                lane_id, Y, count, self._max_green,
            )
            oversaturated = True
            return GreenDecision(
                lane_id        = lane_id,
                green_duration = self._max_green,
                raw_duration   = float("inf"),
                vehicle_count  = count,
                algorithm      = Algorithm.WEBSTER,
                was_clamped    = True,
                clamp_reason   = "oversaturated",
            ), oversaturated

        if Y >= OVERSATURATION_THRESHOLD:
            log.warning(
                "Lane '%s' approaching saturation (Y=%.3f). "
                "Webster result may be unreliable.",
                lane_id, Y,
            )

        if count == 0:
            return self._clamped_decision(
                lane_id, raw=0.0, count=0, algo=Algorithm.WEBSTER
            ), oversaturated

        L = self._lost_time
        C = (1.5 * L + 5.0) / (1.0 - Y)
        g = C - L
        return self._clamped_decision(
            lane_id, raw=g, count=count, algo=Algorithm.WEBSTER
        ), oversaturated

    # ── Clamping helper ────────────────────────────────────────────────────────

    def _clamped_decision(
        self,
        lane_id: str,
        raw:     float,
        count:   int,
        algo:    Algorithm,
    ) -> GreenDecision:
        """Apply [min_green, max_green] safety clamp and return a GreenDecision."""
        if math.isnan(raw) or math.isinf(raw):
            log.warning(
                "Lane '%s': raw duration is non-finite (%s) — clamping to max_green.",
                lane_id, raw,
            )
            clamped, reason = self._max_green, "max"
        elif raw < self._min_green:
            clamped, reason = self._min_green, "min"
        elif raw > self._max_green:
            clamped, reason = self._max_green, "max"
        else:
            clamped, reason = raw, ""

        return GreenDecision(
            lane_id        = lane_id,
            green_duration = round(clamped, 2),
            raw_duration   = round(raw, 4) if math.isfinite(raw) else raw,
            vehicle_count  = count,
            algorithm      = algo,
            was_clamped    = clamped != raw,
            clamp_reason   = reason,
        )

    # ── Validation ─────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_config(
        algorithm, min_green, max_green, vehicles_per_sec, sat_flow, lost_time
    ) -> None:
        try:
            Algorithm(algorithm)
        except ValueError:
            raise TimingError(
                f"Unknown algorithm '{algorithm}'. "
                f"Valid options: {[a.value for a in Algorithm]}."
            )

        for name, val, minimum in [
            ("min_green",        min_green,        1.0),
            ("max_green",        max_green,        1.0),
            ("vehicles_per_sec", vehicles_per_sec, 1e-6),
            ("sat_flow",         sat_flow,         1e-6),
            ("lost_time",        lost_time,        0.0),
        ]:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TimingError(
                    f"{name} must be a number, got {type(val).__name__}."
                )
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