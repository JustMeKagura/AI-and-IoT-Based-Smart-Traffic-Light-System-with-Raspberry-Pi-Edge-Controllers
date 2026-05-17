# =============================================================================
# ATCS - Adaptive Traffic Control System
# edge/logic/state_machine.py | Phase transition validator
#
# Responsibilities:
#   - Track the current traffic light phase
#   - Validate every incoming PhaseDecision against VALID_TRANSITIONS
#   - Reject unsafe transitions and log the reason
#   - Delegate pin control to GPIODriver only after a transition is approved
#   - Emit structured state-change events for the watchdog / logger
#
# Rules:
#   - This module makes NO GPIO calls directly. It calls gpio_driver.apply_phase().
#   - All transition rules live in shared/constants.py (VALID_TRANSITIONS).
#     Do NOT duplicate or override them here.
#   - Thread-safe: MQTT Controller thread is the only caller, but the
#     watchdog may call force_safe_state() concurrently.
#   - Never silently accept an invalid transition. Always raise or log + reject.
# =============================================================================

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from shared.constants import Phase, VALID_TRANSITIONS, BOOT_PHASE
from shared.schemas import PhaseDecision
from edge.hardware.gpio_driver import GPIODriver

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# State Change Event
# Emitted on every successful transition. Consumers (logger, SQLite, watchdog)
# can register a callback via register_listener().
# -----------------------------------------------------------------------------

@dataclass
class StateChangeEvent:
    """
    Fired after every successful phase transition.

    Attributes
    ----------
    previous_phase : Phase
        The phase the system was in before this transition.
    new_phase : Phase
        The phase the system just entered.
    duration_seconds : float
        How long the new phase should be held (from PhaseDecision).
    vehicle_count : int
        Smoothed vehicle count that triggered this decision (0 in fallback).
    sequence : int
        Frame sequence number that originated this decision.
    timestamp : float
        Unix epoch when the transition was applied on the edge.
    source : str
        'server' for normal operation, 'watchdog' for fallback-triggered.
    """
    previous_phase:   Phase
    new_phase:        Phase
    duration_seconds: float
    vehicle_count:    int   = 0
    sequence:         int   = 0
    timestamp:        float = field(default_factory=time.time)
    source:           str   = "server"


# -----------------------------------------------------------------------------
# Rejected Transition Record (for diagnostics / tests)
# -----------------------------------------------------------------------------

@dataclass
class RejectedTransition:
    """Records a transition that was blocked by the state machine."""
    from_phase:  Phase
    to_phase:    Phase
    reason:      str
    timestamp:   float = field(default_factory=time.time)


# =============================================================================
# StateMachine
# =============================================================================

class StateMachine:
    """
    Guards every phase transition on the Edge.

    Flow:
        MQTT callback receives PhaseDecision
            → StateMachine.process_decision()
                → validate intersection_id
                → validate transition via VALID_TRANSITIONS
                → call GPIODriver.apply_phase()
                → emit StateChangeEvent to listeners
                → update last_command_time (resets watchdog)

    Parameters
    ----------
    gpio_driver : GPIODriver
        Initialised GPIO driver. Must have setup() already called.
    intersection_id : str
        This edge node's intersection ID. Decisions for other IDs are rejected.
    """

    def __init__(self, gpio_driver: GPIODriver, intersection_id: str) -> None:
        self._gpio            = gpio_driver
        self._intersection_id = intersection_id
        self._lock            = threading.Lock()

        self._current_phase:     Phase                   = BOOT_PHASE
        self._last_command_time: float                   = time.time()
        self._listeners:         List[Callable[[StateChangeEvent], None]] = []
        self._rejected_log:      List[RejectedTransition]                 = []
        self._transition_count:  int                                      = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_decision(self, decision: PhaseDecision) -> bool:
        """
        Validate and apply an incoming PhaseDecision from the server.

        Parameters
        ----------
        decision : PhaseDecision
            Parsed command from the MQTT control/signal topic.

        Returns
        -------
        bool
            True if the transition was accepted and applied.
            False if it was rejected (wrong ID, unsafe transition, etc.).
        """
        with self._lock:

            # 1. Validate intersection ID — reject commands for other nodes
            if decision.intersection_id != self._intersection_id:
                reason = (
                    f"intersection_id mismatch: expected "
                    f"'{self._intersection_id}', got '{decision.intersection_id}'."
                )
                self._reject(self._current_phase, decision.phase, reason)
                return False

            # 2. Validate the transition is allowed
            allowed_targets = VALID_TRANSITIONS.get(self._current_phase, set())
            if decision.phase not in allowed_targets:
                reason = (
                    f"Transition {self._current_phase.value} → "
                    f"{decision.phase.value} is not in VALID_TRANSITIONS. "
                    f"Allowed: {[p.value for p in allowed_targets]}."
                )
                self._reject(self._current_phase, decision.phase, reason)
                return False

            # 3. Apply to hardware
            previous_phase = self._current_phase
            try:
                self._gpio.apply_phase(decision.phase)
            except RuntimeError as exc:
                logger.error(
                    "GPIO apply_phase(%s) failed: %s. Aborting transition.",
                    decision.phase.value, exc,
                )
                return False

            # 4. Update internal state
            self._current_phase     = decision.phase
            self._last_command_time = time.time()
            self._transition_count += 1

            logger.info(
                "[SM] Transition #%d: %s → %s | duration=%.1fs | "
                "vehicles=%d | seq=%d",
                self._transition_count,
                previous_phase.value,
                self._current_phase.value,
                decision.duration_seconds,
                decision.vehicle_count,
                decision.sequence,
            )

            # 5. Notify listeners (non-blocking: errors are caught per listener)
            event = StateChangeEvent(
                previous_phase   = previous_phase,
                new_phase        = self._current_phase,
                duration_seconds = decision.duration_seconds,
                vehicle_count    = decision.vehicle_count,
                sequence         = decision.sequence,
                source           = "server",
            )
            self._emit(event)

            return True

    def force_safe_state(self, source: str = "watchdog") -> None:
        """
        Immediately drive hardware to ALL_RED and update internal state.
        Called by safety.py watchdog on server timeout, bypassing
        normal transition validation.

        Parameters
        ----------
        source : str
            Label for the StateChangeEvent (e.g. 'watchdog', 'shutdown').
        """
        with self._lock:
            previous_phase = self._current_phase

            if previous_phase == Phase.ALL_RED:
                logger.debug("[SM] force_safe_state(): already ALL_RED, no-op.")
                return

            logger.warning(
                "[SM] force_safe_state(source=%s): %s → ALL_RED.",
                source, previous_phase.value,
            )

            self._gpio.force_all_red()
            self._current_phase = Phase.ALL_RED

            event = StateChangeEvent(
                previous_phase   = previous_phase,
                new_phase        = Phase.ALL_RED,
                duration_seconds = 0.0,
                vehicle_count    = 0,
                source           = source,
            )
            self._emit(event)

    # ------------------------------------------------------------------
    # Listener Registration
    # ------------------------------------------------------------------

    def register_listener(
        self, callback: Callable[[StateChangeEvent], None]
    ) -> None:
        """
        Register a callback to be called on every successful transition.

        The callback receives a StateChangeEvent and must not block.
        Exceptions in callbacks are caught and logged so one bad listener
        cannot break the state machine.

        Parameters
        ----------
        callback : Callable[[StateChangeEvent], None]
            Function to call after each successful phase change.
        """
        self._listeners.append(callback)
        logger.debug("[SM] Listener registered: %s", callback.__name__)

    # ------------------------------------------------------------------
    # Properties / Diagnostics
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> Phase:
        """The phase currently active on the hardware."""
        return self._current_phase

    @property
    def last_command_time(self) -> float:
        """Unix epoch of the last accepted server command. Used by watchdog."""
        return self._last_command_time

    @property
    def seconds_since_last_command(self) -> float:
        """Seconds elapsed since the last accepted server command."""
        return time.time() - self._last_command_time

    @property
    def transition_count(self) -> int:
        """Total number of successful transitions since boot."""
        return self._transition_count

    @property
    def rejected_transitions(self) -> List[RejectedTransition]:
        """Read-only list of all rejected transitions (for diagnostics/tests)."""
        return list(self._rejected_log)

    def reset_last_command_time(self) -> None:
        """
        Manually reset the last command timestamp to now.
        Called after reconnect to prevent an immediate watchdog timeout.
        """
        self._last_command_time = time.time()
        logger.debug("[SM] last_command_time reset to now.")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _reject(self, from_phase: Phase, to_phase: Phase, reason: str) -> None:
        """Log and record a rejected transition."""
        logger.warning("[SM] REJECTED %s → %s: %s", from_phase.value, to_phase.value, reason)
        self._rejected_log.append(
            RejectedTransition(from_phase=from_phase, to_phase=to_phase, reason=reason)
        )

    def _emit(self, event: StateChangeEvent) -> None:
        """Fire all registered listeners. Errors are isolated per listener."""
        for callback in self._listeners:
            try:
                callback(event)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "[SM] Listener '%s' raised an exception: %s",
                    callback.__name__, exc,
                )

    def __repr__(self) -> str:
        return (
            f"StateMachine("
            f"phase={self._current_phase.value}, "
            f"transitions={self._transition_count}, "
            f"rejected={len(self._rejected_log)}, "
            f"last_cmd={self.seconds_since_last_command:.1f}s ago)"
        )


# =============================================================================
# USAGE EXAMPLE
# Simulates a sequence of server commands to verify transition logic.
# Does NOT require real GPIO (uses a mock driver).
# Run from the project root: python edge/logic/state_machine.py
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from shared.config_loader import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg)

    # Use a mock GPIO driver so this runs on any machine (no real GPIO needed)
    mock_gpio = MagicMock(spec=GPIODriver)
    mock_gpio.is_ready = True

    sm = StateMachine(mock_gpio, intersection_id=cfg.intersection.id)

    # Register a simple listener
    events_received = []
    def on_state_change(event: StateChangeEvent) -> None:
        events_received.append(event)
        print(f"    [EVENT] {event.previous_phase.value} → {event.new_phase.value} "
              f"(source={event.source})")

    sm.register_listener(on_state_change)

    print("=== StateMachine Smoke Test ===\n")
    print(f"Initial state: {sm}\n")

    def make_decision(phase: Phase, seq: int = 0) -> PhaseDecision:
        return PhaseDecision(
            intersection_id  = cfg.intersection.id,
            phase            = phase,
            duration_seconds = 30.0,
            vehicle_count    = 5,
            sequence         = seq,
        )

    # Valid sequence: ALL_RED → GREEN → AMBER → RED → GREEN
    valid_steps = [
        (Phase.GREEN, True,  "ALL_RED → GREEN (valid boot transition)"),
        (Phase.AMBER, True,  "GREEN  → AMBER  (mandatory exit)"),
        (Phase.RED,   True,  "AMBER  → RED    (valid)"),
        (Phase.GREEN, True,  "RED    → GREEN  (valid)"),
    ]

    print("[1] Valid transition sequence:")
    for phase, expected, label in valid_steps:
        result = sm.process_decision(make_decision(phase))
        status = "✅" if result == expected else "❌"
        print(f"  {status} {label} → accepted={result}")

    print(f"\n[2] Invalid transition (GREEN → RED, skipping AMBER):")
    # Currently in GREEN after last step
    result = sm.process_decision(make_decision(Phase.RED))
    print(f"  {'✅' if not result else '❌'} Rejected GREEN → RED: {not result}")

    print(f"\n[3] Wrong intersection ID:")
    bad_decision = PhaseDecision(
        intersection_id  = "wrong_intersection",
        phase            = Phase.AMBER,
        duration_seconds = 4.0,
    )
    result = sm.process_decision(bad_decision)
    print(f"  {'✅' if not result else '❌'} Rejected wrong ID: {not result}")

    print(f"\n[4] force_safe_state (watchdog simulation):")
    sm.force_safe_state(source="watchdog")
    print(f"  Current phase: {sm.current_phase.value}")
    assert sm.current_phase == Phase.ALL_RED
    print(f"  ✅ force_safe_state → ALL_RED confirmed")

    print(f"\n[5] Summary:")
    print(f"  {sm}")
    print(f"  Events received : {len(events_received)}")
    print(f"  Rejected        : {len(sm.rejected_transitions)}")
    for r in sm.rejected_transitions:
        print(f"    - {r.from_phase.value} → {r.to_phase.value}: {r.reason}")

    print("\n=== StateMachine smoke test passed ===")
