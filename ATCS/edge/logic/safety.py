# =============================================================================
# ATCS - Adaptive Traffic Control System
# edge/logic/safety.py | Watchdog & fixed-cycle fallback
#
# Responsibilities:
#   - Monitor time since last server command via StateMachine
#   - Trigger ALL_RED + fixed-cycle fallback if server silent > timeout
#   - Run fixed-cycle independently until server reconnects
#   - Hand control back to server cleanly on reconnect
#   - Log all mode transitions (NORMAL ↔ FALLBACK) for audit
#
# Rules:
#   - Runs in its own daemon thread. Never blocks the Controller thread.
#   - Only calls state_machine.force_safe_state() and gpio_driver.apply_phase().
#   - Fixed-cycle timings come from ATCSConfig — no hardcoded values.
#   - Must be startable and stoppable cleanly (for tests and shutdown).
# =============================================================================

import logging
import threading
import time
from enum import Enum, auto
from typing import Optional

from shared.constants import Phase
from shared.config_loader import ATCSConfig
from edge.logic.state_machine import StateMachine
from edge.logic.timer import PhaseTimer
from edge.hardware.gpio_driver import GPIODriver

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Operating Mode
# -----------------------------------------------------------------------------

class OperatingMode(Enum):
    """Current operating mode of the edge node."""
    NORMAL   = auto()   # Server is reachable; commands drive the lights
    FALLBACK = auto()   # Server unreachable; fixed-cycle drives the lights


# =============================================================================
# SafetyWatchdog
# =============================================================================

class SafetyWatchdog:
    """
    Background watchdog thread that monitors server connectivity and
    autonomously runs a fixed traffic cycle when the server is unreachable.

    Normal mode:
        Polls state_machine.seconds_since_last_command every
        WATCHDOG_POLL_INTERVAL_SECONDS. If it exceeds server_timeout_seconds,
        transitions to FALLBACK mode.

    Fallback mode:
        Drives GPIO directly through a fixed RED → GREEN → AMBER cycle using
        durations from config.edge. Keeps cycling until the server sends a
        fresh command, at which point it hands control back (NORMAL mode).

    Parameters
    ----------
    state_machine : StateMachine
        Used to check last_command_time and to force ALL_RED on timeout.
    gpio_driver : GPIODriver
        Used directly in fallback mode to apply fixed-cycle phases.
    config : ATCSConfig
        Source of timeout threshold and fixed-cycle durations.
    """

    # How often the watchdog wakes up to check server connectivity (seconds)
    _POLL_INTERVAL: float = 1.0

    def __init__(
        self,
        state_machine: StateMachine,
        gpio_driver:   GPIODriver,
        config:        ATCSConfig,
    ) -> None:
        self._sm      = state_machine
        self._gpio    = gpio_driver
        self._cfg     = config.edge

        self._timeout = config.edge.server_timeout_seconds

        # Fixed-cycle phase sequence and durations
        self._fixed_cycle = [
            (Phase.GREEN, float(config.edge.fallback_green_duration)),
            (Phase.AMBER, float(config.edge.fallback_amber_duration)),
            (Phase.RED,   float(config.edge.fallback_red_duration)),
        ]

        self._mode:          OperatingMode   = OperatingMode.NORMAL
        self._timer:         PhaseTimer      = PhaseTimer()
        self._cycle_index:   int             = 0   # current position in fixed_cycle

        self._thread:        Optional[threading.Thread] = None
        self._stop_event:    threading.Event            = threading.Event()

        # Counters for diagnostics
        self._fallback_count:  int = 0   # how many times fallback was triggered
        self._reconnect_count: int = 0   # how many times server reconnected

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the watchdog background thread.
        Safe to call only once. Raises RuntimeError if already running.
        """
        if self._thread and self._thread.is_alive():
            raise RuntimeError("SafetyWatchdog is already running.")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target   = self._run,
            name     = "SafetyWatchdog",
            daemon   = True,   # Dies automatically when main thread exits
        )
        self._thread.start()
        logger.info(
            "SafetyWatchdog started. Timeout threshold: %ds.", self._timeout
        )

    def stop(self) -> None:
        """
        Signal the watchdog thread to stop and wait for it to exit.
        Called during graceful shutdown in edge/main.py.
        """
        logger.info("SafetyWatchdog stopping...")
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._POLL_INTERVAL * 3)

        self._timer.stop()
        logger.info("SafetyWatchdog stopped.")

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """
        Watchdog thread entry point. Alternates between monitoring and
        fallback driving based on server connectivity.
        """
        logger.debug("SafetyWatchdog thread running.")

        while not self._stop_event.is_set():
            try:
                if self._mode == OperatingMode.NORMAL:
                    self._run_normal_tick()
                else:
                    self._run_fallback_tick()

            except Exception as exc:    # pylint: disable=broad-except
                # Watchdog must never crash — log and keep going
                logger.error(
                    "SafetyWatchdog unexpected error: %s. Continuing.", exc,
                    exc_info=True,
                )

            self._stop_event.wait(timeout=self._POLL_INTERVAL)

        logger.debug("SafetyWatchdog thread exiting.")

    # ------------------------------------------------------------------
    # Normal Mode Tick
    # ------------------------------------------------------------------

    def _run_normal_tick(self) -> None:
        """
        Check how long since the last server command.
        If threshold exceeded → enter fallback mode.
        """
        elapsed = self._sm.seconds_since_last_command

        if elapsed >= self._timeout:
            logger.warning(
                "SERVER TIMEOUT: no command received for %.1fs "
                "(threshold=%ds). Entering FALLBACK mode.",
                elapsed, self._timeout,
            )
            self._enter_fallback()

    # ------------------------------------------------------------------
    # Fallback Mode Tick
    # ------------------------------------------------------------------

    def _run_fallback_tick(self) -> None:
        """
        Check if a server command has arrived (reconnect), or advance
        the fixed-cycle phase if the current phase timer has expired.
        """
        # Check for server reconnect first
        if self._sm.seconds_since_last_command < self._timeout:
            logger.info(
                "Server reconnected (last command %.1fs ago). "
                "Returning to NORMAL mode.",
                self._sm.seconds_since_last_command,
            )
            self._exit_fallback()
            return

        # Advance fixed cycle when current phase expires
        if self._timer.is_expired:
            self._advance_fixed_cycle()

    # ------------------------------------------------------------------
    # Mode Transitions
    # ------------------------------------------------------------------

    def _enter_fallback(self) -> None:
        """
        Transition from NORMAL → FALLBACK.
        Forces ALL_RED immediately, then starts fixed-cycle from RED.
        """
        self._mode = OperatingMode.FALLBACK
        self._fallback_count += 1

        # Safety: force all-red before starting fixed cycle
        self._sm.force_safe_state(source="watchdog")
        logger.warning(
            "FALLBACK mode active (trigger #%d). "
            "Fixed cycle: GREEN=%ds / AMBER=%ds / RED=%ds.",
            self._fallback_count,
            self._cfg.fallback_green_duration,
            self._cfg.fallback_amber_duration,
            self._cfg.fallback_red_duration,
        )

        # Start fixed cycle from RED (index 2) after ALL_RED buffer
        time.sleep(1.0)   # ALL_RED buffer before first fixed-cycle phase
        self._cycle_index = 2   # Start at RED, not GREEN (safer after outage)
        self._advance_fixed_cycle()

    def _exit_fallback(self) -> None:
        """
        Transition from FALLBACK → NORMAL.
        Stops the fixed-cycle timer and lets the state machine take over.
        """
        self._mode = OperatingMode.NORMAL
        self._reconnect_count += 1
        self._timer.stop()

        # Reset state machine's last_command_time so watchdog
        # doesn't immediately re-trigger fallback
        self._sm.reset_last_command_time()

        logger.info(
            "NORMAL mode restored (reconnect #%d). "
            "Server commands are now in control.",
            self._reconnect_count,
        )

    # ------------------------------------------------------------------
    # Fixed-Cycle Advancement
    # ------------------------------------------------------------------

    def _advance_fixed_cycle(self) -> None:
        """
        Move to the next phase in the fixed cycle and start its timer.
        Cycles: RED → GREEN → AMBER → RED → ...
        """
        self._cycle_index = (self._cycle_index + 1) % len(self._fixed_cycle)
        phase, duration = self._fixed_cycle[self._cycle_index]

        try:
            self._gpio.apply_phase(phase)
        except RuntimeError as exc:
            logger.error(
                "Fixed-cycle GPIO apply_phase(%s) failed: %s",
                phase.value, exc,
            )
            return

        self._timer.start(duration)

        logger.info(
            "[FALLBACK] Phase: %s | Duration: %.0fs | "
            "Remaining cycle position: %d/%d",
            phase.value, duration,
            self._cycle_index + 1, len(self._fixed_cycle),
        )

    # ------------------------------------------------------------------
    # Properties / Diagnostics
    # ------------------------------------------------------------------

    @property
    def mode(self) -> OperatingMode:
        """Current operating mode (NORMAL or FALLBACK)."""
        return self._mode

    @property
    def is_in_fallback(self) -> bool:
        """True if the system is currently in fallback mode."""
        return self._mode == OperatingMode.FALLBACK

    @property
    def fallback_count(self) -> int:
        """How many times fallback mode has been triggered since boot."""
        return self._fallback_count

    @property
    def reconnect_count(self) -> int:
        """How many times the server has reconnected since boot."""
        return self._reconnect_count

    def __repr__(self) -> str:
        return (
            f"SafetyWatchdog("
            f"mode={self._mode.name}, "
            f"fallbacks={self._fallback_count}, "
            f"reconnects={self._reconnect_count}, "
            f"timeout={self._timeout}s)"
        )


# =============================================================================
# USAGE EXAMPLE
# Simulates a server timeout and reconnect using mocks.
# No real GPIO or MQTT needed.
# Run from project root: python edge/logic/safety.py
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from shared.config_loader import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg)

    # Mock dependencies
    mock_gpio = MagicMock(spec=GPIODriver)
    mock_sm   = MagicMock(spec=StateMachine)

    # Simulate server being reachable initially
    mock_sm.seconds_since_last_command = 0.0
    mock_sm.force_safe_state           = MagicMock()
    mock_sm.reset_last_command_time    = MagicMock()

    # Use a very short timeout for the smoke test (3s instead of 30s)
    cfg_edge         = cfg.edge
    short_timeout    = 3

    watchdog = SafetyWatchdog(mock_sm, mock_gpio, cfg)
    # Override timeout for fast test
    watchdog._timeout = short_timeout

    print("=== SafetyWatchdog Smoke Test ===\n")

    watchdog.start()
    print(f"Watchdog started: {watchdog}\n")

    # Phase 1: Normal mode — server is responding
    print("[1] Normal mode (server healthy for 2s)...")
    time.sleep(2)
    assert watchdog.mode == OperatingMode.NORMAL
    print(f"  ✅ Mode: {watchdog.mode.name}\n")

    # Phase 2: Simulate server timeout
    print(f"[2] Simulating server timeout (silent for {short_timeout + 1}s)...")
    mock_sm.seconds_since_last_command = float(short_timeout + 1)
    time.sleep(short_timeout + 1.5)
    assert watchdog.is_in_fallback, "Expected FALLBACK mode after timeout"
    print(f"  ✅ Fallback triggered. Mode: {watchdog.mode.name}")
    print(f"  force_safe_state called: {mock_sm.force_safe_state.called}")
    print(f"  GPIO apply_phase calls: {mock_gpio.apply_phase.call_count}\n")

    # Phase 3: Simulate server reconnect
    print("[3] Simulating server reconnect...")
    mock_sm.seconds_since_last_command = 0.5
    time.sleep(2)
    assert watchdog.mode == OperatingMode.NORMAL, "Expected NORMAL after reconnect"
    print(f"  ✅ Returned to NORMAL mode.")
    print(f"  reset_last_command_time called: {mock_sm.reset_last_command_time.called}\n")

    watchdog.stop()
    print(f"Final state: {watchdog}")
    print(f"\n=== SafetyWatchdog smoke test passed ===")
