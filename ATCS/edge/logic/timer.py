# =============================================================================
# ATCS - Adaptive Traffic Control System
# edge/logic/timer.py | Phase duration countdown timer
#
# Responsibilities:
#   - Track how long the current phase should be held
#   - Tell the Controller thread when a phase has expired
#   - Support reset, pause, and remaining-time queries
#   - Remain thread-safe for concurrent reads from watchdog / logger
#
# Rules:
#   - No GPIO, no MQTT, no config dependencies. Pure timing logic only.
#   - Thread-safe: Controller thread writes, watchdog/logger threads read.
#   - Uses time.monotonic() — immune to system clock adjustments (NTP, DST).
# =============================================================================

import logging
import threading
import time

logger = logging.getLogger(__name__)


class PhaseTimer:
    """
    Countdown timer that tracks how long the current phase should be held.

    The Controller thread starts the timer when a PhaseDecision is applied,
    then polls is_expired() in its loop. When it expires, the Controller
    knows it's time to request the next phase or fall back.

    Uses time.monotonic() throughout — never time.time() — so NTP
    corrections or DST changes cannot cause negative or inflated durations.

    Parameters
    ----------
    None — instantiate with no arguments, then call start() to arm it.
    """

    def __init__(self) -> None:
        self._lock:          threading.Lock  = threading.Lock()
        self._duration:      float           = 0.0
        self._start_mono:    float           = 0.0   # monotonic timestamp of last start
        self._is_running:    bool            = False
        self._elapsed_pause: float           = 0.0   # accumulated elapsed before pause
        self._is_paused:     bool            = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def start(self, duration_seconds: float) -> None:
        """
        Arm the timer for a new phase duration.

        Calling start() while already running resets and restarts from zero.
        This handles the case where a new PhaseDecision arrives before the
        current phase expires (server supersedes its own command).

        Parameters
        ----------
        duration_seconds : float
            How long this phase should be held. Must be > 0.

        Raises
        ------
        ValueError
            If duration_seconds is not positive.
        """
        if duration_seconds <= 0:
            raise ValueError(
                f"PhaseTimer.start(): duration must be > 0, got {duration_seconds}."
            )

        with self._lock:
            self._duration      = duration_seconds
            self._start_mono    = time.monotonic()
            self._is_running    = True
            self._is_paused     = False
            self._elapsed_pause = 0.0

        logger.debug(
            "PhaseTimer started: duration=%.1fs.", duration_seconds
        )

    def stop(self) -> None:
        """
        Stop the timer and reset all state.
        Called on shutdown or when transitioning to a phase with no fixed duration.
        """
        with self._lock:
            self._is_running    = False
            self._is_paused     = False
            self._elapsed_pause = 0.0
            self._duration      = 0.0

        logger.debug("PhaseTimer stopped.")

    def pause(self) -> None:
        """
        Pause the countdown. Elapsed time is preserved.
        Calling pause() on an already-paused or stopped timer is a no-op.
        """
        with self._lock:
            if not self._is_running or self._is_paused:
                return
            # Snapshot elapsed so far
            self._elapsed_pause += time.monotonic() - self._start_mono
            self._is_paused = True

        logger.debug(
            "PhaseTimer paused at %.2fs elapsed.", self._elapsed_pause
        )

    def resume(self) -> None:
        """
        Resume a paused timer from where it left off.
        Calling resume() on a running or stopped timer is a no-op.
        """
        with self._lock:
            if not self._is_running or not self._is_paused:
                return
            self._start_mono = time.monotonic()
            self._is_paused  = False

        logger.debug("PhaseTimer resumed.")

    # ------------------------------------------------------------------
    # State Queries  (all thread-safe, non-blocking)
    # ------------------------------------------------------------------

    @property
    def is_expired(self) -> bool:
        """
        True if the timer is running and the full duration has elapsed.
        False if stopped, paused, or time remains.

        The Controller thread polls this in its loop.
        """
        with self._lock:
            if not self._is_running or self._is_paused:
                return False
            return self._elapsed() >= self._duration

    @property
    def is_running(self) -> bool:
        """True if the timer has been started and not stopped."""
        with self._lock:
            return self._is_running

    @property
    def is_paused(self) -> bool:
        """True if the timer is currently paused."""
        with self._lock:
            return self._is_paused

    @property
    def remaining_seconds(self) -> float:
        """
        Seconds remaining until expiry.
        Returns 0.0 if expired, stopped, or paused with no time left.
        """
        with self._lock:
            if not self._is_running:
                return 0.0
            remaining = self._duration - self._elapsed()
            return max(0.0, remaining)

    @property
    def elapsed_seconds(self) -> float:
        """
        Seconds elapsed since the timer was last started (or resumed).
        Returns 0.0 if the timer is stopped.
        """
        with self._lock:
            if not self._is_running:
                return 0.0
            return min(self._elapsed(), self._duration)

    @property
    def duration(self) -> float:
        """The total duration this timer was started with (seconds)."""
        with self._lock:
            return self._duration

    @property
    def progress(self) -> float:
        """
        Fraction of the duration elapsed, in range [0.0, 1.0].
        Returns 0.0 if the timer is stopped.
        Useful for logging / dashboard display.
        """
        with self._lock:
            if not self._is_running or self._duration == 0:
                return 0.0
            return min(self._elapsed() / self._duration, 1.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _elapsed(self) -> float:
        """
        Total elapsed seconds (accumulated pause + current run).
        Must be called with self._lock held.
        """
        if self._is_paused:
            return self._elapsed_pause
        return self._elapsed_pause + (time.monotonic() - self._start_mono)

    def __repr__(self) -> str:
        with self._lock:
            if not self._is_running:
                return "PhaseTimer(stopped)"
            state = "paused" if self._is_paused else "running"
            return (
                f"PhaseTimer({state}, "
                f"elapsed={self.elapsed_seconds:.1f}s / "
                f"{self._duration:.1f}s, "
                f"remaining={self.remaining_seconds:.1f}s)"
            )


# =============================================================================
# USAGE EXAMPLE
# No hardware or config needed — pure Python.
# Run from anywhere: python edge/logic/timer.py
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=== PhaseTimer Smoke Test ===\n")

    timer = PhaseTimer()

    # ── Test 1: Basic countdown ──────────────────────────────────────
    print("[1] Basic countdown (3s duration, poll every 0.5s):")
    timer.start(3.0)
    while not timer.is_expired:
        print(
            f"  remaining={timer.remaining_seconds:.2f}s  "
            f"elapsed={timer.elapsed_seconds:.2f}s  "
            f"progress={timer.progress:.0%}"
        )
        time.sleep(0.5)
    print(f"  ✅ Timer expired. elapsed={timer.elapsed_seconds:.2f}s\n")

    # ── Test 2: Pause / Resume ───────────────────────────────────────
    print("[2] Pause / Resume test (5s duration, pause at ~1s for 1s):")
    timer.start(5.0)
    time.sleep(1.0)
    snap_before = timer.elapsed_seconds
    timer.pause()
    print(f"  Paused at elapsed={snap_before:.2f}s")

    time.sleep(1.0)   # paused — elapsed should not advance
    snap_paused = timer.elapsed_seconds
    print(f"  Still paused, elapsed={snap_paused:.2f}s (should be ~{snap_before:.2f}s)")
    assert abs(snap_paused - snap_before) < 0.05, "Elapsed advanced while paused!"

    timer.resume()
    print(f"  Resumed. remaining={timer.remaining_seconds:.2f}s")
    time.sleep(4.5)   # wait for rest of 5s to pass
    assert timer.is_expired
    print(f"  ✅ Expired after resume. Total elapsed={timer.elapsed_seconds:.2f}s\n")

    # ── Test 3: Restart mid-flight ───────────────────────────────────
    print("[3] Restart mid-flight (start 10s, restart with 2s after 0.5s):")
    timer.start(10.0)
    time.sleep(0.5)
    print(f"  After 0.5s: remaining={timer.remaining_seconds:.2f}s")
    timer.start(2.0)   # supersede with shorter duration
    print(f"  After restart(2s): remaining={timer.remaining_seconds:.2f}s (should be ~2s)")
    assert timer.remaining_seconds > 1.9
    time.sleep(2.1)
    assert timer.is_expired
    print(f"  ✅ Restarted timer expired correctly.\n")

    # ── Test 4: Stop ─────────────────────────────────────────────────
    print("[4] Stop test:")
    timer.start(60.0)
    timer.stop()
    assert not timer.is_running
    assert not timer.is_expired
    assert timer.remaining_seconds == 0.0
    print(f"  ✅ Stopped timer: running={timer.is_running}, expired={timer.is_expired}\n")

    # ── Test 5: Invalid duration ─────────────────────────────────────
    print("[5] Invalid duration rejection:")
    try:
        timer.start(-1.0)
        print("  ❌ Should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"  ✅ Caught: {e}\n")

    print("=== All PhaseTimer tests passed ===")
