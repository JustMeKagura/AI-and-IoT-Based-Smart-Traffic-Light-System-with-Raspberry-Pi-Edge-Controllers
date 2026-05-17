# =============================================================================
# ATCS - Adaptive Traffic Control System
# edge/hardware/gpio_driver.py | Traffic light LED control via gpiozero
#
# Responsibilities:
#   - Map Phase enum values to physical GPIO pin states
#   - Turn the correct LEDs on/off for each phase transition
#   - Provide a safe all_off() and cleanup() for shutdown
#   - Abstract hardware so state_machine.py never touches GPIO directly
#
# Rules:
#   - Only this file touches GPIO. No other module imports gpiozero.
#   - ALL_RED and boot state must be achievable without a server command.
#   - Must be thread-safe: Controller thread calls apply_phase() while
#     watchdog may call apply_phase(ALL_RED) concurrently.
#   - Pin numbers come from ATCSConfig (BCM numbering via gpiozero default).
# =============================================================================

import logging
import threading
from typing import Dict

from gpiozero import LED
from gpiozero.exc import GPIOZeroError

from shared.constants import Phase
from shared.config_loader import ATCSConfig

logger = logging.getLogger(__name__)


class GPIODriver:
    """
    Controls the three-light traffic signal (Red / Amber / Green LEDs).

    Each LED maps to a BCM GPIO pin configured in config.yaml.
    Phase transitions are applied atomically under a lock to prevent
    the Controller thread and watchdog from racing on pin state.

    Parameters
    ----------
    config : ATCSConfig
        Loaded system config. Pin assignments read from config.edge.gpio_pins.
    """

    # Maps each Phase to which LEDs should be ON (True) or OFF (False)
    # Format: {Phase: {red: bool, amber: bool, green: bool}}
    _PHASE_MAP: Dict[Phase, Dict[str, bool]] = {
        Phase.ALL_RED: {"red": True,  "amber": False, "green": False},
        Phase.RED:     {"red": True,  "amber": False, "green": False},
        Phase.AMBER:   {"red": False, "amber": True,  "green": False},
        Phase.GREEN:   {"red": False, "amber": False, "green": True},
    }

    def __init__(self, config: ATCSConfig) -> None:
        self._pins = config.edge.gpio_pins
        self._lock = threading.Lock()
        self._current_phase: Phase = Phase.ALL_RED
        self._leds: Dict[str, LED] = {}
        self._initialised: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Initialise GPIO pins and boot into ALL_RED state.

        Raises
        ------
        RuntimeError
            If GPIO initialisation fails (wrong pin, hardware fault).
        """
        logger.info(
            "Initialising GPIO pins — R:%d A:%d G:%d (BCM).",
            self._pins.red, self._pins.amber, self._pins.green,
        )

        try:
            self._leds = {
                "red":   LED(self._pins.red),
                "amber": LED(self._pins.amber),
                "green": LED(self._pins.green),
            }
        except GPIOZeroError as exc:
            raise RuntimeError(
                f"GPIO setup failed: {exc}\n"
                f"Check BCM pin assignments: "
                f"R={self._pins.red}, A={self._pins.amber}, G={self._pins.green}."
            ) from exc

        self._initialised = True
        logger.info("GPIO pins initialised. Booting to ALL_RED.")

        # Safety: always boot with all lights known-state
        self._apply_pins(Phase.ALL_RED)

    def cleanup(self) -> None:
        """
        Turn off all LEDs and release GPIO resources.
        Call in a finally block in edge/main.py on shutdown.
        """
        if not self._initialised:
            return

        logger.info("GPIO cleanup: turning off all LEDs.")
        self._all_off()

        for name, led in self._leds.items():
            try:
                led.close()
                logger.debug("LED '%s' closed.", name)
            except GPIOZeroError as exc:
                logger.warning("Error closing LED '%s': %s", name, exc)

        self._leds.clear()
        self._initialised = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_phase(self, phase: Phase) -> None:
        """
        Atomically transition the traffic light to the given phase.

        Thread-safe: acquires lock before touching any pin.
        Called by the Controller thread (normal operation) and by
        safety.py watchdog (fallback mode).

        Parameters
        ----------
        phase : Phase
            Target phase to display. Must be a valid Phase enum member.

        Raises
        ------
        RuntimeError
            If setup() has not been called, or if GPIO write fails.
        """
        if not self._initialised:
            raise RuntimeError(
                "GPIODriver.setup() must be called before apply_phase()."
            )

        with self._lock:
            if phase == self._current_phase:
                logger.debug("apply_phase(%s): already in this phase, no-op.", phase.value)
                return

            logger.info(
                "Phase transition: %s → %s",
                self._current_phase.value, phase.value,
            )
            self._apply_pins(phase)
            self._current_phase = phase

    def force_all_red(self) -> None:
        """
        Emergency method: immediately set ALL_RED regardless of current phase.
        Bypasses the phase equality check in apply_phase().
        Called by safety.py watchdog on server timeout.
        """
        if not self._initialised:
            logger.error("force_all_red() called before setup(). Ignoring.")
            return

        with self._lock:
            logger.warning("SAFETY: force_all_red() called — overriding current phase.")
            self._apply_pins(Phase.ALL_RED)
            self._current_phase = Phase.ALL_RED

    @property
    def current_phase(self) -> Phase:
        """The phase currently displayed on the hardware."""
        return self._current_phase

    @property
    def is_ready(self) -> bool:
        """True if setup() has been called successfully."""
        return self._initialised

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _apply_pins(self, phase: Phase) -> None:
        """
        Write pin states for the given phase.
        Must be called with self._lock held.

        Raises
        ------
        RuntimeError
            If the phase has no entry in _PHASE_MAP (should never happen).
        """
        pin_states = self._PHASE_MAP.get(phase)
        if pin_states is None:
            raise RuntimeError(
                f"No pin mapping defined for phase: {phase.value}. "
                f"Update GPIODriver._PHASE_MAP."
            )

        try:
            # Turn off all first to prevent any brief overlap
            # (e.g. both green and red on simultaneously during transition)
            for led in self._leds.values():
                led.off()

            # Then turn on only the required LED(s)
            for name, should_be_on in pin_states.items():
                if should_be_on:
                    self._leds[name].on()
                    logger.debug("LED '%s' → ON (pin %s).", name,
                                 getattr(self._pins, name))

        except GPIOZeroError as exc:
            raise RuntimeError(f"GPIO write failed during phase {phase.value}: {exc}") from exc

    def _all_off(self) -> None:
        """Turn off every LED unconditionally. No lock needed if called from cleanup()."""
        for name, led in self._leds.items():
            try:
                led.off()
            except GPIOZeroError as exc:
                logger.warning("Could not turn off LED '%s': %s", name, exc)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "ready" if self._initialised else "not initialised"
        return (
            f"GPIODriver(status={status}, "
            f"phase={self._current_phase.value}, "
            f"pins=R{self._pins.red}/A{self._pins.amber}/G{self._pins.green})"
        )


# =============================================================================
# USAGE EXAMPLE
# Run on a Raspberry Pi with LEDs wired to BCM pins 17 (R), 27 (A), 22 (G).
# Cycles through all phases so you can verify wiring visually.
#
#   python hardware/gpio_driver.py
# =============================================================================

if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from shared.config_loader import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg)

    driver = GPIODriver(cfg)

    try:
        driver.setup()
        print(f"Driver ready: {driver}\n")

        cycle = [Phase.ALL_RED, Phase.RED, Phase.GREEN, Phase.AMBER, Phase.RED]

        for phase in cycle:
            print(f"  Applying phase: {phase.value} ...", end=" ", flush=True)
            driver.apply_phase(phase)
            print(f"✅  Current: {driver.current_phase.value}")
            time.sleep(2)

        print("\nTesting force_all_red()...")
        driver.force_all_red()
        print(f"  Current phase after force: {driver.current_phase.value}")
        assert driver.current_phase == Phase.ALL_RED
        print("  ✅ force_all_red OK")

        print("\n✅ GPIO driver test complete.")

    except RuntimeError as e:
        print(f"\n❌ GPIO error: {e}")
        sys.exit(1)

    finally:
        driver.cleanup()
        print("GPIO cleaned up.")
