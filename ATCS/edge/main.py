# =============================================================================
# ATCS - Adaptive Traffic Control System
# edge/main.py | Edge Orchestrator — Raspberry Pi 5
#
# Architecture: Dual-Thread ("Thin Edge")
#
#   ┌─────────────────────────────────────────────────────────┐
#   │                     edge/main.py                        │
#   │                                                         │
#   │  ┌──────────────────┐      ┌───────────────────────┐   │
#   │  │  Reporter Thread  │      │  Controller Thread     │   │
#   │  │                  │      │                        │   │
#   │  │  Camera.capture()│      │  mqtt.get_decision()   │   │
#   │  │  → FramePayload  │      │  → StateMachine        │   │
#   │  │  → mqtt.publish()│      │  → GPIODriver          │   │
#   │  │  (1 FPS loop)    │      │  → PhaseTimer          │   │
#   │  └──────────────────┘      └───────────────────────┘   │
#   │                                                         │
#   │  ┌──────────────────────────────────────────────────┐  │
#   │  │  SafetyWatchdog (daemon thread)                   │  │
#   │  │  Monitors last_command_time → fallback if timeout │  │
#   │  └──────────────────────────────────────────────────┘  │
#   └─────────────────────────────────────────────────────────┘
#
# Startup sequence:
#   1. Load + validate config
#   2. Setup logging
#   3. Init GPIO → ALL_RED (hardware safe state)
#   4. Open camera
#   5. Connect MQTT broker
#   6. Start SafetyWatchdog daemon thread
#   7. Start Reporter thread
#   8. Run Controller loop in main thread
#   9. On KeyboardInterrupt / error → graceful shutdown
#
# Rules:
#   - Reporter thread NEVER waits for server response.
#   - Controller thread NEVER captures frames.
#   - MQTT callbacks NEVER touch GPIO.
#   - All blocking calls have timeouts — nothing hangs forever.
# =============================================================================

import logging
import signal
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — allows running from edge/ or project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config_loader import load_config, setup_logging
from shared.schemas import FramePayload
from shared.version import version_banner

from edge.hardware.camera import Camera
from edge.hardware.gpio_driver import GPIODriver
from edge.logic.state_machine import StateMachine, StateChangeEvent
from edge.logic.timer import PhaseTimer
from edge.logic.safety import SafetyWatchdog
from edge.network.mqtt_client import EdgeMQTTClient

logger = logging.getLogger(__name__)


# =============================================================================
# Reporter Thread
# Captures frames and publishes them to the broker at configured FPS.
# Completely independent of command reception — never waits for server.
# =============================================================================

def reporter_thread(
    camera:      Camera,
    mqtt_client: EdgeMQTTClient,
    intersection_id: str,
    stop_event:  threading.Event,
) -> None:
    """
    Reporter thread entry point.

    Runs a tight capture → encode → publish loop at the configured
    frame rate. Drops frames silently if MQTT is disconnected rather
    than blocking or buffering.

    Parameters
    ----------
    camera : Camera
        Opened camera instance (camera.open() already called).
    mqtt_client : EdgeMQTTClient
        Connected MQTT client (connect() already called).
    intersection_id : str
        Intersection ID to embed in every FramePayload.
    stop_event : threading.Event
        Set by main thread to signal a clean shutdown.
    """
    logger.info("[Reporter] Thread started.")
    sequence = 0

    while not stop_event.is_set():
        try:
            # Rate-limit to configured capture_fps
            camera.wait_until_ready()

            if stop_event.is_set():
                break

            # Capture and encode
            frame_b64, width, height = camera.capture_frame_b64()
            sequence += 1

            # Build payload
            payload = FramePayload(
                intersection_id = intersection_id,
                frame_b64       = frame_b64,
                sequence        = sequence,
                width           = width,
                height          = height,
            )

            # Publish (non-blocking — drops if disconnected)
            published = mqtt_client.publish_frame(payload)

            if published:
                logger.debug(
                    "[Reporter] Frame seq=%d published (%.1f KB).",
                    sequence,
                    len(frame_b64) / 1024,
                )
            else:
                logger.warning(
                    "[Reporter] Frame seq=%d dropped (not connected).",
                    sequence,
                )

        except RuntimeError as exc:
            logger.error("[Reporter] Camera error: %s. Retrying in 2s.", exc)
            time.sleep(2.0)

        except Exception as exc:   # pylint: disable=broad-except
            logger.error(
                "[Reporter] Unexpected error: %s. Retrying in 2s.", exc,
                exc_info=True,
            )
            time.sleep(2.0)

    logger.info(
        "[Reporter] Thread stopped. Total frames published: %d.", sequence
    )


# =============================================================================
# Controller Thread (main thread)
# Drains the MQTT decision queue and applies phase commands via StateMachine.
# =============================================================================

def run_controller(
    mqtt_client:   EdgeMQTTClient,
    state_machine: StateMachine,
    timer:         PhaseTimer,
    stop_event:    threading.Event,
) -> None:
    """
    Controller loop — runs in the main thread.

    Polls the MQTT decision queue and hands each PhaseDecision to the
    StateMachine. The PhaseTimer tracks how long the current phase
    should be held; expiry is logged but the server controls actual
    transitions (the watchdog handles server silence).

    Parameters
    ----------
    mqtt_client : EdgeMQTTClient
        Connected MQTT client to poll for decisions.
    state_machine : StateMachine
        Validates and applies phase transitions.
    timer : PhaseTimer
        Tracks how long the current phase should be held.
    stop_event : threading.Event
        Set externally to signal shutdown.
    """
    logger.info("[Controller] Loop started.")

    while not stop_event.is_set():
        try:
            # Block up to 0.5s waiting for a decision
            decision = mqtt_client.get_pending_decision(timeout=0.5)

            if decision is None:
                # No command arrived — check if current phase has expired
                if timer.is_running and timer.is_expired:
                    logger.info(
                        "[Controller] Phase timer expired "
                        "(phase=%s, duration=%.1fs). "
                        "Awaiting next server command or watchdog fallback.",
                        state_machine.current_phase.value,
                        timer.duration,
                    )
                    timer.stop()
                continue

            # Decision received — apply via state machine
            accepted = state_machine.process_decision(decision)

            if accepted:
                # Start timer for this phase's duration
                if decision.duration_seconds > 0:
                    timer.start(decision.duration_seconds)
                    logger.debug(
                        "[Controller] Timer started: %.1fs for phase %s.",
                        decision.duration_seconds,
                        decision.phase.value,
                    )
            else:
                logger.warning(
                    "[Controller] Decision rejected by state machine: "
                    "phase=%s seq=%d.",
                    decision.phase.value, decision.sequence,
                )

        except Exception as exc:   # pylint: disable=broad-except
            logger.error(
                "[Controller] Unexpected error: %s.", exc, exc_info=True
            )
            time.sleep(0.5)

    logger.info("[Controller] Loop stopped.")


# =============================================================================
# State Change Listener
# Registered with StateMachine — called on every successful transition.
# =============================================================================

def on_state_change(event: StateChangeEvent) -> None:
    """
    Listener called after every successful phase transition.

    Logs a structured summary. In future iterations, this is where
    you would also write to a local SQLite log on the edge for audit.

    Parameters
    ----------
    event : StateChangeEvent
        Details of the transition that just occurred.
    """
    logger.info(
        "[StateChange] %s → %s | duration=%.1fs | vehicles=%d | "
        "seq=%d | source=%s",
        event.previous_phase.value,
        event.new_phase.value,
        event.duration_seconds,
        event.vehicle_count,
        event.sequence,
        event.source,
    )


# =============================================================================
# Graceful Shutdown
# =============================================================================

def shutdown(
    stop_event:   threading.Event,
    reporter:     threading.Thread,
    watchdog:     SafetyWatchdog,
    mqtt_client:  EdgeMQTTClient,
    camera:       Camera,
    gpio_driver:  GPIODriver,
) -> None:
    """
    Orderly shutdown sequence. Called on SIGINT, SIGTERM, or fatal error.

    Order matters:
      1. Signal threads to stop (stop_event)
      2. Stop watchdog (no more fallback cycles)
      3. Disconnect MQTT (no more publishes/callbacks)
      4. Join Reporter thread
      5. Release camera
      6. GPIO → ALL_RED, then cleanup
    """
    logger.info("=== Initiating graceful shutdown ===")

    # 1. Signal all threads
    stop_event.set()

    # 2. Stop watchdog daemon
    try:
        watchdog.stop()
    except Exception as exc:   # pylint: disable=broad-except
        logger.warning("Watchdog stop error: %s", exc)

    # 3. Disconnect MQTT
    try:
        mqtt_client.disconnect()
    except Exception as exc:   # pylint: disable=broad-except
        logger.warning("MQTT disconnect error: %s", exc)

    # 4. Join Reporter thread (give it up to 5s to exit cleanly)
    if reporter.is_alive():
        reporter.join(timeout=5.0)
        if reporter.is_alive():
            logger.warning("Reporter thread did not exit cleanly.")

    # 5. Release camera
    try:
        camera.release()
    except Exception as exc:   # pylint: disable=broad-except
        logger.warning("Camera release error: %s", exc)

    # 6. GPIO safe state + cleanup
    try:
        gpio_driver.force_all_red()
        time.sleep(0.5)
        gpio_driver.cleanup()
    except Exception as exc:   # pylint: disable=broad-except
        logger.warning("GPIO cleanup error: %s", exc)

    logger.info("=== Shutdown complete ===")


# =============================================================================
# Entry Point
# =============================================================================

def main() -> None:
    """
    Edge node entry point.

    Initialises all subsystems, starts background threads, and runs
    the Controller loop until interrupted.
    """

    # ------------------------------------------------------------------
    # 1. Load config + setup logging
    # ------------------------------------------------------------------
    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FATAL] Config load failed: {exc}", file=sys.stderr)
        sys.exit(1)

    setup_logging(config)
    logger.info("=== ATCS Edge Node Starting ===")
    logger.info(version_banner())
    logger.info("Intersection: %s", config.intersection.id)

    # ------------------------------------------------------------------
    # 2. Initialise hardware
    # ------------------------------------------------------------------
    gpio_driver = GPIODriver(config)
    try:
        gpio_driver.setup()   # Boots to ALL_RED immediately
    except RuntimeError as exc:
        logger.critical("GPIO setup failed: %s", exc)
        sys.exit(1)

    camera = Camera(config)
    try:
        camera.open()
    except RuntimeError as exc:
        logger.critical("Camera open failed: %s. Cleaning up GPIO.", exc)
        gpio_driver.cleanup()
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Initialise logic layer
    # ------------------------------------------------------------------
    state_machine = StateMachine(
        gpio_driver     = gpio_driver,
        intersection_id = config.intersection.id,
    )
    state_machine.register_listener(on_state_change)

    timer = PhaseTimer()

    # ------------------------------------------------------------------
    # 4. Connect MQTT
    # ------------------------------------------------------------------
    mqtt_client = EdgeMQTTClient(config)
    try:
        mqtt_client.connect()
    except ConnectionError as exc:
        logger.critical("MQTT connection failed: %s. Cleaning up.", exc)
        camera.release()
        gpio_driver.cleanup()
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Start SafetyWatchdog
    # ------------------------------------------------------------------
    watchdog = SafetyWatchdog(state_machine, gpio_driver, config)
    watchdog.start()

    # ------------------------------------------------------------------
    # 6. Start Reporter thread
    # ------------------------------------------------------------------
    stop_event = threading.Event()

    reporter = threading.Thread(
        target = reporter_thread,
        kwargs = dict(
            camera          = camera,
            mqtt_client     = mqtt_client,
            intersection_id = config.intersection.id,
            stop_event      = stop_event,
        ),
        name   = "ReporterThread",
        daemon = False,   # Must join cleanly on shutdown
    )
    reporter.start()
    logger.info("Reporter thread started.")

    # ------------------------------------------------------------------
    # 7. Register OS signal handlers for clean shutdown
    # ------------------------------------------------------------------
    def _handle_signal(signum: int, frame: object) -> None:
        logger.info("Signal %d received — initiating shutdown.", signum)
        shutdown(stop_event, reporter, watchdog, mqtt_client, camera, gpio_driver)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ------------------------------------------------------------------
    # 8. Run Controller loop (main thread)
    # ------------------------------------------------------------------
    logger.info("=== ATCS Edge Node Running ===")
    logger.info(
        "Reporter: %.1f FPS | Watchdog timeout: %ds | "
        "Fallback green: %ds",
        config.edge.capture_fps,
        config.edge.server_timeout_seconds,
        config.edge.fallback_green_duration,
    )

    try:
        run_controller(mqtt_client, state_machine, timer, stop_event)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")

    except Exception as exc:   # pylint: disable=broad-except
        logger.critical(
            "Controller loop fatal error: %s", exc, exc_info=True
        )

    finally:
        shutdown(stop_event, reporter, watchdog, mqtt_client, camera, gpio_driver)


if __name__ == "__main__":
    main()
