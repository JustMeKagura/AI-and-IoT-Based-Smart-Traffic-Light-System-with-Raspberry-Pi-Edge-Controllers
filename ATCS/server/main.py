"""
server/main.py
─────────────────────────────────────────────────────────────────────────────
Responsibility : Runtime orchestrator for the server-side ATCS pipeline.

Architecture
    ServerApp owns all components and wires them together.
    Mirrors edge/main.py's startup pattern: load_config() once, pass the
    immutable ATCSConfig down to every component.

    Paho's network thread (owned by ServerMQTTClient, loop_start) calls
    on_frame_callback for every incoming FramePayload. Inference runs
    synchronously inside that callback — one frame at a time.

    ┌─────────┐   FramePayload   ┌────────────────────────────────────────┐
    │   Pi    │ ──────────────► │  on_frame_callback()                   │
    │  (edge) │                 │    preprocessor → detector → counter   │
    │         │ ◄────────────── │    → smoother → timing_algo → database │
    └─────────┘  PhaseDecision  └────────────────────────────────────────┘

    Wire formats are shared.schemas.FramePayload / PhaseDecision — built
    and parsed entirely inside ServerMQTTClient. main.py never touches
    raw paho or raw JSON for the network boundary.

Lanes
    config.yaml has no per-deployment lane geometry section validated by
    ATCSConfig (lane regions are server-only and vary per intersection),
    so it's loaded separately, directly from the YAML, by this module.

Graceful shutdown
    SIGINT / SIGTERM → shutdown() → disconnect MQTT, unload model, close DB.

Author : Oussama (server side)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap — allows running from server/ or project root (mirrors edge/main.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # type: ignore

from shared.config_loader import ATCSConfig, load_config, setup_logging
from shared.schemas import FramePayload
from shared.version import version_banner

from server.inference.preprocessor import PreprocessError, preprocess
from server.inference.detector     import Detector, DetectorError
from server.inference.counter      import Counter
from server.logic.smoother         import Smoother
from server.logic.timing_algo      import Algorithm, TimingAlgo, TimingResult
from server.persistence.database   import Database, DatabaseError
from server.network.mqtt_client    import ServerMQTTClient

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────────────

class AppError(RuntimeError):
    """Raised for unrecoverable startup/configuration errors."""


# ──────────────────────────────────────────────────────────────────────────────
# Lane config loader (server-only — not part of ATCSConfig)
# ──────────────────────────────────────────────────────────────────────────────

def load_lane_map(config_path: str | Path = "config.yaml") -> dict[str, tuple]:
    """
    Load the 'lanes:' section directly from config.yaml.

    This lives outside shared.config_loader because lane pixel geometry
    is a server-only concern (the edge side never needs it) and varies
    per physical intersection/camera mount — unlike everything in
    ATCSConfig, which both nodes share and validate identically.

    Returns
    -------
    dict[str, tuple[float, float, float, float]]
        {lane_id: (x1, y1, x2, y2)}

    Raises
    ------
    AppError
        If config.yaml is missing/malformed, or 'lanes:' is empty.
    """
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise AppError(f"Config file not found: '{config_path}'.")
    except yaml.YAMLError as exc:
        raise AppError(f"Malformed config.yaml: {exc}") from exc

    lanes_cfg = (raw or {}).get("lanes", {})
    if not lanes_cfg:
        raise AppError(
            "config.yaml must define at least one lane under 'lanes:'."
        )

    return {lid: tuple(region) for lid, region in lanes_cfg.items()}


# ──────────────────────────────────────────────────────────────────────────────
# ServerApp
# ──────────────────────────────────────────────────────────────────────────────

class ServerApp:
    """
    Full server-side pipeline orchestrator.

    Usage (production)
    ------------------
        config   = load_config()
        lane_map = load_lane_map()
        app = ServerApp.build(config, lane_map)
        app.start()     # blocks until SIGINT / SIGTERM

    Usage (testing)
    ---------------
        app = ServerApp(detector=mock, counter=..., ..., mqtt_client=mock)
        app.handle_frame(frame_payload)   # call directly, no MQTT needed
    """

    def __init__(
        self,
        config:      ATCSConfig,
        detector:    Detector,
        counter:     Counter,
        smoother:    Smoother,
        timing_algo: TimingAlgo,
        database:    Database,
        mqtt_client: ServerMQTTClient,
    ) -> None:
        self._config       = config
        self._detector      = detector
        self._counter       = counter
        self._smoother      = smoother
        self._timing_algo   = timing_algo
        self._database      = database
        self._mqtt          = mqtt_client

        self._running     = False
        self._frames_rx   = 0   # total frames received
        self._frames_ok   = 0   # frames successfully processed
        self._frames_err  = 0   # frames that hit a pipeline error

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        config:   ATCSConfig,
        lane_map: dict[str, tuple],
    ) -> "ServerApp":
        """
        Build a fully-wired ServerApp from an ATCSConfig + lane map.

        Mirrors edge/main.py's component construction: every component
        is built straight from the typed config, no raw dict access.

        Raises AppError if a component fails to initialise.
        """
        try:
            detector = Detector(
                model_path = config.server.model_path,
                conf       = config.server.inference_conf,
                imgsz      = config.server.inference_imgsz,
                device     = "cpu",
            )

            counter = Counter(lane_map)

            smoother = Smoother(
                window = config.server.smoother_window_size,
            )

            timing_algo = TimingAlgo(
                algorithm        = Algorithm.LINEAR,
                min_green        = config.server.min_green_duration,
                max_green        = config.server.max_green_duration,
                vehicles_per_sec = config.server.vehicles_per_second,
            )

            database = Database(
                db_path = config.server.db_path,
            )

        except (ValueError, TypeError) as exc:
            raise AppError(f"Failed to build server components: {exc}") from exc

        app = cls(
            config      = config,
            detector    = detector,
            counter     = counter,
            smoother    = smoother,
            timing_algo = timing_algo,
            database    = database,
            mqtt_client = None,  # type: ignore  # set right below
        )

        # ServerMQTTClient needs the frame callback, which needs `app` —
        # constructed last and attached after the rest of the app exists.
        mqtt_client = ServerMQTTClient(
            config            = config,
            on_frame_callback = app.handle_frame,
        )
        app._mqtt = mqtt_client
        return app

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Open DB, load model, connect MQTT, block until shutdown signal.
        """
        log.info("ServerApp starting…")
        self._running = True

        self._database.open()
        self._detector.load()

        self._mqtt.connect()

        log.info("ServerApp running. Intersection: %s", self._config.intersection.id)

    def shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        log.info(
            "Shutting down — frames_rx=%d ok=%d err=%d",
            self._frames_rx, self._frames_ok, self._frames_err,
        )
        try:
            self._mqtt.disconnect()
        except Exception as exc:
            log.warning("MQTT disconnect error (non-fatal): %s", exc)
        try:
            self._detector.unload()
        except Exception as exc:
            log.warning("Detector unload error (non-fatal): %s", exc)
        try:
            self._database.close()
        except Exception as exc:
            log.warning("Database close error (non-fatal): %s", exc)
        log.info("ServerApp stopped.")

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def handle_frame(self, frame: FramePayload) -> Optional[TimingResult]:
        """
        Full inference pipeline for one incoming FramePayload.

        This is the on_frame_callback passed to ServerMQTTClient — it
        fires from Paho's network thread (mirrors EdgeMQTTClient's
        on_message → _on_frame_cb pattern). Kept synchronous, matching
        the Pi's send-and-wait design: one frame fully processed and
        replied to before the next is accepted.

        Returns TimingResult on success, None if the frame was skipped
        (non-fatal errors like a corrupt image). Raises only on truly
        unexpected errors, which the caller logs and counts as dropped.
        """
        self._frames_rx += 1
        try:
            result = self._process(frame)
            self._frames_ok += 1
            return result
        except Exception as exc:
            self._frames_err += 1
            log.error("Unhandled error in pipeline (frame dropped): %s", exc, exc_info=True)
            return None

    def _process(self, frame: FramePayload) -> Optional[TimingResult]:
        # ── 1. Preprocess ─────────────────────────────────────────────────────
        try:
            pre = preprocess(frame.frame_b64)
        except PreprocessError as exc:
            log.warning("Frame seq=%d dropped — preprocessing failed: %s", frame.sequence, exc)
            return None

        # ── 2. Detect ─────────────────────────────────────────────────────────
        try:
            inference = self._detector.predict(pre.frame)
        except DetectorError as exc:
            log.warning("Frame seq=%d dropped — detection failed: %s", frame.sequence, exc)
            return None

        # ── 3. Count ──────────────────────────────────────────────────────────
        lane_counts = self._counter.count(inference.detections)

        # ── 4. Smooth ─────────────────────────────────────────────────────────
        smoothed = self._smoother.update(lane_counts)

        # ── 5. Decide ─────────────────────────────────────────────────────────
        timing = self._timing_algo.compute(smoothed)

        # ── 6. Publish decisions (one PhaseDecision per lane) ────────────────
        wires = timing.to_wire_decisions(
            self._config.intersection.id,
            frame.sequence,
        )
        for wire in wires:
            self._mqtt.publish_decision(wire)

        # ── 7. Log to DB (non-fatal on failure) ───────────────────────────────
        for decision in timing.decisions:
            try:
                self._database.log_decision(decision, frame_index=timing.frame_index)
            except DatabaseError as exc:
                log.warning("DB write failed (non-fatal, frame not lost): %s", exc)

        log.info(
            "Frame %d processed | vehicles=%d | decisions=%s",
            timing.frame_index,
            smoothed.total,
            {d.lane_id: d.green_duration for d in timing.decisions},
        )

        return timing

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def frames_received(self) -> int:
        return self._frames_rx

    @property
    def frames_ok(self) -> int:
        return self._frames_ok

    @property
    def frames_error(self) -> int:
        return self._frames_err


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load config + setup logging (mirrors edge/main.py exactly)
    # ------------------------------------------------------------------
    try:
        config = load_config()
        lane_map = load_lane_map()
    except (FileNotFoundError, ValueError, AppError) as exc:
        print(f"[FATAL] Config load failed: {exc}", file=sys.stderr)
        sys.exit(1)

    setup_logging(config)
    log.info("=== ATCS Server Node Starting ===")
    log.info(version_banner())
    log.info("Intersection: %s", config.intersection.id)
    log.info("Lanes: %s", list(lane_map.keys()))

    # ------------------------------------------------------------------
    # 2. Build app
    # ------------------------------------------------------------------
    try:
        app = ServerApp.build(config, lane_map)
    except AppError as exc:
        log.critical("Server build failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Register signal handlers for clean shutdown
    # ------------------------------------------------------------------
    import signal

    def _handle_signal(signum: int, frame: object) -> None:
        log.info("Signal %d received — initiating shutdown.", signum)
        app.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ------------------------------------------------------------------
    # 4. Start and block
    # ------------------------------------------------------------------
    try:
        app.start()
        # ServerMQTTClient.connect() runs Paho's network thread (loop_start)
        # in the background; block the main thread here until a signal fires.
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received.")

    except Exception as exc:
        log.critical("Fatal error: %s", exc, exc_info=True)

    finally:
        app.shutdown()


if __name__ == "__main__":
    main()