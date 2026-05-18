"""
server/main.py
─────────────────────────────────────────────────────────────────────────────
Responsibility : Runtime orchestrator for the server-side ATCS pipeline.

Architecture
    ServerApp owns all components and wires them together.
    MQTT loop runs on the main thread (paho's loop_forever).
    Inference runs synchronously inside the MQTT on_message callback —
    one frame at a time, matching the Pi's send-and-wait pattern.

    ┌─────────┐   Base64 JPEG   ┌────────────────────────────────────────┐
    │   Pi    │ ─────────────► │  on_message()                          │
    │  (edge) │                │    preprocessor → detector → counter   │
    │         │ ◄───────────── │    → smoother → timing_algo → database │
    └─────────┘  PhaseDecision └────────────────────────────────────────┘

MQTT message format (incoming, topic: atcs/frames/<lane_id>)
    JSON: { "image": "<base64_jpeg>", "lane_id": "<str>", "ts": <float> }

MQTT message format (outgoing, topic: atcs/decisions)
    JSON: { "lane_id": "<str>", "green_duration": <float>,
            "algorithm": "<str>", "was_clamped": <bool>,
            "clamp_reason": "<str>", "frame_index": <int>, "ts": <float> }

Graceful shutdown
    SIGINT / SIGTERM → _shutdown() → close DB, disconnect MQTT, unload model.

Author : Oussama (server side)
"""

from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import asdict
from typing import Optional

from server.inference.preprocessor import PreprocessError, preprocess
from server.inference.detector     import Detector, DetectorError
from server.inference.counter      import Counter
from server.logic.smoother         import Smoother
from server.logic.timing_algo      import TimingAlgo, TimingResult
from server.persistence.database   import Database, DatabaseError

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────────────

class AppError(RuntimeError):
    """Raised for unrecoverable startup/configuration errors."""


# ──────────────────────────────────────────────────────────────────────────────
# ServerApp
# ──────────────────────────────────────────────────────────────────────────────

class ServerApp:
    """
    Full server-side pipeline orchestrator.

    Usage (production)
    ------------------
        app = ServerApp.from_config("config.yaml")
        app.start()     # blocks until SIGINT / SIGTERM

    Usage (testing)
    ---------------
        app = ServerApp(detector=mock, counter=..., ..., mqtt_client=mock)
        app._handle_message(topic, payload_bytes)   # call directly
    """

    def __init__(
        self,
        detector:    Detector,
        counter:     Counter,
        smoother:    Smoother,
        timing_algo: TimingAlgo,
        database:    Database,
        mqtt_client,                    # paho.mqtt.client.Client (injected)
        publish_topic: str = "atcs/decisions",
        frame_topic:   str = "atcs/frames/#",
    ) -> None:
        self._detector     = detector
        self._counter      = counter
        self._smoother     = smoother
        self._timing_algo  = timing_algo
        self._database     = database
        self._mqtt         = mqtt_client
        self._publish_topic   = publish_topic
        self._frame_topic     = frame_topic

        self._running      = False
        self._frames_rx    = 0      # total frames received
        self._frames_ok    = 0      # frames successfully processed
        self._frames_err   = 0      # frames that hit a pipeline error

        # Wire MQTT callbacks
        self._mqtt.on_connect    = self._on_connect
        self._mqtt.on_disconnect = self._on_disconnect
        self._mqtt.on_message    = self._on_message

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "ServerApp":
        """
        Build a fully-wired ServerApp from a config.yaml file.

        Raises AppError if config is missing, malformed, or components fail
        to initialise.
        """
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise AppError("PyYAML is not installed. Run: pip install pyyaml") from exc

        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            raise AppError(f"Config file not found: '{config_path}'.")
        except yaml.YAMLError as exc:
            raise AppError(f"Malformed config.yaml: {exc}") from exc

        if not isinstance(cfg, dict):
            raise AppError(f"config.yaml must be a YAML mapping, got {type(cfg).__name__}.")

        try:
            return cls._build_from_dict(cfg)
        except (KeyError, TypeError, ValueError) as exc:
            raise AppError(f"Invalid config structure: {exc}") from exc

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Open DB, load model, connect MQTT, block until shutdown signal.
        Registers SIGINT / SIGTERM handlers for graceful shutdown.
        """
        log.info("ServerApp starting…")
        self._running = True

        signal.signal(signal.SIGINT,  lambda *_: self._shutdown())
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown())

        self._database.open()
        self._detector.load()

        try:
            self._mqtt.connect_async(
                self._mqtt._host,   # set by from_config / tests
                self._mqtt._port,
                keepalive=60,
            )
            self._mqtt.loop_forever()
        except Exception as exc:
            log.error("MQTT loop terminated with error: %s", exc)
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        log.info(
            "Shutting down — frames_rx=%d ok=%d err=%d",
            self._frames_rx, self._frames_ok, self._frames_err,
        )
        try:
            self._mqtt.loop_stop()
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

    # ── MQTT callbacks ────────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            log.info("MQTT connected. Subscribing to '%s'.", self._frame_topic)
            client.subscribe(self._frame_topic, qos=1)
        else:
            log.error("MQTT connection refused (rc=%d).", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        if rc != 0:
            log.warning("MQTT unexpectedly disconnected (rc=%d). Resetting smoother.", rc)
            # Stale rolling-average history is misleading after a gap — flush it
            self._smoother.reset()
        else:
            log.info("MQTT disconnected cleanly.")

    def _on_message(self, client, userdata, msg) -> None:
        """Entry point for every incoming MQTT frame."""
        self._frames_rx += 1
        try:
            self._handle_message(msg.topic, msg.payload)
            self._frames_ok += 1
        except Exception as exc:
            self._frames_err += 1
            log.error("Unhandled error in pipeline (frame dropped): %s", exc, exc_info=True)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _handle_message(self, topic: str, payload: bytes) -> Optional[TimingResult]:
        """
        Full inference pipeline for one incoming MQTT message.

        Separated from _on_message so tests can call it directly without
        needing a real paho Message object.

        Returns TimingResult on success, None if the frame was skipped
        (non-fatal errors like corrupt image or missing field).
        Raises only on truly unexpected errors.
        """
        # ── 1. Parse JSON payload ─────────────────────────────────────────────
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("Frame dropped — invalid JSON payload: %s", exc)
            return None

        if not isinstance(data, dict):
            log.warning("Frame dropped — payload is not a JSON object.")
            return None

        image_b64 = data.get("image")
        if not image_b64:
            log.warning("Frame dropped — 'image' field missing or empty.")
            return None

        # ── 2. Preprocess ─────────────────────────────────────────────────────
        try:
            pre = preprocess(image_b64)
        except PreprocessError as exc:
            log.warning("Frame dropped — preprocessing failed: %s", exc)
            return None

        # ── 3. Detect ─────────────────────────────────────────────────────────
        try:
            inference = self._detector.predict(pre.frame)
        except DetectorError as exc:
            log.warning("Frame dropped — detection failed: %s", exc)
            return None

        # ── 4. Count ──────────────────────────────────────────────────────────
        lane_counts = self._counter.count(inference.detections)

        # ── 5. Smooth ─────────────────────────────────────────────────────────
        smoothed = self._smoother.update(lane_counts)

        # ── 6. Decide ─────────────────────────────────────────────────────────
        timing = self._timing_algo.compute(smoothed)

        # ── 7. Publish decisions ──────────────────────────────────────────────
        for decision in timing.decisions:
            msg_dict = {
                "lane_id":        decision.lane_id,
                "green_duration": decision.green_duration,
                "algorithm":      decision.algorithm.value,
                "was_clamped":    decision.was_clamped,
                "clamp_reason":   decision.clamp_reason,
                "frame_index":    timing.frame_index,
                "ts":             time.time(),
            }
            self._mqtt.publish(
                self._publish_topic,
                json.dumps(msg_dict),
                qos=1,
            )

        # ── 8. Log to DB (non-fatal on failure) ───────────────────────────────
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

    # ── Private: config builder ───────────────────────────────────────────────

    @classmethod
    def _build_from_dict(cls, cfg: dict) -> "ServerApp":
        """Construct all components from a parsed config dict."""
        from server.logic.timing_algo import Algorithm

        mqtt_cfg   = cfg.get("mqtt",   {})
        server_cfg = cfg.get("server", {})
        timing_cfg = cfg.get("timing", {})
        lanes_cfg  = cfg.get("lanes",  {})
        db_cfg     = cfg.get("database", {})

        # Validate lanes exist before importing paho so the error is clear
        if not lanes_cfg:
            raise AppError("config.yaml must define at least one lane under 'lanes:'.")

        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except ImportError as exc:
            raise AppError("paho-mqtt is not installed. Run: pip install paho-mqtt") from exc

        # Parse lane regions
        lane_map = {
            lid: tuple(region)
            for lid, region in lanes_cfg.items()
        }

        # Build components
        detector = Detector(
            model_path=server_cfg.get("model_path", "weights/yolov8s.pt"),
            conf=server_cfg.get("conf", 0.15),
            imgsz=server_cfg.get("imgsz", 640),
            device=server_cfg.get("device", "cpu"),
        )

        counter = Counter(lane_map)

        smoother = Smoother(
            window=server_cfg.get("smoother_window", 3),
        )

        timing_algo = TimingAlgo(
            algorithm=Algorithm(timing_cfg.get("algorithm", "webster")),
            min_green=timing_cfg.get("min_green", 10.0),
            max_green=timing_cfg.get("max_green", 60.0),
            sat_flow=timing_cfg.get("sat_flow",   0.5),
            lost_time=timing_cfg.get("lost_time", 2.0),
        )

        database = Database(
            db_path=db_cfg.get("path", "atcs.db"),
        )

        # Build MQTT client
        client = mqtt.Client(client_id="atcs-server")
        client._host = mqtt_cfg.get("host", "localhost")
        client._port = mqtt_cfg.get("port", 1883)

        if mqtt_cfg.get("username"):
            client.username_pw_set(
                mqtt_cfg["username"],
                mqtt_cfg.get("password", ""),
            )

        return cls(
            detector=detector,
            counter=counter,
            smoother=smoother,
            timing_algo=timing_algo,
            database=database,
            mqtt_client=client,
            publish_topic=mqtt_cfg.get("publish_topic", "atcs/decisions"),
            frame_topic=mqtt_cfg.get("frame_topic",   "atcs/frames/#"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = ServerApp.from_config("config.yaml")
    app.start()


if __name__ == "__main__":
    main()