# =============================================================================
# ATCS - Adaptive Traffic Control System
# edge/network/mqtt_client.py | Paho MQTT wrapper for the Edge node
#
# Responsibilities:
#   - Connect to the Mosquitto broker with automatic reconnect
#   - Publish FramePayload messages (called by Reporter thread)
#   - Subscribe to control/signal topic and dispatch PhaseDecision
#     to StateMachine (called by Paho network thread via callback)
#   - Never block the MQTT callback thread (dispatch only, no GPIO here)
#   - Expose clean connect() / disconnect() / publish_frame() API
#
# Rules:
#   - MQTT callbacks must return instantly. Heavy work goes to Controller thread.
#   - Never call gpio_driver directly from this module.
#   - All topic strings built from constants + config, never hardcoded.
#   - Thread-safe: Reporter thread publishes, Paho thread fires callbacks,
#     Controller thread reads from the decision queue.
# =============================================================================

import json
import logging
import queue
import threading
import time
from typing import Callable, Optional

import paho.mqtt.client as mqtt

from shared.constants import (
    TOPIC_FRAMES_RAW,
    TOPIC_CONTROL_SIGNAL,
    MQTT_CLIENT_ID_EDGE,
    PAYLOAD_ENCODING,
)
from shared.schemas import FramePayload, PhaseDecision
from shared.config_loader import ATCSConfig

logger = logging.getLogger(__name__)


class EdgeMQTTClient:
    """
    Paho MQTT wrapper for the Edge (Raspberry Pi) node.

    Threading model:
        - Paho runs its own network thread internally (loop_start()).
        - Reporter thread calls publish_frame() — thread-safe via Paho's
          internal queue.
        - Paho network thread fires on_message() → puts PhaseDecision
          into _decision_queue (non-blocking).
        - Controller thread calls get_pending_decision() to drain the queue.

    Parameters
    ----------
    config : ATCSConfig
        Loaded system config. MQTT and intersection settings used.
    on_decision_callback : Callable[[PhaseDecision], None], optional
        If provided, called from the Controller thread (via get_pending_decision)
        or directly from on_message if you prefer push-style dispatch.
        Defaults to queue-based pull model if None.
    """

    def __init__(
        self,
        config: ATCSConfig,
        on_decision_callback: Optional[Callable[[PhaseDecision], None]] = None,
    ) -> None:
        self._cfg             = config.mqtt
        self._intersection_id = config.intersection.id
        self._qos             = config.mqtt.qos
        self._on_decision_cb  = on_decision_callback

        # Build topic strings once
        self._topic_publish   = TOPIC_FRAMES_RAW.format(
            intersection_id=self._intersection_id
        )
        self._topic_subscribe = TOPIC_CONTROL_SIGNAL.format(
            intersection_id=self._intersection_id
        )

        # Queue for incoming PhaseDecision objects
        # Controller thread drains this via get_pending_decision()
        self._decision_queue: queue.Queue[PhaseDecision] = queue.Queue(maxsize=10)

        # Connection state
        self._connected        = threading.Event()
        self._should_reconnect = True

        # Counters for diagnostics
        self._frames_published  = 0
        self._decisions_received = 0
        self._publish_errors     = 0

        # Initialise Paho client
        self._client = mqtt.Client(
            client_id  = MQTT_CLIENT_ID_EDGE,
            clean_session = True,
        )
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message
        self._client.on_publish    = self._on_publish

        # Exponential backoff state
        self._reconnect_delay = self._cfg.reconnect_delay_min

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Connect to the MQTT broker and start the Paho network thread.
        Blocks until the connection is established or raises on hard failure.

        Raises
        ------
        ConnectionError
            If the broker is unreachable and initial connection fails.
        """
        logger.info(
            "Connecting to MQTT broker at %s:%d (client_id=%s)...",
            self._cfg.broker_host, self._cfg.broker_port, MQTT_CLIENT_ID_EDGE,
        )

        try:
            self._client.connect(
                host      = self._cfg.broker_host,
                port      = self._cfg.broker_port,
                keepalive = self._cfg.keepalive,
            )
        except OSError as exc:
            raise ConnectionError(
                f"Cannot reach MQTT broker at "
                f"{self._cfg.broker_host}:{self._cfg.broker_port}. "
                f"Check broker_host in config.yaml and that Mosquitto is running. "
                f"Details: {exc}"
            ) from exc

        # Start Paho's background network thread
        self._client.loop_start()

        # Wait for on_connect to fire (up to keepalive seconds)
        if not self._connected.wait(timeout=self._cfg.keepalive):
            self._client.loop_stop()
            raise ConnectionError(
                f"MQTT broker connected at socket level but on_connect "
                f"did not fire within {self._cfg.keepalive}s. "
                f"Check broker configuration."
            )

        logger.info("MQTT connected. Publishing to: %s", self._topic_publish)
        logger.info("MQTT subscribed to: %s", self._topic_subscribe)

    def disconnect(self) -> None:
        """
        Gracefully disconnect from the broker and stop the network thread.
        Call in edge/main.py finally block during shutdown.
        """
        logger.info("Disconnecting MQTT client...")
        self._should_reconnect = False
        self._client.loop_stop()
        self._client.disconnect()
        self._connected.clear()
        logger.info("MQTT client disconnected.")

    # ------------------------------------------------------------------
    # Publishing (called by Reporter thread)
    # ------------------------------------------------------------------

    def publish_frame(self, payload: FramePayload) -> bool:
        """
        Publish a FramePayload to the frames/raw topic.

        Non-blocking: Paho queues the message internally and the network
        thread sends it. Safe to call from the Reporter thread at any time.

        Parameters
        ----------
        payload : FramePayload
            The frame payload to publish. Must be fully populated.

        Returns
        -------
        bool
            True if the message was queued successfully, False on error.
        """
        if not self._connected.is_set():
            logger.warning(
                "publish_frame(): not connected. Frame seq=%d dropped.",
                payload.sequence,
            )
            self._publish_errors += 1
            return False

        try:
            json_str = payload.to_json()
            result   = self._client.publish(
                topic   = self._topic_publish,
                payload = json_str.encode(PAYLOAD_ENCODING),
                qos     = self._qos,
            )

            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.error(
                    "publish_frame() failed: rc=%d seq=%d.",
                    result.rc, payload.sequence,
                )
                self._publish_errors += 1
                return False

            self._frames_published += 1
            logger.debug(
                "Frame published: seq=%d, size=%d bytes, topic=%s",
                payload.sequence, len(json_str), self._topic_publish,
            )
            return True

        except Exception as exc:   # pylint: disable=broad-except
            logger.error("publish_frame() unexpected error: %s", exc)
            self._publish_errors += 1
            return False

    # ------------------------------------------------------------------
    # Consuming decisions (called by Controller thread)
    # ------------------------------------------------------------------

    def get_pending_decision(self, timeout: float = 0.5) -> Optional[PhaseDecision]:
        """
        Retrieve the next PhaseDecision from the incoming queue.

        Blocks for up to `timeout` seconds waiting for a decision.
        Returns None if the queue is empty within the timeout.

        The Controller thread calls this in a loop:
            while running:
                decision = mqtt_client.get_pending_decision(timeout=0.5)
                if decision:
                    state_machine.process_decision(decision)

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait. Keep short (0.1–1.0s) so the
            Controller thread remains responsive to stop signals.

        Returns
        -------
        PhaseDecision | None
            The next decision, or None if none arrived within timeout.
        """
        try:
            return self._decision_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Paho Callbacks (run in Paho network thread — must not block)
    # ------------------------------------------------------------------

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: object,
        flags: dict,
        rc: int,
    ) -> None:
        """Fired by Paho when the broker connection is established."""
        if rc == mqtt.CONNACK_ACCEPTED:
            logger.info("MQTT on_connect: broker accepted connection (rc=0).")
            self._connected.set()
            self._reconnect_delay = self._cfg.reconnect_delay_min

            # Subscribe to control topic on every connect (handles reconnects)
            client.subscribe(self._topic_subscribe, qos=self._qos)
            logger.info("Subscribed: %s (QoS=%d)", self._topic_subscribe, self._qos)
        else:
            logger.error(
                "MQTT on_connect: broker rejected connection. rc=%d (%s).",
                rc, mqtt.connack_string(rc),
            )

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: object,
        rc: int,
    ) -> None:
        """Fired by Paho on disconnect (clean or unexpected)."""
        self._connected.clear()

        if rc == 0:
            logger.info("MQTT on_disconnect: clean disconnect.")
        else:
            logger.warning(
                "MQTT on_disconnect: unexpected disconnect (rc=%d). "
                "Paho will attempt to reconnect.",
                rc,
            )
            # Paho's loop_start() handles reconnect automatically when
            # reconnect_delay is configured. We log the current backoff state.
            logger.info(
                "Reconnect backoff: %.1fs (max: %ds).",
                self._reconnect_delay, self._cfg.reconnect_delay_max,
            )
            self._reconnect_delay = min(
                self._reconnect_delay * 2,
                self._cfg.reconnect_delay_max,
            )

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: object,
        msg: mqtt.MQTTMessage,
    ) -> None:
        """
        Fired by Paho when a message arrives on the subscribed topic.

        MUST return quickly — no GPIO, no heavy computation here.
        Parses the PhaseDecision and puts it in the queue for the
        Controller thread to process.
        """
        logger.debug(
            "MQTT message received: topic=%s, size=%d bytes.",
            msg.topic, len(msg.payload),
        )

        # Parse and validate
        try:
            decision = PhaseDecision.from_json(msg.payload)
        except ValueError as exc:
            logger.error(
                "on_message: failed to parse PhaseDecision: %s. "
                "Raw payload: %s",
                exc,
                msg.payload[:200],   # truncate for log safety
            )
            return

        self._decisions_received += 1

        # Push to queue (non-blocking — drop if full to avoid backpressure)
        try:
            self._decision_queue.put_nowait(decision)
            logger.debug(
                "PhaseDecision queued: phase=%s seq=%d",
                decision.phase.value, decision.sequence,
            )
        except queue.Full:
            logger.warning(
                "Decision queue full! Dropping PhaseDecision "
                "phase=%s seq=%d. Controller thread may be overloaded.",
                decision.phase.value, decision.sequence,
            )

        # Also call optional push-style callback if registered
        if self._on_decision_cb:
            try:
                self._on_decision_cb(decision)
            except Exception as exc:   # pylint: disable=broad-except
                logger.error(
                    "on_decision_callback raised: %s", exc, exc_info=True
                )

    def _on_publish(
        self,
        client: mqtt.Client,
        userdata: object,
        mid: int,
    ) -> None:
        """Fired by Paho when a QoS=1 PUBACK is received from the broker."""
        logger.debug("MQTT PUBACK received for mid=%d.", mid)

    # ------------------------------------------------------------------
    # Properties / Diagnostics
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """True if currently connected to the broker."""
        return self._connected.is_set()

    @property
    def frames_published(self) -> int:
        """Total frames successfully queued for publish since connect."""
        return self._frames_published

    @property
    def decisions_received(self) -> int:
        """Total PhaseDecision messages received since connect."""
        return self._decisions_received

    @property
    def publish_errors(self) -> int:
        """Total publish failures since connect."""
        return self._publish_errors

    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return (
            f"EdgeMQTTClient("
            f"status={status}, "
            f"broker={self._cfg.broker_host}:{self._cfg.broker_port}, "
            f"published={self._frames_published}, "
            f"received={self._decisions_received}, "
            f"errors={self._publish_errors})"
        )


# =============================================================================
# USAGE EXAMPLE
# Requires a running Mosquitto broker at the address in config.yaml.
# Publishes 3 dummy frames and prints any PhaseDecision that arrives.
#
# Run from project root:
#   python edge/network/mqtt_client.py
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from shared.config_loader import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg)

    print("=== EdgeMQTTClient Smoke Test ===\n")

    received_decisions = []

    def on_decision(decision: PhaseDecision) -> None:
        received_decisions.append(decision)
        print(f"  [CALLBACK] Decision received: {decision}")

    client = EdgeMQTTClient(cfg, on_decision_callback=on_decision)

    try:
        client.connect()
        print(f"Connected: {client}\n")

        # Publish 3 dummy frames
        for seq in range(1, 4):
            payload = FramePayload(
                intersection_id = cfg.intersection.id,
                frame_b64       = "SGVsbG8gV29ybGQ=",   # dummy base64
                sequence        = seq,
                width           = 640,
                height          = 480,
            )
            success = client.publish_frame(payload)
            print(f"  Frame seq={seq} published: {'✅' if success else '❌'}")
            time.sleep(1.0)

        # Poll for any decisions (server won't respond to dummy frames,
        # but this confirms the queue drain works)
        print("\nListening for PhaseDecisions for 5s...")
        deadline = time.time() + 5
        while time.time() < deadline:
            decision = client.get_pending_decision(timeout=0.5)
            if decision:
                print(f"  Got decision: {decision}")

        print(f"\nFinal state: {client}")
        print(f"Decisions received: {len(received_decisions)}")
        print("\n✅ EdgeMQTTClient smoke test complete.")

    except ConnectionError as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

    finally:
        client.disconnect()
        print("Client disconnected.")
