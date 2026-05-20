# =============================================================================
# ATCS - Adaptive Traffic Control System
# server/network/mqtt_client.py | Paho MQTT wrapper for the Server node
#
# Responsibilities:
#   - Connect to the Mosquitto broker with automatic reconnect + backoff
#   - Subscribe to the frames/raw topic and dispatch FramePayload objects
#     to the inference pipeline (via callback)
#   - Publish PhaseDecision messages back to the edge (control/signal topic)
#   - Never run inference in this module — dispatch only
#   - Expose clean connect() / disconnect() / publish_decision() API
#
# Mirrors:
#   edge/network/mqtt_client.py by Abderrahmane
#   Same threading model (loop_start), same QoS, same topic constants,
#   same config injection, same exponential backoff pattern.
#
# Threading model:
#   - Paho runs its own network thread (loop_start).
#   - on_message fires in the Paho thread → calls _on_frame_cb instantly.
#   - Inference pipeline runs in whatever thread the caller provides.
#   - publish_decision() is safe to call from any thread (Paho-internal queue).
# =============================================================================

import json
import logging
import threading
import time
from typing import Callable, Optional

import paho.mqtt.client as mqtt

from shared.constants import (
    TOPIC_FRAMES_RAW,
    TOPIC_CONTROL_SIGNAL,
    MQTT_CLIENT_ID_SERVER,
    PAYLOAD_ENCODING,
)
from shared.schemas import FramePayload, PhaseDecision
from shared.config_loader import ATCSConfig

logger = logging.getLogger(__name__)


class ServerMQTTClient:
    """
    Paho MQTT wrapper for the Server (laptop/PC) node.

    Threading model
    ---------------
    Paho runs its own network thread internally (loop_start).
    on_message fires in the Paho thread and immediately calls _on_frame_cb —
    the callback must return quickly (no inference here).
    publish_decision() is thread-safe via Paho's internal message queue.

    Parameters
    ----------
    config : ATCSConfig
        Loaded system config. MQTT and intersection settings used.
    on_frame_callback : Callable[[FramePayload], None]
        Called from the Paho thread every time a FramePayload arrives.
        Must return quickly — offload any heavy work to the inference thread.
    """

    def __init__(
        self,
        config: ATCSConfig,
        on_frame_callback: Callable[[FramePayload], None],
    ) -> None:
        self._cfg              = config.mqtt
        self._intersection_id  = config.intersection.id
        self._qos              = config.mqtt.qos
        self._on_frame_cb      = on_frame_callback

        # Build topic strings once — mirrors edge side exactly
        self._topic_subscribe = TOPIC_FRAMES_RAW.format(
            intersection_id=self._intersection_id
        )
        self._topic_publish   = TOPIC_CONTROL_SIGNAL.format(
            intersection_id=self._intersection_id
        )

        # Connection state
        self._connected        = threading.Event()
        self._should_reconnect = True

        # Diagnostics counters
        self._frames_received    = 0
        self._decisions_published = 0
        self._publish_errors      = 0
        self._parse_errors        = 0

        # Exponential backoff state (mirrors edge side)
        self._reconnect_delay = self._cfg.reconnect_delay_min

        # Initialise Paho client
        self._client = mqtt.Client(
            client_id     = MQTT_CLIENT_ID_SERVER,
            clean_session = True,
        )
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message
        self._client.on_publish    = self._on_publish

        if self._cfg.username:
            self._client.username_pw_set(
                self._cfg.username,
                self._cfg.password,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Connect to the MQTT broker and start the Paho network thread.
        Blocks until the connection is acknowledged or raises on failure.

        Raises
        ------
        ConnectionError
            If the broker is unreachable or on_connect does not fire in time.
        """
        logger.info(
            "Connecting to MQTT broker at %s:%d (client_id=%s)...",
            self._cfg.broker_host, self._cfg.broker_port, MQTT_CLIENT_ID_SERVER,
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

        # Start Paho's background network thread (mirrors edge side)
        self._client.loop_start()

        # Block until on_connect fires — same pattern as edge
        if not self._connected.wait(timeout=self._cfg.keepalive):
            self._client.loop_stop()
            raise ConnectionError(
                f"MQTT broker connected at socket level but on_connect "
                f"did not fire within {self._cfg.keepalive}s. "
                f"Check broker configuration."
            )

        logger.info("MQTT connected. Subscribed to: %s", self._topic_subscribe)
        logger.info("Publishing decisions to:        %s", self._topic_publish)

    def disconnect(self) -> None:
        """
        Gracefully stop the network thread and disconnect from the broker.
        Call in server/main.py finally block during shutdown.
        """
        logger.info("Disconnecting ServerMQTTClient...")
        self._should_reconnect = False
        self._client.loop_stop()
        self._client.disconnect()
        self._connected.clear()
        logger.info("ServerMQTTClient disconnected.")

    # ──────────────────────────────────────────────────────────────────────────
    # Publishing (called by inference pipeline thread)
    # ──────────────────────────────────────────────────────────────────────────

    def publish_decision(self, decision: PhaseDecision) -> bool:
        """
        Publish a PhaseDecision to the control/signal topic.

        Non-blocking: Paho queues the message internally and the network
        thread sends it. Safe to call from any thread.

        Parameters
        ----------
        decision : PhaseDecision
            The phase decision to publish. Must be fully populated.

        Returns
        -------
        bool
            True if the message was queued successfully, False on error.
        """
        if not self._connected.is_set():
            logger.warning(
                "publish_decision(): not connected. "
                "Decision for phase=%s dropped.",
                decision.phase.value if hasattr(decision, "phase") else "?",
            )
            self._publish_errors += 1
            return False

        if not isinstance(decision, PhaseDecision):
            logger.error(
                "publish_decision(): expected PhaseDecision, got %s.",
                type(decision).__name__,
            )
            self._publish_errors += 1
            return False

        try:
            json_str = decision.to_json()
            result   = self._client.publish(
                topic   = self._topic_publish,
                payload = json_str.encode(PAYLOAD_ENCODING),
                qos     = self._qos,
            )

            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.error(
                    "publish_decision() failed: rc=%d phase=%s.",
                    result.rc,
                    decision.phase.value if hasattr(decision, "phase") else "?",
                )
                self._publish_errors += 1
                return False

            self._decisions_published += 1
            logger.debug(
                "Decision published: phase=%s green=%.1fs topic=%s",
                decision.phase.value if hasattr(decision, "phase") else "?",
                getattr(decision, "green_duration", 0.0),
                self._topic_publish,
            )
            return True

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("publish_decision() unexpected error: %s", exc)
            self._publish_errors += 1
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Paho callbacks (run in Paho network thread — must not block)
    # ──────────────────────────────────────────────────────────────────────────

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: object,
        flags:    dict,
        rc:       int,
    ) -> None:
        """Fired by Paho when the broker connection is established."""
        if rc == mqtt.CONNACK_ACCEPTED:
            logger.info("MQTT on_connect: broker accepted connection (rc=0).")
            self._connected.set()
            self._reconnect_delay = self._cfg.reconnect_delay_min

            # Subscribe on every connect — handles reconnects transparently
            client.subscribe(self._topic_subscribe, qos=self._qos)
            logger.info(
                "Subscribed: %s (QoS=%d)", self._topic_subscribe, self._qos
            )
        else:
            logger.error(
                "MQTT on_connect: broker rejected connection. rc=%d (%s).",
                rc, mqtt.connack_string(rc),
            )

    def _on_disconnect(
        self,
        client:   mqtt.Client,
        userdata: object,
        rc:       int,
    ) -> None:
        """Fired by Paho on disconnect (clean or unexpected)."""
        self._connected.clear()

        if rc == 0:
            logger.info("MQTT on_disconnect: clean disconnect.")
        else:
            logger.warning(
                "MQTT on_disconnect: unexpected disconnect (rc=%d). "
                "Paho will attempt to reconnect automatically.",
                rc,
            )
            logger.info(
                "Reconnect backoff: %.1fs (max: %ds).",
                self._reconnect_delay, self._cfg.reconnect_delay_max,
            )
            # Mirror edge: exponential backoff tracking
            self._reconnect_delay = min(
                self._reconnect_delay * 2,
                self._cfg.reconnect_delay_max,
            )

    def _on_message(
        self,
        client:   mqtt.Client,
        userdata: object,
        msg:      mqtt.MQTTMessage,
    ) -> None:
        """
        Fired by Paho when a FramePayload arrives on the subscribed topic.

        MUST return quickly — no inference here.
        Parses the FramePayload and hands it off to the inference callback.
        """
        logger.debug(
            "MQTT message received: topic=%s size=%d bytes.",
            msg.topic, len(msg.payload),
        )

        # Parse FramePayload — mirrors edge's PhaseDecision.from_json() pattern
        try:
            frame = FramePayload.from_json(msg.payload)
        except (ValueError, KeyError, TypeError) as exc:
            self._parse_errors += 1
            logger.error(
                "on_message: failed to parse FramePayload: %s. "
                "Raw payload (truncated): %s",
                exc,
                msg.payload[:200],
            )
            return

        self._frames_received += 1

        # Dispatch to inference pipeline — callback must return quickly
        try:
            self._on_frame_cb(frame)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "on_frame_callback raised unexpectedly: %s", exc, exc_info=True
            )

    def _on_publish(
        self,
        client:   mqtt.Client,
        userdata: object,
        mid:      int,
    ) -> None:
        """Fired by Paho when a QoS=1 PUBACK is received from the broker."""
        logger.debug("MQTT PUBACK received for mid=%d.", mid)

    # ──────────────────────────────────────────────────────────────────────────
    # Properties / Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """True if currently connected to the broker."""
        return self._connected.is_set()

    @property
    def frames_received(self) -> int:
        """Total FramePayload messages successfully parsed since connect."""
        return self._frames_received

    @property
    def decisions_published(self) -> int:
        """Total PhaseDecision messages successfully queued since connect."""
        return self._decisions_published

    @property
    def publish_errors(self) -> int:
        """Total publish failures since connect."""
        return self._publish_errors

    @property
    def parse_errors(self) -> int:
        """Total FramePayload parse failures since connect."""
        return self._parse_errors

    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return (
            f"ServerMQTTClient("
            f"status={status}, "
            f"broker={self._cfg.broker_host}:{self._cfg.broker_port}, "
            f"received={self._frames_received}, "
            f"published={self._decisions_published}, "
            f"errors={self._publish_errors})"
        )