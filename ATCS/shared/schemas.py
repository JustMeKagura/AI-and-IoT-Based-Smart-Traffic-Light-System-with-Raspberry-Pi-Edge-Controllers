# =============================================================================
# ATCS - Adaptive Traffic Control System
# shared/schemas.py | Data contract between Edge and Server
#
# Rules:
#   - These dataclasses are the ONLY permitted wire format over MQTT.
#   - Both sides MUST use to_json() to serialise and from_json() to parse.
#   - Never construct raw dicts for MQTT payloads — always go through here.
#   - stdlib only (dataclasses, json, datetime). No third-party deps.
# =============================================================================

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from constants import Phase


# -----------------------------------------------------------------------------
# 1. FramePayload
#    Direction : Edge → Server  (topic: atcs/{id}/frames/raw)
#    Published by : edge/network/mqtt_client.py (Reporter thread)
#    Consumed by  : server/network/mqtt_client.py → inference pipeline
# -----------------------------------------------------------------------------

@dataclass
class FramePayload:
    """
    Wraps a single captured JPEG frame for transit over MQTT.

    Fields
    ------
    intersection_id : str
        Matches the intersection.id in config.yaml.
        Server uses this to route to the correct handler if scaled to N intersections.
    frame_b64 : str
        Base64-encoded JPEG bytes (UTF-8 string). Decode → bytes → cv2.imdecode.
    timestamp : float
        Unix epoch (seconds) at capture time. Used for latency profiling.
    sequence : int
        Monotonically increasing frame counter per session.
        Server can detect dropped frames by checking for gaps.
    width : int
        Frame width in pixels after edge resize. Informational — server re-checks.
    height : int
        Frame height in pixels after edge resize.
    """

    intersection_id: str
    frame_b64: str
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0
    width: int = 640
    height: int = 480

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise to a UTF-8 JSON string ready for MQTT publish."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str | bytes) -> "FramePayload":
        """
        Deserialise from an MQTT message payload.

        Parameters
        ----------
        payload : str | bytes
            Raw bytes from paho on_message callback, or a JSON string.

        Raises
        ------
        ValueError
            If required fields are missing or types are wrong.
        """
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"FramePayload: invalid JSON → {exc}") from exc

        # Validate required fields exist
        required = {"intersection_id", "frame_b64"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"FramePayload: missing required fields: {missing}")

        # Type guards
        if not isinstance(data["intersection_id"], str):
            raise ValueError("FramePayload: 'intersection_id' must be a string.")
        if not isinstance(data["frame_b64"], str):
            raise ValueError("FramePayload: 'frame_b64' must be a base64 string.")

        return cls(
            intersection_id = data["intersection_id"],
            frame_b64       = data["frame_b64"],
            timestamp       = float(data.get("timestamp", time.time())),
            sequence        = int(data.get("sequence", 0)),
            width           = int(data.get("width", 640)),
            height          = int(data.get("height", 480)),
        )

    def latency_ms(self) -> float:
        """Returns milliseconds elapsed since this frame was captured."""
        return (time.time() - self.timestamp) * 1000.0

    def __repr__(self) -> str:
        return (
            f"FramePayload(id={self.intersection_id!r}, "
            f"seq={self.sequence}, "
            f"ts={self.timestamp:.3f}, "
            f"size={len(self.frame_b64)} chars)"
        )


# -----------------------------------------------------------------------------
# 2. PhaseDecision
#    Direction : Server → Edge  (topic: atcs/{id}/control/signal)
#    Published by : server/network/mqtt_client.py
#    Consumed by  : edge/network/mqtt_client.py → state_machine → gpio_driver
# -----------------------------------------------------------------------------

@dataclass
class PhaseDecision:
    """
    Instructs the Edge which phase to activate and for how long.

    Fields
    ------
    intersection_id : str
        Must match the Edge's configured intersection_id. Edge rejects mismatches.
    phase : Phase
        Target phase (GREEN / AMBER / RED / ALL_RED).
        Edge state_machine validates this against VALID_TRANSITIONS.
    duration_seconds : float
        How long the Edge should hold this phase before expecting the next command.
        Edge uses this as a local countdown. If the next command arrives before
        expiry, it supersedes. If it doesn't arrive, fallback watchdog fires.
    vehicle_count : int
        Smoothed vehicle count the server used to calculate this decision.
        Stored in SQLite on server side. Forwarded here for edge-side logging.
    timestamp : float
        Unix epoch (seconds) when this decision was computed on the server.
    sequence : int
        Matches the FramePayload.sequence that triggered this decision.
        Enables end-to-end latency tracing.
    """

    intersection_id: str
    phase: Phase
    duration_seconds: float
    vehicle_count: int = 0
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise to a UTF-8 JSON string ready for MQTT publish."""
        data = asdict(self)
        data["phase"] = self.phase.value   # Enum → str for JSON
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str | bytes) -> "PhaseDecision":
        """
        Deserialise from an MQTT message payload.

        Raises
        ------
        ValueError
            If required fields are missing, types are wrong, or phase is invalid.
        """
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"PhaseDecision: invalid JSON → {exc}") from exc

        # Validate required fields
        required = {"intersection_id", "phase", "duration_seconds"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"PhaseDecision: missing required fields: {missing}")

        # Validate and convert phase string → Phase enum
        try:
            phase = Phase(data["phase"])
        except ValueError:
            valid = [p.value for p in Phase]
            raise ValueError(
                f"PhaseDecision: unknown phase {data['phase']!r}. "
                f"Must be one of {valid}."
            )

        # Validate duration is non-negative
        duration = float(data["duration_seconds"])
        if duration < 0:
            raise ValueError(
                f"PhaseDecision: duration_seconds must be >= 0, got {duration}."
            )

        return cls(
            intersection_id  = str(data["intersection_id"]),
            phase            = phase,
            duration_seconds = duration,
            vehicle_count    = int(data.get("vehicle_count", 0)),
            timestamp        = float(data.get("timestamp", time.time())),
            sequence         = int(data.get("sequence", 0)),
        )

    def latency_ms(self) -> float:
        """Returns milliseconds elapsed since this decision was computed."""
        return (time.time() - self.timestamp) * 1000.0

    def __repr__(self) -> str:
        return (
            f"PhaseDecision(id={self.intersection_id!r}, "
            f"phase={self.phase.value}, "
            f"duration={self.duration_seconds}s, "
            f"vehicles={self.vehicle_count}, "
            f"seq={self.sequence})"
        )


# -----------------------------------------------------------------------------
# USAGE EXAMPLE (run directly to verify round-trip serialisation)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ATCS Schemas Round-Trip Smoke Test ===\n")

    # --- FramePayload ---
    print("[1] FramePayload")
    original_frame = FramePayload(
        intersection_id = "intersection_01",
        frame_b64       = "SGVsbG8gV29ybGQ=",   # "Hello World" in base64
        sequence        = 42,
        width           = 640,
        height          = 480,
    )
    json_str = original_frame.to_json()
    print(f"  Serialised : {json_str}")

    recovered_frame = FramePayload.from_json(json_str)
    print(f"  Recovered  : {recovered_frame}")
    print(f"  Latency    : {recovered_frame.latency_ms():.2f} ms")

    assert recovered_frame.intersection_id == original_frame.intersection_id
    assert recovered_frame.frame_b64       == original_frame.frame_b64
    assert recovered_frame.sequence        == original_frame.sequence
    print("  ✅ FramePayload round-trip OK\n")

    # --- PhaseDecision ---
    print("[2] PhaseDecision")
    original_decision = PhaseDecision(
        intersection_id  = "intersection_01",
        phase            = Phase.GREEN,
        duration_seconds = 35.0,
        vehicle_count    = 7,
        sequence         = 42,
    )
    json_str = original_decision.to_json()
    print(f"  Serialised : {json_str}")

    recovered_decision = PhaseDecision.from_json(json_str)
    print(f"  Recovered  : {recovered_decision}")
    print(f"  Latency    : {recovered_decision.latency_ms():.2f} ms")

    assert recovered_decision.phase            == Phase.GREEN
    assert recovered_decision.duration_seconds == 35.0
    assert recovered_decision.vehicle_count    == 7
    print("  ✅ PhaseDecision round-trip OK\n")

    # --- Error handling ---
    print("[3] Invalid phase rejection")
    try:
        PhaseDecision.from_json(
            '{"intersection_id":"x","phase":"PURPLE","duration_seconds":10}'
        )
    except ValueError as e:
        print(f"  ✅ Caught expected error: {e}\n")

    print("[4] Missing field rejection")
    try:
        FramePayload.from_json('{"intersection_id":"x"}')
    except ValueError as e:
        print(f"  ✅ Caught expected error: {e}\n")

    print("=== All schema tests passed ===")
