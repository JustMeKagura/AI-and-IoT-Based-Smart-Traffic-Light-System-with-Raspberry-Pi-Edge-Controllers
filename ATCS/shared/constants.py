# =============================================================================
# ATCS - Adaptive Traffic Control System
# shared/constants.py | Named constants for the entire system
#
# Rules:
#   - NO magic numbers anywhere else in the codebase. Import from here.
#   - Timing values come from config.yaml. This file covers IDs, enums,
#     labels, and safety limits that are structural (not operator-tunable).
#   - Imported by BOTH edge/ and server/. Keep dependencies minimal
#     (stdlib only — no paho, no ultralytics, no RPi libs here).
# =============================================================================

from enum import Enum, auto


# -----------------------------------------------------------------------------
# 1. MQTT TOPIC TEMPLATES
#    Usage: TOPIC_FRAMES.format(intersection_id="intersection_01")
# -----------------------------------------------------------------------------

TOPIC_FRAMES_RAW: str = "atcs/{intersection_id}/frames/raw"
TOPIC_CONTROL_SIGNAL: str = "atcs/{intersection_id}/control/signal"


# -----------------------------------------------------------------------------
# 2. TRAFFIC LIGHT PHASES
#    Single source of truth for all valid phase identifiers.
#    Used by: state_machine.py (edge), timing_algo.py (server), schemas.py
# -----------------------------------------------------------------------------

class Phase(str, Enum):
    """
    Valid traffic light phases.
    Inherits from str so values are JSON-serialisable directly:
        Phase.GREEN == "GREEN"  → True
    """
    ALL_RED = "ALL_RED"     # Boot state & safety buffer between transitions
    RED     = "RED"         # Standard red — vehicles stopped
    AMBER   = "AMBER"       # Transition phase — never skipped
    GREEN   = "GREEN"       # Vehicles moving


# -----------------------------------------------------------------------------
# 3. STATE MACHINE — VALID TRANSITIONS
#    Edge state_machine.py MUST reject any command not in this map.
#    Key   = current phase
#    Value = set of phases the system is allowed to move to next
#
#    Safety rule: GREEN can never jump directly to RED or GREEN again.
#                 AMBER is always the mandatory intermediate step.
# -----------------------------------------------------------------------------

VALID_TRANSITIONS: dict[Phase, set[Phase]] = {
    Phase.ALL_RED: {Phase.RED, Phase.GREEN},   # Boot → normal operation
    Phase.RED:     {Phase.GREEN},              # Red → Green (server command)
    Phase.GREEN:   {Phase.AMBER},              # Green → must pass through Amber
    Phase.AMBER:   {Phase.RED, Phase.ALL_RED}, # Amber → Red or safety reset
}


# -----------------------------------------------------------------------------
# 4. COCO CLASS FILTER
#    YOLOv8s is trained on COCO-80. We only count vehicle classes.
#    Reference: https://cocodataset.org/#explore
#       2  → car
#       3  → motorcycle
#       5  → bus
#       7  → truck
#    Kept here (not only in config.yaml) so server inference code can import
#    a typed constant rather than a raw list from config.
# -----------------------------------------------------------------------------

COCO_VEHICLE_CLASS_IDS: frozenset[int] = frozenset({2, 3, 5, 7})

COCO_CLASS_LABELS: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# -----------------------------------------------------------------------------
# 5. SAFETY LIMITS (hard-coded — NOT overridable via config.yaml)
#    These are physical/legal safety floors. Operator config can only
#    tighten them, never loosen below these values.
# -----------------------------------------------------------------------------

SAFETY_MIN_GREEN_SECONDS: int = 5      # Absolute minimum green — pedestrian safety
SAFETY_MAX_GREEN_SECONDS: int = 90     # Absolute maximum green — starvation prevention
SAFETY_MIN_AMBER_SECONDS: int = 3      # Below 3s amber is unsafe (reaction time)
SAFETY_MAX_AMBER_SECONDS: int = 6      # Above 6s amber causes driver confusion
SAFETY_ALL_RED_BUFFER_SECONDS: int = 1 # Minimum ALL_RED clearance between phases


# -----------------------------------------------------------------------------
# 6. EDGE BOOT & WATCHDOG
# -----------------------------------------------------------------------------

BOOT_PHASE: Phase = Phase.ALL_RED      # Edge always boots into ALL_RED
WATCHDOG_POLL_INTERVAL_SECONDS: float = 1.0  # How often watchdog checks last_seen


# -----------------------------------------------------------------------------
# 7. MQTT CLIENT IDs
#    Must be unique per broker connection. Using fixed IDs is fine for a
#    single-intersection deployment.
# -----------------------------------------------------------------------------

MQTT_CLIENT_ID_EDGE: str   = "atcs-edge-intersection_01"
MQTT_CLIENT_ID_SERVER: str = "atcs-server-intersection_01"


# -----------------------------------------------------------------------------
# 8. PAYLOAD ENCODING
# -----------------------------------------------------------------------------

PAYLOAD_ENCODING: str = "utf-8"        # JSON payloads are UTF-8 encoded strings
FRAME_ENCODING: str   = "base64"       # Frame bytes are Base64 before JSON wrap


# -----------------------------------------------------------------------------
# USAGE EXAMPLE (run this file directly to verify imports are clean)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ATCS Constants Smoke Test ===\n")

    print(f"[Topics]")
    print(f"  Frames : {TOPIC_FRAMES_RAW.format(intersection_id='intersection_01')}")
    print(f"  Control: {TOPIC_CONTROL_SIGNAL.format(intersection_id='intersection_01')}")

    print(f"\n[Phases]")
    for phase in Phase:
        targets = VALID_TRANSITIONS.get(phase, set())
        print(f"  {phase.value:8s} → {[t.value for t in targets]}")

    print(f"\n[COCO Vehicle Classes]")
    for cid, label in COCO_CLASS_LABELS.items():
        print(f"  ID {cid}: {label}")

    print(f"\n[Safety Limits]")
    print(f"  Green : {SAFETY_MIN_GREEN_SECONDS}s – {SAFETY_MAX_GREEN_SECONDS}s")
    print(f"  Amber : {SAFETY_MIN_AMBER_SECONDS}s – {SAFETY_MAX_AMBER_SECONDS}s")
    print(f"  ALL_RED buffer: {SAFETY_ALL_RED_BUFFER_SECONDS}s")

    print(f"\n[Boot Phase] {BOOT_PHASE.value}")
    print("\n✅ All constants loaded successfully.")
