# =============================================================================
# ATCS - Adaptive Traffic Control System
# shared/config_loader.py | Parses config.yaml + validates all values
#
# Rules:
#   - This is the ONLY place config.yaml is read. Never call open("config.yaml")
#     anywhere else in the codebase.
#   - Validates types AND safety bounds on load. Fails loudly at startup
#     rather than silently misbehaving at runtime.
#   - Supports env var overrides for broker_host and broker_port only
#     (useful for Docker / CI). Format: ATCS_BROKER_HOST, ATCS_BROKER_PORT.
#   - Returns a single immutable ATCSConfig object. Import and call
#     load_config() once in main.py, then pass the object down.
#   - stdlib only: pathlib, os, dataclasses, PyYAML (already in requirements).
# =============================================================================

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import yaml

from shared.constants import (
    SAFETY_MIN_GREEN_SECONDS,
    SAFETY_MAX_GREEN_SECONDS,
    SAFETY_MIN_AMBER_SECONDS,
    SAFETY_MAX_AMBER_SECONDS,
    SAFETY_ALL_RED_BUFFER_SECONDS,
)

logger = logging.getLogger(__name__)

# Default path: config.yaml sits at the project root (one level above shared/)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


# =============================================================================
# Typed sub-config dataclasses (mirrors config.yaml structure exactly)
# =============================================================================

@dataclass(frozen=True)
class MQTTConfig:
    broker_host: str
    broker_port: int
    keepalive: int
    qos: int
    reconnect_delay_min: int
    reconnect_delay_max: int


@dataclass(frozen=True)
class IntersectionConfig:
    id: str


@dataclass(frozen=True)
class GPIOPins:
    red: int
    amber: int
    green: int


@dataclass(frozen=True)
class EdgeConfig:
    camera_index: int
    frame_width: int
    frame_height: int
    jpeg_quality: int
    capture_fps: float
    server_timeout_seconds: int
    fallback_green_duration: int
    fallback_amber_duration: int
    fallback_red_duration: int
    gpio_pins: GPIOPins


@dataclass(frozen=True)
class ServerConfig:
    model_path: str
    inference_conf: float
    inference_imgsz: int
    coco_vehicle_classes: List[int]
    clahe_clip_limit: float
    clahe_tile_grid_size: Tuple[int, int]
    smoother_window_size: int
    min_green_duration: int
    max_green_duration: int
    amber_duration: int
    red_clearance_duration: int
    vehicles_per_second: float
    pipeline_timeout_ms: int
    db_path: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    log_to_file: bool
    log_file_path: str


@dataclass(frozen=True)
class ATCSConfig:
    """
    Single immutable config object passed through the entire application.
    Constructed once by load_config() and never mutated.
    """
    mqtt: MQTTConfig
    intersection: IntersectionConfig
    edge: EdgeConfig
    server: ServerConfig
    logging: LoggingConfig


# =============================================================================
# Internal Validators
# =============================================================================

def _require_keys(section: dict, keys: list[str], section_name: str) -> None:
    """Raise ValueError if any required key is missing from a config section."""
    missing = [k for k in keys if k not in section]
    if missing:
        raise ValueError(
            f"config.yaml [{section_name}] is missing required keys: {missing}"
        )


def _validate_mqtt(cfg: dict) -> MQTTConfig:
    _require_keys(cfg, [
        "broker_host", "broker_port", "keepalive",
        "qos", "reconnect_delay_min", "reconnect_delay_max"
    ], "mqtt")

    # Allow env var override for broker_host and broker_port (Docker / CI)
    host = os.environ.get("ATCS_BROKER_HOST", cfg["broker_host"])
    port = int(os.environ.get("ATCS_BROKER_PORT", cfg["broker_port"]))

    if not isinstance(host, str) or not host:
        raise ValueError("mqtt.broker_host must be a non-empty string.")
    if not (1 <= port <= 65535):
        raise ValueError(f"mqtt.broker_port must be 1–65535, got {port}.")
    if cfg["qos"] not in (0, 1, 2):
        raise ValueError(f"mqtt.qos must be 0, 1, or 2, got {cfg['qos']}.")

    return MQTTConfig(
        broker_host        = host,
        broker_port        = port,
        keepalive          = int(cfg["keepalive"]),
        qos                = int(cfg["qos"]),
        reconnect_delay_min= int(cfg["reconnect_delay_min"]),
        reconnect_delay_max= int(cfg["reconnect_delay_max"]),
    )


def _validate_intersection(cfg: dict) -> IntersectionConfig:
    _require_keys(cfg, ["id"], "intersection")
    if not isinstance(cfg["id"], str) or not cfg["id"].strip():
        raise ValueError("intersection.id must be a non-empty string.")
    return IntersectionConfig(id=cfg["id"].strip())


def _validate_edge(cfg: dict) -> EdgeConfig:
    _require_keys(cfg, [
        "camera_index", "frame_width", "frame_height", "jpeg_quality",
        "capture_fps", "server_timeout_seconds",
        "fallback_green_duration", "fallback_amber_duration",
        "fallback_red_duration", "gpio_pins"
    ], "edge")

    # Validate GPIO pins sub-section
    pins_cfg = cfg["gpio_pins"]
    _require_keys(pins_cfg, ["red", "amber", "green"], "edge.gpio_pins")
    pins = [pins_cfg["red"], pins_cfg["amber"], pins_cfg["green"]]
    for pin in pins:
        if not (1 <= int(pin) <= 27):
            raise ValueError(
                f"edge.gpio_pins: BCM pin {pin} is out of range (1–27)."
            )
    if len(set(pins)) != 3:
        raise ValueError("edge.gpio_pins: red, amber, green pins must be unique.")

    # Validate JPEG quality
    quality = int(cfg["jpeg_quality"])
    if not (0 <= quality <= 100):
        raise ValueError(f"edge.jpeg_quality must be 0–100, got {quality}.")

    # Validate fallback timings against safety floors
    fb_green = int(cfg["fallback_green_duration"])
    fb_amber = int(cfg["fallback_amber_duration"])

    if fb_green < SAFETY_MIN_GREEN_SECONDS:
        raise ValueError(
            f"edge.fallback_green_duration ({fb_green}s) is below the "
            f"safety minimum ({SAFETY_MIN_GREEN_SECONDS}s)."
        )
    if fb_amber < SAFETY_MIN_AMBER_SECONDS:
        raise ValueError(
            f"edge.fallback_amber_duration ({fb_amber}s) is below the "
            f"safety minimum ({SAFETY_MIN_AMBER_SECONDS}s)."
        )

    return EdgeConfig(
        camera_index           = int(cfg["camera_index"]),
        frame_width            = int(cfg["frame_width"]),
        frame_height           = int(cfg["frame_height"]),
        jpeg_quality           = quality,
        capture_fps            = float(cfg["capture_fps"]),
        server_timeout_seconds = int(cfg["server_timeout_seconds"]),
        fallback_green_duration= fb_green,
        fallback_amber_duration= fb_amber,
        fallback_red_duration  = int(cfg["fallback_red_duration"]),
        gpio_pins = GPIOPins(
            red   = int(pins_cfg["red"]),
            amber = int(pins_cfg["amber"]),
            green = int(pins_cfg["green"]),
        ),
    )


def _validate_server(cfg: dict) -> ServerConfig:
    _require_keys(cfg, [
        "model_path", "inference_conf", "inference_imgsz",
        "coco_vehicle_classes", "clahe_clip_limit", "clahe_tile_grid_size",
        "smoother_window_size", "min_green_duration", "max_green_duration",
        "amber_duration", "red_clearance_duration", "vehicles_per_second",
        "pipeline_timeout_ms", "db_path"
    ], "server")

    # Confidence threshold
    conf = float(cfg["inference_conf"])
    if not (0.0 < conf < 1.0):
        raise ValueError(f"server.inference_conf must be between 0 and 1, got {conf}.")

    # Green duration bounds — operator values clamped by safety constants
    min_green = int(cfg["min_green_duration"])
    max_green = int(cfg["max_green_duration"])
    amber     = int(cfg["amber_duration"])

    if min_green < SAFETY_MIN_GREEN_SECONDS:
        logger.warning(
            "server.min_green_duration (%ds) is below safety floor (%ds). "
            "Clamping to %ds.",
            min_green, SAFETY_MIN_GREEN_SECONDS, SAFETY_MIN_GREEN_SECONDS
        )
        min_green = SAFETY_MIN_GREEN_SECONDS

    if max_green > SAFETY_MAX_GREEN_SECONDS:
        logger.warning(
            "server.max_green_duration (%ds) exceeds safety ceiling (%ds). "
            "Clamping to %ds.",
            max_green, SAFETY_MAX_GREEN_SECONDS, SAFETY_MAX_GREEN_SECONDS
        )
        max_green = SAFETY_MAX_GREEN_SECONDS

    if min_green >= max_green:
        raise ValueError(
            f"server.min_green_duration ({min_green}s) must be less than "
            f"max_green_duration ({max_green}s)."
        )

    if not (SAFETY_MIN_AMBER_SECONDS <= amber <= SAFETY_MAX_AMBER_SECONDS):
        raise ValueError(
            f"server.amber_duration ({amber}s) must be between "
            f"{SAFETY_MIN_AMBER_SECONDS}s and {SAFETY_MAX_AMBER_SECONDS}s."
        )

    # CLAHE tile grid
    tile = cfg["clahe_tile_grid_size"]
    if not (isinstance(tile, list) and len(tile) == 2):
        raise ValueError("server.clahe_tile_grid_size must be a list of 2 integers.")

    # COCO classes
    coco_classes = list(cfg["coco_vehicle_classes"])
    if not coco_classes:
        raise ValueError("server.coco_vehicle_classes must not be empty.")

    return ServerConfig(
        model_path            = str(cfg["model_path"]),
        inference_conf        = conf,
        inference_imgsz       = int(cfg["inference_imgsz"]),
        coco_vehicle_classes  = coco_classes,
        clahe_clip_limit      = float(cfg["clahe_clip_limit"]),
        clahe_tile_grid_size  = (int(tile[0]), int(tile[1])),
        smoother_window_size  = int(cfg["smoother_window_size"]),
        min_green_duration    = min_green,
        max_green_duration    = max_green,
        amber_duration        = amber,
        red_clearance_duration= int(cfg["red_clearance_duration"]),
        vehicles_per_second   = float(cfg["vehicles_per_second"]),
        pipeline_timeout_ms   = int(cfg["pipeline_timeout_ms"]),
        db_path               = str(cfg["db_path"]),
    )


def _validate_logging(cfg: dict) -> LoggingConfig:
    _require_keys(cfg, ["level", "log_to_file", "log_file_path"], "logging")

    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    level = str(cfg["level"]).upper()
    if level not in valid_levels:
        raise ValueError(
            f"logging.level must be one of {valid_levels}, got {level!r}."
        )

    return LoggingConfig(
        level         = level,
        log_to_file   = bool(cfg["log_to_file"]),
        log_file_path = str(cfg["log_file_path"]),
    )


# =============================================================================
# Public API
# =============================================================================

def load_config(path: Path | str | None = None) -> ATCSConfig:
    """
    Load, parse, and validate config.yaml.

    Parameters
    ----------
    path : Path | str | None
        Path to config.yaml. Defaults to <project_root>/config.yaml.
        Pass an explicit path in tests to use a fixture config file.

    Returns
    -------
    ATCSConfig
        Fully validated, immutable config object.

    Raises
    ------
    FileNotFoundError
        If config.yaml does not exist at the resolved path.
    ValueError
        If any section is missing, has wrong types, or violates safety bounds.
    yaml.YAMLError
        If the file is not valid YAML.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at: {config_path}\n"
            f"Ensure config.yaml is at the project root or pass an explicit path."
        )

    logger.info("Loading config from: %s", config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("config.yaml must be a YAML mapping (key: value pairs).")

    # Validate presence of top-level sections
    top_level = ["mqtt", "intersection", "edge", "server", "logging"]
    _require_keys(raw, top_level, "root")

    config = ATCSConfig(
        mqtt         = _validate_mqtt(raw["mqtt"]),
        intersection = _validate_intersection(raw["intersection"]),
        edge         = _validate_edge(raw["edge"]),
        server       = _validate_server(raw["server"]),
        logging      = _validate_logging(raw["logging"]),
    )

    logger.info(
        "Config loaded OK — broker=%s:%d, intersection=%s",
        config.mqtt.broker_host,
        config.mqtt.broker_port,
        config.intersection.id,
    )
    return config


def setup_logging(config: ATCSConfig) -> None:
    """
    Configure the root logger using settings from ATCSConfig.
    Call this once at the top of main.py, before any other imports log.

    Parameters
    ----------
    config : ATCSConfig
        The loaded config object (from load_config()).
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if config.logging.log_to_file:
        handlers.append(logging.FileHandler(config.logging.log_file_path))

    logging.basicConfig(
        level   = config.logging.level,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers= handlers,
    )
    logger.debug("Logging initialised at level: %s", config.logging.level)


# =============================================================================
# USAGE EXAMPLE (run directly to verify config loads cleanly)
# =============================================================================

if __name__ == "__main__":
    import sys

    # Accept optional path argument: python config_loader.py ../../config.yaml
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        cfg = load_config(path_arg)
        setup_logging(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Config load failed: {e}")
        sys.exit(1)

    print("=== ATCS Config Loader Smoke Test ===\n")
    print(f"[MQTT]")
    print(f"  Broker     : {cfg.mqtt.broker_host}:{cfg.mqtt.broker_port}")
    print(f"  QoS        : {cfg.mqtt.qos}")
    print(f"  Keepalive  : {cfg.mqtt.keepalive}s")

    print(f"\n[Intersection]")
    print(f"  ID         : {cfg.intersection.id}")

    print(f"\n[Edge]")
    print(f"  Camera     : index {cfg.edge.camera_index} @ {cfg.edge.capture_fps} FPS")
    print(f"  Resolution : {cfg.edge.frame_width}x{cfg.edge.frame_height}")
    print(f"  Timeout    : {cfg.edge.server_timeout_seconds}s → fallback mode")
    print(f"  GPIO Pins  : R={cfg.edge.gpio_pins.red} "
          f"A={cfg.edge.gpio_pins.amber} "
          f"G={cfg.edge.gpio_pins.green}")

    print(f"\n[Server]")
    print(f"  Model      : {cfg.server.model_path}")
    print(f"  Confidence : {cfg.server.inference_conf}")
    print(f"  Green      : {cfg.server.min_green_duration}s – "
          f"{cfg.server.max_green_duration}s")
    print(f"  Amber      : {cfg.server.amber_duration}s (fixed)")
    print(f"  Smoother   : {cfg.server.smoother_window_size}-frame window")
    print(f"  DB         : {cfg.server.db_path}")

    print(f"\n[Logging]")
    print(f"  Level      : {cfg.logging.level}")
    print(f"  To file    : {cfg.logging.log_to_file}")

    print("\n✅ Config loaded and validated successfully.")