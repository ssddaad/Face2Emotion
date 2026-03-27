import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class ServiceConfig:
    host: str
    port: int
    log_level: str
    source_type: str
    source_value: str
    confidence: float
    iou: float
    image_size: int
    model_path: Path
    yolo_device: str
    min_face_size: int
    max_track_distance: float
    micro_ema_alpha: float
    max_fps: float
    emotion_interval: int
    stale_timeout_sec: float
    mirror_input: bool
    reconnect_cooldown_sec: float
    api_key: str


def _pick(data: dict, key: str, default):
    value = data.get(key, default)
    return default if value is None else value


def _env(name: str):
    value = os.getenv(name)
    return None if value is None or value == "" else value


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _validate(config: ServiceConfig) -> ServiceConfig:
    if config.source_type not in {"camera", "rtsp", "file"}:
        raise ValueError("source_type 必须是 camera/rtsp/file")
    if config.source_type == "camera":
        int(config.source_value)
    if not config.yolo_device:
        raise ValueError("yolo_device 不能为空")
    if not (0.01 <= config.confidence <= 1.0):
        raise ValueError("confidence 必须在 [0.01, 1.0]")
    if not (0.01 <= config.iou <= 1.0):
        raise ValueError("iou 必须在 [0.01, 1.0]")
    if config.image_size < 160:
        raise ValueError("image_size 建议 >= 160")
    if config.min_face_size < 8:
        raise ValueError("min_face_size 必须 >= 8")
    if not (0.01 <= config.micro_ema_alpha <= 1.0):
        raise ValueError("micro_ema_alpha 必须在 [0.01, 1.0]")
    if not (1.0 <= config.max_fps <= 120.0):
        raise ValueError("max_fps 必须在 [1, 120]")
    if config.emotion_interval < 1:
        raise ValueError("emotion_interval 必须 >= 1")
    if config.stale_timeout_sec <= 0:
        raise ValueError("stale_timeout_sec 必须 > 0")
    if config.reconnect_cooldown_sec < 0.1:
        raise ValueError("reconnect_cooldown_sec 必须 >= 0.1")
    return config


def load_service_config(config_path: str | None = None) -> ServiceConfig:
    default_path = os.getenv("F2E_CONFIG_PATH", "settings.yaml")
    path = Path(config_path) if config_path else Path(default_path)
    raw: dict = {}
    if path.exists():
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(content, dict):
            raw = content

    raw_mirror = _env("F2E_MIRROR_INPUT")
    if raw_mirror is None:
        raw_mirror = _pick(raw, "mirror_input", True)

    config = ServiceConfig(
        host=str(_env("F2E_HOST") or _pick(raw, "host", "0.0.0.0")),
        port=int(_env("F2E_PORT") or _pick(raw, "port", 8000)),
        log_level=str(_env("F2E_LOG_LEVEL") or _pick(raw, "log_level", "info")),
        source_type=str(_env("F2E_SOURCE_TYPE") or _pick(raw, "source_type", "camera")),
        source_value=str(_env("F2E_SOURCE_VALUE") or _pick(raw, "source_value", "0")),
        confidence=float(_env("F2E_CONF") or _pick(raw, "confidence", 0.4)),
        iou=float(_env("F2E_IOU") or _pick(raw, "iou", 0.5)),
        image_size=int(_env("F2E_IMGSZ") or _pick(raw, "image_size", 640)),
        model_path=Path(_env("F2E_MODEL") or _pick(raw, "model_path", "models/yolov8n-face.pt")),
        yolo_device=str(_env("F2E_YOLO_DEVICE") or _pick(raw, "yolo_device", "cuda:0")),
        min_face_size=int(_env("F2E_MIN_FACE_SIZE") or _pick(raw, "min_face_size", 40)),
        max_track_distance=float(_env("F2E_MAX_TRACK_DISTANCE") or _pick(raw, "max_track_distance", 80.0)),
        micro_ema_alpha=float(_env("F2E_MICRO_EMA_ALPHA") or _pick(raw, "micro_ema_alpha", 0.35)),
        max_fps=float(_env("F2E_MAX_FPS") or _pick(raw, "max_fps", 20.0)),
        emotion_interval=int(_env("F2E_EMOTION_INTERVAL") or _pick(raw, "emotion_interval", 3)),
        stale_timeout_sec=float(_env("F2E_STALE_TIMEOUT_SEC") or _pick(raw, "stale_timeout_sec", 2.5)),
        mirror_input=_as_bool(raw_mirror, True),
        reconnect_cooldown_sec=float(
            _env("F2E_RECONNECT_COOLDOWN_SEC") or _pick(raw, "reconnect_cooldown_sec", 1.0)
        ),
        api_key=str(_env("F2E_API_KEY") or _pick(raw, "api_key", "")),
    )

    return _validate(config)
