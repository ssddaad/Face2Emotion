import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    camera_id: int
    confidence: float
    iou: float
    image_size: int
    model_path: Path
    window_name: str
    show_fps: bool
    min_face_size: int
    max_track_distance: float
    micro_ema_alpha: float
    yolo_device: str


def build_arg_parser():
    p = argparse.ArgumentParser(description="face micro-expression & emotion detection")
    p.add_argument("--camera",             type=int,   default=0)
    p.add_argument("--conf",               type=float, default=0.4)
    p.add_argument("--iou",                type=float, default=0.5)
    p.add_argument("--imgsz",              type=int,   default=640)
    p.add_argument("--model",              type=str,   default="models/yolov8n-face.pt")
    p.add_argument("--window",             type=str,   default="Face2Emotion")
    p.add_argument("--hide-fps",           action="store_true")
    p.add_argument("--min-face-size",      type=int,   default=40)
    p.add_argument("--max-track-distance", type=float, default=80.0)
    p.add_argument("--micro-ema-alpha",    type=float, default=0.35)
    p.add_argument("--yolo-device",        type=str,   default="cuda:0")
    return p


def config_from_args(ns):
    return AppConfig(
        camera_id=ns.camera,
        confidence=ns.conf,
        iou=ns.iou,
        image_size=ns.imgsz,
        model_path=Path(ns.model),
        window_name=ns.window,
        show_fps=not ns.hide_fps,
        min_face_size=ns.min_face_size,
        max_track_distance=ns.max_track_distance,
        micro_ema_alpha=ns.micro_ema_alpha,
        yolo_device=ns.yolo_device,
    )
