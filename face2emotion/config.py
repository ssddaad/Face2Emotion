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
    # 微表情 ROI 权重
    micro_roi_brow_weight: float = 1.25
    micro_roi_eye_weight: float = 1.45
    micro_roi_mouth_weight: float = 1.60
    # 动作捕捉配置
    enable_motion_capture: bool = False
    motion_complexity: int = 1
    motion_vote_window: int = 10
    motion_interval: int = 1
    motion_landmark_ema_alpha: float = 0.45
    show_skeleton: bool = True
    show_hands: bool = True
    show_action: bool = True


def build_arg_parser():
    p = argparse.ArgumentParser(description="face micro-expression & emotion detection")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--model", type=str, default="models/yolov8n-face.pt")
    p.add_argument("--window", type=str, default="Face2Emotion")
    p.add_argument("--hide-fps", action="store_true")
    p.add_argument("--min-face-size", type=int, default=40)
    p.add_argument("--max-track-distance", type=float, default=80.0)
    p.add_argument("--micro-ema-alpha", type=float, default=0.35)
    p.add_argument("--micro-roi-brow", type=float, default=1.25, help="微表情眉区权重")
    p.add_argument("--micro-roi-eye", type=float, default=1.45, help="微表情眼区权重")
    p.add_argument("--micro-roi-mouth", type=float, default=1.60, help="微表情嘴区权重")
    p.add_argument("--yolo-device", type=str, default="cuda:0")
    p.add_argument("--motion-capture", action="store_true", help="启用全身动作捕捉")
    p.add_argument("--motion-complexity", type=int, default=1, help="MediaPipe 复杂度 0/1/2")
    p.add_argument("--motion-vote-window", type=int, default=10, help="动作分类时序投票窗口")
    p.add_argument("--motion-interval", type=int, default=1, help="每隔 N 帧执行一次动作捕捉")
    p.add_argument("--motion-lm-alpha", type=float, default=0.45, help="姿态关键点 EMA 平滑系数")
    p.add_argument("--hide-skeleton", action="store_true", help="隐藏骨骼线条")
    p.add_argument("--hide-hands", action="store_true", help="隐藏手部关键点")
    p.add_argument("--hide-action", action="store_true", help="隐藏动作标签")
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
        micro_roi_brow_weight=ns.micro_roi_brow,
        micro_roi_eye_weight=ns.micro_roi_eye,
        micro_roi_mouth_weight=ns.micro_roi_mouth,
        yolo_device=ns.yolo_device,
        enable_motion_capture=ns.motion_capture,
        motion_complexity=ns.motion_complexity,
        motion_vote_window=ns.motion_vote_window,
        motion_interval=max(1, ns.motion_interval),
        motion_landmark_ema_alpha=ns.motion_lm_alpha,
        show_skeleton=not ns.hide_skeleton,
        show_hands=not ns.hide_hands,
        show_action=not ns.hide_action,
    )
