# 基于 21 点手部关键点的规则手势识别
from __future__ import annotations

import math

from typing import List

from .motion_capture import HandEstimate, HandLandmark2D


def _dist(a: HandLandmark2D, b: HandLandmark2D) -> float:
    dx = a.x_norm - b.x_norm
    dy = a.y_norm - b.y_norm
    return math.sqrt(dx * dx + dy * dy)


def _finger_extended(tip: HandLandmark2D, pip: HandLandmark2D, mcp: HandLandmark2D, wrist: HandLandmark2D) -> bool:
    # 指尖到腕距离大于 PIP 到腕，且 TIP 在 MCP 外侧（简化）
    return _dist(tip, wrist) > _dist(pip, wrist) * 1.02


def _classify(pts: List[HandLandmark2D]) -> tuple[str, float]:
    if len(pts) < 21:
        return "unknown", 0.0
    wrist = pts[0]
    thumb_tip, index_tip = pts[4], pts[8]
    middle_tip, ring_tip, pinky_tip = pts[12], pts[16], pts[20]
    index_pip, middle_pip = pts[6], pts[10]

    ext = [
        _finger_extended(pts[8], pts[6], pts[5], wrist),
        _finger_extended(pts[12], pts[10], pts[9], wrist),
        _finger_extended(pts[16], pts[14], pts[13], wrist),
        _finger_extended(pts[20], pts[18], pts[17], wrist),
    ]
    n_ext = sum(ext)

    # 拇指张开：拇指尖远离小指侧
    thumb_out = _dist(thumb_tip, pinky_tip) > _dist(pts[3], pinky_tip)

    if n_ext == 0 and not thumb_out:
        return "fist", 0.85
    if n_ext == 4 and thumb_out:
        return "open", 0.88
    if n_ext == 1 and ext[0]:
        return "point", 0.8
    if n_ext == 2 and ext[0] and ext[1]:
        return "peace", 0.78
    if thumb_out and n_ext <= 1 and _dist(thumb_tip, index_pip) < _dist(index_tip, index_pip) * 0.6:
        return "thumbs_up", 0.75
    if n_ext >= 3:
        return "open", 0.65
    return "unknown", 0.5


def apply_gesture(hand: HandEstimate) -> None:
    g, c = _classify(hand.landmarks)
    hand.gesture = g
    hand.confidence = c
