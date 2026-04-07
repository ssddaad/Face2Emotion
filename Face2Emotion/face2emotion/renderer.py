# renderer.py — 渲染模块
# 渲染内容：人脸检测框+情绪标签+微表情分数、全身骨骼、手部关键点+手势、动作标签、FPS
from functools import lru_cache
from pathlib import Path
from typing import List

import cv2
import numpy as np
from .config import AppConfig
from .schema import FaceInference

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
    _PIL_OK = False

# 调色板
_PALETTE=[
    (  0, 255, 128),(  0, 200, 255),(255, 200,   0),(255,  80,  80),
    (180, 100, 255),(255, 255,   0),( 80, 255, 200),(255, 140,   0),
]

# 骨骼颜色分区（BGR）
_COL_TORSO     =( 50, 205,  50)  # 躯干：亮绿
_COL_LEFT_ARM  =( 30, 144, 255)  # 左臂：道奇蓝
_COL_RIGHT_ARM =(255, 100,  30)  # 右臂：橙
_COL_LEFT_LEG  =( 60, 220, 220)  # 左腿：青
_COL_RIGHT_LEG =(220,  60, 220)  # 右腿：品红
_COL_FACE_CHAIN=(180, 180, 180)  # 脸部：灰
_COL_HAND_L    =(  0, 230, 115)  # 左手：绿
_COL_HAND_R    =(115, 115, 255)  # 右手：蓝紫
_COL_KP        =(255, 255, 255)  # 关键点：白
_COL_ACTION    =( 20, 220, 220)  # 动作标签：亮青
_COL_GESTURE   =(255, 215,   0)  # 手势标签：金

_F=cv2.FONT_HERSHEY_PLAIN
_FS=1.0
_LH=14

_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/msyhbd.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
]

# 连接索引 -> 颜色区域映射（与 motion_capture.py 的 POSE_CONNECTIONS 对应）
_BONE_COLORS={
    # 脸
    (0,1):"face",(1,2):"face",(2,3):"face",(3,7):"face",
    (0,4):"face",(4,5):"face",(5,6):"face",(6,8):"face",(9,10):"face",
    # 躯干
    (11,12):"torso",(11,23):"torso",(12,24):"torso",(23,24):"torso",
    # 左臂
    (11,13):"larm",(13,15):"larm",(15,17):"larm",(15,19):"larm",(15,21):"larm",(17,19):"larm",
    # 右臂
    (12,14):"rarm",(14,16):"rarm",(16,18):"rarm",(16,20):"rarm",(16,22):"rarm",(18,20):"rarm",
    # 左腿
    (23,25):"lleg",(25,27):"lleg",(27,29):"lleg",(29,31):"lleg",(27,31):"lleg",
    # 右腿
    (24,26):"rleg",(26,28):"rleg",(28,30):"rleg",(30,32):"rleg",(28,32):"rleg",
}
_ZONE_COL={
    "face":  _COL_FACE_CHAIN,
    "torso": _COL_TORSO,
    "larm":  _COL_LEFT_ARM,
    "rarm":  _COL_RIGHT_ARM,
    "lleg":  _COL_LEFT_LEG,
    "rleg":  _COL_RIGHT_LEG,
}


@lru_cache(maxsize=8)
def _pick_font(size: int):
    if not _PIL_OK:
        return None
    for p in _FONT_CANDIDATES:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    return None


def _need_unicode(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


def _txt(img, s, x, y, col=(200, 230, 200), scale=_FS):
    if _need_unicode(s):
        font = _pick_font(max(14, int(16 * scale)))
        if font is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil)
            draw.text((x, y - 12), s, font=font, fill=(int(col[2]), int(col[1]), int(col[0])))
            img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            return
    cv2.putText(img, s, (x, y), _F, scale, col, 1, cv2.LINE_AA)


def _txt_bg(img, s, x, y, col, bg=(0, 0, 0), alpha=0.55, scale=_FS):
    _txt(img, s, x, y, col, scale)


def _estimate_distance_m(box) -> float:
    # 基于人脸框宽度的粗略单目距离估计（默认等效焦距）
    face_w_px=max(1.0, float(box.width))
    focal_px=700.0
    real_face_w_m=0.16
    d=(focal_px*real_face_w_m)/face_w_px
    return float(np.clip(d, 0.2, 4.0))


def _color_desc(color_vitality: float) -> str:
    if color_vitality >= 68:
        return "红润"
    if color_vitality >= 45:
        return "正常"
    if color_vitality >= 30:
        return "偏苍白"
    return "晦暗"


def _spirit_desc(mental_score: float) -> str:
    if mental_score >= 72:
        return "有神"
    if mental_score >= 52:
        return "一般"
    return "无神"


def _eye_desc(eye_focus: float, perclos: float) -> str:
    if perclos > 0.34 or eye_focus < 35:
        return "目光呆滞/眼睑疲劳"
    if perclos > 0.22 or eye_focus < 55:
        return "轻度疲劳"
    return "有神"


def _draw_pose(frame, pose, show_skeleton: bool, show_action: bool):
    if pose is None:
        return
    lms=pose.landmarks
    h, w=frame.shape[:2]
    if show_skeleton:
        from .motion_capture import POSE_CONNECTIONS
        for (a, b) in POSE_CONNECTIONS:
            if a>=len(lms) or b>=len(lms):
                continue
            zone=_BONE_COLORS.get((a, b)) or _BONE_COLORS.get((b, a), "torso")
            if zone=="face":
                continue
            la, lb=lms[a], lms[b]
            if not (la.visible and lb.visible):
                continue
            cv2.line(frame, (la.x_px, la.y_px), (lb.x_px, lb.y_px), _ZONE_COL[zone], 2, cv2.LINE_AA)
        for i, lm in enumerate(lms):
            if i<=10:
                continue
            if lm.visible:
                cv2.circle(frame, (lm.x_px, lm.y_px), 3, _COL_KP, -1)
    if show_action and pose.action_label:
        label=f"Action: {pose.action_label}  ({pose.action_conf:.0%})"
        (tw, _), _=cv2.getTextSize(label, _F, 1.3, 1)
        x=max(w//2-tw//2, 4)
        _txt_bg(frame, label, x, 24, _COL_ACTION, scale=1.3)
    quality_s=f"Pose {pose.pose_quality:.0%}"
    _txt(frame, quality_s, 4, h-6, (160, 160, 160), scale=0.9)


def _draw_hand(frame, hand, show_hands: bool):
    if hand is None or not show_hands:
        return
    from .motion_capture import HAND_CONNECTIONS
    lms=hand.landmarks
    col=_COL_HAND_L if hand.side=="left" else _COL_HAND_R
    for (a, b) in HAND_CONNECTIONS:
        if a>=len(lms) or b>=len(lms):
            continue
        la, lb=lms[a], lms[b]
        cv2.line(frame, (la.x_px, la.y_px), (lb.x_px, lb.y_px), col, 1, cv2.LINE_AA)
    for lm in lms:
        cv2.circle(frame, (lm.x_px, lm.y_px), 2, col, -1)
    if hand.gesture and hand.gesture!="unknown":
        wrist=lms[0]
        label=f"{hand.side[0].upper()}: {hand.gesture}"
        _txt_bg(frame, label, wrist.x_px, wrist.y_px-8, _COL_GESTURE, scale=1.0)


class Renderer:
    def __init__(self, cfg: AppConfig):
        self.cfg=cfg

    def draw(self, frame, faces: List[FaceInference], fps: float):
        self._draw_summary(frame, faces)
        for f in faces:
            self._draw_face(frame, f)
            if f.motion is not None:
                _draw_pose(frame, f.motion.pose, self.cfg.show_skeleton, self.cfg.show_action)
                _draw_hand(frame, f.motion.left_hand,  self.cfg.show_hands)
                _draw_hand(frame, f.motion.right_hand, self.cfg.show_hands)
        if self.cfg.show_fps:
            self._draw_fps(frame, fps)

    def _draw_face(self, frame, inf: FaceInference):
        b=inf.box
        col=_PALETTE[inf.track_id % len(_PALETTE)]
        cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), col, 1)

        dist_m=_estimate_distance_m(b)
        _txt(frame, f"expr={inf.emotion.label}", b.x1, max(b.y1-_LH, _LH), col)
        _txt(frame, f"dist={dist_m:.2f}m", b.x1, max(b.y1-2, _LH*2), (180, 220, 255))

        if inf.mental_state is not None:
            _txt(
                frame,
                f"mental={inf.mental_state.score:.0f} {inf.mental_state.level}",
                b.x1,
                max(b.y1 + _LH, _LH * 3),
                (80, 220, 240),
            )

    def _draw_summary(self, frame, faces: List[FaceInference]):
        if not faces:
            return
        main=max(faces, key=lambda f: f.box.width * f.box.height)
        if main.mental_state is None:
            return

        ms=main.mental_state
        color_s=_color_desc(ms.color_vitality)
        spirit_s=_spirit_desc(ms.score)
        eye_s=_eye_desc(ms.eye_focus, ms.perclos)

        x, y=8, 22
        _txt_bg(frame, f"面色: {color_s}", x, y, (120, 220, 255), scale=1.1)
        _txt_bg(frame, f"神态: {spirit_s}", x, y+20, (120, 255, 180), scale=1.1)
        _txt_bg(frame, f"眼睛: {eye_s}", x, y+40, (255, 230, 150), scale=1.1)

    @staticmethod
    def _draw_fps(frame, fps):
        h, w=frame.shape[:2]
        s=f"{fps:.1f} fps"
        (tw, _), _=cv2.getTextSize(s, _F, _FS, 1)
        _txt(frame, s, w-tw-8, 16, (160, 160, 160))
