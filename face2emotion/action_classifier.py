# 基于关节角度与速度的时序投票动作分类
from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Dict, Set

from .motion_capture import PoseEstimate


def _infer_frame(pose: PoseEstimate) -> tuple[str, float]:
    v = pose.velocity
    lm = {x.name: x for x in pose.joint_angles}

    def ang(name: str, default: float = 180.0) -> float:
        return lm[name].angle_deg if name in lm else default

    # 高整体运动优先
    if v.overall_motion > 0.26:
        return "moving", min(1.0, 0.58 + v.overall_motion * 0.36)

    le, re = ang("left_elbow"), ang("right_elbow")
    lk, rk = ang("left_knee"), ang("right_knee")

    # 深蹲：双膝明显弯曲
    if lk < 138 and rk < 138 and pose.pose_quality > 0.42:
        return "squat", min(1.0, 0.62 + (180 - max(lk, rk)) / 280.0)

    # 双臂上举：肘部张开且手腕高于肩（用 y 归一化：图像向下增大 → 手腕 y 小于肩）
    if len(pose.landmarks) > 16:
        ls, rs = pose.landmarks[11], pose.landmarks[12]
        lw, rw = pose.landmarks[15], pose.landmarks[16]
        if ls.visible and rs.visible and lw.visible and rw.visible:
            shoulder_y = (ls.y_norm + rs.y_norm) * 0.5
            wrist_y = (lw.y_norm + rw.y_norm) * 0.5
            if wrist_y < shoulder_y - 0.045 and le > 100 and re > 100:
                return "arms_up", 0.84

    # 躯干侧倾：双肩高度差
    if len(pose.landmarks) > 12:
        ls, rs = pose.landmarks[11], pose.landmarks[12]
        if ls.visible and rs.visible and abs(ls.y_norm - rs.y_norm) > 0.04:
            return "lean", 0.74

    # 行走感：踝部相对手腕更活跃
    if v.ankle_left_speed + v.ankle_right_speed > v.wrist_left_speed + v.wrist_right_speed + 0.06:
        return "walking", 0.68

    return "idle", min(0.95, 0.57 + pose.pose_quality * 0.33)


class ActionClassifier:
    def __init__(self, vote_window: int = 10):
        self._win = max(1, vote_window)
        self._buf: Dict[int, Deque[str]] = {}

    def forget(self, alive_ids: Set[int]) -> None:
        for k in list(self._buf.keys()):
            if k not in alive_ids:
                del self._buf[k]

    def apply(self, pose: PoseEstimate, track_id: int) -> None:
        label, conf = _infer_frame(pose)
        if track_id not in self._buf:
            self._buf[track_id] = deque(maxlen=self._win)
        self._buf[track_id].append(label)
        votes = Counter(self._buf[track_id])
        best, cnt = votes.most_common(1)[0]

        dynamic_ratio = 0.45 if pose.velocity.overall_motion > 0.28 else 0.60
        if cnt / len(self._buf[track_id]) >= dynamic_ratio:
            pose.action_label = best
        else:
            pose.action_label = label
        pose.action_conf = min(1.0, conf * (cnt / len(self._buf[track_id])))
