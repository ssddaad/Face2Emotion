# 全身姿态 + 双手关键点（MediaPipe Tasks API）
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as mp_base

# 与 MediaPipe Pose 一致（33 点）
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

POSE_LANDMARK_NAMES: Tuple[str, ...] = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)

HAND_LANDMARK_NAMES: Tuple[str, ...] = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)


def _models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def _resolve_pose_asset(complexity: int) -> Path:
    root = _models_dir()
    order: List[str]
    if complexity <= 0:
        order = ["pose_landmarker_lite.task"]
    elif complexity == 1:
        order = ["pose_landmarker_full.task", "pose_landmarker_lite.task"]
    else:
        order = ["pose_landmarker_heavy.task", "pose_landmarker_full.task", "pose_landmarker_lite.task"]
    for name in order:
        p = root / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"未找到姿态模型，请将 pose_landmarker_lite.task（或 full/heavy）放到 {root}"
    )


def _hand_asset() -> Path:
    p = _models_dir() / "hand_landmarker.task"
    if not p.is_file():
        raise FileNotFoundError(f"未找到手部模型，请将 hand_landmarker.task 放到 {p.parent}")
    return p


@dataclass(slots=True)
class Landmark2D:
    name: str
    x_norm: float
    y_norm: float
    z_norm: float
    visibility: float
    x_px: int
    y_px: int

    @property
    def visible(self) -> bool:
        return self.visibility >= 0.5


@dataclass(slots=True)
class HandLandmark2D:
    name: str
    x_norm: float
    y_norm: float
    x_px: int
    y_px: int


@dataclass(slots=True)
class JointAngle:
    name: str
    angle_deg: float
    confidence: float


@dataclass(slots=True)
class Velocity:
    wrist_left_speed: float
    wrist_right_speed: float
    ankle_left_speed: float
    ankle_right_speed: float
    overall_motion: float


@dataclass(slots=True)
class PoseEstimate:
    landmarks: List[Landmark2D]
    joint_angles: List[JointAngle]
    velocity: Velocity
    action_label: str
    action_conf: float
    pose_quality: float


@dataclass(slots=True)
class HandEstimate:
    side: str
    gesture: str
    confidence: float
    landmarks: List[HandLandmark2D]


@dataclass(slots=True)
class MotionCaptureResult:
    timestamp: float
    pose: Optional[PoseEstimate]
    left_hand: Optional[HandEstimate]
    right_hand: Optional[HandEstimate]


def _angle_deg_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return float("nan")
    cos = float(np.dot(ba, bc) / (nba * nbc))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def _lm_xy(lms: List[Landmark2D], i: int) -> Optional[np.ndarray]:
    if i >= len(lms):
        return None
    lm = lms[i]
    if not lm.visible:
        return None
    return np.array([float(lm.x_px), float(lm.y_px)], dtype=np.float64)


class MotionCaptureEngine:
    def __init__(self, model_complexity: int = 1, landmark_ema_alpha: float = 0.45):
        pose_path = str(_resolve_pose_asset(model_complexity))
        hand_path = str(_hand_asset())
        base_pose = mp_base.BaseOptions(model_asset_path=pose_path)
        base_hand = mp_base.BaseOptions(model_asset_path=hand_path)
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base_pose,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_hand,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._ts_ms = int(time.time() * 1000)
        self._prev_norm: Dict[int, np.ndarray] = {}
        self._prev_t: Dict[int, float] = {}
        self._lm_ema_alpha = float(min(1.0, max(0.01, landmark_ema_alpha)))
        self._lm_smooth: Dict[int, np.ndarray] = {}

    def forget(self, alive_ids: set[int]) -> None:
        for k in list(self._prev_norm.keys()):
            if k not in alive_ids:
                del self._prev_norm[k]
                self._prev_t.pop(k, None)
                self._lm_smooth.pop(k, None)

    def close(self) -> None:
        self._pose.close()
        self._hand.close()

    def process(self, frame_bgr, track_id: int = -1) -> Optional[MotionCaptureResult]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms = max(self._ts_ms + 1, int(time.time() * 1000))
        now = time.time()

        pose_out = self._pose.detect_for_video(mp_image, self._ts_ms)
        hand_out = self._hand.detect_for_video(mp_image, self._ts_ms)

        pose_est: Optional[PoseEstimate] = None
        if pose_out.pose_landmarks and len(pose_out.pose_landmarks) > 0:
            raw = pose_out.pose_landmarks[0]
            smoothed_xy = self._smooth_landmarks(raw, track_id)
            landmarks: List[Landmark2D] = []
            for i, lm in enumerate(raw):
                vis = float(lm.visibility) if lm.visibility is not None else 0.0
                xn = float(smoothed_xy[i][0]) if i < len(smoothed_xy) else (float(lm.x) if lm.x is not None else 0.0)
                yn = float(smoothed_xy[i][1]) if i < len(smoothed_xy) else (float(lm.y) if lm.y is not None else 0.0)
                zn = float(lm.z) if lm.z is not None else 0.0
                landmarks.append(
                    Landmark2D(
                        name=POSE_LANDMARK_NAMES[i] if i < len(POSE_LANDMARK_NAMES) else f"lm{i}",
                        x_norm=xn,
                        y_norm=yn,
                        z_norm=zn,
                        visibility=vis,
                        x_px=int(round(xn * w)),
                        y_px=int(round(yn * h)),
                    )
                )
            joint_angles = _compute_joint_angles(landmarks)
            vel = self._compute_velocity(landmarks, track_id, now)
            pq = _pose_quality(landmarks)
            pose_est = PoseEstimate(
                landmarks=landmarks,
                joint_angles=joint_angles,
                velocity=vel,
                action_label="idle",
                action_conf=0.5,
                pose_quality=pq,
            )

        left_h: Optional[HandEstimate] = None
        right_h: Optional[HandEstimate] = None
        if hand_out.hand_landmarks:
            for idx, hlms in enumerate(hand_out.hand_landmarks):
                side = "right"
                if hand_out.handedness and idx < len(hand_out.handedness):
                    cats = hand_out.handedness[idx]
                    if cats and len(cats) > 0 and cats[0].category_name:
                        side = cats[0].category_name.lower()
                pts: List[HandLandmark2D] = []
                for j, lm in enumerate(hlms):
                    xn = float(lm.x) if lm.x is not None else 0.0
                    yn = float(lm.y) if lm.y is not None else 0.0
                    pts.append(
                        HandLandmark2D(
                            name=HAND_LANDMARK_NAMES[j] if j < len(HAND_LANDMARK_NAMES) else f"h{j}",
                            x_norm=xn,
                            y_norm=yn,
                            x_px=int(round(xn * w)),
                            y_px=int(round(yn * h)),
                        )
                    )
                he = HandEstimate(side=side, gesture="unknown", confidence=0.75, landmarks=pts)
                if side == "left":
                    left_h = he
                else:
                    right_h = he

        if pose_est is None and left_h is None and right_h is None:
            return None
        return MotionCaptureResult(timestamp=now, pose=pose_est, left_hand=left_h, right_hand=right_h)

    def _smooth_landmarks(self, raw_landmarks, track_id: int) -> np.ndarray:
        cur = np.array(
            [[float(lm.x or 0.0), float(lm.y or 0.0)] for lm in raw_landmarks],
            dtype=np.float64,
        )
        prev = self._lm_smooth.get(track_id)
        if prev is None or prev.shape != cur.shape:
            self._lm_smooth[track_id] = cur
            return cur
        a = self._lm_ema_alpha
        smoothed = a * cur + (1.0 - a) * prev
        self._lm_smooth[track_id] = smoothed
        return smoothed

    def _compute_velocity(self, landmarks: List[Landmark2D], track_id: int, now: float) -> Velocity:
        idxs = (15, 16, 27, 28)
        pts: List[List[float]] = []
        for i in idxs:
            if i >= len(landmarks):
                continue
            lm = landmarks[i]
            pts.append([lm.x_norm, lm.y_norm])
        cur = np.array(pts, dtype=np.float64) if pts else np.zeros((0, 2))
        tid = track_id
        prev = self._prev_norm.get(tid)
        prev_t = self._prev_t.get(tid, now)
        dt = max(now - prev_t, 1e-6)
        self._prev_norm[tid] = cur.copy()
        self._prev_t[tid] = now

        if prev is None or prev.shape != cur.shape or cur.size == 0:
            return Velocity(0.0, 0.0, 0.0, 0.0, 0.0)

        dists = np.linalg.norm(cur - prev, axis=1) / dt
        scale = 2.0
        wl = min(1.0, float(dists[0] * scale)) if dists.size > 0 else 0.0
        wr = min(1.0, float(dists[1] * scale)) if dists.size > 1 else 0.0
        al = min(1.0, float(dists[2] * scale)) if dists.size > 2 else 0.0
        ar = min(1.0, float(dists[3] * scale)) if dists.size > 3 else 0.0
        om = min(1.0, float(np.mean(dists) * scale * 0.5))
        return Velocity(
            wrist_left_speed=wl,
            wrist_right_speed=wr,
            ankle_left_speed=al,
            ankle_right_speed=ar,
            overall_motion=om,
        )


def _pose_quality(landmarks: List[Landmark2D]) -> float:
    body_idx = list(range(11, 17)) + list(range(23, 29))
    vis = [landmarks[i].visibility for i in body_idx if i < len(landmarks)]
    if not vis:
        return 0.0
    return float(max(0.0, min(1.0, sum(vis) / len(vis))))


def _compute_joint_angles(landmarks: List[Landmark2D]) -> List[JointAngle]:
    out: List[JointAngle] = []
    triples = [
        ("left_elbow", 11, 13, 15),
        ("right_elbow", 12, 14, 16),
        ("left_knee", 23, 25, 27),
        ("right_knee", 24, 26, 28),
        ("left_hip", 11, 23, 25),
        ("right_hip", 12, 24, 26),
    ]
    for name, ia, ib, ic in triples:
        a = _lm_xy(landmarks, ia)
        b = _lm_xy(landmarks, ib)
        c = _lm_xy(landmarks, ic)
        if a is None or b is None or c is None:
            continue
        ang = _angle_deg_2d(a, b, c)
        if np.isnan(ang):
            continue
        conf = min(landmarks[ib].visibility, landmarks[ia].visibility, landmarks[ic].visibility)
        out.append(JointAngle(name=name, angle_deg=ang, confidence=float(conf)))
    return out
