from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .schema import MentalStateResult


class MentalStateEstimator:
    def __init__(self, trend_window: int = 90, baseline_frames: int = 120, eye_window: int = 120):
        self._trend_window = max(10, trend_window)
        self._baseline_frames = max(20, baseline_frames)
        self._eye_window = max(30, eye_window)

        self._score_hist: dict[int, deque[float]] = {}
        self._baseline_sum: dict[int, float] = {}
        self._baseline_cnt: dict[int, int] = {}

        # 眼部时序状态（用于 PERCLOS + 眨眼率）
        self._eye_closed_hist: dict[int, deque[int]] = {}
        self._blink_count: dict[int, int] = {}
        self._prev_closed: dict[int, bool] = {}
        self._frame_seen: dict[int, int] = {}

    def forget(self, alive_ids: set[int]) -> None:
        for tid in list(self._score_hist.keys()):
            if tid not in alive_ids:
                self._score_hist.pop(tid, None)
                self._baseline_sum.pop(tid, None)
                self._baseline_cnt.pop(tid, None)
                self._eye_closed_hist.pop(tid, None)
                self._blink_count.pop(tid, None)
                self._prev_closed.pop(tid, None)
                self._frame_seen.pop(tid, None)

    def infer(self, track_id: int, face_bgr: np.ndarray, micro_score: float, emotion_conf: float) -> MentalStateResult:
        if face_bgr is None or face_bgr.size == 0:
            return MentalStateResult(
                score=0.0,
                level="Low",
                color_vitality=0.0,
                eye_focus=0.0,
                facial_energy=0.0,
                trend_score=0.0,
                risk_level="High Risk",
                perclos=1.0,
                blink_rate=0.0,
            )

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        color_v = _color_vitality_score(face_bgr)
        eye_focus_raw, eye_closed = _eye_focus_and_closed(gray)
        perclos, blink_rate = self._update_eye_metrics(track_id, eye_closed)
        eye_f = _eye_focus_fused(eye_focus_raw, perclos, blink_rate)
        facial_e = _facial_energy_score(micro_score, emotion_conf)

        raw = float(0.35 * color_v + 0.40 * eye_f + 0.25 * facial_e)
        base_adjusted = self._baseline_adjust(track_id, raw)
        trend = self._trend(track_id, base_adjusted)
        final_score = float(np.clip(0.70 * base_adjusted + 0.30 * trend, 0.0, 100.0))

        return MentalStateResult(
            score=final_score,
            level=_level(final_score),
            color_vitality=color_v,
            eye_focus=eye_f,
            facial_energy=facial_e,
            trend_score=trend,
            risk_level=_risk(final_score),
            perclos=perclos,
            blink_rate=blink_rate,
        )

    def _update_eye_metrics(self, track_id: int, eye_closed: bool) -> tuple[float, float]:
        if track_id not in self._eye_closed_hist:
            self._eye_closed_hist[track_id] = deque(maxlen=self._eye_window)
            self._blink_count[track_id] = 0
            self._prev_closed[track_id] = False
            self._frame_seen[track_id] = 0

        hist = self._eye_closed_hist[track_id]
        prev = self._prev_closed[track_id]
        hist.append(1 if eye_closed else 0)

        # 眨眼事件：开 -> 闭 的沿触发
        if eye_closed and not prev:
            self._blink_count[track_id] += 1

        self._prev_closed[track_id] = eye_closed
        self._frame_seen[track_id] += 1

        perclos = float(np.mean(hist)) if hist else 0.0
        # 以 30fps 近似，窗口秒数=window/30
        win_sec = max(1e-6, len(hist) / 30.0)
        blink_rate = float(self._blink_count[track_id] * (60.0 / win_sec))

        # 防止过短序列下 blink_rate 发散
        if self._frame_seen[track_id] < 20:
            blink_rate = 0.0

        return perclos, float(min(80.0, blink_rate))

    def _baseline_adjust(self, track_id: int, raw: float) -> float:
        cnt = self._baseline_cnt.get(track_id, 0)
        if cnt < self._baseline_frames:
            self._baseline_sum[track_id] = self._baseline_sum.get(track_id, 0.0) + raw
            self._baseline_cnt[track_id] = cnt + 1
            return raw

        base = self._baseline_sum[track_id] / max(1, self._baseline_cnt[track_id])
        return float(np.clip(50.0 + (raw - base) * 1.6, 0.0, 100.0))

    def _trend(self, track_id: int, score: float) -> float:
        if track_id not in self._score_hist:
            self._score_hist[track_id] = deque(maxlen=self._trend_window)
        self._score_hist[track_id].append(score)
        return float(np.mean(self._score_hist[track_id]))


def _safe_patch(img, x1: int, y1: int, x2: int, y2: int):
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def _level(score: float) -> str:
    if score >= 75:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 45:
        return "Fair"
    return "Low"


def _risk(score: float) -> str:
    if score < 40:
        return "High Risk"
    if score < 60:
        return "Medium Risk"
    return "Low Risk"


def _color_vitality_score(face_bgr: np.ndarray) -> float:
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l = lab[..., 0].astype(np.float32)
    a = lab[..., 1].astype(np.float32)
    b = lab[..., 2].astype(np.float32)

    l_mean = float(np.mean(l))
    l_std = float(np.std(l))
    chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)
    c_mean = float(np.mean(chroma))

    light_score = np.clip((l_mean - 60.0) / 80.0, 0.0, 1.0)
    texture_score = np.clip(l_std / 35.0, 0.0, 1.0)
    chroma_score = np.clip(c_mean / 24.0, 0.0, 1.0)

    return float((0.50 * light_score + 0.20 * texture_score + 0.30 * chroma_score) * 100.0)


def _eye_focus_and_closed(gray_face: np.ndarray) -> tuple[float, bool]:
    h, w = gray_face.shape[:2]
    eye_band = _safe_patch(gray_face, int(w * 0.15), int(h * 0.12), int(w * 0.85), int(h * 0.50))
    if eye_band is None or eye_band.size == 0:
        return 50.0, False

    lap = cv2.Laplacian(eye_band, cv2.CV_32F)
    sharp = float(np.var(lap))
    contrast = float(np.std(eye_band.astype(np.float32)))

    # 粗闭眼估计：眼带高频纹理和对比度显著下降
    closed = (sharp < 35.0 and contrast < 18.0)

    sharp_s = np.clip(sharp / 260.0, 0.0, 1.0)
    contrast_s = np.clip(contrast / 42.0, 0.0, 1.0)
    focus = float((0.65 * sharp_s + 0.35 * contrast_s) * 100.0)
    return focus, closed


def _eye_focus_fused(base_focus: float, perclos: float, blink_rate: float) -> float:
    # 正常 blink_rate 约 10-25 次/分钟，过低或过高都扣分
    blink_norm = np.clip(1.0 - abs(blink_rate - 16.0) / 24.0, 0.0, 1.0)
    perclos_penalty = np.clip(1.0 - perclos / 0.45, 0.0, 1.0)

    return float(np.clip(0.60 * (base_focus / 100.0) + 0.25 * perclos_penalty + 0.15 * blink_norm, 0.0, 1.0) * 100.0)


def _facial_energy_score(micro_score: float, emotion_conf: float) -> float:
    m = np.clip(float(micro_score) / 14.0, 0.0, 1.0)
    e = np.clip(float(emotion_conf), 0.0, 1.0)
    return float((0.72 * m + 0.28 * e) * 100.0)
