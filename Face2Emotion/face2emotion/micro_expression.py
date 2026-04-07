import cv2
import numpy as np

_LK = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01),
)
# 降低 qualityLevel，提高微弱阶段可追踪点数量
_SHI = dict(maxCorners=120, qualityLevel=0.03, minDistance=3, blockSize=3)

MIN_PTS = 3         # 细微阶段允许更少有效点
SCALE = 28.0        # 对早期小位移更敏感
NORM_SZ = (96, 96)  # 提高归一化分辨率，保留局部细节


# 人脸区域分区（眉眼嘴更高）
# 归一化坐标系：x/y in [0, 1]
ROI_BOUNDS = (
    (0.00, 0.00, 1.00, 0.28),  # 额头/眉区
    (0.15, 0.18, 0.85, 0.48),  # 眼区
    (0.18, 0.56, 0.82, 0.90),  # 嘴区
)
ROI_WEIGHTS = [1.25, 1.45, 1.60]


def set_roi_weights(brow: float, eye: float, mouth: float):
    ROI_WEIGHTS[0] = float(max(0.1, brow))
    ROI_WEIGHTS[1] = float(max(0.1, eye))
    ROI_WEIGHTS[2] = float(max(0.1, mouth))


def _norm(a, b):
    return (
        cv2.resize(a, NORM_SZ, interpolation=cv2.INTER_LINEAR),
        cv2.resize(b, NORM_SZ, interpolation=cv2.INTER_LINEAR),
    )


def _point_weight(pts_xy: np.ndarray) -> np.ndarray:
    # pts_xy: (N,2), 值域在像素坐标；转换到 [0,1] 后按 ROI 给权重
    if pts_xy.size == 0:
        return np.zeros((0,), dtype=np.float64)

    w, h = float(NORM_SZ[0]), float(NORM_SZ[1])
    nx = np.clip(pts_xy[:, 0] / max(w - 1.0, 1.0), 0.0, 1.0)
    ny = np.clip(pts_xy[:, 1] / max(h - 1.0, 1.0), 0.0, 1.0)

    out = np.ones((pts_xy.shape[0],), dtype=np.float64)
    for i, (x1, y1, x2, y2) in enumerate(ROI_BOUNDS):
        wt = ROI_WEIGHTS[i]
        mask = (nx >= x1) & (nx <= x2) & (ny >= y1) & (ny <= y2)
        out[mask] = np.maximum(out[mask], wt)
    return out


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    ws = np.maximum(weights, 1e-6)
    mean_v = float(np.sum(values * ws) / np.sum(ws))

    # 近似高分位：按值排序后累计权重到 75%
    order = np.argsort(values)
    sv = values[order]
    sw = ws[order]
    cum = np.cumsum(sw) / np.sum(sw)
    idx = int(np.searchsorted(cum, 0.75, side="left"))
    p75_v = float(sv[min(idx, sv.size - 1)])

    max_v = float(np.max(values))
    return mean_v, p75_v, max_v


def _lk_score(cur, prev):
    cn, pn = _norm(cur, prev)
    pts = cv2.goodFeaturesToTrack(pn, mask=None, **_SHI)
    if pts is None or len(pts) < MIN_PTS:
        return None

    dpts, st, _ = cv2.calcOpticalFlowPyrLK(pn, cn, pts, None, **_LK)
    if dpts is None or st is None:
        return None

    ok = st.ravel() == 1
    if ok.sum() < MIN_PTS:
        return None

    src = pts[ok].reshape(-1, 2)
    dst = dpts[ok].reshape(-1, 2)
    disp_vec = dst - src

    # 去除全局平移分量，抑制头部整体移动对微表情评分的干扰
    global_shift = np.median(disp_vec, axis=0)
    rel_vec = disp_vec - global_shift
    disp = np.linalg.norm(rel_vec, axis=1)

    wts = _point_weight(src)
    mean_d, p75_d, max_d = _weighted_stats(disp, wts)

    # ROI 加权统计 + 尾部位移，提升 onset 阶段触发能力
    raw = (0.58 * mean_d + 0.30 * p75_d + 0.12 * max_d) * SCALE
    # 非线性增强：让低位移区间也有可见分数增益
    score = 100.0 * (1.0 - np.exp(-raw * 0.08))
    return min(float(score), 100.0)


def _diff_score(cur, prev):
    # 光流失败兜底：均值 + 高分位帧差，避免细微变化被均值吞掉
    cn, pn = _norm(cur, prev)
    d = cv2.absdiff(cn, pn)
    mean_s = float(np.mean(d) / 255.0 * 100.0)
    p90_s = float(np.percentile(d, 90) / 255.0 * 100.0)
    return min(100.0, 0.35 * mean_s + 0.65 * p90_s)


def micro_expression_score(cur_gray, prev_gray):
    if prev_gray is None or prev_gray.size == 0 or cur_gray.size == 0:
        return 0.0
    s = _lk_score(cur_gray, prev_gray)
    return s if s is not None else _diff_score(cur_gray, prev_gray)


def micro_expression_level(score):
    if score < 2.0:
        return "Very Low"
    if score < 4.5:
        return "Low"
    if score < 8.0:
        return "Medium"
    return "High"


def ema(new_val, last_val, alpha):
    if last_val is None:
        return new_val
    return alpha * new_val + (1.0 - alpha) * last_val
