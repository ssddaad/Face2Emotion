import cv2
import numpy as np

_LK = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
# Shi-Tomasi角点，qualityLevel=0.2在小人脸图上能取到足够数量的点
_SHI = dict(maxCorners=80, qualityLevel=0.2, minDistance=5, blockSize=5)

MIN_PTS = 5     # 有效追踪点不足时放弃光流，降级到帧差兜底
SCALE   = 20.0  # 把平均像素位移映射到0~100分值
NORM_SZ = (64, 64)  # 归一化尺寸，消除人脸远近导致的位移量差异


def _norm(a, b):
    return (cv2.resize(a, NORM_SZ, interpolation=cv2.INTER_LINEAR),
            cv2.resize(b, NORM_SZ, interpolation=cv2.INTER_LINEAR))


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
    # 只取追踪成功的点，计算欧氏位移均值
    disp = np.linalg.norm(dpts[ok] - pts[ok], axis=2).ravel()
    return min(float(np.mean(disp)) * SCALE, 100.0)


def _diff_score(cur, prev):
    # 光流失败的兜底方案：像素帧差，精度低但稳定
    cn, pn = _norm(cur, prev)
    return float(np.mean(cv2.absdiff(cn, pn)) / 255.0 * 100.0)


def micro_expression_score(cur_gray, prev_gray):
    if prev_gray is None or prev_gray.size == 0 or cur_gray.size == 0:
        return 0.0
    s = _lk_score(cur_gray, prev_gray)
    return s if s is not None else _diff_score(cur_gray, prev_gray)


def micro_expression_level(score):
    if score < 3.0:   return "Very Low"
    if score < 6.0:   return "Low"
    if score < 10.0:  return "Medium"
    return "High"


def ema(new_val, last_val, alpha):
    if last_val is None:
        return new_val
    return alpha * new_val + (1.0 - alpha) * last_val
