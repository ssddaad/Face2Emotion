import logging
import cv2
import numpy as np

from .schema import EmotionResult

_log = logging.getLogger("face2emotion")

# must match hsemotion enet_b2_8 output order
_LABELS = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise",
]
_LOWER = {
    "Anger": "angry", "Contempt": "contempt", "Disgust": "disgust",
    "Fear": "fear",   "Happiness": "happy",   "Neutral": "neutral",
    "Sadness": "sad", "Surprise": "surprise",
}

# Contempt训练样本极少，分类边界模糊，直接并入Neutral避免误判
_CI = _LABELS.index("Contempt")
_NI = _LABELS.index("Neutral")

_W_NEW = 0.6   # EMA新帧权重
_W_OLD = 0.4
_MIN_SZ = 30

# FaceMesh左右眼内外眼角索引，取均值定位眼球中心
_L_EYE = [33, 133]
_R_EYE = [362, 263]


def _eye_centers(lm, w, h):
    try:
        lx = sum(lm[i].x * w for i in _L_EYE) / 2
        ly = sum(lm[i].y * h for i in _L_EYE) / 2
        rx = sum(lm[i].x * w for i in _R_EYE) / 2
        ry = sum(lm[i].y * h for i in _R_EYE) / 2
        return (lx, ly), (rx, ry)
    except Exception:
        return None


def _align(bgr, lm):
    """仿射旋转对齐：以两眼连线为基准，消除头部偏转对识别精度的影响"""
    h, w = bgr.shape[:2]
    pts = _eye_centers(lm, w, h)
    if not pts:
        return None
    (lx, ly), (rx, ry) = pts
    angle = np.degrees(np.arctan2(ry - ly, rx - lx))
    cx, cy = (lx + rx) / 2, (ly + ry) / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR)


def _preprocess(bgr, lm=None):
    h, w = bgr.shape[:2]
    if h < _MIN_SZ or w < _MIN_SZ:
        return bgr
    if lm is not None:
        aligned = _align(bgr, lm)
        if aligned is not None:
            return aligned
    return bgr


def _to7(p8):
    # fold 8-class probs into 7, re-normalize
    p = p8.copy()
    p[_NI] += p[_CI]
    out = {}; tot = 0.0
    for i, lbl in enumerate(_LABELS):
        if lbl == "Contempt": continue
        v = float(p[i])
        out[_LOWER[lbl]] = v
        tot += v
    if tot > 1e-9:
        out = {k: v/tot for k, v in out.items()}
    return out


class EmotionRecognizer:
    def __init__(self):
        self._model = None
        self._mesh  = None
        self._buf: dict[int, dict[str, float]] = {}  # per-track EMA history

    def _get_model(self):
        if self._model is None:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            self._model = HSEmotionRecognizer(model_name="enet_b2_8")
        return self._model

    def _get_mesh(self):
        if self._mesh is None:
            import mediapipe as mp
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        return self._mesh

    def _landmarks(self, bgr):
        try:
            res = self._get_mesh().process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                return res.multi_face_landmarks[0].landmark
        except Exception as e:
            _log.debug("landmark err: %s", e)
        return None

    def _smooth(self, tid, dist):
        # tid=-1是warmup调用，不维护历史
        if tid < 0: return dict(dist)
        old = self._buf.get(tid)
        if old is None:
            self._buf[tid] = dict(dist)
            return dict(dist)
        # 指数加权平均：平衡响应速度与输出稳定性
        blended = {k: _W_NEW*dist.get(k,0.) + _W_OLD*old.get(k,0.) for k in dist}
        self._buf[tid] = blended
        return blended

    def predict(self, bgr, track_id=-1):
        if bgr is None or bgr.size == 0:
            return self._fallback(track_id)
        try:
            lm  = self._landmarks(bgr)
            img = _preprocess(bgr, lm)
            _, scores = self._get_model().predict_emotions(img, logits=False)
            arr = np.asarray(scores, dtype=np.float32).ravel()
            sm  = self._smooth(track_id, _to7(arr))
            k, v = max(sm.items(), key=lambda x: x[1])
            return EmotionResult(label=k, score=float(v))
        except Exception as e:
            _log.warning("predict failed tid=%s: %s", track_id, e, exc_info=True)
            return self._fallback(track_id)

    def forget(self, active: set):
        for k in [k for k in self._buf if k not in active]:
            del self._buf[k]

    def _fallback(self, tid):
        h = self._buf.get(tid)
        if h:
            k, v = max(h.items(), key=lambda x: x[1])
            return EmotionResult(label=k, score=float(v))
        return EmotionResult(label="unknown", score=0.0)
