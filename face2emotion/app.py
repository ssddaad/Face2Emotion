import time
import cv2
import numpy as np

from .config import build_arg_parser, config_from_args
from .detector import FaceDetector
from .emotion import EmotionRecognizer
from .logging_utils import setup_logger
from .micro_expression import ema, micro_expression_level, micro_expression_score
from .model_store import ensure_face_model
from .renderer import Renderer
from .schema import EmotionResult, FaceInference
from .tracker import FaceTracker

INFER_EVERY_N = 3


def _warmup(recog, log):
    log.info("warming up emotion model, first run may take ~30s")
    rng = np.random.default_rng(42)
    # 随机dummy图触发ONNX/PyTorch第一次forward，避免第一帧真人脸进来时卡顿
    dummy = rng.integers(100, 200, (260, 260, 3), dtype=np.uint8)
    recog.predict(dummy, track_id=-1)
    log.info("warmup done")


def main():
    args = build_arg_parser().parse_args()
    cfg  = config_from_args(args)
    log  = setup_logger()

    ensure_face_model(cfg.model_path, log)
    log.info("face model: %s", cfg.model_path)

    detector   = FaceDetector(cfg, logger=log)
    recognizer = EmotionRecognizer()
    tracker    = FaceTracker(cfg.max_track_distance)
    renderer   = Renderer(cfg)
    _warmup(recognizer, log)

    cap = cv2.VideoCapture(cfg.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {cfg.camera_id}")
    log.info("running -- press Q to quit")

    prev_gray  = None
    t_prev     = time.time()
    n          = 0
    mu_hist:   dict[int, float]             = {}
    emo_cache: dict[int, tuple[str, float]] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            log.warning("frame read failed")
            break

        n += 1
        frame    = cv2.flip(frame, 1)   # 镜像，符合自拍直觉
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        boxes  = detector.detect(frame)
        tracks = tracker.assign(boxes)

        # 用alive_ids而非当帧tracks做清理，保留短暂消失轨迹的历史
        # 否则轨迹闪烁时情绪历史丢失，导致下一帧输出跳变
        alive     = tracker.alive_ids
        mu_hist   = {k: v for k, v in mu_hist.items()   if k in alive}
        emo_cache = {k: v for k, v in emo_cache.items() if k in alive}
        recognizer.forget(alive)

        results = []
        for fid, box in tracks:
            bgr_patch  = frame[box.y1:box.y2, box.x1:box.x2]
            gray_patch = cur_gray[box.y1:box.y2, box.x1:box.x2]
            prev_patch = prev_gray[box.y1:box.y2, box.x1:box.x2] if prev_gray is not None else None

            # 抽帧推理：每N帧跑一次，新轨迹首帧必须强制跑
            if n % INFER_EVERY_N == 0 or fid not in emo_cache:
                r = recognizer.predict(bgr_patch, track_id=fid)
                emo_cache[fid] = (r.label, r.score)

            lbl, sc = emo_cache[fid]

            raw  = micro_expression_score(gray_patch, prev_patch)
            smth = ema(raw, mu_hist.get(fid), cfg.micro_ema_alpha)
            mu_hist[fid] = smth

            results.append(FaceInference(
                track_id=fid, box=box,
                emotion=EmotionResult(label=lbl, score=sc),
                micro_score=smth,
                micro_level=micro_expression_level(smth),
            ))

        now    = time.time()
        fps    = 1.0 / max(now - t_prev, 1e-9)
        t_prev = now

        renderer.draw(frame, results, fps)
        cv2.imshow(cfg.window_name, frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

        prev_gray = cur_gray

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
