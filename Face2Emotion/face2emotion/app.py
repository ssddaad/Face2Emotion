import time
import cv2

import numpy as np

from .action_classifier import ActionClassifier
from .config import build_arg_parser, config_from_args
from .detector import FaceDetector
from .emotion import EmotionRecognizer
from .gesture import apply_gesture
from .logging_utils import setup_logger
from .mental_state import MentalStateEstimator
from .micro_expression import ema, micro_expression_level, micro_expression_score, set_roi_weights
from .model_store import ensure_face_model
from .motion_capture import MotionCaptureEngine
from .renderer import Renderer
from .schema import EmotionResult, FaceInference
from .tracker import FaceTracker

INFER_EVERY_N = 1


def _warmup(recog, log):
    log.info("warming up emotion model, first run may take ~30s")
    rng = np.random.default_rng(42)
    dummy = rng.integers(100, 200, (260, 260, 3), dtype=np.uint8)
    recog.predict(dummy, track_id=-1)
    log.info("warmup done")


def main():
    args = build_arg_parser().parse_args()
    cfg = config_from_args(args)
    log = setup_logger()

    ensure_face_model(cfg.model_path, log)
    log.info("face model: %s", cfg.model_path)
    log.info(
        "micro ROI weights: brow=%.2f eye=%.2f mouth=%.2f",
        cfg.micro_roi_brow_weight,
        cfg.micro_roi_eye_weight,
        cfg.micro_roi_mouth_weight,
    )

    detector = FaceDetector(cfg, logger=log)
    recognizer = EmotionRecognizer()
    set_roi_weights(cfg.micro_roi_brow_weight, cfg.micro_roi_eye_weight, cfg.micro_roi_mouth_weight)
    tracker = FaceTracker(cfg.max_track_distance)
    mental_est = MentalStateEstimator()
    renderer = Renderer(cfg)
    _warmup(recognizer, log)

    # 动作捕捉引擎（按需启用）
    mc_engine = None
    ac_classify = None
    mc_last = None
    if cfg.enable_motion_capture:
        log.info(
            "motion capture enabled (complexity=%d, vote_window=%d, interval=%d)",
            cfg.motion_complexity,
            cfg.motion_vote_window,
            cfg.motion_interval,
        )
        mc_engine = MotionCaptureEngine(
            model_complexity=cfg.motion_complexity,
            landmark_ema_alpha=cfg.motion_landmark_ema_alpha,
        )
        ac_classify = ActionClassifier(vote_window=cfg.motion_vote_window)

    cap = cv2.VideoCapture(cfg.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {cfg.camera_id}")
    log.info("running -- press Q to quit")

    prev_gray = None
    t_prev = time.time()
    n = 0
    mu_hist: dict[int, float] = {}
    emo_cache: dict[int, tuple[str, float]] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            log.warning("frame read failed")
            break

        n += 1
        frame = cv2.flip(frame, 1)
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        boxes = detector.detect(frame)
        tracks = tracker.assign(boxes)

        alive = tracker.alive_ids
        mu_hist = {k: v for k, v in mu_hist.items() if k in alive}
        emo_cache = {k: v for k, v in emo_cache.items() if k in alive}
        recognizer.forget(alive)
        if ac_classify:
            ac_classify.forget(alive)
        if mc_engine:
            mc_engine.forget(alive)
        mental_est.forget(alive)

        # 动作捕捉：按 interval 降频执行，减少延迟
        mc_results: dict[int, object] = {}
        if mc_engine is not None:
            if n % cfg.motion_interval == 0 or mc_last is None:
                primary_tid = max(
                    tracks,
                    key=lambda item: (item[1].x2 - item[1].x1) * (item[1].y2 - item[1].y1),
                )[0] if tracks else -1
                mc_last = mc_engine.process(frame, track_id=primary_tid)
                if mc_last is not None:
                    if mc_last.left_hand:
                        apply_gesture(mc_last.left_hand)
                    if mc_last.right_hand:
                        apply_gesture(mc_last.right_hand)
                    if mc_last.pose and ac_classify:
                        ac_classify.apply(mc_last.pose, track_id=primary_tid)
            if mc_last is not None:
                for fid, _ in tracks:
                    mc_results[fid] = mc_last

        results = []
        for fid, box in tracks:
            bgr_patch = frame[box.y1:box.y2, box.x1:box.x2]
            gray_patch = cur_gray[box.y1:box.y2, box.x1:box.x2]
            prev_patch = prev_gray[box.y1:box.y2, box.x1:box.x2] if prev_gray is not None else None

            if n % INFER_EVERY_N == 0 or fid not in emo_cache:
                r = recognizer.predict(bgr_patch, track_id=fid)
                emo_cache[fid] = (r.label, r.score)

            lbl, sc = emo_cache[fid]

            raw = micro_expression_score(gray_patch, prev_patch)
            smth = ema(raw, mu_hist.get(fid), cfg.micro_ema_alpha)
            mu_hist[fid] = smth

            results.append(FaceInference(
                track_id=fid,
                box=box,
                emotion=EmotionResult(label=lbl, score=sc),
                micro_score=smth,
                micro_level=micro_expression_level(smth),
                mental_state=mental_est.infer(fid, bgr_patch, smth, sc),
                motion=mc_results.get(fid),
            ))

        now = time.time()
        fps = 1.0 / max(now - t_prev, 1e-9)
        t_prev = now

        renderer.draw(frame, results, fps)
        cv2.imshow(cfg.window_name, frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

        prev_gray = cur_gray

    cap.release()
    if mc_engine:
        mc_engine.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
