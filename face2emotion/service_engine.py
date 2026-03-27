import threading
import time

import cv2
import numpy as np

from .config import AppConfig
from .detector import FaceDetector
from .emotion import EmotionRecognizer
from .metrics import (
    camera_errors_total, engine_running, engine_stale,
    faces_detected, frames_total, inference_errors_total,
    inference_latency_seconds, processed_faces_total, realtime_fps,
)
from .micro_expression import ema, micro_expression_level, micro_expression_score
from .model_store import ensure_face_model
from .service_config import ServiceConfig
from .tracker import FaceTracker


class RealtimeInferenceEngine:
    """Inference engine running in a background thread.
    Public API: start / stop / restart / snapshot.
    """

    def __init__(self, config: ServiceConfig, logger):
        self.cfg = config
        self.log = logger
        self._mu   = threading.Lock()
        self._th:  threading.Thread | None = None
        self._quit = threading.Event()

        self._st: dict = {
            "running": False, "started_at": None, "last_frame_at": None,
            "fps": 0.0, "face_count": 0, "frame_count": 0, "faces": [],
            "last_error": None,
            "source_type": config.source_type,
            "source_value": config.source_value,
            "yolo_device": config.yolo_device,
        }

        ensure_face_model(self.cfg.model_path, self.log)

        # ServiceConfig字段和AppConfig不完全一致，手动做适配
        dcfg = AppConfig(
            camera_id=0, confidence=config.confidence, iou=config.iou,
            image_size=config.image_size, model_path=config.model_path,
            window_name="", show_fps=False, min_face_size=config.min_face_size,
            max_track_distance=config.max_track_distance,
            micro_ema_alpha=config.micro_ema_alpha, yolo_device=config.yolo_device,
        )
        self.detector   = FaceDetector(dcfg, logger=self.log)
        self.recognizer = EmotionRecognizer()
        self.tracker    = FaceTracker(config.max_track_distance)

    # ---- public ----

    def start(self):
        with self._mu:
            if self._th and self._th.is_alive():
                return
            self._quit.clear()
            self._st["last_error"] = None
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()

    def stop(self):
        self._quit.set()
        if self._th and self._th.is_alive():
            self._th.join(timeout=3)

    def restart(self):
        self.stop()
        self.start()

    def snapshot(self):
        with self._mu:
            lfa = self._st["last_frame_at"]
            sa  = self._st["started_at"]
            now = time.time()
            # 引擎在跑但超过stale_timeout没有新帧，视为卡住
            stale = self._st["running"] and (lfa is None or now - lfa > self.cfg.stale_timeout_sec)
            return {
                "running": self._st["running"], "stale": stale,
                "started_at": sa, "last_frame_at": lfa,
                "uptime_sec": (now - sa) if sa else 0.0,
                "fps": self._st["fps"], "face_count": self._st["face_count"],
                "frame_count": self._st["frame_count"],
                "faces": list(self._st["faces"]),
                "last_error": self._st["last_error"],
                "source_type": self._st["source_type"],
                "source_value": self._st["source_value"],
                "yolo_device": self._st["yolo_device"],
            }

    # ---- internal ----

    def _open(self):
        if self.cfg.source_type == "camera":
            # VideoCapture接受int或string，camera模式必须传int
            return cv2.VideoCapture(int(self.cfg.source_value))
        return cv2.VideoCapture(self.cfg.source_value)

    def _loop(self):
        prev_gray = None
        t_last    = time.time()
        mu_buf:  dict[int, float]             = {}
        emo_buf: dict[int, tuple[str, float]] = {}
        fidx   = 0
        dt_min = 1.0 / max(self.cfg.max_fps, 1.0)

        with self._mu:
            self._st["running"]    = True
            self._st["started_at"] = time.time()
        engine_running.set(1)
        self.log.info("engine start: %s=%s", self.cfg.source_type, self.cfg.source_value)

        cap = None
        while not self._quit.is_set():
            if cap is None or not cap.isOpened():
                cap = self._open()
                if not cap.isOpened():
                    camera_errors_total.inc()
                    self._err(f"cannot open: {self.cfg.source_type}={self.cfg.source_value}")
                    engine_stale.set(1)
                    time.sleep(self.cfg.reconnect_cooldown_sec)
                    continue
                self.log.info("source connected")

            t0 = time.time()
            try:
                ok, frame = cap.read()
                if not ok:
                    camera_errors_total.inc()
                    engine_stale.set(1)
                    self._err("frame read failed, reconnecting")
                    if self.cfg.source_type == "file":
                        self.log.info("EOF, engine stopping")
                        break
                    cap.release()
                    cap = None
                    time.sleep(self.cfg.reconnect_cooldown_sec)
                    continue

                frames_total.inc()
                fidx += 1
                self._clear_err()

                if self.cfg.mirror_input:
                    frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                boxes  = self.detector.detect(frame)
                tracks = self.tracker.assign(boxes)

                # 必须用alive_ids而非当帧tracks：tracker有_MISS_LIMIT容忍机制，
                # 短暂消失的轨迹尚未被evict，不能提前删掉它们的情绪/微表情缓存
                alive   = self.tracker.alive_ids
                mu_buf  = {k: v for k, v in mu_buf.items()  if k in alive}
                emo_buf = {k: v for k, v in emo_buf.items() if k in alive}
                self.recognizer.forget(alive)

                faces = []
                for tid, box in tracks:
                    processed_faces_total.inc()
                    pb = frame[box.y1:box.y2, box.x1:box.x2]
                    pg = gray[box.y1:box.y2, box.x1:box.x2]
                    pp = prev_gray[box.y1:box.y2, box.x1:box.x2] if prev_gray is not None else None

                    # 新轨迹首帧强制推理；之后按emotion_interval抽帧，降低GPU负载
                    if fidx % self.cfg.emotion_interval == 0 or tid not in emo_buf:
                        r = self.recognizer.predict(pb, track_id=tid)
                        emo_buf[tid] = (r.label, r.score)
                    lbl, sc = emo_buf[tid]

                    raw = micro_expression_score(pg, pp)
                    smu = ema(raw, mu_buf.get(tid), self.cfg.micro_ema_alpha)
                    mu_buf[tid] = smu

                    faces.append({
                        "track_id": tid,
                        "bbox": {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2},
                        "emotion": {"label": lbl, "score": sc},
                        "micro_expression": {"score": smu, "level": micro_expression_level(smu)},
                    })

                now    = time.time()
                fps    = 1.0 / max(now - t_last, 1e-6)
                t_last = now
                prev_gray = gray

                engine_stale.set(0)
                faces_detected.set(len(faces))
                realtime_fps.set(fps)
                inference_latency_seconds.observe(now - t0)

                with self._mu:
                    self._st["last_frame_at"] = now
                    self._st["fps"]           = fps
                    self._st["face_count"]    = len(faces)
                    self._st["frame_count"]  += 1
                    self._st["faces"]         = faces

            except Exception as e:
                inference_errors_total.inc()
                self.log.exception("inference error: %s", e)
                self._err(str(e))
                engine_stale.set(1)
                time.sleep(0.05)

            elapsed = time.time() - t0
            if elapsed < dt_min:
                time.sleep(dt_min - elapsed)

        if cap and cap.isOpened():
            cap.release()

        with self._mu:
            self._st["running"] = False
        engine_running.set(0)
        engine_stale.set(0)
        self.log.info("engine stopped")

    # ---- error helpers ----

    def _err(self, msg):
        with self._mu:
            self._st["last_error"] = msg

    def _clear_err(self):
        with self._mu:
            self._st["last_error"] = None
