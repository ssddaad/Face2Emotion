import threading

import time
import cv2
import numpy as np

from .action_classifier import ActionClassifier
from .config import AppConfig
from .detector import FaceDetector
from .emotion import EmotionRecognizer
from .gesture import apply_gesture
from .metrics import (
    camera_errors_total, engine_running, engine_stale,
    faces_detected, frames_total, inference_errors_total,
    inference_latency_seconds, processed_faces_total, realtime_fps,
    motion_capture_errors_total, motion_capture_latency_seconds,
    pose_detected, actions_classified, gestures_classified,
)
from .mental_state import MentalStateEstimator
from .micro_expression import ema, micro_expression_level, micro_expression_score, set_roi_weights
from .model_store import ensure_face_model
from .motion_capture import MotionCaptureEngine
from .service_config import ServiceConfig
from .tracker import FaceTracker


class RealtimeInferenceEngine:
    # 后台线程推理引擎
    def __init__(self, config: ServiceConfig, logger):
        self.cfg=config
        self.log=logger
        self._mu=threading.Lock()
        self._th: threading.Thread | None=None
        self._quit=threading.Event()
        self._st: dict={
            "running": False, "started_at": None, "last_frame_at": None,
            "fps": 0.0, "face_count": 0, "frame_count": 0, "faces": [],
            "last_error": None,
            "source_type": config.source_type,
            "source_value": config.source_value,
            "yolo_device": config.yolo_device,
            "motion_capture_enabled": config.enable_motion_capture,
        }
        ensure_face_model(self.cfg.model_path, self.log)
        dcfg=AppConfig(
            camera_id=0, confidence=config.confidence, iou=config.iou,
            image_size=config.image_size, model_path=config.model_path,
            window_name="", show_fps=False, min_face_size=config.min_face_size,
            max_track_distance=config.max_track_distance,
            micro_ema_alpha=config.micro_ema_alpha,
            micro_roi_brow_weight=config.micro_roi_brow_weight,
            micro_roi_eye_weight=config.micro_roi_eye_weight,
            micro_roi_mouth_weight=config.micro_roi_mouth_weight,
            yolo_device=config.yolo_device,
            enable_motion_capture=config.enable_motion_capture,
            motion_complexity=config.motion_complexity,
            motion_vote_window=config.motion_vote_window,
            motion_interval=config.motion_interval,
            motion_landmark_ema_alpha=config.motion_landmark_ema_alpha,
        )
        self.detector=FaceDetector(dcfg, logger=self.log)
        self.recognizer=EmotionRecognizer()
        set_roi_weights(
            config.micro_roi_brow_weight,
            config.micro_roi_eye_weight,
            config.micro_roi_mouth_weight,
        )
        self.log.info(
            "micro ROI weights: brow=%.2f eye=%.2f mouth=%.2f",
            config.micro_roi_brow_weight,
            config.micro_roi_eye_weight,
            config.micro_roi_mouth_weight,
        )
        self.tracker=FaceTracker(config.max_track_distance)
        self.mental_est=MentalStateEstimator()
        self.mc_engine: MotionCaptureEngine | None=None
        self.ac_classify: ActionClassifier | None=None
        if config.enable_motion_capture:
            self.mc_engine=MotionCaptureEngine(
                model_complexity=config.motion_complexity,
                landmark_ema_alpha=config.motion_landmark_ema_alpha,
            )
            self.ac_classify=ActionClassifier(vote_window=config.motion_vote_window)
            self.log.info(
                "motion capture enabled (complexity=%d, interval=%d)",
                config.motion_complexity,
                config.motion_interval,
            )

    def start(self):
        with self._mu:
            if self._th and self._th.is_alive():
                return
            self._quit.clear()
            self._st["last_error"]=None
            self._th=threading.Thread(target=self._loop, daemon=True)
            self._th.start()

    def stop(self):
        self._quit.set()
        if self._th and self._th.is_alive():
            self._th.join(timeout=3)
        if self.mc_engine:
            self.mc_engine.close()

    def restart(self):
        self.stop()
        if self.cfg.enable_motion_capture:
            self.mc_engine=MotionCaptureEngine(
                model_complexity=self.cfg.motion_complexity,
                landmark_ema_alpha=self.cfg.motion_landmark_ema_alpha,
            )
        self.start()

    def snapshot(self):
        with self._mu:
            lfa=self._st["last_frame_at"]
            sa=self._st["started_at"]
            now=time.time()
            stale=self._st["running"] and (lfa is None or now-lfa>self.cfg.stale_timeout_sec)
            return {
                "running": self._st["running"], "stale": stale,
                "started_at": sa, "last_frame_at": lfa,
                "uptime_sec": (now-sa) if sa else 0.0,
                "fps": self._st["fps"], "face_count": self._st["face_count"],
                "frame_count": self._st["frame_count"],
                "faces": list(self._st["faces"]),
                "last_error": self._st["last_error"],
                "source_type": self._st["source_type"],
                "source_value": self._st["source_value"],
                "yolo_device": self._st["yolo_device"],
                "motion_capture_enabled": self._st["motion_capture_enabled"],
            }

    def _open(self):
        if self.cfg.source_type=="camera":
            return cv2.VideoCapture(int(self.cfg.source_value))
        return cv2.VideoCapture(self.cfg.source_value)

    def _loop(self):
        prev_gray=None
        t_last=time.time()
        mu_buf:  dict[int, float]            ={}
        emo_buf: dict[int, tuple[str, float]]={}
        fidx=0
        dt_min=1.0/max(self.cfg.max_fps, 1.0)
        mc_last=None
        with self._mu:
            self._st["running"]=True
            self._st["started_at"]=time.time()
        engine_running.set(1)
        self.log.info("engine start: %s=%s", self.cfg.source_type, self.cfg.source_value)
        cap=None
        while not self._quit.is_set():
            if cap is None or not cap.isOpened():
                cap=self._open()
                if not cap.isOpened():
                    camera_errors_total.inc()
                    self._err(f"cannot open: {self.cfg.source_type}={self.cfg.source_value}")
                    engine_stale.set(1)
                    time.sleep(self.cfg.reconnect_cooldown_sec)
                    continue
                self.log.info("source connected")
            t0=time.time()
            try:
                ok, frame=cap.read()
                if not ok:
                    camera_errors_total.inc()
                    engine_stale.set(1)
                    self._err("frame read failed, reconnecting")
                    if self.cfg.source_type=="file":
                        self.log.info("EOF, engine stopping")
                        break
                    cap.release(); cap=None
                    time.sleep(self.cfg.reconnect_cooldown_sec)
                    continue
                frames_total.inc()
                fidx+=1
                self._clear_err()
                if self.cfg.mirror_input:
                    frame=cv2.flip(frame, 1)
                gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes=self.detector.detect(frame)
                tracks=self.tracker.assign(boxes)
                alive=self.tracker.alive_ids
                mu_buf={k: v for k, v in mu_buf.items()  if k in alive}
                emo_buf={k: v for k, v in emo_buf.items() if k in alive}
                self.recognizer.forget(alive)
                if self.ac_classify:
                    self.ac_classify.forget(alive)
                if self.mc_engine:
                    self.mc_engine.forget(alive)
                self.mental_est.forget(alive)
                # 动作捕捉（按 interval 执行，降低追踪延迟）
                mc_results: dict[int, object]={}
                if self.mc_engine is not None:
                    t_mc=time.time()
                    try:
                        if fidx % self.cfg.motion_interval==0 or mc_last is None:
                            primary_tid=max(
                                tracks,
                                key=lambda item: (item[1].x2-item[1].x1)*(item[1].y2-item[1].y1),
                            )[0] if tracks else -1
                            mc_last=self.mc_engine.process(frame, track_id=primary_tid)
                            if mc_last is not None:
                                if mc_last.left_hand:
                                    apply_gesture(mc_last.left_hand)
                                    gestures_classified.labels(gesture=mc_last.left_hand.gesture, side="left").inc()
                                if mc_last.right_hand:
                                    apply_gesture(mc_last.right_hand)
                                    gestures_classified.labels(gesture=mc_last.right_hand.gesture, side="right").inc()
                                if mc_last.pose and self.ac_classify:
                                    self.ac_classify.apply(mc_last.pose, track_id=primary_tid)
                                    actions_classified.labels(action=mc_last.pose.action_label).inc()
                        if mc_last is not None:
                            for tid, _ in tracks:
                                mc_results[tid]=mc_last
                        motion_capture_latency_seconds.observe(time.time()-t_mc)
                        pose_detected.set(1 if mc_last is not None and mc_last.pose else 0)
                    except Exception as me:
                        motion_capture_errors_total.inc()
                        self.log.warning("motion capture error: %s", me)
                # 情绪 + 微表情
                faces=[]
                for tid, box in tracks:
                    processed_faces_total.inc()
                    pb=frame[box.y1:box.y2, box.x1:box.x2]
                    pg=gray[box.y1:box.y2, box.x1:box.x2]
                    pp=prev_gray[box.y1:box.y2, box.x1:box.x2] if prev_gray is not None else None
                    if fidx % self.cfg.emotion_interval==0 or tid not in emo_buf:
                        r=self.recognizer.predict(pb, track_id=tid)
                        emo_buf[tid]=(r.label, r.score)
                    lbl, sc=emo_buf[tid]
                    raw=micro_expression_score(pg, pp)
                    smu=ema(raw, mu_buf.get(tid), self.cfg.micro_ema_alpha)
                    mu_buf[tid]=smu
                    mc_serial=_serialize_mc(mc_results.get(tid))
                    faces.append({
                        "track_id": tid,
                        "bbox": {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2},
                        "emotion": {"label": lbl, "score": sc},
                        "micro_expression": {"score": smu, "level": micro_expression_level(smu)},
                        "mental_state": _serialize_mental(self.mental_est.infer(tid, pb, smu, sc)),
                        "motion_capture": mc_serial,
                    })
                now=time.time()
                fps=1.0/max(now-t_last, 1e-6)
                t_last=now
                prev_gray=gray
                engine_stale.set(0)
                faces_detected.set(len(faces))
                realtime_fps.set(fps)
                inference_latency_seconds.observe(now-t0)
                with self._mu:
                    self._st["last_frame_at"]=now
                    self._st["fps"]=fps
                    self._st["face_count"]=len(faces)
                    self._st["frame_count"]+=1
                    self._st["faces"]=faces
            except Exception as e:
                inference_errors_total.inc()
                self.log.exception("inference error: %s", e)
                self._err(str(e))
                engine_stale.set(1)
                time.sleep(0.05)
            elapsed=time.time()-t0
            if elapsed<dt_min:
                time.sleep(dt_min-elapsed)
        if cap and cap.isOpened():
            cap.release()
        with self._mu:
            self._st["running"]=False
        engine_running.set(0)
        engine_stale.set(0)
        self.log.info("engine stopped")

    def _err(self, msg):
        with self._mu:
            self._st["last_error"]=msg

    def _clear_err(self):
        with self._mu:
            self._st["last_error"]=None


def _serialize_mental(ms) -> dict:
    return {
        "score": round(ms.score, 2),
        "level": ms.level,
        "color_vitality": round(ms.color_vitality, 2),
        "eye_focus": round(ms.eye_focus, 2),
        "facial_energy": round(ms.facial_energy, 2),
        "trend_score": round(ms.trend_score, 2),
        "risk_level": ms.risk_level,
        "perclos": round(ms.perclos, 4),
        "blink_rate": round(ms.blink_rate, 2),
    }


# 序列化辅助（将 dataclass 转为 JSON 兼容 dict）
def _serialize_mc(mc_res) -> dict | None:
    if mc_res is None:
        return None
    out: dict={"timestamp": mc_res.timestamp}
    if mc_res.pose:
        p=mc_res.pose
        out["pose"]={
            "action_label": p.action_label,
            "action_conf":  round(p.action_conf,  4),
            "pose_quality": round(p.pose_quality, 4),
            "velocity": {
                "wrist_left_speed":  round(p.velocity.wrist_left_speed,  2),
                "wrist_right_speed": round(p.velocity.wrist_right_speed, 2),
                "ankle_left_speed":  round(p.velocity.ankle_left_speed,  2),
                "ankle_right_speed": round(p.velocity.ankle_right_speed, 2),
                "overall_motion":    round(p.velocity.overall_motion,    2),
            },
            "joint_angles": [
                {"name": a.name, "angle_deg": round(a.angle_deg, 2), "confidence": round(a.confidence, 3)}
                for a in p.joint_angles
            ],
            "landmarks": [
                {"name": lm.name, "x": round(lm.x_norm, 4), "y": round(lm.y_norm, 4),
                 "z": round(lm.z_norm, 4), "visibility": round(lm.visibility, 3)}
                for lm in p.landmarks
            ],
        }
    else:
        out["pose"]=None
    for side, hand in [("left", mc_res.left_hand), ("right", mc_res.right_hand)]:
        if hand:
            out[f"{side}_hand"]={
                "gesture":    hand.gesture,
                "confidence": round(hand.confidence, 3),
                "landmarks": [
                    {"name": lm.name, "x": round(lm.x_norm, 4), "y": round(lm.y_norm, 4)}
                    for lm in hand.landmarks
                ],
            }
        else:
            out[f"{side}_hand"]=None
    return out
