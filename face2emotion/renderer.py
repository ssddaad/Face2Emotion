import cv2
import numpy as np

from .config import AppConfig
from .schema import FaceInference

_PALETTE = [
    (  0, 255, 128), (  0, 200, 255), (255, 200,   0), (255,  80,  80),
    (180, 100, 255), (255, 255,   0), ( 80, 255, 200), (255, 140,   0),
]

_F  = cv2.FONT_HERSHEY_PLAIN
_FS = 1.0
_LH = 14


def _txt(img, s, x, y, col=(200, 230, 200)):
    cv2.putText(img, s, (x, y), _F, _FS, col, 1, cv2.LINE_AA)


class Renderer:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def draw(self, frame, faces, fps):
        for f in faces:
            self._draw_face(frame, f)
        if self.cfg.show_fps:
            self._draw_fps(frame, fps)

    def _draw_face(self, frame, inf: FaceInference):
        b   = inf.box
        col = _PALETTE[inf.track_id % len(_PALETTE)]

        cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), col, 1)

        tag = f"[{inf.track_id}] {inf.emotion.label} {inf.emotion.score:.2f}"
        _txt(frame, tag, b.x1, max(b.y1 - _LH, _LH), col)

        _txt(frame, f"mu={inf.micro_score:.1f} {inf.micro_level}",
             b.x1, max(b.y1 - 2, _LH * 2), (140, 190, 140))

    @staticmethod
    def _draw_fps(frame, fps):
        h, w = frame.shape[:2]
        s = f"{fps:.1f} fps"
        (tw, _), _ = cv2.getTextSize(s, _F, _FS, 1)
        _txt(frame, s, w - tw - 8, 16, (160, 160, 160))
