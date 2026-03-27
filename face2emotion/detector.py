from ultralytics import YOLO

from .config import AppConfig
from .schema import FaceBox


class FaceDetector:
    def __init__(self, config: AppConfig, logger=None) -> None:
        self.config = config
        self.logger = logger
        self.model = YOLO(str(config.model_path))
        self.device = self._resolve_device(config.yolo_device)

    def _resolve_device(self, preferred: str) -> str:
        try:
            import torch

            if preferred.startswith("cuda") and not torch.cuda.is_available():
                if self.logger:
                    self.logger.warning("CUDA 不可用，YOLO 自动回退到 CPU")
                return "cpu"

            if preferred.startswith("cuda"):
                if self.logger:
                    self.logger.info("YOLO 使用设备: %s (%s)", preferred, torch.cuda.get_device_name(0))
                return preferred

            if self.logger:
                self.logger.info("YOLO 使用设备: %s", preferred)
            return preferred
        except Exception:
            if self.logger and preferred.startswith("cuda"):
                self.logger.warning("Torch/CUDA 环境检测失败，YOLO 自动回退到 CPU")
            return "cpu" if preferred.startswith("cuda") else preferred

    def detect(self, frame) -> list[FaceBox]:
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence,
            iou=self.config.iou,
            verbose=False,
            imgsz=self.config.image_size,
            device=self.device,
        )

        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []
        face_boxes: list[FaceBox] = []

        h, w = frame.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            face = FaceBox(x1=x1, y1=y1, x2=x2, y2=y2)
            if face.width < self.config.min_face_size or face.height < self.config.min_face_size:
                continue
            face_boxes.append(face)

        return face_boxes
