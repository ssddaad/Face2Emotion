from dataclasses import dataclass


@dataclass(slots=True)
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(slots=True)
class EmotionResult:
    label: str
    score: float


@dataclass(slots=True)
class FaceInference:
    track_id: int
    box: FaceBox
    emotion: EmotionResult
    micro_score: float
    micro_level: str
