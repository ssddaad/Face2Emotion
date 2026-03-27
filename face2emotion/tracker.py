import math
from typing import List, Tuple

from .schema import FaceBox

_MISS_LIMIT = 5  # 连续消失超过此帧数才真正丢弃轨迹，容忍短暂遮挡


class FaceTracker:
    """
    Greedy nearest-neighbor ID tracker.
    候选配对按距离升序排列后贪心匹配，保证最近的配对优先处理，
    效果接近最优二分匹配且实现简单。
    """

    def __init__(self, max_dist: float):
        self.max_dist = max_dist
        self._nid  = 1
        self._ctrs: dict[int, Tuple[float, float]] = {}
        self._miss: dict[int, int] = {}

    @property
    def alive_ids(self) -> set[int]:
        return set(self._ctrs.keys())

    def assign(self, boxes: List[FaceBox]) -> List[Tuple[int, FaceBox]]:
        self._evict()

        if not boxes:
            for tid in list(self._ctrs):
                self._miss[tid] = self._miss.get(tid, 0) + 1
            return []

        if not self._ctrs:
            return self._register(boxes)

        old = list(self._ctrs.keys())
        cands = []
        for bi, box in enumerate(boxes):
            cx, cy = box.center
            for tid in old:
                px, py = self._ctrs[tid]
                d = math.hypot(cx - px, cy - py)
                if d <= self.max_dist:
                    cands.append((d, bi, tid))
        # tuple自然排序，第一元素是距离，直接sort即可
        cands.sort()

        used_b: set[int] = set()
        used_t: set[int] = set()
        out: list[tuple[int, FaceBox]] = []

        for _, bi, tid in cands:
            if bi in used_b or tid in used_t:
                continue
            used_b.add(bi); used_t.add(tid)
            out.append((tid, boxes[bi]))
            self._miss[tid] = 0

        for bi, box in enumerate(boxes):
            if bi not in used_b:
                nid = self._nid; self._nid += 1
                out.append((nid, box))
                self._miss[nid] = 0

        for tid in old:
            if tid not in used_t:
                self._miss[tid] = self._miss.get(tid, 0) + 1

        for tid, box in out:
            self._ctrs[tid] = box.center

        return out

    def _evict(self):
        dead = [t for t, c in self._miss.items() if c >= _MISS_LIMIT]
        for t in dead:
            self._ctrs.pop(t, None)
            self._miss.pop(t, None)

    def _register(self, boxes):
        out = []
        for box in boxes:
            tid = self._nid; self._nid += 1
            self._ctrs[tid] = box.center
            self._miss[tid] = 0
            out.append((tid, box))
        return out
