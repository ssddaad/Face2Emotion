import math
from .schema import FaceBox

# 连续消失超过此帧数才真正丢弃轨迹，容忍短暂遮挡
_MISS_LIMIT = 5


class FaceTracker:
    # 贪心最近邻 ID 跟踪器（简版 Kalman：常速度预测 + 尺度约束）
    def __init__(self, max_dist: float):
        self.max_dist = max_dist
        self._nid = 1
        self._ctrs: dict[int, tuple[float, float]] = {}
        self._sizes: dict[int, float] = {}
        self._vel: dict[int, tuple[float, float]] = {}
        self._miss: dict[int, int] = {}

    @property
    def alive_ids(self) -> set[int]:
        return set(self._ctrs.keys())

    def assign(self, boxes: list[FaceBox]) -> list[tuple[int, FaceBox]]:
        self._evict()
        if not boxes:
            for tid in list(self._ctrs):
                self._miss[tid] = self._miss.get(tid, 0) + 1
                vx, vy = self._vel.get(tid, (0.0, 0.0))
                self._vel[tid] = (vx * 0.82, vy * 0.82)
            return []

        if not self._ctrs:
            return self._register(boxes)

        old = list(self._ctrs.keys())
        cands = []
        for bi, box in enumerate(boxes):
            cx, cy = box.center
            area = float(max(1, box.width * box.height))
            for tid in old:
                px, py = self._predict_center(tid)
                prev_area = self._sizes.get(tid, area)

                ratio = area / max(prev_area, 1.0)
                if ratio < 0.45 or ratio > 2.2:
                    continue

                d = math.hypot(cx - px, cy - py)
                dynamic_dist = max(self.max_dist, 0.12 * math.sqrt(prev_area))
                if d > dynamic_dist:
                    continue

                size_penalty = abs(math.log(max(ratio, 1e-6)))
                score = d + 22.0 * size_penalty
                cands.append((score, bi, tid))

        cands.sort()
        used_b: set[int] = set()
        used_t: set[int] = set()
        out: list[tuple[int, FaceBox]] = []

        for _, bi, tid in cands:
            if bi in used_b or tid in used_t:
                continue
            used_b.add(bi)
            used_t.add(tid)
            out.append((tid, boxes[bi]))
            self._miss[tid] = 0

        for bi, box in enumerate(boxes):
            if bi not in used_b:
                nid = self._nid
                self._nid += 1
                out.append((nid, box))
                self._miss[nid] = 0

        for tid in old:
            if tid not in used_t:
                self._miss[tid] = self._miss.get(tid, 0) + 1
                vx, vy = self._vel.get(tid, (0.0, 0.0))
                self._vel[tid] = (vx * 0.86, vy * 0.86)

        for tid, box in out:
            prev = self._ctrs.get(tid, box.center)
            cur = box.center
            obs_vx = cur[0] - prev[0]
            obs_vy = cur[1] - prev[1]
            pvx, pvy = self._vel.get(tid, (0.0, 0.0))
            self._vel[tid] = (0.65 * pvx + 0.35 * obs_vx, 0.65 * pvy + 0.35 * obs_vy)
            self._ctrs[tid] = cur
            self._sizes[tid] = float(max(1, box.width * box.height))

        return out

    def _predict_center(self, tid: int) -> tuple[float, float]:
        cx, cy = self._ctrs[tid]
        vx, vy = self._vel.get(tid, (0.0, 0.0))
        # 遮挡缺失帧越多，预测步长越大
        step = 1.0 + min(2.5, float(self._miss.get(tid, 0)))
        return cx + vx * step, cy + vy * step

    def _evict(self):
        dead = [t for t, c in self._miss.items() if c >= _MISS_LIMIT]
        for t in dead:
            self._ctrs.pop(t, None)
            self._sizes.pop(t, None)
            self._vel.pop(t, None)
            self._miss.pop(t, None)

    def _register(self, boxes):
        out = []
        for box in boxes:
            tid = self._nid
            self._nid += 1
            self._ctrs[tid] = box.center
            self._sizes[tid] = float(max(1, box.width * box.height))
            self._vel[tid] = (0.0, 0.0)
            self._miss[tid] = 0
            out.append((tid, box))
        return out
