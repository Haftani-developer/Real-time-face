"""IoU face tracker with smoothed boxes and identity voting (no more flickering names)."""

from __future__ import annotations

from collections import Counter, deque
from typing import List, Optional, Tuple

import numpy as np


def iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


class Track:
    VOTES = 7

    def __init__(self, tid: int, row: np.ndarray):
        self.id = tid
        self.raw = np.array(row, dtype=np.float32)    # latest detection, for accurate alignment
        self.face = np.array(row, dtype=np.float32)   # smoothed copy, for drawing
        self.missed = 0
        self.hits = 1
        self.last_rec = -10 ** 9
        self.votes: deque = deque(maxlen=self.VOTES)

    @classmethod
    def static(cls, row, name: Optional[str], score: float, tid: int = 0) -> "Track":
        track = cls(tid, row)
        track.votes.append((name, score))
        return track

    @property
    def box(self) -> np.ndarray:
        return self.face[:4]

    def update(self, row: np.ndarray, alpha: float) -> None:
        row = np.asarray(row, dtype=np.float32)
        self.raw = row.copy()
        self.face[:14] = alpha * row[:14] + (1.0 - alpha) * self.face[:14]
        self.face[14] = row[14]
        self.missed = 0
        self.hits += 1

    def add_vote(self, name: Optional[str], score: float, frame_idx: int) -> None:
        self.votes.append((name, score))
        self.last_rec = frame_idx

    def needs_recognition(self, frame_idx: int, every: int) -> bool:
        return len(self.votes) < 3 or frame_idx - self.last_rec >= every

    def identity(self) -> Tuple[Optional[str], float]:
        """Majority vote over the last few recognitions -> (name or None, mean similarity)."""
        if not self.votes:
            return None, 0.0
        winner, _ = Counter(v[0] for v in self.votes).most_common(1)[0]
        scores = [v[1] for v in self.votes if v[0] == winner]
        return winner, float(np.mean(scores))


class Tracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 20, smoothing: float = 0.6):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.alpha = smoothing
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, rows: np.ndarray) -> List[Track]:
        rows = np.asarray(rows, dtype=np.float32).reshape(-1, 15)
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, row in enumerate(rows):
                value = iou(track.box, row[:4])
                if value >= self.iou_threshold:
                    pairs.append((value, ti, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            self.tracks[ti].update(rows[di], self.alpha)
            used_t.add(ti)
            used_d.add(di)

        for ti, track in enumerate(self.tracks):
            if ti not in used_t:
                track.missed += 1
        for di, row in enumerate(rows):
            if di not in used_d:
                self.tracks.append(Track(self._next_id, row))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]
        return [t for t in self.tracks if t.missed == 0]
