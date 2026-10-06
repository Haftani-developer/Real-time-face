"""Glue: detect -> track -> recognise (re-verified periodically) -> vote."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .database import FaceDB
from .detector import FaceDetector
from .recognizer import DEFAULT_THRESHOLD, FaceRecognizer
from .tracker import Track, Tracker


@dataclass
class Config:
    models_dir: str = ""
    db_path: str = "data/faces.npz"
    det_score: float = 0.7
    scale: float = 1.0           # <1.0 detects on a downscaled frame (faster)
    threshold: float = DEFAULT_THRESHOLD
    min_face: int = 40           # px; smaller faces are tracked but not recognised
    rec_every: int = 15          # re-verify identity every N frames
    max_age: int = 20            # frames a lost track survives


class Pipeline:
    def __init__(self, detector: FaceDetector, recognizer: FaceRecognizer, db: FaceDB, cfg: Config):
        self.detector, self.recognizer, self.db, self.cfg = detector, recognizer, db, cfg
        self.tracker = Tracker(max_age=cfg.max_age)

    @classmethod
    def build(cls, cfg: Config, quiet: bool = False) -> "Pipeline":
        from .models import ensure_models

        paths = ensure_models(cfg.models_dir or None, quiet=quiet)
        return cls(
            FaceDetector(paths["detector"], score=cfg.det_score, scale=cfg.scale),
            FaceRecognizer(paths["recognizer"]),
            FaceDB(cfg.db_path),
            cfg,
        )

    def detect(self, frame: np.ndarray) -> np.ndarray:
        return self.detector.detect(frame)

    def identify(self, frame: np.ndarray, row: np.ndarray):
        return self.db.match(self.recognizer.embed(frame, row), self.cfg.threshold)

    def process(self, frame: np.ndarray, frame_idx: int) -> List[Track]:
        """Video mode: tracked faces with smoothed boxes and stable identities."""
        tracks = self.tracker.update(self.detect(frame))
        for t in tracks:
            if t.raw[2] >= self.cfg.min_face and t.needs_recognition(frame_idx, self.cfg.rec_every):
                name, score = self.identify(frame, t.raw)
                t.add_vote(name, score, frame_idx)
        return tracks

    def analyze(self, frame: np.ndarray) -> List[Track]:
        """Still-image mode: no tracking, every face identified once."""
        out = []
        for i, row in enumerate(self.detect(frame), 1):
            name, score = (None, 0.0)
            if row[2] >= self.cfg.min_face:
                name, score = self.identify(frame, row)
            out.append(Track.static(row, name, score, tid=i))
        return out
