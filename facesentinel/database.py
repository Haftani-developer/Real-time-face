"""Tiny on-disk face database: one .npz file with (label, embedding) pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


class FaceDB:
    def __init__(self, path):
        self.path = Path(path)
        self.labels = np.empty(0, dtype="<U64")
        self.embs = np.empty((0, 0), dtype=np.float32)
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as data:
                self.labels = data["labels"].astype("<U64")
                self.embs = data["embs"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def names(self) -> Dict[str, int]:
        """name -> number of stored samples"""
        out: Dict[str, int] = {}
        for label in self.labels:
            out[str(label)] = out.get(str(label), 0) + 1
        return dict(sorted(out.items()))

    def add(self, name: str, embeddings: Iterable[np.ndarray]) -> int:
        new = np.asarray(list(embeddings), dtype=np.float32)
        if new.size == 0:
            return 0
        self.embs = new.copy() if self.embs.size == 0 else np.vstack([self.embs, new])
        self.labels = np.concatenate([self.labels, np.array([name] * len(new), dtype="<U64")])
        return len(new)

    def remove(self, name: str) -> int:
        keep = self.labels != name
        removed = int((~keep).sum())
        if removed:
            self.labels, self.embs = self.labels[keep], self.embs[keep]
        return removed

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, labels=self.labels, embs=self.embs)
        tmp.replace(self.path)

    def best(self, feat: np.ndarray) -> Tuple[Optional[str], float]:
        """Closest stored person and cosine similarity, ignoring any threshold."""
        if len(self) == 0:
            return None, 0.0
        sims = self.embs @ feat
        i = int(np.argmax(sims))
        return str(self.labels[i]), float(sims[i])

    def match(self, feat: np.ndarray, threshold: float) -> Tuple[Optional[str], float]:
        name, score = self.best(feat)
        return (name, score) if name is not None and score >= threshold else (None, score)
