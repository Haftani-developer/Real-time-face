"""Face embeddings with SFace (cv2.FaceRecognizerSF)."""

from __future__ import annotations

import cv2
import numpy as np

# Cosine-similarity threshold recommended by the OpenCV Model Zoo for SFace.
DEFAULT_THRESHOLD = 0.363


class FaceRecognizer:
    def __init__(self, model_path=None, engine=None):
        if engine is None:
            engine = cv2.FaceRecognizerSF.create(str(model_path), "")
        self.engine = engine

    def embed(self, frame: np.ndarray, row: np.ndarray) -> np.ndarray:
        """Align the face using its landmarks and return an L2-normalised embedding."""
        row = np.ascontiguousarray(row, dtype=np.float32)
        aligned = self.engine.alignCrop(frame, row)
        feat = np.asarray(self.engine.feature(aligned), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(feat))
        return feat / norm if norm > 0 else feat
