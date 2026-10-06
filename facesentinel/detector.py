"""Face detection with YuNet (cv2.FaceDetectorYN).

Every detection is a float32 row of 15 values:
    x, y, w, h,  right-eye(x,y), left-eye(x,y), nose(x,y),
    right-mouth(x,y), left-mouth(x,y),  score
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

EMPTY = np.empty((0, 15), dtype=np.float32)


class FaceDetector:
    def __init__(self, model_path=None, score: float = 0.7, nms: float = 0.3,
                 top_k: int = 5000, scale: float = 1.0, engine=None):
        if engine is None:
            engine = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score, nms, top_k)
        self.engine = engine
        self.scale = float(scale)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect faces; results are always in the coordinates of the original frame."""
        h, w = frame.shape[:2]
        img = frame
        if self.scale != 1.0:
            img = cv2.resize(frame, (max(1, int(w * self.scale)), max(1, int(h * self.scale))),
                             interpolation=cv2.INTER_AREA)
        ih, iw = img.shape[:2]
        self.engine.setInputSize((iw, ih))
        _, faces = self.engine.detect(img)
        if faces is None or len(faces) == 0:
            return EMPTY.copy()

        faces = np.array(faces, dtype=np.float32)
        if (iw, ih) != (w, h):
            faces[:, 0:14:2] *= w / iw
            faces[:, 1:14:2] *= h / ih
        order = np.argsort(-(faces[:, 2] * faces[:, 3]))  # biggest face first
        return faces[order]


def face_quality(row: np.ndarray, min_size: int = 80, min_score: float = 0.85,
                 max_yaw: float = 0.35) -> Tuple[bool, str]:
    """Is this face good enough to enroll? Returns (ok, human-readable reason)."""
    if row[2] < min_size or row[3] < min_size:
        return False, "move closer"
    if row[14] < min_score:
        return False, "face not clear"
    eye_r, eye_l, nose = row[4:6], row[6:8], row[8:10]
    eye_dist = abs(eye_l[0] - eye_r[0])
    if eye_dist < 1:
        return False, "face not clear"
    yaw = (nose[0] - (eye_r[0] + eye_l[0]) / 2.0) / eye_dist
    if abs(yaw) > max_yaw:
        return False, "look at the camera"
    return True, "good"
