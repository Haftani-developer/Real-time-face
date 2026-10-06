"""Image I/O that survives non-ASCII paths on Windows (cv2.imread/imwrite do not)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def imread(path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, img: np.ndarray) -> bool:
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return bool(ok)
