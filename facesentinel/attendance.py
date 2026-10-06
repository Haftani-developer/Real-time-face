"""CSV attendance log with a per-person cooldown (opens cleanly in Excel, Persian names included)."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Optional


class AttendanceLog:
    def __init__(self, path, cooldown: float = 60.0):
        self.path = Path(path)
        self.cooldown = cooldown
        self._last: Dict[str, float] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8-sig") as fh:
                csv.writer(fh).writerow(["timestamp", "name", "similarity"])

    def mark(self, name: str, score: float, now: Optional[float] = None) -> bool:
        """Log a sighting unless this person was already logged within the cooldown."""
        now = time.time() if now is None else now
        if now - self._last.get(name, float("-inf")) < self.cooldown:
            return False
        self._last[name] = now
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        with open(self.path, "a", newline="", encoding="utf-8-sig") as fh:
            csv.writer(fh).writerow([stamp, name, f"{score:.3f}"])
        return True
