"""Drawing: sci-fi corner brackets, landmarks, labels, stats panel and privacy filters."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
UNKNOWN_COLOR = (70, 70, 255)  # BGR
PRIVACY_MODES = ("off", "pixelate", "blur")

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]


def color_for(name: Optional[str]) -> Tuple[int, int, int]:
    """Stable neon colour per person; unknown faces are red."""
    if name is None:
        return UNKNOWN_COLOR
    hue = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:4], 16) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ----------------------------------------------------------------------------- text
def _is_ascii(text: str) -> bool:
    return all(ord(c) < 128 for c in text)


@lru_cache(maxsize=8)
def _load_font(font_path: Optional[str], px: int):
    from PIL import ImageFont

    for path in ([font_path] if font_path else []) + _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    raise RuntimeError("no Unicode-capable TTF font found; pass --font /path/to/font.ttf")


def _unicode_patch(text: str, fg, bg, font_path, px: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    font = _load_font(font_path, px)
    try:  # proper Persian/Arabic shaping + right-to-left, if the optional extras are installed
        import arabic_reshaper
        from bidi.algorithm import get_display

        text = get_display(arabic_reshaper.reshape(text))
    except Exception:
        pass
    left, top, right, bottom = font.getbbox(text)
    img = Image.new("RGB", (right - left + 14, bottom - top + 10), (bg[2], bg[1], bg[0]))
    ImageDraw.Draw(img).text((7 - left, 5 - top), text, font=font, fill=(fg[2], fg[1], fg[0]))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


@lru_cache(maxsize=256)
def render_label(text: str, fg: tuple, bg: tuple, font_path: Optional[str] = None,
                 scale: float = 0.55) -> np.ndarray:
    """Render text into a small BGR patch (OpenCV's putText cannot draw Persian/Unicode)."""
    if not _is_ascii(text):
        try:
            return _unicode_patch(text, fg, bg, font_path, int(30 * scale) + 2)
        except Exception:
            text = "".join(c if ord(c) < 128 else "?" for c in text)
    (w, h), base = cv2.getTextSize(text, FONT, scale, 1)
    patch = np.full((h + base + 10, w + 14, 3), bg, dtype=np.uint8)
    cv2.putText(patch, text, (7, h + 5), FONT, scale, fg, 1, cv2.LINE_AA)
    return patch


def overlay(img: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Paste `patch` at (x, y), clipped to the image."""
    ih, iw = img.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(iw, x + pw), min(ih, y + ph)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = patch[y0 - y:y1 - y, x0 - x:x1 - x]


# ------------------------------------------------------------------------- shapes
def draw_brackets(img, x, y, w, h, color, thickness=2) -> None:
    length = int(max(10, min(w, h) * 0.22))
    for cx, dx in ((x, 1), (x + w, -1)):
        for cy, dy in ((y, 1), (y + h, -1)):
            cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)


def apply_privacy(img: np.ndarray, box: Sequence[float], mode: str) -> None:
    """Pixelate or blur a face region in place (with a little padding)."""
    if mode == "off":
        return
    ih, iw = img.shape[:2]
    x, y, w, h = [int(v) for v in box]
    pad = int(0.12 * max(w, h))
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(iw, x + w + pad), min(ih, y + h + pad)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return
    rh, rw = roi.shape[:2]
    if mode == "blur":
        k = max(3, (rw // 2) | 1)
        img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    else:
        block = max(8, min(rw, rh) // 10)
        small = cv2.resize(roi, (max(1, rw // block), max(1, rh // block)), interpolation=cv2.INTER_LINEAR)
        img[y0:y1, x0:x1] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)


def _panel(img, lines, color=(0, 255, 160)) -> None:
    pad, lh = 10, 20
    width = max(cv2.getTextSize(s, FONT, 0.5, 1)[0][0] for s in lines) + 2 * pad
    height = lh * len(lines) + pad
    height, width = min(height, img.shape[0]), min(width, img.shape[1])
    roi = img[8:8 + height, 8:8 + width]
    roi[:] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (8 + pad, 8 + pad + lh * i + 6), FONT, 0.5,
                    color if i == 0 else (235, 235, 235), 1, cv2.LINE_AA)


# -------------------------------------------------------------------------- main
def render(frame: np.ndarray, tracks, info: Optional[Dict] = None, privacy: str = "off",
           landmarks: bool = True, show_hud: bool = True, font_path: Optional[str] = None) -> np.ndarray:
    """Annotate `frame` in place and return it. `tracks` need .id, .face and .identity()."""
    known = unknown = 0
    resolved = []
    for t in tracks:
        name, score = t.identity()
        resolved.append((t, name, score))
        if name is None:
            unknown += 1
            apply_privacy(frame, t.face[:4], privacy)   # always before drawing anything
        else:
            known += 1

    for t, name, score in resolved:
        x, y, w, h = [int(round(v)) for v in t.face[:4]]
        color = color_for(name)
        draw_brackets(frame, x, y, w, h, color, 2)
        if landmarks:
            for i in range(5):
                px, py = int(t.face[4 + 2 * i]), int(t.face[5 + 2 * i])
                cv2.circle(frame, (px, py), max(2, w // 40), (0, 255, 255), -1, cv2.LINE_AA)
        text = f"#{t.id}  {name}  {int(score * 100)}%" if name else f"#{t.id}  UNKNOWN"
        dark = (20, 20, 20)
        patch = render_label(text, dark, color, font_path)
        ly = y - patch.shape[0] - 2
        overlay(frame, patch, x, ly if ly >= 0 else y + h + 2)

    if show_hud:
        info = info or {}
        lines = ["FACESENTINEL"]
        if "fps" in info:
            lines.append(f"FPS      {info['fps']:5.1f}")
        if "ms" in info:
            lines.append(f"Latency  {info['ms']:5.1f} ms")
        lines += [f"Faces    {len(resolved)}", f"Known    {known}", f"Unknown  {unknown}",
                  f"Privacy  {privacy}"]
        if info.get("recording"):
            lines.append("REC")
        _panel(frame, lines)
    return frame
