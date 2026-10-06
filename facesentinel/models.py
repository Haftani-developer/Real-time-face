"""Model files (YuNet detector + SFace recognizer from the OpenCV Model Zoo)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODELS = {
    "detector": (
        "face_detection_yunet_2023mar.onnx",
        f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
    "recognizer": (
        "face_recognition_sface_2021dec.onnx",
        f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
}

# Anything smaller is almost certainly an HTML error page or a Git-LFS pointer file.
_MIN_BYTES = 50_000


class ModelError(RuntimeError):
    """Raised when a model file is missing or cannot be downloaded."""


def default_dir() -> Path:
    return Path(os.environ.get("FACESENTINEL_MODELS", "models"))


def model_path(key: str, directory: Optional[os.PathLike] = None) -> Path:
    return (Path(directory) if directory else default_dir()) / MODELS[key][0]


def _manual_help(directory: Path) -> str:
    lines = ["Download these files manually and put them in '%s':" % directory]
    lines += [f"  {url}" for _, url in MODELS.values()]
    return "\n".join(lines)


def _download(url: str, dest: Path, quiet: bool) -> None:
    import urllib.request

    tmp = dest.with_name(dest.name + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "facesentinel"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if not quiet:
                    pct = f"{done * 100 // total:3d}%" if total else f"{done // 1024} KB"
                    sys.stdout.write(f"\r  {dest.name}  {pct}")
                    sys.stdout.flush()
        if not quiet:
            sys.stdout.write("\n")
        if tmp.stat().st_size < _MIN_BYTES:
            raise ModelError(f"Downloaded file for {dest.name} is too small to be a model.")
        tmp.replace(dest)
    except OSError as exc:  # URLError is an OSError
        raise ModelError(f"Could not download {dest.name}: {exc}\n{_manual_help(dest.parent)}") from exc
    finally:
        if tmp.exists():
            tmp.unlink()


def ensure_models(directory: Optional[os.PathLike] = None, quiet: bool = False) -> Dict[str, Path]:
    """Return paths of both models, downloading whichever is missing."""
    folder = Path(directory) if directory else default_dir()
    folder.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for key, (fname, url) in MODELS.items():
        path = folder / fname
        if not (path.exists() and path.stat().st_size >= _MIN_BYTES):
            if not quiet:
                print(f"Downloading {fname} ...")
            _download(url, path, quiet)
        paths[key] = path
    return paths
