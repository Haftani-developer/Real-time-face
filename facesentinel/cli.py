"""Command line interface:  python -m facesentinel <command> --help"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from . import __version__, hud
from .detector import face_quality
from .imageio import imread, imwrite
from .models import ModelError, ensure_models
from .pipeline import Config, Pipeline
from .recognizer import DEFAULT_THRESHOLD

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW = "FaceSentinel"


# ----------------------------------------------------------------------- helpers
class ThreadedCapture:
    """Reads frames on a background thread so processing always sees the *newest* frame
    (no stale buffered frames, which is what makes webcams/RTSP feel laggy)."""

    def __init__(self, src, width: int = 0, height: int = 0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source: {src!r}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._lock = threading.Lock()
        self._frame = None
        self._seq = self._last = 0
        self._alive = True
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                self._alive = False
                return
            with self._lock:
                self._frame, self._seq = frame, self._seq + 1

    def read(self, timeout: float = 5.0):
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if self._frame is not None and self._seq != self._last:
                    self._last = self._seq
                    return True, self._frame.copy()
            if not self._alive:
                return False, None
            time.sleep(0.002)
        return False, None

    def release(self) -> None:
        self._stop = True
        time.sleep(0.05)
        self.cap.release()


def _source(value: str):
    return int(value) if value.isdigit() else value


def _config(a) -> Config:
    return Config(models_dir=a.models, db_path=a.db, det_score=a.det_score, scale=a.scale,
                  threshold=a.threshold, min_face=a.min_face, rec_every=a.rec_every)


def _fail(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


def _label(img, text: str, x: int, y: int, color=(0, 220, 120)) -> None:
    hud.overlay(img, hud.render_label(text, (20, 20, 20), color), x, y)


# ---------------------------------------------------------------------- commands
def cmd_models(a) -> int:
    paths = ensure_models(a.models or None)
    for key, path in paths.items():
        print(f"{key:<11} {path}  ({path.stat().st_size // 1024} KB)")
    return 0


def cmd_list(a) -> int:
    from .database import FaceDB

    names = FaceDB(a.db).names()
    if not names:
        print("Database is empty. Add someone with:  facesentinel enroll --name <name>")
    for name, n in names.items():
        print(f"{name:<30} {n} samples")
    return 0


def cmd_remove(a) -> int:
    from .database import FaceDB

    db = FaceDB(a.db)
    removed = db.remove(a.name.strip())
    if not removed:
        return _fail(f"no such person: {a.name!r}")
    db.save()
    print(f"Removed {a.name!r} ({removed} samples).")
    return 0


def _enroll_images(pipe: Pipeline, folder: Path) -> List[np.ndarray]:
    embs = []
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not files:
        print(f"no images found in {folder}")
    for path in files:
        img = imread(path)
        faces = pipe.detect(img) if img is not None else []
        if len(faces) == 0:
            print(f"  skip {path.name}: no face found")
            continue
        if len(faces) > 1:
            print(f"  note {path.name}: {len(faces)} faces, using the largest")
        embs.append(pipe.recognizer.embed(img, faces[0]))
        print(f"  ok   {path.name}")
    return embs


def _enroll_camera(pipe: Pipeline, a) -> List[np.ndarray]:
    cap = ThreadedCapture(_source(a.source), a.width, a.height)
    embs: List[np.ndarray] = []
    last = 0.0
    try:
        while len(embs) < a.samples:
            ok, frame = cap.read()
            if not ok:
                break
            if a.mirror:
                frame = cv2.flip(frame, 1)
            faces = pipe.detect(frame)
            view = frame.copy()
            msg, color = "show your face", (0, 170, 255)
            if len(faces) > 1:
                msg = "one person only"
            elif len(faces) == 1:
                ok_q, reason = face_quality(faces[0])
                msg = reason
                color = (0, 220, 120) if ok_q else (0, 170, 255)
                x, y, w, h = [int(v) for v in faces[0][:4]]
                hud.draw_brackets(view, x, y, w, h, color, 2)
                if ok_q and time.time() - last >= a.interval:
                    embs.append(pipe.recognizer.embed(frame, faces[0]))
                    last = time.time()
            vh, vw = view.shape[:2]
            cv2.rectangle(view, (0, vh - 8), (int(vw * len(embs) / a.samples), vh), (0, 220, 120), -1)
            _label(view, f"{a.name}: {len(embs)}/{a.samples} - {msg}", 10, 10, color)
            cv2.imshow(WINDOW, view)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                print("Enrollment cancelled.")
                return []
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return embs


def cmd_enroll(a) -> int:
    a.name = a.name.strip()
    if not a.name or len(a.name) > 64:
        return _fail("--name must be 1-64 characters")
    pipe = Pipeline.build(_config(a))
    embs = _enroll_images(pipe, Path(a.images)) if a.images else _enroll_camera(pipe, a)
    if not embs:
        return _fail("no usable face samples were captured")

    for e in embs[:3]:  # safety net: does this face already belong to someone else?
        other, score = pipe.db.best(e)
        if other and other != a.name and score >= a.threshold:
            print(f"warning: this face looks like existing person {other!r} (similarity {score:.2f}).")
            break
    pipe.db.add(a.name, embs)
    pipe.db.save()
    print(f"Saved {len(embs)} samples for {a.name!r}  ->  {a.db}")
    return 0


def cmd_image(a) -> int:
    img = imread(a.input)
    if img is None:
        return _fail(f"cannot read image: {a.input}")
    pipe = Pipeline.build(_config(a))
    tracks = pipe.analyze(img)

    results = []
    for t in tracks:
        name, score = t.identity()
        x, y, w, h = [float(v) for v in t.face[:4]]
        results.append({"box": [round(x), round(y), round(w), round(h)],
                        "detection_score": round(float(t.face[14]), 3),
                        "name": name or "unknown", "similarity": round(score, 3),
                        "landmarks": [[round(float(t.face[4 + 2 * i])), round(float(t.face[5 + 2 * i]))]
                                      for i in range(5)]})
    if a.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"{len(results)} face(s)")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['name']:<20} similarity={r['similarity']:.2f}  box={r['box']}")

    out = Path(a.output) if a.output else Path(a.input).with_name(Path(a.input).stem + "_out.jpg")
    hud.render(img, tracks, info=None, privacy=a.privacy, show_hud=False, font_path=a.font)
    if not imwrite(out, img):
        return _fail(f"cannot write {out}")
    print(f"Annotated image: {out}")
    return 0


def cmd_video(a) -> int:
    cap = cv2.VideoCapture(_source(a.input))
    if not cap.isOpened():
        return _fail(f"cannot open video: {a.input}")
    pipe = Pipeline.build(_config(a))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer, idx, seen = None, 0, set()
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tracks = pipe.process(frame, idx)
            for t in tracks:
                name, _ = t.identity()
                if name:
                    seen.add(name)
            hud.render(frame, tracks, {"fps": idx / max(time.time() - t0, 1e-6)},
                       privacy=a.privacy, font_path=a.font)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
            idx += 1
            if idx % 30 == 0:
                print(f"\r  frame {idx}" + (f"/{total}" if total else ""), end="", flush=True)
    finally:
        cap.release()
        if writer:
            writer.release()
    print(f"\nDone: {idx} frames in {time.time() - t0:.1f}s -> {a.output}")
    print("Recognised: " + (", ".join(sorted(seen)) if seen else "nobody"))
    return 0


def cmd_live(a) -> int:
    from .attendance import AttendanceLog

    pipe = Pipeline.build(_config(a))
    cap = ThreadedCapture(_source(a.source), a.width, a.height)
    log = AttendanceLog(a.log_file, a.cooldown) if a.log else None
    privacy, show_lm, show_hud = a.privacy, True, not a.no_hud
    writer, paused, last_view = None, False, None
    fps, idx, ms = 0.0, 0, 0.0
    Path(a.captures).mkdir(parents=True, exist_ok=True)
    print("Keys: q/ESC quit | p privacy | l landmarks | h HUD | s snapshot | r record | space pause")

    try:
        prev = time.time()
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("video source ended")
                    break
                if a.mirror:
                    frame = cv2.flip(frame, 1)
                t0 = time.perf_counter()
                tracks = pipe.process(frame, idx)
                ms = ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1 if idx else (time.perf_counter() - t0) * 1000
                now = time.time()
                fps = fps * 0.9 + (1.0 / max(now - prev, 1e-6)) * 0.1 if idx else 0.0
                prev, idx = now, idx + 1

                if log:
                    for t in tracks:
                        name, score = t.identity()
                        if name and t.hits >= 5:
                            if log.mark(name, score):
                                print(f"[attendance] {name}  ({score:.2f})")
                hud.render(frame, tracks, {"fps": fps, "ms": ms, "recording": writer is not None},
                           privacy=privacy, landmarks=show_lm, show_hud=show_hud, font_path=a.font)
                if writer is not None:
                    writer.write(frame)
                last_view = frame
            if last_view is not None:
                cv2.imshow(WINDOW, last_view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("p"):
                privacy = hud.PRIVACY_MODES[(hud.PRIVACY_MODES.index(privacy) + 1) % 3]
            elif key == ord("l"):
                show_lm = not show_lm
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord(" "):
                paused = not paused
            elif key == ord("s") and last_view is not None:
                out = Path(a.captures) / time.strftime("snap_%Y%m%d_%H%M%S.jpg")
                imwrite(out, last_view)
                print(f"saved {out}")
            elif key == ord("r") and last_view is not None:
                if writer is None:
                    h, w = last_view.shape[:2]
                    out = Path(a.captures) / time.strftime("rec_%Y%m%d_%H%M%S.mp4")
                    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
                    print(f"recording -> {out}")
                else:
                    writer.release()
                    writer = None
                    print("recording stopped")
    finally:
        if writer:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
    return 0


def cmd_bench(a) -> int:
    img = imread(a.image)
    if img is None:
        return _fail(f"cannot read image: {a.image}")
    pipe = Pipeline.build(_config(a), quiet=True)
    faces = pipe.detect(img)
    if len(faces) == 0:
        return _fail("no face in the benchmark image")

    def timed(fn) -> List[float]:
        for _ in range(5):
            fn()
        out = []
        for _ in range(a.n):
            t0 = time.perf_counter()
            fn()
            out.append((time.perf_counter() - t0) * 1000)
        return out

    def report(label: str, ms: List[float]) -> None:
        print(f"{label:<28} mean {np.mean(ms):7.2f} ms   p95 {np.percentile(ms, 95):7.2f} ms   "
              f"~{1000 / np.mean(ms):6.1f} /s")

    print(f"image {img.shape[1]}x{img.shape[0]}, {len(faces)} face(s), {a.n} runs, OpenCV {cv2.__version__}")
    report("detect", timed(lambda: pipe.detect(img)))
    report("embed (1 face)", timed(lambda: pipe.recognizer.embed(img, faces[0])))
    report("detect + embed + match", timed(lambda: pipe.analyze(img)))
    return 0


# ------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models", default="", help="model folder (default: ./models or $FACESENTINEL_MODELS)")
    common.add_argument("--db", default="data/faces.npz", help="face database file")
    common.add_argument("--det-score", type=float, default=0.7, help="detection confidence threshold")
    common.add_argument("--scale", type=float, default=1.0, help="detect on a scaled frame, e.g. 0.5 = faster")
    common.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="cosine match threshold")
    common.add_argument("--min-face", type=int, default=40, help="smallest face (px) to recognise")
    common.add_argument("--rec-every", type=int, default=15, help="re-verify each tracked face every N frames")

    cam = argparse.ArgumentParser(add_help=False)
    cam.add_argument("--source", default="0", help="camera index, video file or RTSP/HTTP URL")
    cam.add_argument("--width", type=int, default=0)
    cam.add_argument("--height", type=int, default=0)
    cam.add_argument("--mirror", action="store_true", help="flip horizontally (selfie view)")

    look = argparse.ArgumentParser(add_help=False)
    look.add_argument("--privacy", choices=hud.PRIVACY_MODES, default="off",
                      help="hide the faces of unknown people")
    look.add_argument("--font", default=None, help="TTF font for non-English names (e.g. Persian)")

    p = argparse.ArgumentParser(prog="facesentinel", description="Real-time face detection, tracking & recognition.")
    p.add_argument("--version", action="version", version=f"facesentinel {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", parents=[common], help="download the ONNX models").set_defaults(fn=cmd_models)

    s = sub.add_parser("live", parents=[common, cam, look], help="live recognition from a camera/stream")
    s.add_argument("--log", action="store_true", help="write recognised people to a CSV attendance log")
    s.add_argument("--log-file", default="logs/attendance.csv")
    s.add_argument("--cooldown", type=float, default=60.0, help="seconds before logging the same person again")
    s.add_argument("--captures", default="captures", help="folder for snapshots and recordings")
    s.add_argument("--no-hud", action="store_true", help="hide the stats panel")
    s.set_defaults(fn=cmd_live)

    s = sub.add_parser("enroll", parents=[common, cam], help="add a person (camera or a folder of photos)")
    s.add_argument("--name", required=True)
    s.add_argument("--images", help="folder of photos instead of the camera")
    s.add_argument("--samples", type=int, default=12)
    s.add_argument("--interval", type=float, default=0.5, help="seconds between captured samples")
    s.set_defaults(fn=cmd_enroll)

    s = sub.add_parser("image", parents=[common, look], help="analyse one image")
    s.add_argument("input")
    s.add_argument("-o", "--output")
    s.add_argument("--json", action="store_true", help="print machine-readable results")
    s.set_defaults(fn=cmd_image)

    s = sub.add_parser("video", parents=[common, look], help="process a video file")
    s.add_argument("input")
    s.add_argument("-o", "--output", required=True)
    s.set_defaults(fn=cmd_video)

    sub.add_parser("list", parents=[common], help="list enrolled people").set_defaults(fn=cmd_list)

    s = sub.add_parser("remove", parents=[common], help="delete a person from the database")
    s.add_argument("--name", required=True)
    s.set_defaults(fn=cmd_remove)

    s = sub.add_parser("bench", parents=[common], help="measure speed on your machine")
    s.add_argument("image")
    s.add_argument("-n", type=int, default=100)
    s.set_defaults(fn=cmd_bench)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ModelError, RuntimeError) as exc:
        return _fail(str(exc))
    except KeyboardInterrupt:
        return 130
