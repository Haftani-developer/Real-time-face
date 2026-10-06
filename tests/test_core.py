"""Run with `pytest` (or plain `python tests/test_core.py`). No model files or camera needed."""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facesentinel import hud
from facesentinel.attendance import AttendanceLog
from facesentinel.cli import build_parser, main
from facesentinel.database import FaceDB
from facesentinel.detector import FaceDetector, face_quality
from facesentinel.pipeline import Config, Pipeline
from facesentinel.tracker import Track, Tracker, iou


def row(x, y, w, h, score=0.95, nose_shift=0.0):
    ex = w * 0.25
    cx, cy = x + w / 2, y + h * 0.4
    return np.array([x, y, w, h, cx - ex, cy, cx + ex, cy, cx + nose_shift * 2 * ex, cy + h * 0.15,
                     cx - ex / 2, y + h * 0.75, cx + ex / 2, y + h * 0.75, score], dtype=np.float32)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    assert 0.3 < iou((0, 0, 10, 10), (5, 0, 10, 10)) < 0.4


def test_tracker_keeps_ids_and_expires():
    tr = Tracker(max_age=2)
    a = tr.update(np.stack([row(100, 100, 80, 80), row(400, 100, 80, 80)]))
    assert sorted(t.id for t in a) == [1, 2]
    b = tr.update(np.stack([row(104, 102, 80, 80), row(404, 101, 80, 80)]))
    assert sorted(t.id for t in b) == [1, 2]
    c = tr.update(np.stack([row(106, 103, 80, 80)]))          # second face disappears
    assert [t.id for t in c] == [1]
    for _ in range(3):
        tr.update(np.empty((0, 15)))
    assert tr.tracks == []                                     # everything expired
    d = tr.update(np.stack([row(106, 103, 80, 80)]))
    assert d[0].id == 3                                        # ids are never reused


def test_identity_voting_resists_flicker():
    t = Track(1, row(0, 0, 80, 80))
    for name in ["ali", "ali", None, "ali", "ali", None, "ali"]:
        t.add_vote(name, 0.6, 0)
    assert t.identity()[0] == "ali"
    t2 = Track(2, row(0, 0, 80, 80))
    assert t2.identity() == (None, 0.0)
    assert t2.needs_recognition(0, 15)


def test_database_roundtrip_and_match():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "faces.npz"
        db = FaceDB(path)
        assert db.match(unit([1, 0, 0]), 0.3) == (None, 0.0)
        db.add("علی", [unit([1, 0.1, 0]), unit([1, 0, 0.1])])
        db.add("sara", [unit([0, 1, 0])])
        db.save()

        db2 = FaceDB(path)
        assert db2.names() == {"sara": 1, "علی": 2}
        name, score = db2.match(unit([0.9, 0.1, 0]), 0.5)
        assert name == "علی" and score > 0.9
        assert db2.match(unit([0, 0, 1]), 0.5)[0] is None          # nobody close enough
        assert db2.remove("sara") == 1 and db2.remove("sara") == 0
        assert len(db2) == 2


def test_attendance_cooldown_and_single_bom():
    with tempfile.TemporaryDirectory() as d:
        log = AttendanceLog(Path(d) / "a.csv", cooldown=60)
        assert log.mark("ali", 0.8, now=1000.0)
        assert not log.mark("ali", 0.8, now=1030.0)
        assert log.mark("ali", 0.8, now=1061.0)
        assert log.mark("سارا", 0.7, now=1062.0)
        raw = (Path(d) / "a.csv").read_bytes()
        assert raw.count(b"\xef\xbb\xbf") == 1                     # BOM only once
        rows = list(csv.reader(raw.decode("utf-8-sig").splitlines()))
        assert len(rows) == 4 and rows[3][1] == "سارا"


def test_face_quality_gate():
    assert face_quality(row(0, 0, 120, 120))[0]
    assert face_quality(row(0, 0, 40, 40))[1] == "move closer"
    assert face_quality(row(0, 0, 120, 120, score=0.5))[1] == "face not clear"
    assert face_quality(row(0, 0, 120, 120, nose_shift=0.8))[1] == "look at the camera"


class FakeYuNet:
    def __init__(self):
        self.size = None

    def setInputSize(self, size):
        self.size = size

    def detect(self, img):
        h, w = img.shape[:2]
        assert (w, h) == self.size
        return 1, np.stack([row(w * 0.25, h * 0.25, w * 0.5, h * 0.5)])


def test_detector_scale_maps_back_to_original_coordinates():
    frame = np.zeros((400, 800, 3), np.uint8)
    full = FaceDetector(engine=FakeYuNet()).detect(frame)[0]
    half = FaceDetector(engine=FakeYuNet(), scale=0.5).detect(frame)[0]
    assert np.allclose(full[:14], half[:14], atol=1.0)
    assert half[14] == full[14]


class FakeRecognizer:
    def embed(self, frame, r):
        return unit([1, 0, 0]) if r[0] < 300 else unit([0, 1, 0])


class FakeDetector:
    def detect(self, frame):
        return np.stack([row(100, 100, 100, 100), row(500, 100, 100, 100)])


def test_pipeline_end_to_end_with_fakes():
    with tempfile.TemporaryDirectory() as d:
        db = FaceDB(Path(d) / "f.npz")
        db.add("ali", [unit([1, 0, 0])])
        pipe = Pipeline(FakeDetector(), FakeRecognizer(), db, Config())
        frame = np.zeros((300, 800, 3), np.uint8)
        for i in range(4):
            tracks = pipe.process(frame, i)
        names = sorted(str(t.identity()[0]) for t in tracks)
        assert names == ["None", "ali"]
        still = pipe.analyze(frame)
        assert sorted(str(t.identity()[0]) for t in still) == ["None", "ali"]


def test_hud_render_privacy_and_unicode_do_not_crash():
    frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    before = frame[100:200, 100:200].copy()
    unknown = Track.static(row(100, 100, 100, 100), None, 0.1, 1)
    known = Track.static(row(400, 100, 100, 100), "سارا", 0.8, 2)   # non-ASCII name
    hud.render(frame, [unknown, known], {"fps": 30.0, "ms": 12.0, "recording": True}, privacy="pixelate")
    assert not np.array_equal(before, frame[100:200, 100:200])       # unknown face was hidden
    hud.render(frame, [unknown], privacy="blur", show_hud=False)
    hud.render(np.zeros((20, 20, 3), np.uint8), [unknown])           # tiny frame, off-screen label
    assert hud.color_for("ali") == hud.color_for("ali") != hud.color_for(None)


def test_cli_parser_and_offline_commands():
    p = build_parser()
    assert p.parse_args(["live", "--privacy", "blur", "--log", "--scale", "0.5"]).privacy == "blur"
    assert p.parse_args(["enroll", "--name", "x", "--images", "photos"]).images == "photos"
    with tempfile.TemporaryDirectory() as d:
        db = str(Path(d) / "faces.npz")
        assert main(["list", "--db", db]) == 0
        assert main(["remove", "--name", "nobody", "--db", db]) == 1


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"\n{len(tests)} tests passed")
