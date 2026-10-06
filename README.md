<div align="center">

# 🛰️ FaceSentinel

**Real-time face detection, tracking and recognition — built on pure OpenCV.**
No dlib. No TensorFlow. No PyTorch. Two small ONNX models and one `pip install`.

[![tests](https://github.com/YOUR_USER/facesentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USER/facesentinel/actions)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![opencv](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

[فارسی](README.fa.md) · [Quick start](#-quick-start) · [How it works](#-how-it-works) · [Privacy & ethics](#-privacy--ethics)

</div>

---

## ✨ Why FaceSentinel

Most "OpenCV face recognition" demos are a Haar cascade and a flickering box. FaceSentinel is the full pipeline, done properly:

| | |
|---|---|
| 🎯 **Modern detector** | YuNet (`cv2.FaceDetectorYN`): faces plus 5 landmarks, robust to angle and scale |
| 🧬 **Deep recognition** | SFace (`cv2.FaceRecognizerSF`): landmark-aligned 128-D embeddings, cosine matching |
| 🧵 **Stable identities** | IoU tracker with box smoothing and **majority-vote identity**: names don't flicker |
| ⚡ **Built for speed** | Background capture thread (always the newest frame), recognition only when needed per track, `--scale` for faster detection |
| 🙈 **Privacy mode** | Pixelate or blur every *unknown* face live (`--privacy pixelate`) |
| 🗒️ **Attendance log** | CSV with per-person cooldown, Excel-friendly UTF-8 |
| 📸 **Smart enrollment** | Captures only good samples (size, sharpness, frontal pose); warns if the face already belongs to someone else |
| 🎬 **Everything in one CLI** | Live camera / RTSP, images, video files, benchmark, JSON output |
| 🌍 **Unicode names** | Persian / Arabic / any script on screen (optional extra) |
| 🧪 **Tested** | Unit tests run without a camera or model files |

## 🚀 Quick start

```bash
git clone https://github.com/YOUR_USER/facesentinel.git
cd facesentinel
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .                                       # add ".[unicode]" for Persian/Arabic names

facesentinel models                                    # downloads the two ONNX models (once)
facesentinel enroll --name "Sara" --mirror             # look at the camera, ~10 seconds
facesentinel live --mirror --log                       # recognise live + write attendance
```

Without installing, use `python -m facesentinel <command>` instead of `facesentinel <command>`.

## 🧰 Commands

| Command | What it does |
|---|---|
| `models` | Download YuNet + SFace into `./models` (or `$FACESENTINEL_MODELS`) |
| `enroll --name N` | Add a person from the camera, or `--images folder/` from photos |
| `live` | Live recognition from a webcam, video file or RTSP/HTTP stream (`--source`) |
| `image photo.jpg` | Annotate one image; `--json` prints boxes, landmarks, names, similarities |
| `video in.mp4 -o out.mp4` | Process a whole video |
| `list` / `remove --name N` | Manage the database |
| `bench photo.jpg` | Measure detect / embed / total speed **on your machine** |

Useful options: `--privacy off\|pixelate\|blur` · `--scale 0.5` · `--threshold 0.363` · `--min-face 40` · `--rec-every 15` · `--source rtsp://…` · `--width 1280 --height 720` · `--font C:\Windows\Fonts\tahoma.ttf`

**Live keys:** `q`/`Esc` quit · `p` cycle privacy · `l` landmarks · `h` HUD · `s` snapshot · `r` record · `Space` pause

### Examples

```bash
# Enroll from a folder of photos (3-10 clear photos of one person work best)
facesentinel enroll --name "Ali" --images ./photos/ali

# Hide everyone who is not enrolled, on an IP camera
facesentinel live --source "rtsp://user:pass@192.168.1.20/stream" --privacy pixelate

# Use the result in your own scripts
facesentinel image group.jpg --json > faces.json

# Faster on a weak CPU: detect on a half-size frame, re-verify less often
facesentinel live --scale 0.5 --rec-every 30
```

## 🧠 How it works

```mermaid
flowchart LR
    A[Camera / RTSP / file] -->|newest frame| B[YuNet detector]
    B -->|boxes + 5 landmarks| C[IoU tracker<br/>smoothing + IDs]
    C -->|new or due for re-check| D[Align + SFace embedding]
    D -->|cosine similarity| E[(Face database .npz)]
    E --> F[Identity vote<br/>last 7 results]
    C --> G[HUD / privacy filter]
    F --> G
    F --> H[Attendance CSV]
```

1. **Detect:** YuNet returns a box, 5 landmarks and a confidence per face.
2. **Track:** faces are matched frame-to-frame by IoU, so each person keeps a stable ID and a smoothed box.
3. **Recognise only when needed:** a new track is checked on its first few frames, then re-verified every `--rec-every` frames. Recognition is the expensive step, and this keeps FPS high with many faces.
4. **Vote:** the displayed name is the majority of the last 7 results, so one bad frame can't flip an identity.
5. **Decide:** an embedding matches a person if its cosine similarity to a stored sample is at least `--threshold` (default **0.363**, the value the OpenCV Model Zoo recommends for SFace).

### Tuning

- **Strangers get recognised as someone** → raise `--threshold` (e.g. `0.45`) and enroll more varied samples.
- **Known people show as UNKNOWN** → lower `--threshold` a little, enroll in the lighting you'll use, make sure `--min-face` isn't larger than the face in the frame.
- **Low FPS** → `--scale 0.5`, a lower camera resolution, or a higher `--rec-every`. Run `facesentinel bench photo.jpg` to see where the time goes.

## 🌍 Persian / Arabic / Unicode names

OpenCV's `putText` can't draw non-Latin scripts, so FaceSentinel renders those labels with Pillow:

```bash
pip install -e ".[unicode]"        # pillow + arabic-reshaper + python-bidi
facesentinel live --font C:\Windows\Fonts\tahoma.ttf
```

Without the extras, non-ASCII characters are replaced by `?` on screen (recognition and the CSV log are unaffected). Database files and image paths with Unicode names work on Windows too.

## 🔒 Privacy & ethics

Face embeddings are **biometric data**. Please treat them that way.

- Only enroll people who **gave consent**, and tell people when a camera is recognising faces.
- Many places regulate biometric processing (e.g. GDPR in the EU). You are responsible for complying with the laws that apply to you.
- `data/`, `models/`, `*.npz` and `captures/` are in `.gitignore`. **Never commit a face database** or captured footage.
- Embeddings are stored unencrypted; protect the `data/` folder with OS permissions or disk encryption.
- Face recognition makes mistakes, and error rates differ across lighting, camera quality and demographic groups. Don't use it as the only basis for decisions that affect people (access, discipline, law enforcement).

## 🗂️ Project layout

```
facesentinel/
├── detector.py     YuNet wrapper, scale handling, enrollment quality gate
├── recognizer.py   SFace embeddings
├── database.py     .npz face store (atomic save, no pickle)
├── tracker.py      IoU tracker, smoothing, identity voting
├── pipeline.py     detect → track → recognise glue
├── hud.py          brackets, landmarks, labels, privacy filters, Unicode text
├── attendance.py   CSV log with cooldown
├── models.py       model download + validation
└── cli.py          commands, threaded camera capture
tests/              unit tests (no camera/models needed)
```

## 🧪 Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests use fake detector/recognizer objects, so they run anywhere, including CI.

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `Could not download …onnx` | Download the two files from the URLs printed in the error and put them in `./models/` |
| `cannot open video source: 0` | Try `--source 1`; close other apps using the camera; on Linux check `/dev/video*` permissions |
| Window doesn't open / `cv2.imshow` error | You installed `opencv-python-headless`; use `opencv-python` instead |
| Persian names show as `?` | `pip install -e ".[unicode]"` and pass `--font` pointing at a font that has Persian glyphs |
| Laggy RTSP stream | Use the sub-stream (lower resolution) of your camera and `--scale 0.5` |

## 🗺️ Roadmap

- [ ] Headless mode with a small HTTP/WebSocket event stream
- [ ] Mask and liveness (anti-spoofing) checks
- [ ] Multi-camera support
- [ ] Encrypted face database
- [ ] Optional GPU backends (CUDA / OpenVINO) switch

Contributions are welcome: open an issue first for big changes.

## 🙏 Credits

Models come from the [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo): **YuNet** (face detection) and **SFace** (face recognition). Please check the zoo for the models' own licences. This project's code is [MIT](LICENSE).
