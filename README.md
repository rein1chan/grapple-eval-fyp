# Grapple Eval – final year project

Automatic guard-position indexing for Brazilian Jiu-Jitsu video: open a sparring or competition video,
press **Analyse**, and jump straight to each stand-up, guard or scramble from a clickable, colour-coded list.

| File | What it is |
|---|---|
| [`Grapple-Eval report.pdf`](Grapple-Eval%20report.pdf) | The project report |
| [`Grapple-Eval demo video.mp4`](Grapple-Eval%20demo%20video.mp4) | 4½-minute demonstration of the working app |
| [`code/`](code/) | The app's source code – see [`code/README.md`](code/README.md) for installation and use |

## Quick start (Windows)

```powershell
cd code
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Pipeline: YOLOv11 pose + ByteTrack → bidirectional Kalman filter → kinematic middleware → rule-based
reasoning (optionally checked by Llama 3 via Ollama), with optional Whisper audio keywords.

The demo video contains short excerpts of 2024 ADCC World Championships footage (FloGrappling),
used for academic demonstration only.
