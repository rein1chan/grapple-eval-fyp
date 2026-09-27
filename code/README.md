# Grapple Eval

Open a Brazilian Jiu-Jitsu video, press **Analyse**, and get a clickable, colour-coded list of the
positions in it: Stand-up, Closed Guard, Half Guard, Open Guard, Butterfly Guard, Guard Pass / Scramble,
and a plain "Guard" when the legs are clearly tangled but the exact guard is unclear.
Click any entry and the video jumps straight to that moment.

It runs on a normal laptop. If the computer has an NVIDIA graphics card, it is used automatically.

## 1. Install (Windows)

1. Install **Python 3.11** from https://www.python.org/downloads/ (tick *"Add python.exe to PATH"*).
   Python 3.12 and 3.13 also work.
2. Open **PowerShell** in this folder (Shift + right-click the folder → *Open in Terminal*) and run:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
3. Start the app:
   ```powershell
   python main.py
   ```
   The pose model (`yolo11n-pose.pt`, 6 MB) is included in this folder.

**Using an NVIDIA graphics card (optional).** `pip install -r requirements.txt` installs the CPU-only
version of PyTorch. To use the graphics card instead:
1. Run `nvidia-smi` and look at "CUDA Version" in the top right (13.0 or higher for the command below;
   otherwise update the driver from nvidia.com, or use `cu128` instead of `cu130`).
2. With the `.venv` active:
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps`
3. Check: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

## 1b. Install (Mac)

```bash
brew install python@3.11            # or use the installer from python.org
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```
If you see "No module named _tkinter", run `brew install python-tk@3.11`.
(On a Mac the pose model runs on the processor.)

## 2. Optional extras

**Listening to the audio (Whisper + FFmpeg).** Picks up coaching or commentary words such as "half guard".
* Windows: `winget install Gyan.FFmpeg`, then close and reopen PowerShell. Mac: `brew install ffmpeg`.
* `openai-whisper` is installed by `requirements.txt`. The first run downloads its "base" model (about 140 MB).

**Llama 3 as a second opinion (Ollama).**
* Install Ollama from https://ollama.com/download and start it.
* In a terminal: `ollama pull llama3:8b` (about 4.7 GB).
* Tick **Llama 3 (Ollama)** in the app.

Every extra checks itself. If something is missing, the app says so in its log and carries on without it.

## 3. Using the app

| To... | Do this |
|---|---|
| Load a video | **Open Video** (MP4 recommended) |
| Analyse it | **Analyse**. *Every Nth frame*: 3 = fast (default), 1 = most detail |
| Jump to a position | Click a row in the list, or click anywhere on the colour timeline |
| See what the system sees | *Show skeleton*: filled dots = seen by the camera, hollow dots = hidden joint filled in by the Kalman filter |
| Hide uncertain results | *Only show intervals with confidence at least* (default 0.75) → **Apply** (no need to analyse again) |
| Fix mistakes | **Add** (starts at the current time), **Edit** (or double-click a row), **Delete** |
| Save or reload the list | **Export JSON**, **Export CSV**, **Load JSON** |
| Save the smoothing plot | **Save Kalman plot (PNG)** (report Figure 4.3) |

**Status badges** (second row): *Running on: GPU* (green) or *CPU* (orange), and *Llama 3: ready* (green)
or *not available* (grey; click it to see why). The **Decided by** column shows who made each decision:
`Llama 3 + Rules` (both agreed), `Llama 3 only` (they disagreed, so low confidence), `Rules`,
`Rules + Audio` (the commentary agreed), `Rules (legs tangled)` (plain Guard) or `Manual` (you edited it).

## 4. How it avoids wrong answers

* **Stand-up:** both athletes upright with their feet well below their hips.
* **No guard without someone on the mat**, and **no guard without the legs**: the bottom athlete must be lying
  or sitting and their legs must touch the top athlete. Side control, mount and back control are therefore
  left out rather than called a guard.
* **Referee filter** (*Ignore referee*): the referee wears long dark trousers and no-gi athletes have bare
  shins, so the app checks how bright each standing person's shins are and removes the referee. In a gi match
  where the athletes also wear dark trousers, it switches itself off. The athletes are then taken as the
  biggest person on the mat plus whoever is grappling with them.
* **Honest confidence:** high only when the measurements are clear-cut and both athletes are well visible;
  lowered when a second disagrees with the seconds around it or when Llama 3 and the rules disagree.
  Uncertain seconds and anything shorter than 2 seconds are left out.

## 5. Files and the report sections they implement

| File | What it does | Report section |
|---|---|---|
| `main.py` | The desktop app (video player, timeline, list, editing, export) | 3.7, 4.10 |
| `pipeline.py` | Runs all the steps in order | 3.1, 4.1 |
| `pose.py` | `VisualWorker`: finds the athletes' body points, follows them, ignores the referee | 3.2, 4.2, 4.8 |
| `kalman.py` | `BidirectionalJointKalmanFilter`: smooths the body points and fills in hidden ones | 3.4, 4.3 |
| `audio.py` | `AudioWorker`: turns speech into text and looks for BJJ words (optional) | 3.3, 4.4 |
| `features.py` | `GrappleMiddlewareBridge`: body measurements and 1-second fact sheets | 3.5, 3.6, 4.5, 4.9 |
| `reasoning.py` | Rules, Llama 3 second opinion, confidence and joining seconds into intervals | 3.6, 3.7, 4.6, 4.9 |
| `utils.py` | Position names, colours and time formatting | – |

## 6. Known limitations

* When the athletes are tightly tangled, the pose model often sees them as one person. Those moments can't
  be classified and stay as gaps (the log says how much of the video this affected).
* The rule limits (top of `reasoning.py`) were set by hand, not learned from labelled data. Stand-up and
  plain Guard are reliable; the specific guard types (closed, half, open, butterfly) are not yet.
* Differences from the full design in the report (Table 4.2): no server stack (FastAPI, vLLM, TensorRT);
  smaller models (YOLO11n-pose, Whisper base, Llama 3 8B through Ollama); only every Nth frame is analysed;
  back-to-back 1-second windows instead of an overlapping sliding window.
