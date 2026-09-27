"""Runs all the steps of Grapple Eval in order (report section 3.1, Figure 3.1).

video -> find body points -> smooth them -> listen to the audio (optional)
      -> turn them into measurements -> decide the positions -> list of intervals

The app runs this in the background so the window doesn't freeze. Progress and log
messages are passed back to the app through two small functions.
"""

import numpy as np

from audio import AudioWorker
from features import GrappleMiddlewareBridge
from kalman import BidirectionalJointKalmanFilter, smooth_all_keypoints
from pose import VisualWorker
from reasoning import LlamaGuardReasoner, RuleBasedGuardClassifier, merge_windows_into_intervals


class GrappleEvalPipeline:
    def __init__(self, frame_step=3, use_audio=True, use_llm=True, min_confidence=0.75,
                 ignore_referee=True, progress_cb=None, log_cb=None, stop_event=None):
        self.frame_step = frame_step          # only look at every Nth frame
        self.ignore_referee = ignore_referee  # leave out the person in long dark trousers (the referee)
        self.min_confidence = min_confidence  # intervals less sure than this are left out
        self.use_audio = use_audio
        self.use_llm = use_llm
        self.progress_cb = progress_cb or (lambda p, msg="": None)
        self.log = log_cb or print
        self.stop_event = stop_event

    def run(self, video_path):
        # ---- Step 1: find the athletes' body points (progress bar 0% -> 75%) ----
        visual = VisualWorker(frame_step=self.frame_step, ignore_referee=self.ignore_referee)
        self.log(f"Visual worker: YOLOv11 pose on {visual.device_name}, every {self.frame_step} frame(s).")
        self.progress_cb(0.0, "Loading pose model...")
        visual.load_model()
        pose = visual.run(video_path,
                          progress_cb=lambda p: self.progress_cb(0.75 * p, "Detecting poses..."),
                          stop_event=self.stop_event)
        raw = pose["keypoints"]
        if "failed" in pose["device"]:
            self.log(f"Note: pose model ran on {pose['device']}.")
        st = pose["stats"]
        if self.ignore_referee:
            self.log(f"Referee filter {st.get('referee_filter', 'on')}: "
                     f"ignored {st.get('referee_detections_removed', 0)} referee detections.")

        # Stop early with a clear message if nobody was ever found
        people_per_frame = (~np.all(np.isnan(raw[..., 2]), axis=2)).sum(axis=1)  # 0, 1 or 2 per frame
        if people_per_frame.max() == 0:
            raise RuntimeError("No people were detected in this video.")
        both = float((people_per_frame == 2).mean())
        self.log(f"Detected both athletes in {both:.0%} of analysed frames.")
        if both < 0.7:
            self.log(f"Note: in {1 - both:.0%} of frames the athletes were too tangled (or off-screen) "
                     "for the pose model to separate them - those moments can't be classified.")
        if both < 0.05:
            raise RuntimeError("Two athletes were almost never visible together - "
                               "can't measure guard positions in this video.")

        # ---- Step 2: smooth the body points and fill in hidden ones (75% -> 82%) ----
        self.progress_cb(0.75, "Kalman smoothing...")
        dt = self.frame_step / pose["fps"]  # seconds between two analysed frames
        smoothed = smooth_all_keypoints(raw, dt)

        # ---- Step 3: listen to the audio (optional) (82% -> 88%) ----
        audio_segments = []
        if self.use_audio:
            self.progress_cb(0.82, "Transcribing audio (Whisper)...")
            audio_segments, msg = AudioWorker().transcribe(video_path)
            self.log(msg)
        else:
            self.log("Audio step turned off.")

        # ---- Step 4: turn body points into measurements, one fact sheet per second (88% -> 90%) ----
        self.progress_cb(0.88, "Building fact sheets...")
        bridge = GrappleMiddlewareBridge(window_size=30, frame_step=self.frame_step)
        sheets = bridge.run(pose["timestamps"], pose["frame_indices"], smoothed, raw, audio_segments)
        self.log(f"Middleware: {len(sheets)} fact sheets (1 s windows).")

        # ---- Step 5: decide the position of every second, then join them into intervals (90% -> 100%) ----
        rules = RuleBasedGuardClassifier()
        labels, engine = None, "Rules only"
        if self.use_llm:
            llama = LlamaGuardReasoner()
            ok, msg = llama.check_available()
            if ok:
                self.log(f"Llama 3: {msg} Sending {len(sheets)} windows...")
                try:
                    labels, n_ok = llama.classify_all(
                        sheets, rules,
                        progress_cb=lambda p, m: self.progress_cb(0.9 + 0.09 * p, m))
                    engine = f"Llama 3 answered {n_ok}/{len(sheets)} windows (checked against rules)"
                except Exception as e:
                    self.log(f"Llama 3 FAILED ({type(e).__name__}: {e}) - using the rules instead.")
                    engine = "Rules only (Llama 3 failed)"
                    labels = None
            else:
                self.log(f"Llama 3 NOT USED: {msg}")
                engine = "Rules only (Llama 3 not available)"
        if labels is None:
            labels = [rules.classify(s) for s in sheets]
        intervals = merge_windows_into_intervals(sheets, labels, min_confidence=self.min_confidence)
        self.log(f"Reasoning engine: {engine}.")
        self.log(f"Kept {len(intervals)} intervals with confidence >= {self.min_confidence:.2f} "
                 f"(uncertain parts left out).")
        self.progress_cb(1.0, "Done.")

        return {
            "intervals": intervals,
            "window_labels": labels,   # the answer for every second, so the app can re-filter without re-analysing
            "engine": engine,
            "fact_sheets": sheets,
            "frame_step": pose["frame_step"],
            "frame_indices": pose["frame_indices"],
            "timestamps": pose["timestamps"],
            "raw_keypoints": raw,
            "smoothed_keypoints": smoothed,
            "device": pose["device"],
        }


def save_kalman_plot(result, path, athlete=0, joint=15, joint_name="left ankle"):
    """Save a picture comparing the raw and the smoothed path of one joint (report Figure 4.3).

    Points the model was unsure about (ignored by the filter) are drawn as grey crosses.
    """
    from matplotlib.figure import Figure  # this way of drawing is safe to use inside the app

    t = result["timestamps"]
    raw = result["raw_keypoints"][:, athlete, joint]
    sm = result["smoothed_keypoints"][:, athlete, joint]
    thr = BidirectionalJointKalmanFilter(1.0).conf_threshold
    confident = raw[:, 2] >= thr
    hidden = (raw[:, 2] < thr) & ~np.isnan(raw[:, 2])

    fig = Figure(figsize=(10, 6), dpi=120)
    for row, (axis_i, axis_name) in enumerate([(0, "x"), (1, "y")]):
        ax = fig.add_subplot(2, 1, row + 1)
        ax.plot(t[confident], raw[confident, axis_i], ".", ms=4, color="#e67e22",
                label="Raw YOLO keypoint (conf >= 0.35)")
        ax.plot(t[hidden], raw[hidden, axis_i], "x", ms=4, color="#95a5a6",
                label="Raw keypoint ignored (conf < 0.35)")
        ax.plot(t, sm[:, axis_i], "-", lw=1.8, color="#2e86de",
                label="Bidirectional Kalman (RTS) smoothed")
        ax.set_ylabel(f"{axis_name} position (px)")
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title(f"Raw vs. Kalman-smoothed trajectory - athlete {athlete + 1}, {joint_name}")
            ax.legend(loc="best", fontsize=8)
    fig.axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path)
