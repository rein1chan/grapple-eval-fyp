"""Grapple Eval - the desktop app you start with:  python main.py

How to use it:  Open Video -> Analyse -> click a position in the list to jump to it.

What is on screen:
  top            : Open Video, Analyse, options, progress bar, status badges
  left           : the video player, the colour timeline, play/pause, slider and time
  right          : the list of positions found, edit/export buttons, and a log
"""

import csv
import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from pose import SKELETON_EDGES
from utils import GUARD_COLOURS, GUARD_LABELS, format_time, parse_time

VIDEO_TYPES = [("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")]
DISPLAY_W, DISPLAY_H = 800, 450      # size of the video picture on screen
TIMELINE_H = 34                      # height of the colour timeline under the video
ATHLETE_COLOURS = [(0, 200, 255), (255, 80, 200)]  # skeleton colours for athlete A (orange) and B (purple)


class GrappleEvalApp:
    def __init__(self, root):
        self.root = root
        root.title("Grapple-Eval - BJJ Guard Position Indexer")
        root.minsize(1150, 640)

        # --- the video that is open ---
        self.cap = None
        self.video_path = None
        self.fps = 30.0
        self.n_frames = 0
        self.current_frame = 0
        self.playing = False
        self.last_frame_img = None   # the frame on screen (so it can be redrawn without reading the video again)
        self._slider_busy = False    # stops the slider from reacting to its own movement

        # --- the analysis results ---
        self.result = None           # everything the analysis produced (body points, fact sheets, ...)
        self.intervals = []          # the positions shown in the list (position, start, end, confidence, who decided)
        self.edited = False          # True once the user changed the list by hand
        self.worker = None
        self.stop_event = threading.Event()
        self.msg_queue = queue.Queue()  # messages from the background analysis to the window

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self._poll_worker)

    # ================================================================== building the window
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top, text="Open Video", command=self.open_video).pack(side=tk.LEFT)
        self.analyse_btn = ttk.Button(top, text="Analyse", command=self.start_analysis)
        self.analyse_btn.pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Every Nth frame:").pack(side=tk.LEFT, padx=(12, 2))
        self.step_var = tk.IntVar(value=3)
        ttk.Spinbox(top, from_=1, to=15, width=4, textvariable=self.step_var).pack(side=tk.LEFT)
        self.audio_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Audio (Whisper)", variable=self.audio_var).pack(side=tk.LEFT, padx=6)
        self.llm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Llama 3 (Ollama)", variable=self.llm_var,
                        command=self._on_llama_toggle).pack(side=tk.LEFT)
        self.ref_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Ignore referee", variable=self.ref_var).pack(side=tk.LEFT, padx=(6, 0))
        self.skel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Show skeleton", variable=self.skel_var,
                        command=self._redraw).pack(side=tk.LEFT, padx=6)

        self.progress = ttk.Progressbar(top, length=180, mode="determinate", maximum=100)
        self.progress.pack(side=tk.LEFT, padx=8)
        self.status_var = tk.StringVar(value="Open a video to begin.")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT)

        # ---- second row: status badges and the confidence filter ----
        info = ttk.Frame(self.root, padding=(6, 0, 6, 0))
        info.pack(side=tk.TOP, fill=tk.X)
        badge = dict(fg="white", bg="#7f8c8d", padx=8, pady=2, font=("Segoe UI", 9, "bold"))
        # Hardware badge: green = using the graphics card (GPU), orange = using the normal processor (CPU)
        self.device_label = tk.Label(info, text="Device: checking...", **badge)
        self.device_label.pack(side=tk.LEFT)
        # Llama 3 badge: green = ready, grey = not available. Click it to check again and see why.
        self.llama_label = tk.Label(info, text="Llama 3: checking...", cursor="hand2", **badge)
        self.llama_label.pack(side=tk.LEFT, padx=6)
        self.llama_label.bind("<Button-1>", lambda e: self._check_llama(show_popup=True))

        # Confidence filter: only positions at least this sure are shown
        ttk.Button(info, text="Apply", width=6, command=self.apply_confidence_filter).pack(side=tk.RIGHT)
        self.minconf_var = tk.DoubleVar(value=0.75)
        ttk.Spinbox(info, from_=0.5, to=0.95, increment=0.05, width=5, format="%.2f",
                    textvariable=self.minconf_var).pack(side=tk.RIGHT, padx=4)
        ttk.Label(info, text="Only show intervals with confidence at least:").pack(side=tk.RIGHT)

        threading.Thread(target=self._detect_device, daemon=True).start()
        self._check_llama()

        body = ttk.Frame(self.root, padding=6)
        body.pack(fill=tk.BOTH, expand=True)

        # ---------------- left side: the video player ----------------
        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(left, bg="black", width=DISPLAY_W, height=DISPLAY_H)
        self.video_label.pack()

        self.timeline = tk.Canvas(left, width=DISPLAY_W, height=TIMELINE_H, bg="#dfe4ea",
                                  highlightthickness=0)
        self.timeline.pack(pady=(4, 0))
        self.timeline.bind("<Button-1>", self._on_timeline_click)

        legend = ttk.Frame(left)
        legend.pack(pady=2)
        for label in GUARD_LABELS:
            tk.Label(legend, bg=GUARD_COLOURS[label], width=2).pack(side=tk.LEFT, padx=(8, 2))
            ttk.Label(legend, text=label).pack(side=tk.LEFT)

        controls = ttk.Frame(left)
        controls.pack(fill=tk.X, pady=4)
        self.play_btn = ttk.Button(controls, text="Play", width=8, command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT)
        self.slider = ttk.Scale(controls, from_=0, to=1, orient=tk.HORIZONTAL, command=self._on_slider)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.time_var = tk.StringVar(value="00:00.00 / 00:00.00")
        ttk.Label(controls, textvariable=self.time_var, width=20).pack(side=tk.LEFT)

        # ---------------- right side: the list of positions ----------------
        right = ttk.Frame(body, padding=(8, 0, 0, 0))
        right.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(right, text="Guard intervals (click to jump, double-click to edit)").pack(anchor=tk.W)

        cols = ("position", "start", "end", "conf", "engine")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=16, selectmode="browse")
        for c, text, w in zip(cols, ("Position", "Start", "End", "Conf.", "Decided by"),
                              (140, 62, 62, 44, 105)):
            self.tree.heading(c, text=text)
            self.tree.column(c, width=w, anchor=tk.W if c == "position" else tk.CENTER)
        for label in GUARD_LABELS:  # give each row the same colour as on the timeline
            self.tree.tag_configure(label, background=_lighten(GUARD_COLOURS[label]))
        self.tree.pack(fill=tk.Y, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_row_select)
        self.tree.bind("<Double-1>", lambda e: self.edit_interval())

        btns = ttk.Frame(right)
        btns.pack(fill=tk.X, pady=4)
        for text, cmd in (("Add", self.add_interval), ("Edit", self.edit_interval),
                          ("Delete", self.delete_interval)):
            ttk.Button(btns, text=text, command=cmd).pack(side=tk.LEFT, expand=True, fill=tk.X)
        btns2 = ttk.Frame(right)
        btns2.pack(fill=tk.X)
        for text, cmd in (("Export JSON", self.export_json), ("Export CSV", self.export_csv),
                          ("Load JSON", self.load_json)):
            ttk.Button(btns2, text=text, command=cmd).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(right, text="Save Kalman plot (PNG)", command=self.save_plot).pack(fill=tk.X, pady=4)

        self.log_box = tk.Text(right, height=8, width=46, state=tk.DISABLED, wrap=tk.WORD,
                               font=("Consolas", 8))
        self.log_box.pack(fill=tk.X)

        self._show_placeholder("No video loaded")

    # ================================================================== small helpers
    def _detect_device(self):
        """Check in the background whether the graphics card can be used (this takes a few seconds)."""
        from pose import pick_device
        _, name = pick_device()
        self.msg_queue.put(("device", name))

    def _show_device(self, name):
        on_gpu = name.startswith("GPU")
        self.device_label.configure(text=f"Running on: {name}", bg="#27ae60" if on_gpu else "#e67e22")

    def _check_llama(self, show_popup=False):
        """Ask Ollama (in the background) whether Llama 3 is ready, then update the badge."""
        self.llama_label.configure(text="Llama 3: checking...", bg="#7f8c8d")

        def work():
            from reasoning import LlamaGuardReasoner
            llama = LlamaGuardReasoner()
            ok, msg = llama.check_available()
            self.msg_queue.put(("llama", ok, msg, llama.model, show_popup))
        threading.Thread(target=work, daemon=True).start()

    def _show_llama(self, ok, msg, model, show_popup):
        if ok:
            self.llama_label.configure(text=f"Llama 3: ready ({model})", bg="#27ae60")
        else:
            self.llama_label.configure(text="Llama 3: not available (click for why)", bg="#7f8c8d")
            if self.llm_var.get():  # the user ticked it but it isn't there -> untick it and explain why
                self.llm_var.set(False)
                show_popup = True
        if show_popup:
            (messagebox.showinfo if ok else messagebox.showwarning)(
                "Llama 3", msg + ("" if ok else "\n\nThe rule-based engine will be used instead."))

    def _on_llama_toggle(self):
        if self.llm_var.get():
            self._check_llama()  # check again now; if it isn't available the box is unticked and the reason shown

    def log(self, text):
        self.log_box.configure(state=tk.NORMAL)
        self.log_box.insert(tk.END, text + "\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def duration(self):
        return self.n_frames / self.fps if self.fps else 0.0

    def _show_placeholder(self, text):
        img = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
        cv2.putText(img, text, (30, DISPLAY_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        self._put_image(img)

    def _put_image(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=photo, width=bgr.shape[1], height=bgr.shape[0])
        self.video_label.image = photo  # keep hold of the picture, otherwise it disappears from the screen

    # ================================================================== playing the video
    def open_video(self):
        path = filedialog.askopenfilename(title="Open BJJ sparring video", filetypes=VIDEO_TYPES)
        if not path:
            return
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if not ok:
            cap.release()
            messagebox.showerror("Unsupported file",
                                 f"Could not read video frames from:\n{path}\n\nTry an MP4 (H.264) file.")
            return
        if self.cap is not None:
            self.cap.release()
        self.pause()
        self.cap, self.video_path = cap, path
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 < fps < 240 else 30.0
        self.n_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.result, self.intervals, self.edited = None, [], False
        self._refresh_intervals()
        self._slider_busy = True
        self.slider.configure(to=max(1, self.n_frames - 1))
        self._slider_busy = False
        self.seek_frame(0)
        name = os.path.basename(path)
        self.status_var.set(f"Loaded {name} ({format_time(self.duration())}, {self.fps:.0f} fps)")
        self.log(f"Opened {name}")

    def seek_frame(self, idx):
        if self.cap is None:
            return
        idx = int(min(max(idx, 0), self.n_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if ok:
            self.current_frame = idx
            self.last_frame_img = frame
            self._redraw()

    def seek_time(self, seconds):
        self.seek_frame(round(seconds * self.fps))

    def toggle_play(self):
        if self.cap is None:
            messagebox.showinfo("No video", "Please open a video first.")
            return
        if self.playing:
            self.pause()
        else:
            self.playing = True
            self.play_btn.configure(text="Pause")
            self._play_step()

    def pause(self):
        self.playing = False
        self.play_btn.configure(text="Play")

    def _play_step(self):
        if not self.playing or self.cap is None:
            return
        t0 = time.perf_counter()
        ok, frame = self.cap.read()
        if not ok:  # reached the end of the video
            self.pause()
            return
        self.current_frame += 1
        self.last_frame_img = frame
        self._redraw()
        # wait just long enough so the video plays at its normal speed
        spent_ms = (time.perf_counter() - t0) * 1000
        self.root.after(max(1, int(1000 / self.fps - spent_ms)), self._play_step)

    def _redraw(self):
        """Show the current frame (with the skeleton if switched on) and move the slider, clock and timeline marker."""
        if self.last_frame_img is None:
            return
        frame = self.last_frame_img.copy()
        if self.skel_var.get() and self.result is not None:
            self._draw_skeleton(frame)
        h, w = frame.shape[:2]
        scale = min(DISPLAY_W / w, DISPLAY_H / h)
        frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        self._put_image(frame)

        self._slider_busy = True
        self.slider.set(self.current_frame)
        self._slider_busy = False
        now = self.current_frame / self.fps
        self.time_var.set(f"{format_time(now)} / {format_time(self.duration())}")
        self._draw_timeline()

    def _draw_skeleton(self, frame):
        """Draw the smoothed skeleton from the nearest analysed frame on top of the video.

        Filled dots = the camera saw the joint; hollow dots = the joint was hidden and its
        position was filled in by the Kalman filter.
        """
        idxs = self.result["frame_indices"]
        i = int(np.clip(np.searchsorted(idxs, self.current_frame), 0, len(idxs) - 1))
        if i > 0 and abs(idxs[i - 1] - self.current_frame) < abs(idxs[i] - self.current_frame):
            i -= 1
        if abs(idxs[i] - self.current_frame) > 2 * self.result["frame_step"]:
            return  # no analysed frame close enough to this moment
        raw = self.result["raw_keypoints"][i]
        sm = self.result["smoothed_keypoints"][i]
        thick = max(2, frame.shape[1] // 400)
        for a in range(sm.shape[0]):
            if np.all(np.isnan(raw[a, :, 2])):
                continue  # this athlete wasn't found in this frame
            colour = ATHLETE_COLOURS[a % 2]
            pts = sm[a, :, :2]
            for j1, j2 in SKELETON_EDGES:
                if not np.isnan(pts[[j1, j2]]).any():
                    cv2.line(frame, tuple(pts[j1].astype(int)), tuple(pts[j2].astype(int)), colour, thick)
            for j in range(pts.shape[0]):
                if np.isnan(pts[j]).any():
                    continue
                seen = raw[a, j, 2] >= 0.35
                cv2.circle(frame, tuple(pts[j].astype(int)), thick * 2, colour, -1 if seen else thick)

    def _on_slider(self, value):
        if self._slider_busy or self.cap is None:
            return
        self.seek_frame(float(value))

    # ================================================================== the colour timeline
    def _draw_timeline(self):
        c = self.timeline
        c.delete("all")
        W, dur = DISPLAY_W, self.duration()
        if dur <= 0:
            return
        for iv in self.intervals:
            try:
                x0 = parse_time(iv["t_start"]) / dur * W
                x1 = parse_time(iv["t_end"]) / dur * W
            except ValueError:
                continue
            colour = GUARD_COLOURS.get(iv["guard_position"], "#7f8c8d")
            c.create_rectangle(x0, 2, max(x1, x0 + 2), TIMELINE_H - 2, fill=colour, outline="")
        x = self.current_frame / max(1, self.n_frames - 1) * W  # the black line showing where we are
        c.create_line(x, 0, x, TIMELINE_H, fill="black", width=2)

    def _on_timeline_click(self, event):
        if self.cap is not None:
            self.seek_time(event.x / DISPLAY_W * self.duration())

    # ================================================================== running the analysis
    def start_analysis(self):
        if self.cap is None:
            messagebox.showwarning("No video", "Please open a video before analysing.")
            return
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("Busy", "An analysis is already running.")
            return
        try:
            step = max(1, int(self.step_var.get()))
        except (tk.TclError, ValueError):
            step = 3
        self.pause()
        self.analyse_btn.configure(state=tk.DISABLED)
        self.progress["value"] = 0
        self.stop_event.clear()
        self.log(f"--- Analysing (every {step} frame(s)) ---")

        def progress(p, msg=""):
            self.msg_queue.put(("progress", p, msg))

        def log(text):
            self.msg_queue.put(("log", text))

        # Read the tick boxes now: the background task is not allowed to touch the window itself
        use_audio, use_llm, video_path = self.audio_var.get(), self.llm_var.get(), self.video_path
        min_conf = self._min_confidence()
        ignore_referee = self.ref_var.get()
        self.log("Reasoning engine requested: " + ("Llama 3 (checked against rules)" if use_llm else "Rules"))

        def work():  # runs in the background so the window doesn't freeze
            try:
                from pipeline import GrappleEvalPipeline  # loading the AI libraries here keeps the window responsive
                pipe = GrappleEvalPipeline(frame_step=step, use_audio=use_audio,
                                           use_llm=use_llm, min_confidence=min_conf,
                                           ignore_referee=ignore_referee, progress_cb=progress,
                                           log_cb=log, stop_event=self.stop_event)
                self.msg_queue.put(("done", pipe.run(video_path)))
            except Exception as e:
                self.msg_queue.put(("error", f"{e}"))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _poll_worker(self):
        """Ten times a second, pick up news from the background analysis and update the window."""
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg[0] == "progress":
                    self.progress["value"] = msg[1] * 100
                    if msg[2]:
                        self.status_var.set(msg[2])
                elif msg[0] == "log":
                    self.log(msg[1])
                elif msg[0] == "device":
                    self._show_device(msg[1])
                    self.log(f"Hardware: {msg[1]}")
                elif msg[0] == "llama":
                    self._show_llama(*msg[1:])
                elif msg[0] == "done":
                    self.result = msg[1]
                    self.intervals = self.result["intervals"]
                    self.edited = False
                    self._show_device(self.result["device"])  # shows if it had to switch to the CPU
                    self._refresh_intervals()
                    self.analyse_btn.configure(state=tk.NORMAL)
                    self.status_var.set(f"Done: {len(self.intervals)} intervals | {self.result['engine']}")
                    self._redraw()
                    if not self.intervals:
                        messagebox.showinfo(
                            "Nothing confident found",
                            "No position was recognised with enough confidence.\n\n"
                            "You can lower 'Only show intervals with confidence at least' and press Apply.")
                elif msg[0] == "error":
                    self.analyse_btn.configure(state=tk.NORMAL)
                    self.progress["value"] = 0
                    self.status_var.set("Analysis failed.")
                    self.log("ERROR: " + msg[1])
                    messagebox.showerror("Analysis failed", msg[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_worker)

    def _min_confidence(self):
        try:
            return min(max(float(self.minconf_var.get()), 0.0), 0.99)
        except (tk.TclError, ValueError):
            return 0.75

    def apply_confidence_filter(self):
        """Rebuild the list with a new confidence limit, using the saved answers (no need to analyse again)."""
        if self.result is None:
            messagebox.showinfo("Confidence filter", "Run 'Analyse' first.")
            return
        if self.edited and not messagebox.askyesno(
                "Confidence filter", "This rebuilds the list from the analysis and\n"
                                     "throws away your manual changes. Continue?"):
            return
        from reasoning import merge_windows_into_intervals
        min_conf = self._min_confidence()
        self.intervals = merge_windows_into_intervals(
            self.result["fact_sheets"], self.result["window_labels"], min_confidence=min_conf)
        self.edited = False
        self._refresh_intervals()
        self.log(f"Filter: {len(self.intervals)} intervals with confidence >= {min_conf:.2f}")
        self.status_var.set(f"{len(self.intervals)} intervals with confidence >= {min_conf:.2f}")

    # ================================================================== the list of positions
    def _refresh_intervals(self):
        self.intervals.sort(key=lambda iv: _safe_time(iv["t_start"]))
        self.tree.delete(*self.tree.get_children())
        for i, iv in enumerate(self.intervals):
            self.tree.insert("", tk.END, iid=str(i), tags=(iv["guard_position"],), values=(
                iv["guard_position"], iv["t_start"], iv["t_end"], f"{float(iv['confidence']):.2f}",
                iv.get("source", "")))
        self._draw_timeline()

    def _selected_index(self):
        sel = self.tree.selection()
        return int(sel[0]) if sel else None

    def _on_row_select(self, _event):
        i = self._selected_index()
        if i is None or self.cap is None:
            return
        try:
            self.seek_time(parse_time(self.intervals[i]["t_start"]))  # the main feature: jump straight to that moment
        except ValueError:
            pass

    def add_interval(self):
        now = self.current_frame / self.fps if self.cap else 0.0
        new = {"guard_position": GUARD_LABELS[0], "t_start": format_time(now),
               "t_end": format_time(min(now + 5, self.duration() or now + 5)), "confidence": 1.0,
               "source": "Manual"}
        result = IntervalDialog(self.root, "Add interval", new).result
        if result:
            self.intervals.append(result)
            self.edited = True
            self._refresh_intervals()

    def edit_interval(self):
        i = self._selected_index()
        if i is None:
            messagebox.showinfo("Edit", "Select an interval in the list first.")
            return
        result = IntervalDialog(self.root, "Edit interval", self.intervals[i]).result
        if result:
            self.intervals[i] = result
            self.edited = True
            self._refresh_intervals()

    def delete_interval(self):
        i = self._selected_index()
        if i is None:
            messagebox.showinfo("Delete", "Select an interval in the list first.")
            return
        iv = self.intervals[i]
        if messagebox.askyesno("Delete", f"Delete {iv['guard_position']} {iv['t_start']}-{iv['t_end']}?"):
            del self.intervals[i]
            self.edited = True
            self._refresh_intervals()

    # ================================================================== saving and loading
    def _default_name(self, ext):
        base = os.path.splitext(os.path.basename(self.video_path))[0] if self.video_path else "guards"
        return f"{base}_guards{ext}"

    def export_json(self):
        if not self.intervals:
            messagebox.showinfo("Export", "There are no intervals to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile=self._default_name(".json"),
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        data = {"video": os.path.basename(self.video_path) if self.video_path else None,
                "intervals": self.intervals}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.log(f"Saved {path}")
        except OSError as e:
            messagebox.showerror("Export failed", str(e))

    def export_csv(self):
        if not self.intervals:
            messagebox.showinfo("Export", "There are no intervals to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=self._default_name(".csv"),
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["guard_position", "t_start", "t_end", "confidence", "source"],
                                   restval="")
                w.writeheader()
                w.writerows(self.intervals)
            self.log(f"Saved {path}")
        except OSError as e:
            messagebox.showerror("Export failed", str(e))

    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            items = data["intervals"] if isinstance(data, dict) else data  # accept both file layouts
            loaded = []
            for it in items:
                parse_time(it["t_start"]), parse_time(it["t_end"])  # check the times are valid
                loaded.append({"guard_position": str(it["guard_position"]),
                               "t_start": format_time(parse_time(it["t_start"])),
                               "t_end": format_time(parse_time(it["t_end"])),
                               "confidence": float(it.get("confidence", 1.0)),
                               "source": str(it.get("source", "Loaded"))})
        except (OSError, ValueError, KeyError, TypeError) as e:
            messagebox.showerror("Load failed", f"Not a valid Grapple-Eval JSON file:\n{e}")
            return
        self.intervals = loaded
        self.edited = True  # this list didn't come from the analysis, so protect it from the confidence filter
        self._refresh_intervals()
        self.log(f"Loaded {len(loaded)} intervals from {os.path.basename(path)}")

    def save_plot(self):
        if self.result is None:
            messagebox.showinfo("Kalman plot", "Run 'Analyse' first - the plot needs the keypoint data.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=self._default_name("_kalman.png"),
                                            filetypes=[("PNG image", "*.png")])
        if not path:
            return
        try:
            from pipeline import save_kalman_plot
            # Use the athlete whose left ankle was seen most often, so the plot has something to show
            conf = self.result["raw_keypoints"][:, :, 15, 2]
            athlete = int(np.argmax(np.sum(~np.isnan(conf), axis=0)))
            save_kalman_plot(self.result, path, athlete=athlete, joint=15, joint_name="left ankle")
            self.log(f"Saved Kalman plot to {path}")
        except Exception as e:
            messagebox.showerror("Plot failed", str(e))

    def on_close(self):
        self.stop_event.set()  # tell an analysis that is still running to stop
        self.playing = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


class IntervalDialog:
    """Small pop-up window to add or edit one position. Its result is None if the user presses Cancel."""

    def __init__(self, parent, title, interval):
        self.result = None
        top = self.top = tk.Toplevel(parent)
        top.title(title)
        top.transient(parent)
        top.resizable(False, False)
        frm = ttk.Frame(top, padding=10)
        frm.pack()

        self.label_var = tk.StringVar(value=interval["guard_position"])
        self.start_var = tk.StringVar(value=interval["t_start"])
        self.end_var = tk.StringVar(value=interval["t_end"])
        self.conf_var = tk.StringVar(value=str(interval["confidence"]))

        ttk.Label(frm, text="Guard position").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(frm, textvariable=self.label_var, values=GUARD_LABELS, width=24).grid(row=0, column=1)
        for r, (text, var) in enumerate((("Start (mm:ss.cc)", self.start_var),
                                         ("End (mm:ss.cc)", self.end_var),
                                         ("Confidence (0-1)", self.conf_var)), start=1):
            ttk.Label(frm, text=text).grid(row=r, column=0, sticky=tk.W, pady=2)
            ttk.Entry(frm, textvariable=var, width=26).grid(row=r, column=1)
        bf = ttk.Frame(frm)
        bf.grid(row=4, column=0, columnspan=2, pady=(8, 0))
        ttk.Button(bf, text="OK", command=self._ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text="Cancel", command=top.destroy).pack(side=tk.LEFT)

        top.grab_set()          # the main window can't be used until this pop-up closes
        parent.wait_window(top)

    def _ok(self):
        try:
            start, end = parse_time(self.start_var.get()), parse_time(self.end_var.get())
            conf = float(self.conf_var.get())
            label = self.label_var.get().strip()
            if not label:
                raise ValueError("Guard position can't be empty.")
            if end <= start:
                raise ValueError("End time must be after start time.")
            if not 0 <= conf <= 1:
                raise ValueError("Confidence must be between 0 and 1.")
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=self.top)
            return
        self.result = {"guard_position": label, "t_start": format_time(start),
                       "t_end": format_time(end), "confidence": round(conf, 2), "source": "Manual"}
        self.top.destroy()


def _safe_time(text):
    try:
        return parse_time(text)
    except ValueError:
        return 0.0


def _lighten(hex_colour, amount=0.65):
    """Make a colour paler so the text in the list stays easy to read on top of it."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (int(v + (255 - v) * amount) for v in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


if __name__ == "__main__":
    root = tk.Tk()
    GrappleEvalApp(root)
    root.mainloop()
