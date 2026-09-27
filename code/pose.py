"""Finding the athletes' body points (report sections 3.2 and 4.2).

Video frames are shown to a small ready-trained AI model (YOLOv11 Pose). For every
person it sees, it returns 17 body points (nose, shoulders, hips, knees, ankles, ...)
and how sure it is about each one. A built-in tracker (ByteTrack) gives each person
an ID number that stays the same from frame to frame, so we can keep following the
SAME two athletes and ignore the referee and the people watching.
"""

import cv2
import numpy as np

N_JOINTS = 17  # the model always returns 17 body points per person

# Numbers of the body points we use (so the code reads like English)
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Which body points to join with lines when drawing the skeleton on the video
SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),       # shoulders and arms
    (5, 11), (6, 12), (11, 12),                    # body
    (11, 13), (13, 15), (12, 14), (14, 16),        # legs
    (0, 5), (0, 6),                                # head to shoulders
]


def pick_device():
    """Use the NVIDIA graphics card if one is available, otherwise the normal processor."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0, f"GPU ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    return "cpu", "CPU"


class VisualWorker:
    """Runs the pose model and tracker over a video and keeps the two athletes."""

    def __init__(self, model_name="yolo11n-pose.pt", frame_step=3, det_conf=0.15, nms_iou=0.9,
                 ignore_referee=True):
        self.model_name = model_name
        self.frame_step = max(1, int(frame_step))  # only look at every Nth frame (faster, less detail)
        self.det_conf = det_conf
        # The model normally throws away boxes that overlap a lot, thinking they are the same
        # person twice. Tangled athletes overlap a lot too, so we allow much more overlap and
        # remove true doubles ourselves (see _remove_duplicates).
        self.nms_iou = nms_iou
        self.ignore_referee = ignore_referee
        self.device, self.device_name = pick_device()
        self.model = None
        self.stats = {}
        self._athlete_ids = set()

    def load_model(self):
        """Load the pose model (it downloads itself, about 6 MB, the very first time)."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("The 'ultralytics' package is not installed. "
                               "Run: pip install -r requirements.txt") from e
        self.model = YOLO(self.model_name)

    def run(self, video_path, progress_cb=None, stop_event=None):
        """Go through the video and return the body points of the two athletes.

        For every analysed frame we get 2 athletes x 17 body points x (x, y, confidence),
        left empty when an athlete or a point was not found.
        """
        if self.model is None:
            self.load_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1 or fps > 240:  # some files report a nonsense frame rate
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # Step 1: find everybody in every analysed frame
        frame_indices, frames_people = [], []
        frame_idx = 0
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError("Analysis cancelled.")
                # grab() skips over a frame quickly; we only fully read the frames we analyse
                if not cap.grab():
                    break
                if frame_idx % self.frame_step == 0:
                    ok, frame = cap.retrieve()
                    if ok:
                        frame_indices.append(frame_idx)
                        frames_people.append(self._detect(frame))
                    if progress_cb and total:
                        progress_cb(frame_idx / total)
                frame_idx += 1
        finally:
            cap.release()

        if not frame_indices:
            raise RuntimeError("The video has no readable frames (unsupported or corrupt file?).")

        # Step 2: work out who the referee is over the whole video, then follow the two athletes
        referee_ids = self._find_referee_ids(frames_people) if self.ignore_referee else set()
        slots = _AthleteSlots()  # keeps "athlete A" and "athlete B" the same people over time
        all_kps, removed = [], 0
        for people in frames_people:
            athletes = [p for p in people if not self._is_referee(p, referee_ids)]
            removed += len(people) - len(athletes)
            all_kps.append(slots.assign(athletes))
        self.stats["referee_detections_removed"] = removed

        keypoints = np.array(all_kps, dtype=float)
        return {
            "fps": fps,
            "frame_step": self.frame_step,
            "frame_indices": np.array(frame_indices),
            "timestamps": np.array(frame_indices) / fps,
            "keypoints": keypoints,
            "device": self.device_name,  # what was really used (it may have switched to the CPU)
            "stats": self.stats,
        }

    # ------------------------------------------------------------------ the referee
    def _find_referee_ids(self, frames_people):
        """Which tracked people are the referee?

        The referee wears long dark trousers while no-gi athletes have bare shins, so we
        check how bright each person's shins are. Each tracked person gets a vote over the
        whole video, so one misleading frame can't turn an athlete into the referee.
        """
        votes = {}
        for people in frames_people:
            for p in _on_the_mat(people):  # the crowd in the background doesn't get a vote
                if p["id"] is not None and p["dark_shins"] is not None:
                    votes.setdefault(p["id"], []).append(p["dark_shins"])
        voted = {pid: np.mean(v) for pid, v in votes.items() if len(v) >= 3}
        referee_ids = {pid for pid, dark in voted.items() if dark >= 0.6}
        self._athlete_ids = {pid for pid, dark in voted.items() if dark < 0.4}  # clearly bare shins

        # Safety check: removing the referee should only ever remove ONE extra person. If it
        # often leaves fewer than two people on the mat, the athletes must be wearing dark
        # trousers too (for example a gi match), so the filter is switched off.
        before = sum(len(_on_the_mat(people)) >= 2 for people in frames_people)
        after = sum(len(_on_the_mat([p for p in people if not self._is_referee(p, referee_ids)])) >= 2
                    for people in frames_people)
        if before and after < 0.6 * before:
            self.stats["referee_filter"] = "off (athletes seem to wear dark trousers too)"
            self.ignore_referee = False
            return set()
        self.stats["referee_filter"] = "on"
        return referee_ids

    def _is_referee(self, p, referee_ids):
        if not self.ignore_referee:
            return False
        if p["id"] in referee_ids:
            return True
        if p["id"] in self._athlete_ids:
            return False  # the whole-video vote says "athlete", so ignore one odd frame
        return p["dark_shins"] is True  # not tracked long enough to vote: judge by this frame only

    # ------------------------------------------------------------------ one frame
    def _detect(self, frame):
        """Run the pose model and tracker on one frame. Returns the people found."""
        try:
            results = self.model.track(
                frame, persist=True, tracker="bytetrack.yaml", conf=self.det_conf,
                iou=self.nms_iou, imgsz=640, device=self.device, verbose=False,
            )
        except Exception:
            if self.device == "cpu":
                raise
            # Problem with the graphics card (e.g. out of memory): switch to the processor and retry
            self.device, self.device_name = "cpu", "CPU (GPU failed, switched to CPU)"
            return self._detect(frame)
        r = results[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return []
        kps = r.keypoints.data.cpu().numpy()          # body points of each person: x, y, confidence
        boxes = r.boxes.xyxy.cpu().numpy()            # box around each person: left, top, right, bottom
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(boxes)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # a copy of the frame where brightness is easy to read
        people = []
        for k, b, tid in zip(kps, boxes, ids):
            if k.shape[0] != N_JOINTS:
                continue
            area = (b[2] - b[0]) * (b[3] - b[1])
            people.append({"id": tid, "kps": k, "area": area,
                           "diag": float(np.hypot(b[2] - b[0], b[3] - b[1])),  # rough body "length"
                           "center": np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]),
                           "dark_shins": _dark_shins(hsv, k)})
        return _remove_duplicates(people)


def _on_the_mat(people, min_rel_area=0.25):
    """People big enough to be on the mat (not the crowd far in the background)."""
    if not people:
        return []
    biggest = max(p["area"] for p in people)
    return [p for p in people if p["area"] >= min_rel_area * biggest]


def _upright(k, min_conf=0.5):
    """Is this person standing? Shoulders above hips, hips above knees, knees above ankles.
    (In pictures, the y value gets bigger going down.) Only clearly seen points are used."""
    def y(a, b):
        if k[a, 2] < min_conf or k[b, 2] < min_conf:
            return None
        return (k[a, 1] + k[b, 1]) / 2
    sh, hip = y(L_SHOULDER, R_SHOULDER), y(L_HIP, R_HIP)
    knee, ank = y(L_KNEE, R_KNEE), y(L_ANKLE, R_ANKLE)
    if None in (sh, hip, knee, ank):
        return False
    return sh < hip < knee < ank


def _dark_shins(hsv, k, dark_level=70):
    """True = shins covered by dark trousers (the referee), False = bare or light shins,
    None = can't tell. We check how bright the picture is at a few spots between each
    knee and ankle (0 = black, 255 = white).

    Only checked for people standing up: the referee always stands, and when the athletes
    are tangled on the mat the model often puts its "shins" on their shorts by mistake."""
    if not _upright(k):
        return None
    values = []
    for knee, ankle in ((L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE)):
        if k[knee, 2] < 0.5 or k[ankle, 2] < 0.5:
            continue
        for t in (0.3, 0.5, 0.7):
            x, y = (k[knee, :2] * (1 - t) + k[ankle, :2] * t).astype(int)
            patch = hsv[max(0, y - 3):y + 4, max(0, x - 3):x + 4, 2]
            if patch.size:
                values.append(patch.mean())
    if not values:
        return None
    return bool(np.median(values) < dark_level)


def _remove_duplicates(people):
    """Drop a detection that is really the same body as a bigger one we already kept
    (same skeleton in the same place). Two tangled athletes overlap too, but their
    skeletons are in different places, so both of them are kept."""
    kept = []
    for p in sorted(people, key=lambda q: q["area"], reverse=True):
        dup = False
        for q in kept:
            both = (p["kps"][:, 2] > 0.3) & (q["kps"][:, 2] > 0.3)  # points both detections found
            if both.sum() >= 5:
                gap = np.linalg.norm(p["kps"][both, :2] - q["kps"][both, :2], axis=1).mean()
                if gap < 0.1 * q["diag"]:
                    dup = True
                    break
        if not dup:
            kept.append(p)
    return kept


class _AthleteSlots:
    """Keeps two "slots" (athlete A and athlete B) filled with the SAME two athletes over time.

    The two biggest people are not always the athletes: the referee often stands close to
    the camera and looks big. So in every frame:
      * the biggest person on the mat is taken as one athlete (after the referee has been
        removed, this is an athlete - or both athletes merged into one detection);
      * every other person is scored as their partner:
          engaged        - athletes who grapple touch or are right next to each other;
          same as before - we prefer the people we were already following
                           (same tracker ID, or standing where they just were);
          size           - a small bonus for big people;
      * if the best partner is far away, unknown and small, they are probably in the crowd,
        so the second slot is left empty instead of guessing.
    People much smaller than the biggest person (far in the background) are ignored.
    """

    MIN_REL_AREA = 0.25                    # smaller than a quarter of the biggest person = background
    W_ENGAGED, W_SAME, W_SIZE = 2.0, 1.5, 0.5

    def __init__(self):
        self.ids = [None, None]
        self.centers = [None, None]
        self.diags = [None, None]

    def assign(self, people):
        out = np.full((2, N_JOINTS, 3), np.nan)
        if not people:
            return out
        biggest = max(p["area"] for p in people)
        cands = [p for p in people if p["area"] >= self.MIN_REL_AREA * biggest]
        cands = sorted(cands, key=lambda p: p["area"], reverse=True)[:5]

        if len(cands) == 1:
            # Only one person visible: put them in whichever slot they fit best
            slot = min((0, 1), key=lambda s: self._slot_cost(cands[0], s))
            self._put(out, slot, cands[0])
            return out

        # Pair the biggest person with whoever looks most like their grappling partner
        lead = cands[0]
        a, b = max(((lead, p) for p in cands[1:]), key=lambda ab: self._pair_score(ab[0], ab[1], biggest))

        # Only accept the partner if they touch the lead person, were already being followed,
        # or are nearly as big. Otherwise it is probably someone in the crowd while the
        # athletes are merged into one detection, so the second slot stays empty (an honest gap).
        gap = np.linalg.norm(a["center"] - b["center"]) / ((a["diag"] + b["diag"]) / 2)
        if gap >= 1.0 and not self._is_known(b) and b["area"] < 0.5 * biggest:
            slot = min((0, 1), key=lambda s: self._slot_cost(a, s))
            self._put(out, slot, a)
            return out

        # Keep athlete A as A and B as B (swap them if that matches the earlier frames better)
        if self._slot_cost(a, 1) + self._slot_cost(b, 0) < self._slot_cost(a, 0) + self._slot_cost(b, 1):
            a, b = b, a
        self._put(out, 0, a)
        self._put(out, 1, b)
        return out

    def _is_known(self, p):
        """Is this one of the two people we were following a moment ago?"""
        if p["id"] is not None and p["id"] in self.ids:
            return True
        for c, d in zip(self.centers, self.diags):  # tracker ID changed? check the position instead
            if c is not None and np.linalg.norm(c - p["center"]) < 0.3 * d:
                return True
        return False

    def _pair_score(self, a, b, biggest):
        # "engaged" is 1.0 when the two bodies are on top of each other, 0 when a body-length apart
        gap = np.linalg.norm(a["center"] - b["center"]) / ((a["diag"] + b["diag"]) / 2)
        engaged = max(0.0, 1.0 - gap)
        same = (self._is_known(a) + self._is_known(b)) / 2.0
        size = (a["area"] + b["area"]) / (2 * biggest)
        return self.W_ENGAGED * engaged + self.W_SAME * same + self.W_SIZE * size

    def _slot_cost(self, p, slot):
        """Low number = this person fits this slot well (same ID, or close to where it was last)."""
        if p["id"] is not None and self.ids[slot] == p["id"]:
            return 0.0
        if self.centers[slot] is None:
            return 1.0
        return 1.0 + float(np.linalg.norm(self.centers[slot] - p["center"])) / self.diags[slot]

    def _put(self, out, slot, person):
        out[slot] = person["kps"]
        self.ids[slot] = person["id"]
        self.centers[slot] = person["center"]
        self.diags[slot] = person["diag"]
