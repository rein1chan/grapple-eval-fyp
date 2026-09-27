"""Turning body points into useful measurements (report sections 3.5, 3.6 and 4.5).

Raw positions in pixels mean little on their own - a close-up camera makes everything
look far apart. So for every frame the body points are turned into measurements that
don't depend on camera distance or body size:
  * hip proximity - how close the two athletes' hips are, measured in "torso lengths"
  * knee flexion  - how bent the bottom athlete's knees are (180 degrees = straight leg)
  * ankle spread  - how far apart the bottom athlete's feet are, in "torso lengths"
plus three yes/no checks: is each athlete standing, is the bottom athlete down on the
mat, and are their legs touching the other athlete.
Every second of video (30 frames) is then summarised - the average and how much each
value wobbled - in a small "fact sheet" that the reasoning step reads.
"""

from collections import deque

import numpy as np

from pose import (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)
from utils import format_time

METRICS = ["hip_proximity", "knee_flexion_deg", "ankle_spread"]


def _mid(kps, a, b):
    """The point halfway between two body points (e.g. the centre of the hips)."""
    return (kps[a, :2] + kps[b, :2]) / 2.0


def joint_angle(a, b, c):
    """The angle at point b (in degrees) made by the lines b-a and b-c, e.g. the knee angle
    between hip, knee and ankle (the arccos formula of section 3.6)."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6 or np.isnan(n1) or np.isnan(n2):
        return np.nan
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)  # keeps tiny rounding errors in range
    return float(np.degrees(np.arccos(cos)))


def _body_shape(kps):
    """Returns (upright, feet_below, knee_angle) for one athlete, or None if unknown.

    upright    : 1.0 = body perfectly vertical with the shoulders on top
    feet_below : how far the ankles are below the hips, in torso lengths
    knee_angle : average hip-knee-ankle angle of both legs (180 = straight)
    (In pictures, the y value gets bigger going down.)
    """
    sh = _mid(kps, L_SHOULDER, R_SHOULDER)
    hip = _mid(kps, L_HIP, R_HIP)
    ank = _mid(kps, L_ANKLE, R_ANKLE)
    torso = np.linalg.norm(sh - hip)
    if np.isnan(torso) or torso < 1e-3 or np.isnan(ank).any():
        return None
    knees = [joint_angle(kps[L_HIP, :2], kps[L_KNEE, :2], kps[L_ANKLE, :2]),
             joint_angle(kps[R_HIP, :2], kps[R_KNEE, :2], kps[R_ANKLE, :2])]
    if np.all(np.isnan(knees)):
        return None
    return (hip[1] - sh[1]) / torso, (ank[1] - hip[1]) / torso, float(np.nanmean(knees))


def is_standing(kps):
    """True if this athlete looks like they are up on their feet.

    Three checks, all in torso lengths so the camera distance doesn't matter:
      * body upright      - shoulders roughly straight above the hips (within about 45 degrees);
      * feet far below    - ankles more than 1.2 torso lengths below the hips
                            (kneeling or sitting puts the feet much closer to hip height);
      * legs fairly straight - knee angle above 120 degrees (kneeling is about 90).
    """
    shape = _body_shape(kps)
    if shape is None:
        return False
    upright, feet_below, knee = shape
    return upright > 0.7 and feet_below > 1.2 and knee > 120


def is_grounded(kps):
    """True if this athlete is down on the mat (lying, sitting or on their back).

    A guard player's feet are roughly level with, or above, their hips. People who are
    standing or kneeling have their feet clearly below their hips.
    """
    shape = _body_shape(kps)
    if shape is None:
        return False
    _, feet_below, _ = shape
    return feet_below < 0.8 and not is_standing(kps)


class GrappleMiddlewareBridge:
    """Turns the smoothed body points into measurements for every frame, and into one
    fact sheet for every second of video."""

    def __init__(self, window_size=30, frame_step=1):
        # One window = 30 video frames = 1 second at 30 fps. If only every 3rd frame is
        # analysed, that is 10 analysed frames per window.
        self.window_size = window_size
        self.samples_per_window = max(1, window_size // max(1, frame_step))
        self.frame_buffer = deque(maxlen=self.samples_per_window)  # holds the frames of the current second
        self.previous_sheet = None

    # ------------------------------------------------------------------ one frame
    def process_frame_kinematics(self, frame_id, timestamp, kps_a, kps_b, seen_a=True, seen_b=True):
        """Work out the measurements for one frame and add them to the current second.

        kps_a / kps_b: the smoothed body points of the two athletes.
        seen_a / seen_b: whether the pose model really found that athlete in this frame.
        """
        entry = {"frame_id": int(frame_id), "timestamp": float(timestamp),
                 "hip_proximity": np.nan, "knee_flexion_deg": np.nan, "ankle_spread": np.nan,
                 "standing_count": np.nan, "bottom_grounded": np.nan, "leg_contact": np.nan}

        if seen_a and seen_b:
            # How many of the two athletes are on their feet right now (0, 1 or 2)
            entry["standing_count"] = int(is_standing(kps_a)) + int(is_standing(kps_b))

            torso_a = np.linalg.norm(_mid(kps_a, L_SHOULDER, R_SHOULDER) - _mid(kps_a, L_HIP, R_HIP))
            torso_b = np.linalg.norm(_mid(kps_b, L_SHOULDER, R_SHOULDER) - _mid(kps_b, L_HIP, R_HIP))
            avg_scale = (torso_a + torso_b) / 2.0  # the average torso length is our "ruler"

            if avg_scale > 1e-3 and not np.isnan(avg_scale):
                # 1) Hip proximity: distance between the two hip centres, in torso lengths
                hip_dist = np.linalg.norm(_mid(kps_a, L_HIP, R_HIP) - _mid(kps_b, L_HIP, R_HIP))
                entry["hip_proximity"] = float(hip_dist / avg_scale)

                # Who is the bottom (guard) player? The one whose shoulders are lower in the picture
                ya = _mid(kps_a, L_SHOULDER, R_SHOULDER)[1]
                yb = _mid(kps_b, L_SHOULDER, R_SHOULDER)[1]
                bottom = kps_a if ya >= yb else kps_b

                # Is the bottom player really down on the mat? (there is no guard without that)
                entry["bottom_grounded"] = int(is_grounded(bottom))

                # Are the legs tangled up? Yes if one of the bottom player's knees or ankles is
                # within half a torso length of the top player's hips, knees or ankles.
                top = kps_b if bottom is kps_a else kps_a
                legs = bottom[[L_KNEE, R_KNEE, L_ANKLE, R_ANKLE], :2]
                targets = top[[L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE], :2]
                dists = np.linalg.norm(legs[:, None, :] - targets[None, :, :], axis=2)
                if not np.all(np.isnan(dists)):
                    entry["leg_contact"] = int(np.nanmin(dists) / avg_scale < 0.5)

                # 2) Knee flexion: the hip-knee-ankle angle, averaged over both legs
                left = joint_angle(bottom[L_HIP, :2], bottom[L_KNEE, :2], bottom[L_ANKLE, :2])
                right = joint_angle(bottom[R_HIP, :2], bottom[R_KNEE, :2], bottom[R_ANKLE, :2])
                if not (np.isnan(left) and np.isnan(right)):
                    entry["knee_flexion_deg"] = float(np.nanmean([left, right]))

                # 3) Ankle spread: distance between the bottom player's feet, in torso lengths
                spread = np.linalg.norm(bottom[L_ANKLE, :2] - bottom[R_ANKLE, :2])
                entry["ankle_spread"] = float(spread / avg_scale)

        self.frame_buffer.append(entry)
        return entry

    def is_buffer_complete(self):
        return len(self.frame_buffer) == self.samples_per_window

    # ------------------------------------------------------------------ one second
    def build_payload(self, audio_segments=None):
        """Summarise the current second into a fact sheet (a simple dictionary)."""
        frames = list(self.frame_buffer)
        t0, t1 = frames[0]["timestamp"], frames[-1]["timestamp"]
        sheet = {
            "window_start_s": round(t0, 3),
            "window_end_s": round(t1, 3),
            "window_start_timestamp": format_time(t0),
            "window_end_timestamp": format_time(t1),
        }
        valid = 0
        for m in METRICS:
            vals = np.array([f[m] for f in frames], dtype=float)
            vals = vals[~np.isnan(vals)]
            sheet[f"mean_{m}"] = round(float(vals.mean()), 3) if len(vals) else None   # the average
            sheet[f"std_{m}"] = round(float(vals.std()), 3) if len(vals) else None     # how much it wobbled
            if m == "hip_proximity":
                valid = len(vals)
        # Share of frames in which both athletes were visible (how well we could see)
        sheet["valid_frame_ratio"] = round(valid / len(frames), 2)

        # Share of (visible) frames in which BOTH athletes were standing -> "Stand-up"
        standing = np.array([f["standing_count"] for f in frames], dtype=float)
        standing = standing[~np.isnan(standing)]
        sheet["frac_both_standing"] = round(float((standing == 2).mean()), 2) if len(standing) else 0.0

        # Share of (visible) frames in which the bottom player was down on the mat
        grounded = np.array([f["bottom_grounded"] for f in frames], dtype=float)
        grounded = grounded[~np.isnan(grounded)]
        sheet["frac_bottom_grounded"] = round(float(grounded.mean()), 2) if len(grounded) else 0.0

        # Share of (visible) frames in which the athletes' legs were tangled together
        contact = np.array([f["leg_contact"] for f in frames], dtype=float)
        contact = contact[~np.isnan(contact)]
        sheet["frac_leg_contact"] = round(float(contact.mean()), 2) if len(contact) else 0.0

        # How much the hip distance changed since the previous second (a sign of fast movement)
        prev = self.previous_sheet
        if prev and prev.get("mean_hip_proximity") is not None and sheet["mean_hip_proximity"] is not None:
            sheet["delta_hip_proximity"] = round(sheet["mean_hip_proximity"] - prev["mean_hip_proximity"], 3)
        else:
            sheet["delta_hip_proximity"] = 0.0

        # Add any speech that was heard during this second, and the BJJ words in it
        texts, keywords = [], []
        for seg in audio_segments or []:
            if seg["end"] >= t0 and seg["start"] <= t1 + 1e-6:
                texts.append(seg["text"])
                keywords += [k for k in seg["keywords"] if k not in keywords]
        sheet["aligned_audio_transcript"] = " ".join(texts)
        sheet["audio_keywords"] = keywords

        self.previous_sheet = sheet
        return sheet

    # ------------------------------------------------------------------ the whole video
    def run(self, timestamps, frame_indices, smoothed_kps, raw_kps, audio_segments=None):
        """Go through every analysed frame and return one fact sheet per second of video.

        The seconds follow each other without overlapping, to keep things simple.
        """
        sheets = []
        self.frame_buffer.clear()
        self.previous_sheet = None
        for i in range(len(timestamps)):
            # An athlete counts as "seen" if the pose model found any body points for them
            seen_a = not np.all(np.isnan(raw_kps[i, 0, :, 2]))
            seen_b = not np.all(np.isnan(raw_kps[i, 1, :, 2]))
            self.process_frame_kinematics(frame_indices[i], timestamps[i],
                                          smoothed_kps[i, 0], smoothed_kps[i, 1], seen_a, seen_b)
            if self.is_buffer_complete():
                sheets.append(self.build_payload(audio_segments))
                self.frame_buffer.clear()
        if self.frame_buffer:  # the last, shorter piece at the end of the video
            sheets.append(self.build_payload(audio_segments))
            self.frame_buffer.clear()
        return sheets
