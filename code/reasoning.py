"""Deciding the position (report sections 3.6, 3.7 and 4.6).

This looks at each 1-second fact sheet and decides which position it shows.
  * By default, simple hand-written rules based on the body measurements of section 3.6
    (both on their feet = Stand-up, hips close + knees bent = Closed/Half Guard,
    further apart = Open/Butterfly, numbers jumping around = Guard Pass / Scramble).
    Words heard in the audio can back up the answer.
  * Optionally, Llama 3 (running on this computer through Ollama) gives its own answer,
    and the rules act as a second opinion. When the two agree we are confident; when
    they disagree the confidence is lowered.

Only showing positions we are fairly sure about:
  1. every answer gets an honest confidence score - high only when the measurements are
     clearly on one side of a rule, low when they are borderline;
  2. a second that disagrees with the seconds on both sides of it loses confidence;
  3. seconds below the minimum confidence are left out - EXCEPT when we are sure the legs
     are tangled in a guard but not sure which one: those become a plain "Guard";
  4. stretches shorter than MIN_INTERVAL_S seconds are left out.
"""

import json
import urllib.request

import numpy as np

from utils import GUARD_LABELS, format_time

# ---- Rule limits (distances in "torso lengths", angles in degrees). Change these to fine-tune. ----
STANDUP_FRACTION = 0.6    # both athletes standing for at least 60% of the second = Stand-up
MIN_GROUNDED_FRACTION = 0.5  # the bottom player must be on the mat at least half the second for a guard
CLOSE_HIPS = 1.2          # hips closer than 1.2 torso lengths = a close guard
NO_CONTACT_HIPS = 3.5     # hips further apart than this = not really grappling, so not a guard
CLOSED_ANKLES = 0.9       # feet this close together = ankles locked behind the opponent's back
CLOSED_KNEE_MAX = 110     # in closed guard the knees are bent more than this
BUTTERFLY_KNEE_MAX = 90   # butterfly guard: sitting up, knees sharply bent, feet used as hooks
BUTTERFLY_ANKLES = 1.2
BUTTERFLY_HIPS_MAX = 2.2
SCRAMBLE_HIP_STD = 0.30   # the hip distance wobbles a lot within one second
SCRAMBLE_HIP_JUMP = 0.60  # the hip distance changed a lot since the previous second
SCRAMBLE_KNEE_STD = 25.0  # the knee angle wobbles a lot within one second
MIN_VALID_RATIO = 0.5     # both athletes must be visible for at least half the second
MIN_LEG_CONTACT = 0.5     # legs tangled for at least half the second = "they are in some guard"

# The four specific guards. If we can't tell which one it is, we fall back to a plain "Guard".
SPECIFIC_GUARDS = ["Closed Guard", "Half Guard", "Open Guard", "Butterfly Guard"]
# Llama 3 has to pick a specific answer; the plain "Guard" is only our own fallback
LLAMA_LABELS = [label for label in GUARD_LABELS if label != "Guard"]

# ---- Filtering (the confidence limit can be changed in the app) ----
DEFAULT_MIN_CONFIDENCE = 0.75
MIN_INTERVAL_S = 2.0      # anything shorter than this is left out (too brief to trust)


def _sure(value, threshold, scale):
    """How clearly is a value on one side of a limit? 0 = right on the limit, 1 = clearly past it."""
    return float(np.clip(abs(value - threshold) / scale, 0.0, 1.0))


def guard_certainty(sheet):
    """How sure are we that the athletes are in SOME guard (whichever one it is)? From 0 to 0.99.

    This needs: both athletes visible, not both standing, the bottom player on the mat,
    and their legs tangled together.
    """
    valid = sheet.get("valid_frame_ratio", 0.0)
    grounded = sheet.get("frac_bottom_grounded", 0.0)
    contact = sheet.get("frac_leg_contact", 0.0)
    if (valid < MIN_VALID_RATIO or sheet.get("frac_both_standing", 0.0) >= STANDUP_FRACTION
            or grounded < MIN_GROUNDED_FRACTION or contact < MIN_LEG_CONTACT):
        return 0.0
    certainty = min(_sure(grounded, MIN_GROUNDED_FRACTION, 0.4), _sure(contact, MIN_LEG_CONTACT, 0.4))
    return round(min((0.6 + 0.4 * certainty) * (0.7 + 0.3 * valid), 0.99), 2)


class RuleBasedGuardClassifier:
    """The default decision-maker, which always works (no extra software needed)."""

    def classify(self, sheet):
        """Returns (position or None, confidence, who decided). None = not a guard / can't tell."""
        hip = sheet.get("mean_hip_proximity")
        knee = sheet.get("mean_knee_flexion_deg")
        ankle = sheet.get("mean_ankle_spread")
        valid = sheet.get("valid_frame_ratio", 0.0)
        keywords = sheet.get("audio_keywords") or []

        # The camera couldn't see both athletes -> don't guess (speech alone gets a low score)
        if hip is None or valid < MIN_VALID_RATIO:
            return (keywords[0], 0.5, "Audio") if keywords else (None, 0.0, "Rules")

        label, certainty = self._decide(sheet, hip, knee, ankle)
        if label is None:
            return None, 0.0, "Rules"

        # Confidence = how clear-cut the measurements were x how well we could see the athletes
        conf = (0.6 + 0.4 * certainty) * (0.7 + 0.3 * valid)
        source = "Rules"
        if keywords and label in keywords:  # the commentator said the same thing -> more sure
            conf += 0.1
            source = "Rules + Audio"
        return label, round(min(conf, 0.99), 2), source

    def _decide(self, sheet, hip, knee, ankle):
        """The rules themselves. Returns (position, how clear-cut it was from 0 to 1)."""
        # 1) Both athletes are on their feet
        both_up = sheet.get("frac_both_standing", 0.0)
        if both_up >= STANDUP_FRACTION:
            return "Stand-up", _sure(both_up, STANDUP_FRACTION, 1 - STANDUP_FRACTION)

        # There can be no guard (or guard pass) unless the bottom player is actually on the mat
        if sheet.get("frac_bottom_grounded", 0.0) < MIN_GROUNDED_FRACTION:
            return None, 0.0

        # 2) The numbers are changing fast = someone is passing the guard or scrambling
        hip_std = sheet.get("std_hip_proximity") or 0.0
        hip_jump = abs(sheet.get("delta_hip_proximity") or 0.0)
        knee_std = sheet.get("std_knee_flexion_deg") or 0.0
        excess = max(hip_std / SCRAMBLE_HIP_STD, hip_jump / SCRAMBLE_HIP_JUMP, knee_std / SCRAMBLE_KNEE_STD)
        if excess > 1.0:
            return "Guard Pass / Scramble", float(np.clip(excess - 1.0, 0, 1))

        # From here on we need the leg measurements; without them we can't tell the guards apart
        if knee is None or ankle is None:
            return None, 0.0

        # A guard means the bottom player is USING THEIR LEGS on the opponent. If the legs don't
        # touch the top player, it is side control, mount or back control - not a guard.
        if sheet.get("frac_leg_contact", 0.0) < MIN_LEG_CONTACT:
            return None, 0.0

        # 3) The athletes are far apart (and not standing) = nobody is in a guard
        if hip > NO_CONTACT_HIPS:
            return None, 0.0

        m_hip = _sure(hip, CLOSE_HIPS, 0.5)
        # 4) Hips close together = Closed Guard (feet locked + knees bent) or Half Guard
        if hip < CLOSE_HIPS:
            m_ankle = _sure(ankle, CLOSED_ANKLES, 0.4)
            m_knee = _sure(knee, CLOSED_KNEE_MAX, 30)
            if ankle < CLOSED_ANKLES and knee < CLOSED_KNEE_MAX:
                return "Closed Guard", min(m_hip, m_ankle, m_knee)
            # Half guard: at least one of the closed-guard checks clearly fails
            why_not_closed = max(m_ankle if ankle >= CLOSED_ANKLES else 0.0,
                                 m_knee if knee >= CLOSED_KNEE_MAX else 0.0)
            return "Half Guard", min(m_hip, why_not_closed)

        # 5) Hips further apart = Butterfly Guard (sitting, knees sharply bent) or Open Guard
        checks = [(knee, BUTTERFLY_KNEE_MAX, 30), (ankle, BUTTERFLY_ANKLES, 0.4),
                  (hip, BUTTERFLY_HIPS_MAX, 0.5)]
        if all(v < t for v, t, _ in checks):
            return "Butterfly Guard", min([m_hip] + [_sure(v, t, s) for v, t, s in checks])
        why_not_butterfly = max(_sure(v, t, s) for v, t, s in checks if v >= t)
        return "Open Guard", min(m_hip, why_not_butterfly)


class LlamaGuardReasoner:
    """Optional: asks Llama 3 (running on this computer through Ollama) to label each second."""

    # The instructions sent to Llama 3 with every request
    SYSTEM_PROMPT = (
        "You are a Brazilian Jiu-Jitsu video analyst. You receive a list of 1-second "
        "windows, each with body-geometry measurements of two grappling athletes:\n"
        "- frac_both_standing: fraction of the second where BOTH athletes were on their feet "
        "(above ~0.6 = Stand-up)\n"
        "- frac_bottom_grounded: fraction of the second the bottom player was down on the mat "
        "(below ~0.5 = nobody on the ground, so it is NOT a guard)\n"
        "- frac_leg_contact: fraction of the second the bottom player's legs touched the top "
        "player's hips or legs (high = legs entangled in a guard)\n"
        "- mean_hip_proximity: distance between the athletes' hips in torso lengths "
        "(below ~1.2 = chest-to-chest contact, above ~3.5 = not engaged)\n"
        "- mean_knee_flexion_deg: bottom player's hip-knee-ankle angle (180 = straight leg)\n"
        "- mean_ankle_spread: distance between the bottom player's feet in torso lengths "
        "(small = ankles crossed)\n"
        "- std_* values: how much each value changed inside the window (high = fast movement)\n"
        "- aligned_audio_transcript: what the coach said (may be empty)\n"
        "Rules of thumb: both standing = Stand-up; close hips + small angles = Closed Guard "
        "(ankles together) or Half Guard; larger distances = Open Guard or Butterfly Guard "
        "(knees sharply bent, feet close); fast-changing values = Guard Pass / Scramble. "
        "Use audio keywords to confirm. If you are unsure, give a low confidence.\n"
        f"Allowed labels (use exactly one of these): {', '.join(LLAMA_LABELS)}.\n"
        'Reply ONLY with JSON: {"windows": [{"index": <int>, "guard_position": "<label>", '
        '"confidence": <0..1>}, ...]} with one entry per input window.'
    )

    def __init__(self, model="llama3:8b", host="http://127.0.0.1:11434", chunk_size=15):
        self.model = model
        self.host = host.rstrip("/")
        self.chunk_size = chunk_size  # how many seconds are sent to Llama 3 in one go

    def check_available(self):
        """Returns (True, message) if Ollama is running and Llama 3 is downloaded."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as r:
                names = [m.get("name", "") for m in json.load(r).get("models", [])]
        except Exception:
            return False, ("Ollama is not running. Start the Ollama app "
                           "(or run 'ollama serve'), then try again.")
        if self.model not in names:
            # Also accept another Llama 3 version, e.g. 'llama3:latest'
            alt = next((n for n in names if n.startswith("llama3")), None)
            if not alt:
                return False, (f"Ollama is running but '{self.model}' isn't downloaded. "
                               "Run: ollama pull llama3:8b")
            self.model = alt
        return True, f"Ollama is running and '{self.model}' is ready."

    def _ask(self, windows):
        """Send some fact sheets to Llama 3 and return its answers."""
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "format": "json",               # makes Ollama reply in valid JSON
            "options": {"temperature": 0},  # no randomness: the same question gets the same answer
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(windows)},
            ],
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            content = json.load(r)["message"]["content"]
        return json.loads(content).get("windows", [])

    def classify_all(self, sheets, fallback, progress_cb=None):
        """Label every second with Llama 3, using the rules as a second opinion.

        Returns (answers, how many seconds Llama 3 answered). Each answer is
        (position, confidence, who decided).
        """
        rules = [fallback.classify(s) for s in sheets]
        results = list(rules)
        keep = ["frac_both_standing", "frac_bottom_grounded", "frac_leg_contact",
                "mean_hip_proximity", "std_hip_proximity",
                "mean_knee_flexion_deg", "std_knee_flexion_deg", "mean_ankle_spread",
                "std_ankle_spread", "valid_frame_ratio", "aligned_audio_transcript"]
        n_ok = 0
        n_batches = max(1, -(-len(sheets) // self.chunk_size))  # divide, rounding up
        for b, start in enumerate(range(0, len(sheets), self.chunk_size)):
            if progress_cb:
                progress_cb(b / n_batches, f"Llama 3 is thinking... batch {b + 1} of {n_batches}")
            chunk = []
            for i in range(start, min(start + self.chunk_size, len(sheets))):
                if sheets[i].get("mean_hip_proximity") is None:
                    continue  # the athletes weren't visible: nothing for Llama 3 to judge
                w = {"index": i, "t_start": sheets[i]["window_start_timestamp"]}
                w.update({k: sheets[i].get(k) for k in keep})
                chunk.append(w)
            if not chunk:
                continue
            for a in self._ask(chunk):  # if this fails, the pipeline falls back to the rules
                try:
                    i, label = int(a["index"]), a["guard_position"]
                    conf = float(a.get("confidence", 0.7))
                except (KeyError, TypeError, ValueError):
                    continue
                if label not in LLAMA_LABELS or not 0 <= i < len(sheets):
                    continue
                n_ok += 1
                rule_label, rule_conf, _ = rules[i]
                conf = float(np.clip(conf, 0.0, 1.0))
                if label == rule_label:
                    # Two independent methods agree -> trust it
                    results[i] = (label, round(max(conf, rule_conf), 2), "Llama 3 + Rules")
                else:
                    # They disagree -> keep Llama 3's answer but mark it as unsure
                    results[i] = (label, round(min(conf, rule_conf, 0.6), 2), "Llama 3 only")
        if progress_cb:
            progress_cb(1.0, "Llama 3 finished.")
        return results, n_ok


def apply_temporal_consistency(labels):
    """A second that disagrees with BOTH of its neighbours is probably a mistake, so lower its score.

    labels: one (position, confidence, who decided) per second. Returns a new list.
    """
    out = []
    for i, (label, conf, src) in enumerate(labels):
        if label is None:
            out.append((label, conf, src))
            continue
        neighbours = [labels[j][0] for j in (i - 1, i + 1) if 0 <= j < len(labels)]
        agree = sum(n == label for n in neighbours) / max(1, len(neighbours))
        out.append((label, round(conf * (0.75 + 0.25 * agree), 2), src))
    return out


def apply_guard_fallback(sheets, labels, min_confidence):
    """Not sure WHICH guard, but sure the legs are tangled in one? Then call it a plain "Guard".

    labels: one (position, confidence, who decided) per second. Returns a new list.
    """
    out = []
    for sheet, (label, conf, src) in zip(sheets, labels):
        unsure_guard = label is None or (label in SPECIFIC_GUARDS and conf < min_confidence)
        if unsure_guard:
            g = guard_certainty(sheet)
            if g >= min_confidence:
                out.append(("Guard", g, "Rules (legs tangled)"))
                continue
        out.append((label, conf, src))

    # A short run (1-2 seconds) of plain "Guard" between two confident stretches of the SAME
    # specific guard is almost certainly that guard too, so don't let it split that stretch.
    i = 0
    while i < len(out):
        if out[i][0] != "Guard":
            i += 1
            continue
        j = i
        while j < len(out) and out[j][0] == "Guard":
            j += 1
        before = out[i - 1] if i > 0 else (None, 0, "")
        after = out[j] if j < len(out) else (None, 0, "")
        if (j - i <= 2 and before[0] == after[0] and before[0] in SPECIFIC_GUARDS
                and before[1] >= min_confidence and after[1] >= min_confidence):
            for k in range(i, j):
                out[k] = (before[0], out[k][1], before[2])
        i = j
    return out


def merge_windows_into_intervals(sheets, labels, min_confidence=DEFAULT_MIN_CONFIDENCE,
                                 min_duration=MIN_INTERVAL_S, max_gap=1.2):
    """Join back-to-back confident seconds with the same position into intervals.

    labels: one (position or None, confidence, who decided) per second.
    Each interval looks like:
      {"guard_position": "Half Guard", "t_start": "00:34.10", "t_end": "01:12.80",
       "confidence": 0.94, "source": "Llama 3 + Rules"}
    """
    labels = apply_temporal_consistency(labels)
    labels = apply_guard_fallback(sheets, labels, min_confidence)

    intervals, current = [], None
    for sheet, (label, conf, src) in zip(sheets, labels):
        if label is None or conf < min_confidence:
            continue  # not sure -> leave a gap (one unsure second in the middle can be bridged)
        t0, t1 = sheet["window_start_s"], sheet["window_end_s"]
        if current and current["label"] == label and t0 - current["end"] <= max_gap:
            current["end"] = t1
            current["confs"].append(conf)
            current["sources"].append(src)
        else:
            if current and t0 - current["end"] < 0.25:
                current["end"] = t0  # make intervals that follow straight on touch on the timeline
            current = {"label": label, "start": t0, "end": t1, "confs": [conf], "sources": [src]}
            intervals.append(current)

    result = []
    for iv in intervals:
        # Each second ends at its last analysed frame, so allow a small margin (2 seconds = about 1.9 s)
        if iv["end"] - iv["start"] + 0.15 < min_duration:
            continue  # too short to trust
        result.append({
            "guard_position": iv["label"],
            "t_start": format_time(iv["start"]),
            "t_end": format_time(iv["end"]),
            "confidence": round(sum(iv["confs"]) / len(iv["confs"]), 2),
            "source": max(set(iv["sources"]), key=iv["sources"].count),  # whoever decided most of it
        })
    return result
