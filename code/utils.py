"""Small helpers used by several files: the position names, their colours, and time formatting."""

# The positions the app can report: the five guards from the report (section 3.7 /
# Chapter 5), "Stand-up" for when both athletes are on their feet, and a plain
# "Guard" for when the legs are clearly tangled but the exact guard is unclear.
GUARD_LABELS = [
    "Stand-up",
    "Closed Guard",
    "Half Guard",
    "Open Guard",
    "Butterfly Guard",
    "Guard Pass / Scramble",
    "Guard",
]

# One colour per position, used for the timeline under the video (like Figure 3.5)
GUARD_COLOURS = {
    "Stand-up": "#57606f",
    "Closed Guard": "#2e86de",
    "Half Guard": "#10ac84",
    "Open Guard": "#f39c12",
    "Butterfly Guard": "#8e44ad",
    "Guard Pass / Scramble": "#e74c3c",
    "Guard": "#b8327e",
}


def format_time(seconds):
    """Turn seconds into the report's 'minutes:seconds' style, e.g. 34.1 -> '00:34.10'."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    # Round first, so we never show something like '00:60.00'
    rest = round(rest, 2)
    if rest >= 60:
        minutes += 1
        rest -= 60
    return f"{minutes:02d}:{rest:05.2f}"


def parse_time(text):
    """Turn '01:12.80', '1:02:03.5' or just '72.8' back into seconds.

    If the text isn't a valid time, this raises an error so the app can show a message.
    """
    text = str(text).strip()
    if not text:
        raise ValueError("empty time")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"bad time '{text}'")
    total = 0.0
    for part in parts:  # works for seconds, minutes:seconds and hours:minutes:seconds
        total = total * 60 + float(part)
    if total < 0:
        raise ValueError("time can't be negative")
    return total
