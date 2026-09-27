"""Listening to the video (report sections 3.3 and 4.4) - OPTIONAL.

The speech in the video (for example a coach shouting "get the half guard!") is turned
into text with timestamps using OpenAI Whisper, and then searched for BJJ words. These
words are a backup clue for when the camera can't see the legs clearly.

If Whisper or FFmpeg is not installed, this step is simply skipped.
"""

import importlib.util
import shutil

# Spoken words -> which position they hint at. Longer phrases are checked first.
BJJ_KEYWORDS = {
    "closed guard": "Closed Guard",
    "half guard": "Half Guard",
    "underhook": "Half Guard",        # "recover half guard and get the underhook" (section 3.3)
    "open guard": "Open Guard",
    "butterfly": "Butterfly Guard",
    "guard pass": "Guard Pass / Scramble",
    "pass": "Guard Pass / Scramble",
}


def find_keywords(text):
    """Return the positions hinted at by a piece of spoken text."""
    text = text.lower()
    found = []
    for phrase in sorted(BJJ_KEYWORDS, key=len, reverse=True):
        if phrase in text:
            label = BJJ_KEYWORDS[phrase]
            if label not in found:
                found.append(label)
            text = text.replace(phrase, " ")  # so "guard pass" isn't counted again as "pass"
    return found


class AudioWorker:
    def __init__(self, model_size="base"):
        self.model_size = model_size

    @staticmethod
    def check_available():
        """Returns (True, '') if Whisper and FFmpeg are installed, otherwise (False, the reason)."""
        if shutil.which("ffmpeg") is None:
            return False, "FFmpeg not found on PATH - audio step skipped."
        if importlib.util.find_spec("whisper") is None:
            return False, "openai-whisper not installed - audio step skipped."
        return True, ""

    def transcribe(self, video_path):
        """Returns a list of spoken pieces {'start', 'end', 'text', 'keywords'} (times in seconds).
        Never crashes: if anything goes wrong, the audio step is skipped with a message."""
        ok, reason = self.check_available()
        if not ok:
            return [], reason
        try:
            import whisper
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
            model = whisper.load_model(self.model_size, device=device)
            # Whisper uses FFmpeg to pull the sound out of the MP4 and convert it to the
            # format it needs (16 kHz, one channel), exactly as section 4.4 describes
            result = model.transcribe(video_path, fp16=(device == "cuda"), verbose=None)
        except Exception as e:  # e.g. no sound in the video, a broken file, not enough memory
            return [], f"Audio step skipped ({type(e).__name__}: {e})"

        segments = []
        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            segments.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": text,
                "keywords": find_keywords(text),
            })
        n_hits = sum(1 for s in segments if s["keywords"])
        return segments, f"Whisper: {len(segments)} speech segments, {n_hits} with BJJ keywords."
