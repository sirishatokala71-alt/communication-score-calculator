"""
communication_score.py

C7. Communication Score Calculator
-----------------------------------
Takes a raw audio file of a candidate's interview response and produces a
communication effectiveness score based on:
  - speech rate (words per minute)
  - filler word usage
  - response length

Output (per spec):
    {"communication_score": 78}

IMPORTANT (per task constraints):
This score is a communication-style signal, NOT a hiring decision input.
It should be surfaced as coaching/feedback context alongside other interview
signals, never as a standalone pass/fail gate.

Usage:
    from communication_score import compute_communication_score
    result = compute_communication_score("path/to/response.wav")
    print(result)  # {"communication_score": 78}
"""

import re
import string
from dataclasses import dataclass
from typing import Optional, Dict, Any

import config


# ===========================================================================
# Data containers
# ===========================================================================

@dataclass
class TranscriptionResult:
    text: str
    duration_seconds: float


@dataclass
class FillerStats:
    hard_count: int
    soft_count: int
    weighted_count: float
    total_words: int
    weighted_ratio: float


# ===========================================================================
# Step 1: Transcription
# ===========================================================================

def transcribe_audio(audio_path: str, model_size: str = config.WHISPER_MODEL_SIZE) -> TranscriptionResult:
    """
    Transcribes a raw audio file to text and reports its duration.

    Requires the `openai-whisper` package and `ffmpeg` to be installed:
        pip install openai-whisper
        (ffmpeg must be available on PATH)

    Raises:
        ImportError: if whisper isn't installed.
        FileNotFoundError: if audio_path doesn't exist.
    """
    try:
        import whisper  # local import: only required when actually transcribing
    except ImportError as e:
        raise ImportError(
            "openai-whisper is not installed. Run `pip install openai-whisper` "
            "(and make sure ffmpeg is installed and on your PATH)."
        ) from e

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)

    text = result.get("text", "").strip()
    duration = _get_audio_duration(audio_path, whisper_result=result)

    return TranscriptionResult(text=text, duration_seconds=duration)


def _get_audio_duration(audio_path: str, whisper_result: Optional[dict] = None) -> float:
    """
    Determines audio duration in seconds. Prefers the timestamp of the last
    whisper segment (cheap, no extra dependency); falls back to librosa if
    that's unavailable or looks unreliable (e.g. trailing silence trimmed).
    """
    if whisper_result and whisper_result.get("segments"):
        return float(whisper_result["segments"][-1]["end"])

    try:
        import librosa
        return float(librosa.get_duration(path=audio_path))
    except ImportError:
        raise ImportError(
            "Could not determine audio duration. Install librosa as a fallback: "
            "pip install librosa"
        )


# ===========================================================================
# Step 2: Feature extraction
# ===========================================================================

def _clean_words(text: str) -> list:
    """Lowercases and strips punctuation, returns list of word tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation.replace("'", "")))
    return [w for w in text.split() if w]


def calculate_speech_rate(word_count: int, duration_seconds: float) -> float:
    """Returns words per minute. Caller is responsible for sanity-checking duration."""
    if duration_seconds <= 0:
        return 0.0
    return word_count / (duration_seconds / 60.0)


def detect_filler_words(text: str) -> FillerStats:
    """
    Counts filler word usage in a transcript.

    Hard fillers (um, uh, etc.) count at full weight since they're almost
    never legitimate words. Soft fillers (like, actually, etc.) count at
    half weight since they're often used non-filler-style; this avoids
    over-penalizing normal vocabulary.
    """
    lower_text = text.lower()
    words = _clean_words(text)
    total_words = len(words)

    hard_count = sum(words.count(w) for w in config.HARD_FILLER_WORDS)

    # Soft fillers include multi-word phrases ("you know"), so search the
    # raw lowercased text rather than the tokenized word list.
    soft_count = 0
    for phrase in config.SOFT_FILLER_PHRASES:
        soft_count += len(re.findall(r"\b" + re.escape(phrase) + r"\b", lower_text))

    weighted_count = hard_count + (soft_count * config.SOFT_FILLER_WEIGHT)
    weighted_ratio = weighted_count / total_words if total_words > 0 else 0.0

    return FillerStats(
        hard_count=hard_count,
        soft_count=soft_count,
        weighted_count=weighted_count,
        total_words=total_words,
        weighted_ratio=weighted_ratio,
    )


def calculate_response_length(text: str) -> int:
    """Returns word count of the response."""
    return len(_clean_words(text))


# ===========================================================================
# Step 3: Component scoring (each returns 0-100)
# ===========================================================================

def _plateau_score(value: float, floor: float, ideal_min: float, ideal_max: float, ceil: float) -> float:
    """
    Generic "plateau" scoring curve shared by speech rate and response length:

        0 ---taper-up--- 100 ===plateau=== 100 ---taper-down--- 0
       floor          ideal_min        ideal_max              ceil

    This shape is the key mechanism for "handling different speaking styles":
    a wide band scores full marks, and deviations are penalized gradually
    rather than with a hard cutoff.
    """
    if value <= floor or value >= ceil:
        return 0.0
    if value < ideal_min:
        return 100.0 * (value - floor) / (ideal_min - floor)
    if value > ideal_max:
        return 100.0 * (ceil - value) / (ceil - ideal_max)
    return 100.0


def score_speech_rate(wpm: float) -> float:
    return _plateau_score(
        wpm,
        config.SPEECH_RATE_FLOOR,
        config.SPEECH_RATE_IDEAL_MIN,
        config.SPEECH_RATE_IDEAL_MAX,
        config.SPEECH_RATE_CEIL,
    )


def score_filler_words(weighted_ratio: float) -> float:
    good, bad = config.FILLER_RATIO_GOOD, config.FILLER_RATIO_BAD
    if weighted_ratio <= good:
        return 100.0
    if weighted_ratio >= bad:
        return 0.0
    return 100.0 * (bad - weighted_ratio) / (bad - good)


def score_response_length(word_count: int) -> float:
    return _plateau_score(
        word_count,
        config.RESPONSE_LENGTH_FLOOR,
        config.RESPONSE_LENGTH_IDEAL_MIN,
        config.RESPONSE_LENGTH_IDEAL_MAX,
        config.RESPONSE_LENGTH_CEIL,
    )


# ===========================================================================
# Step 4: Orchestration
# ===========================================================================

def compute_communication_score(audio_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Main entry point. Transcribes the audio, extracts features, scores them,
    and returns the final result.

    Args:
        audio_path: path to a raw audio file (wav/mp3/m4a/etc, anything
                     ffmpeg can decode).
        verbose: if True, includes the feature breakdown in the output for
                  debugging/QA. The default (False) returns exactly the
                  spec'd shape: {"communication_score": <int>}.

    Returns:
        dict matching the task's output spec, or an error dict if there
        wasn't enough speech to score reliably (see EDGE CASES below).
    """
    transcription = transcribe_audio(audio_path)
    return _score_from_transcript(transcription.text, transcription.duration_seconds, verbose)


def compute_communication_score_from_transcript(
    text: str, duration_seconds: float, verbose: bool = False
) -> Dict[str, Any]:
    """
    Same as compute_communication_score but skips transcription -- useful
    for testing, or if the rest of the pipeline already produces a
    transcript + duration upstream (e.g. a shared ASR step feeding multiple
    scoring modules).
    """
    return _score_from_transcript(text, duration_seconds, verbose)


def _score_from_transcript(text: str, duration_seconds: float, verbose: bool) -> Dict[str, Any]:
    word_count = calculate_response_length(text)

    # --- Edge case: not enough speech to score reliably -------------------
    if duration_seconds < config.MIN_DURATION_SECONDS or word_count < config.MIN_WORDS_FOR_SCORING:
        return {
            "communication_score": None,
            "error": "insufficient_speech_detected",
            "detail": (
                f"Only {word_count} word(s) over {duration_seconds:.1f}s. "
                f"Need at least {config.MIN_WORDS_FOR_SCORING} words and "
                f"{config.MIN_DURATION_SECONDS}s of speech to score reliably."
            ),
        }

    wpm = calculate_speech_rate(word_count, duration_seconds)
    filler_stats = detect_filler_words(text)

    rate_score = score_speech_rate(wpm)
    filler_score = score_filler_words(filler_stats.weighted_ratio)
    length_score = score_response_length(word_count)

    final_score = (
        rate_score * config.WEIGHTS["speech_rate"]
        + filler_score * config.WEIGHTS["filler_words"]
        + length_score * config.WEIGHTS["response_length"]
    )
    final_score = round(max(0.0, min(100.0, final_score)))

    result = {"communication_score": final_score}

    if verbose:
        result["breakdown"] = {
            "speech_rate_wpm": round(wpm, 1),
            "speech_rate_score": round(rate_score, 1),
            "filler_word_count": filler_stats.hard_count + filler_stats.soft_count,
            "filler_weighted_ratio": round(filler_stats.weighted_ratio, 3),
            "filler_score": round(filler_score, 1),
            "response_word_count": word_count,
            "response_length_score": round(length_score, 1),
            "duration_seconds": round(duration_seconds, 1),
        }

    return result


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python communication_score.py <audio_file> [--verbose]")
        sys.exit(1)

    verbose_flag = "--verbose" in sys.argv
    output = compute_communication_score(sys.argv[1], verbose=verbose_flag)
    print(json.dumps(output, indent=2))
