"""
test_communication_score.py

Tests the scoring logic directly (no audio/whisper required) using
compute_communication_score_from_transcript, so this can run in any
environment -- including CI without GPU/audio dependencies.

Run with:  python -m pytest test_communication_score.py -v
       or: python test_communication_score.py
"""

from communication_score import (
    compute_communication_score_from_transcript,
    detect_filler_words,
    calculate_speech_rate,
    score_speech_rate,
    score_filler_words,
    score_response_length,
)


def make_transcript(word: str, count: int) -> str:
    return " ".join([word] * count)


# ---------------------------------------------------------------------------
# Component-level tests
# ---------------------------------------------------------------------------

def test_speech_rate_calculation():
    assert calculate_speech_rate(150, 60) == 150.0       # 150 words in 1 min
    assert calculate_speech_rate(75, 30) == 150.0         # 75 words in 0.5 min
    assert calculate_speech_rate(0, 60) == 0.0
    assert calculate_speech_rate(100, 0) == 0.0           # no div-by-zero crash


def test_speech_rate_plateau_scores_full_marks_in_ideal_band():
    assert score_speech_rate(110) == 100.0
    assert score_speech_rate(140) == 100.0
    assert score_speech_rate(165) == 100.0


def test_speech_rate_tapers_outside_band():
    slow = score_speech_rate(90)    # between floor(70) and ideal_min(110)
    fast = score_speech_rate(190)   # between ideal_max(165) and ceil(220)
    assert 0 < slow < 100
    assert 0 < fast < 100
    assert score_speech_rate(70) == 0.0     # at floor
    assert score_speech_rate(220) == 0.0    # at ceiling
    assert score_speech_rate(20) == 0.0     # below floor
    assert score_speech_rate(300) == 0.0    # above ceiling


def test_filler_word_detection_hard_fillers():
    text = "um so I think uh the answer is um clear"
    stats = detect_filler_words(text)
    assert stats.hard_count == 3   # um, uh, um
    assert stats.total_words == 10


def test_filler_word_detection_soft_fillers_are_half_weighted():
    text = "I was like really happy you know about it"
    stats = detect_filler_words(text)
    assert stats.soft_count == 2   # "like", "you know"
    assert stats.hard_count == 0
    assert stats.weighted_count == 1.0  # 2 * 0.5


def test_filler_score_clean_speech_scores_high():
    clean_text = make_transcript("word", 100)  # 0% fillers
    assert score_filler_words(detect_filler_words(clean_text).weighted_ratio) == 100.0


def test_filler_score_heavy_filler_use_scores_low():
    heavy_filler_text = " ".join(["um"] * 20 + ["word"] * 80)  # 20% hard fillers
    ratio = detect_filler_words(heavy_filler_text).weighted_ratio
    assert score_filler_words(ratio) == 0.0


def test_response_length_plateau():
    assert score_response_length(60) == 100.0
    assert score_response_length(150) == 100.0
    assert score_response_length(220) == 100.0
    assert score_response_length(10) == 0.0     # below floor
    assert score_response_length(400) == 0.0    # above ceiling
    assert 0 < score_response_length(30) < 100  # tapering zone, too short
    assert 0 < score_response_length(300) < 100 # tapering zone, too long


# ---------------------------------------------------------------------------
# End-to-end tests via compute_communication_score_from_transcript
# ---------------------------------------------------------------------------

def test_end_to_end_strong_response():
    """Clean, well-paced, well-sized answer -> high score."""
    text = make_transcript("word", 150)  # 150 words, no fillers
    duration = 60.0  # 150 wpm, in ideal band; 150 words in ideal length band
    result = compute_communication_score_from_transcript(text, duration)
    assert result["communication_score"] >= 90


def test_end_to_end_filler_heavy_response():
    """Same length/pace, but heavy filler use -> noticeably lower score."""
    text = " ".join(["um"] * 30 + ["word"] * 120)  # 150 words, 20% hard fillers
    duration = 60.0
    result = compute_communication_score_from_transcript(text, duration)
    assert result["communication_score"] < 70


def test_end_to_end_too_short_response():
    """A near-empty / very short answer should score low on length."""
    text = "I don't know"
    duration = 3.0
    result = compute_communication_score_from_transcript(text, duration)
    # 3 words is below MIN_WORDS_FOR_SCORING -> insufficient data path
    assert result["communication_score"] is None
    assert result["error"] == "insufficient_speech_detected"


def test_end_to_end_insufficient_speech_returns_error_not_crash():
    result = compute_communication_score_from_transcript("", 0.5)
    assert result["communication_score"] is None
    assert "error" in result


def test_end_to_end_score_is_always_in_valid_range():
    for n_words in [5, 20, 60, 150, 220, 300, 500]:
        text = make_transcript("word", n_words)
        duration = max(3.0, n_words / 2.5)  # roughly 150 wpm
        result = compute_communication_score_from_transcript(text, duration)
        score = result["communication_score"]
        if score is not None:
            assert 0 <= score <= 100


def test_verbose_mode_includes_breakdown():
    text = make_transcript("word", 100)
    result = compute_communication_score_from_transcript(text, 50.0, verbose=True)
    assert "breakdown" in result
    assert "speech_rate_wpm" in result["breakdown"]
    assert "filler_weighted_ratio" in result["breakdown"]


def test_default_output_matches_exact_spec_shape():
    """Non-verbose output must be exactly {'communication_score': <int>}."""
    text = make_transcript("word", 100)
    result = compute_communication_score_from_transcript(text, 50.0)
    assert set(result.keys()) == {"communication_score"}
    assert isinstance(result["communication_score"], int)


if __name__ == "__main__":
    import sys
    import traceback

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except AssertionError:
            print(f"FAIL: {test.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
