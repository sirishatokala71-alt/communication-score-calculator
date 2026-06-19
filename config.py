"""
config.py

All tunable thresholds live here so the scoring logic in communication_score.py
never has magic numbers baked into it. If your team collects real interview
data later, recalibrate these constants against that data rather than guessing.

NOTE ON SOURCE OF THESE DEFAULTS:
Speech rate ranges are based on commonly cited conversational English norms
(~110-160 words per minute for clear, deliberate speech in an interview
setting -- slower than casual conversation, which runs faster). Filler-word
and response-length ranges are reasonable starting heuristics, not validated
against labeled data. Treat them as a v1 baseline and revisit once you have
ground truth (e.g. human-rated transcripts) to compare against.
"""

# ---------------------------------------------------------------------------
# Speech rate (words per minute)
# ---------------------------------------------------------------------------
# Full score plateau: anything in [IDEAL_MIN, IDEAL_MAX] scores 100 on this
# dimension. Below FLOOR or above CEIL scores 0. Linear taper in between.
# A *plateau* (not a single ideal point) is what lets us "handle different
# speaking styles" -- naturally slower or faster speakers within a wide
# reasonable band aren't penalized at all.
SPEECH_RATE_FLOOR = 70      # wpm — very slow / halting speech
SPEECH_RATE_IDEAL_MIN = 110
SPEECH_RATE_IDEAL_MAX = 165
SPEECH_RATE_CEIL = 220      # wpm — very fast / rushed speech

# Minimum amount of speech needed before we trust the rate calculation at all
MIN_DURATION_SECONDS = 3.0
MIN_WORDS_FOR_SCORING = 5

# ---------------------------------------------------------------------------
# Filler words
# ---------------------------------------------------------------------------
# "Hard" fillers are almost never used in a non-filler sense -> full weight.
# "Soft" fillers (e.g. "like", "actually") are common, legitimate words most
# of the time and only sometimes used as verbal tics -> half weight, to avoid
# punishing normal usage too harshly. This is a heuristic, not NLP-perfect.
HARD_FILLER_WORDS = [
    "um", "umm", "ummm", "uh", "uhh", "uhm", "er", "err", "erm", "hmm",
]

SOFT_FILLER_PHRASES = [
    "like", "you know", "i mean", "sort of", "kind of", "basically",
    "actually", "literally", "so yeah", "right", "okay so",
]

SOFT_FILLER_WEIGHT = 0.5

# Ratio (weighted filler count / total words) thresholds.
# At or below GOOD -> full score. At or above BAD -> zero score. Linear in between.
FILLER_RATIO_GOOD = 0.03   # ~3% of words being fillers reads as natural, fluent speech
FILLER_RATIO_BAD = 0.15    # ~15%+ reads as noticeably distracting

# ---------------------------------------------------------------------------
# Response length (word count)
# ---------------------------------------------------------------------------
# Same plateau idea: too short suggests under-elaboration, too long suggests
# rambling/lack of structure, but a wide healthy middle band is fully scored.
RESPONSE_LENGTH_FLOOR = 15      # words — likely a non-answer / "I don't know"
RESPONSE_LENGTH_IDEAL_MIN = 60
RESPONSE_LENGTH_IDEAL_MAX = 220
RESPONSE_LENGTH_CEIL = 350      # words — likely rambling without structure

# ---------------------------------------------------------------------------
# Final weighted score
# ---------------------------------------------------------------------------
# Must sum to 1.0. Response length is weighted slightly lower because it's
# the weakest signal of the three (a long answer isn't necessarily a bad
# one -- it depends on the question), so we don't want it dominating.
WEIGHTS = {
    "speech_rate": 0.35,
    "filler_words": 0.35,
    "response_length": 0.30,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"

# Whisper model size to use for transcription. "base" is a good speed/accuracy
# tradeoff for short interview clips; use "small" or "medium" for higher
# accuracy if you have the compute budget.
WHISPER_MODEL_SIZE = "base"
