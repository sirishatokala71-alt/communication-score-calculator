# C7. Communication Score Calculator

Generates a communication effectiveness score from a candidate's raw audio
interview response, as one signal among others in the larger AI interview
system.

## Output

```json
{ "communication_score": 78 }
```

## How it works

1. **Transcribe** the audio (`openai-whisper`) → get transcript text + duration.
2. **Extract three signals:**
   - **Speech rate** — words per minute.
   - **Filler words** — weighted count of filler usage (see below).
   - **Response length** — word count.
3. **Score each signal 0–100** using a "plateau" curve (see below).
4. **Combine** with weights (`speech_rate` 0.35, `filler_words` 0.35,
   `response_length` 0.30 — see `config.py`) into the final 0–100 score.

## Usage

```bash
pip install -r requirements.txt   # also requires ffmpeg on your system
python communication_score.py path/to/response.wav
python communication_score.py path/to/response.wav --verbose   # see breakdown
```

```python
from communication_score import compute_communication_score
result = compute_communication_score("response.wav")
# {"communication_score": 78}
```

If the rest of your pipeline already has a transcript + duration (e.g. a
shared ASR step feeds multiple scoring modules), skip re-transcribing:

```python
from communication_score import compute_communication_score_from_transcript
result = compute_communication_score_from_transcript(transcript_text, duration_seconds=42.0)
```

## Design decisions worth knowing about

**Why a "plateau" scoring curve instead of one ideal number?**
Each signal scores 100 across a *band* of acceptable values, tapering
gradually to 0 outside it, rather than rewarding only one "perfect" value:

```
0 ---taper-up--- 100 ===== plateau ===== 100 ---taper-down--- 0
```

This is the main mechanism for satisfying the **"handle different speaking
styles"** requirement — a naturally slower or faster speaker, a more
talkative or more concise person, all fall inside a wide full-credit band
rather than being compared against one rigid target. Only genuine outliers
(very halting speech, extreme rambling, near-silence) lose points, and they
lose them gradually, not via a cliff-edge cutoff.

**Why split filler words into "hard" and "soft"?**
Words like *um, uh, erm* are almost never anything but filler → full weight.
Words like *like, actually, basically* are normal vocabulary most of the
time and only sometimes a verbal tic → half weight. This avoids penalizing
someone for ordinary word choice. It's a heuristic, not true NLP intent
detection — if this becomes a priority, a learned filler-classifier would
do better than keyword matching.

**Why are all the thresholds in `config.py`, not hardcoded?**
The task spec didn't define exact "good" values for speech rate, filler
ratio, etc. — I used commonly-cited conversational speech norms (e.g.
~110–165 wpm) as a reasonable v1 baseline, but these are guesses, not
validated against labeled interview data. Once your team has real
recordings + human ratings, recalibrate these constants against that data
rather than trusting my defaults blindly.

## Constraints & edge cases (from the task doc)

- **"Use score as hiring decision criteria" — explicitly out of scope.**
  This module returns a signal, not a verdict. Nothing in the code enforces
  this (it's a downstream usage policy, not a technical constraint) — flag
  it clearly wherever this score gets surfaced or stored, e.g. with a label
  like "communication style feedback" rather than "candidate score."
- **Different speaking styles** — handled via the plateau curves above.
- **Insufficient speech** (near-silence, one-word "I don't know" answers) —
  rather than returning a misleadingly low/high number from too little
  data, the function returns:
  ```json
  { "communication_score": null, "error": "insufficient_speech_detected" }
  ```
  Thresholds for "insufficient" live in `config.py`
  (`MIN_DURATION_SECONDS`, `MIN_WORDS_FOR_SCORING`).

## Known limitations (good to flag in your write-up)

- Filler detection is keyword-based, not semantic — it can't tell "like" used
  as a filler from "like" used as a verb/preposition. Soft-weighting
  mitigates but doesn't eliminate this.
- Transcription quality (and thus all downstream scoring) depends on
  Whisper's accuracy, which can degrade with heavy accents, background
  noise, or overlapping speech — none of which this module currently
  detects or corrects for.
- "Response length" here means raw word count for the whole clip. If the
  upstream system passes one audio file per question, this is fine; if a
  clip contains multiple Q&A turns, word count alone won't separate them.

## Files

| File | Purpose |
|---|---|
| `communication_score.py` | Core module: transcription, feature extraction, scoring |
| `config.py` | All tunable thresholds and weights |
| `test_communication_score.py` | Unit + end-to-end tests (no audio dependencies needed) |
| `requirements.txt` | `openai-whisper`, `librosa` (+ ffmpeg, installed separately) |

Run tests: `python test_communication_score.py` (or `pytest test_communication_score.py -v`)
