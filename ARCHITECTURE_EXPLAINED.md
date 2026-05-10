# Architecture Diagram — Stage-by-Stage Explanation

This document describes what happens at each stage of the Startup Pitch Evaluation pipeline, as shown in the architecture diagram.

---

## Stage 1 — Input (Client / User)

**What enters the system:**

- A startup pitch payload submitted via the web UI, REST API (`POST /evaluate`), or CLI (`scripts/infer_cli.py`).
- The payload contains:
  - `title` — name of the pitch.
  - `transcript` — the spoken text of the pitch (typed or auto-transcribed).
  - `language_hint` — language tag such as `en`, `ta`, or `ta-en` (Tamil-English mix).
  - `presenter_profile` — optional background information about the presenter.
  - `slide_text` — text extracted from presentation slides.
  - `video` — optional object with file name, format, duration, and embedded transcript.
  - `slides` — optional list of structured slide objects.
  - `user_details` — optional submitter metadata.

**What happens:**

The `FastAPI` entry point (`backend/app/main.py`) receives the request. The shared inference service (`backend/app/services/inference.py`) builds a deterministic `request_id` from a hash of the payload so that duplicate submissions produce consistent IDs. It then hands the payload to the pipeline.

---

## Stage 2 — Pre-processing and Temporal Segmentation

**What enters this stage:** Raw `PitchInput` payload.

**What happens (`backend/app/services/preprocessing.py`):**

1. **Transcript resolution** — Uses the payload transcript if present; otherwise falls back to the video's embedded `transcript_text`.
2. **Slide context resolution** — Merges `slides` list or raw `slide_text` into a single context string.
3. **Media file resolution** — Finds the video file path using the configured `SPE_MEDIA_LOOKUP_DIR` directory.
4. **Timeline chunking** — Splits the full transcript/pitch into fixed-length time windows (default **5 seconds**, controlled by `SPE_CHUNK_WINDOW_SECONDS`). Each window becomes a `PitchChunk` object containing:
   - `start_sec` / `end_sec` — the time boundary.
   - `chunk_text` — the transcript portion for this window.
   - `audio_ref` / `video_ref` — pointers to the corresponding media segment.
   - `alignment_meta` — word-level timing and confidence metadata.

**Why it matters:** Every downstream model operates on these chunks, not the full pitch, enabling fine-grained time-aligned scoring.

---

## Stage 3 — Transcription (optional / conditional)

**What enters this stage:** Chunk reference to the audio/video file.

**What happens (`backend/app/services/transcriber.py`):**

- If no transcript was supplied, automatic speech recognition (ASR) runs on the audio.
- Supports three backends via `SPE_TRANSCRIBER_BACKEND`:
  - `faster-whisper` — local, runs on CPU or GPU.
  - `openai` — calls the OpenAI Whisper API (`whisper-1`).
  - `auto` — prefers `faster-whisper`, falls back to OpenAI if unavailable.
- Returns structured fields: `backend`, `status`, `reason`, `confidence`.
- Deterministic fallback states handle problem scenarios:
  - **silence** — audio segment is silent.
  - **low quality** — audio quality below `SPE_TRANSCRIBER_MIN_AUDIO_QUALITY`.
  - **missing audio file** — file not found.
  - **empty transcript** — ASR returned no text.

**Why it matters:** Ensures the system can score a pitch even when the user does not supply a typed transcript.

---

## Stage 4 — Video Extraction

**What enters this stage:** Per-chunk video reference.

**What happens (`backend/app/services/video_processor.py`):**

- Samples representative frames from each 5-second chunk using **OpenCV**.
- Runs optional **MediaPipe** pose/face detection to compute:
  - `face_ratio` — fraction of frame occupied by the presenter's face.
  - `motion_score` — how much the presenter moves between frames.
  - `eye_contact_score` — estimated gaze direction toward camera.
  - `pose_ratio` — upper-body visibility.
  - `gesture_energy` — hand and arm movement intensity.
- Saves extracted frames to `backend/outputs/frames/`.
- Falls back with deterministic `extraction_status` values when video or dependencies are unavailable.

**Why it matters:** Provides the visual evidence (confidence, engagement, presence) needed by the visual encoder.

---

## Stage 5 — Audio Extraction

**What enters this stage:** Per-chunk audio reference.

**What happens (`backend/app/services/audio_processor.py`):**

- Extracts a WAV clip for each chunk using **ffmpeg** (system install or `imageio-ffmpeg` fallback).
- Computes waveform-level quality and prosody features:
  - `silence_ratio` — fraction of the chunk that is silent.
  - `clipping_ratio` — fraction of samples that are clipped/distorted.
  - `speech_density` — proportion of active speech frames.
  - `pitch_variation` — standard deviation of fundamental frequency.
  - `energy_variation` — variation in loudness across the chunk.
  - `audio_quality_score` — composite reliability score.
- Saves audio clips to `backend/outputs/audio_chunks/`.
- Falls back with deterministic status when extraction fails.

**Why it matters:** Produces the chunk-level audio reliability and prosody signals that feed the audio encoder.

---

## Stage 6 — Feature Extraction (three parallel branches)

All three branches run **in parallel** via a thread pool for every chunk (`backend/app/services/extractors.py`).

### 6a — Text Feature Extraction (`backend/models/text_encoder.py`)

- Detects language using script ratios, lexical keyword evidence, and `langdetect` fallback.
- Embeds the chunk text and scores six business metrics:
  | Metric | What it measures |
  |---|---|
  | Problem Clarity | How clearly the problem is stated |
  | Market Opportunity | Size and evidence of the target market |
  | Solution Uniqueness | Differentiation from alternatives |
  | Traction Evidence | Proof of early adoption or validation |
  | Business Model Strength | Revenue logic and sustainability |
  | Team Readiness | Founder background and capability signals |
- **Heuristic mode:** deterministic hashed embeddings + rule-based scoring.
- **Neural mode:** `sentence-transformers` embedding + lightweight MLP head.

### 6b — Visual Feature Extraction (`backend/models/visual_encoder.py`)

- Takes the visual metadata from Stage 4 (face, motion, gaze, etc.).
- Produces two presentation quality metrics:
  | Metric | What it measures |
  |---|---|
  | Delivery Clarity | Slide readability, visual structure |
  | Presenter Confidence | Eye contact, stable posture, minimal fidgeting |
- **Heuristic mode:** rule-based scoring from raw visual metadata and slide stage context.
- **Neural mode:** MobileNetV3-small feature backbone → projection layer → scoring MLP.

### 6c — Audio Feature Extraction (`backend/models/audio_encoder.py`)

- Takes the waveform features from Stage 5.
- Produces two vocal delivery metrics:
  | Metric | What it measures |
  |---|---|
  | Voice Pace | Speaking rate relative to expected WPM |
  | Voice Prosody | Variation in pitch and energy (expressiveness) |
- **Heuristic mode:** text-pace calculation + waveform quality proxies.
- **Neural mode:** MFCC spectrogram → CNN → scoring MLP.

---

## Stage 7 — Modality Fusion

**What enters this stage:** Text embedding, visual embedding, audio embedding (all for one chunk).

**What happens (`backend/models/fusion_head.py`, `backend/app/services/fusion.py`):**

- Combines the three embeddings into a single **fused multimodal vector**.
- Computes **attention weights** for each modality (`text_weight`, `visual_weight`, `audio_weight`) so the final result is explainable.
- **Heuristic mode:** weights each modality by its embedding energy (magnitude).
- **Neural mode:** learned cross-modal attention that adapts weights based on content.

**Why it matters:** Produces one unified per-chunk representation while preserving which modalities drove the score.

---

## Stage 8 — Chunk Scoring

**What enters this stage:** Fused vector + per-modality outputs.

**What happens (`backend/models/scoring_head.py`, `backend/app/services/scoring.py`):**

- Maps the fused representation into **10 rubric metrics** (6 text + 4 AV):
  - Text/business: Problem Clarity, Market Opportunity, Solution Uniqueness, Traction Evidence, Business Model Strength, Team Readiness.
  - AV/delivery: Delivery Clarity, Presenter Confidence, Voice Pace, Voice Prosody.
- Computes per-chunk:
  - `aggregate_score` — weighted combination of all 10 metrics.
  - `confidence_score` — how reliable the scoring is given available data.
- **Heuristic mode:** explicit weighted formula over text/AV/fusion signals.
- **Neural mode:** learned scoring network loaded from a `.pt` checkpoint.

---

## Stage 9 — Risk Flag Detection

**What enters this stage:** Chunk text and aggregate score.

**What happens (`backend/app/services/risk.py`):**

- Runs rule-based checks and emits flags when problems are detected:
  | Flag | Meaning |
  |---|---|
  | `low-quality-signal` | Audio/video quality was too poor to trust features |
  | `competitive-landscape-missing` | No mention of competitors or market alternatives |
  | `overclaim-risk` | Language suggests unsupported or exaggerated claims |
  | `traction-evidence-weak` | Insufficient proof of adoption or user validation |
- Flags are attached to the chunk report and surfaced in the final output.

**Why it matters:** Adds investor-relevant diligence signals that go beyond a numeric score.

---

## Stage 10 — Feedback and Reporting

**What enters this stage:** Scored chunk reports + risk flags.

**What happens (`backend/app/services/reporting.py`):**

- Generates natural-language feedback:
  - **Strengths** — dimensions where scores are notably high.
  - **Weaknesses** — dimensions with low scores or active risk flags.
  - **Suggestions** — actionable improvements the founder can take.
- Builds dashboard data series:
  - `quantitative_scores` — bar chart data for all 10 rubric metrics.
  - `modality_weights` — pie/radar showing text vs visual vs audio contribution.
  - `risk_distribution` — breakdown of flag categories.

---

## Stage 11 — Aggregation and Final Output

**What enters this stage:** All per-chunk reports.

**What happens (`backend/app/pipeline.py`):**

- Aggregates chunk-level outputs across the full pitch into a final summary:
  | Field | Description |
  |---|---|
  | `overall_score` | Weighted average across chunks |
  | `confidence_score` | Pipeline-level reliability estimate |
  | `investment_band` | `high-potential`, `watchlist`, or `early-risk` |
  | `language_detected` | Dominant language (`en`, `ta`, `ta-en`) |
  | `processing_option` | `heuristic` or `neural` |
  | `processing_notes` | Any warnings or fallback messages |

- Returns a complete `EvaluationResponse` with:
  - `summary` — overall result fields above.
  - `chunk_reports[]` — per-chunk text/AV metrics, risk flags, attention weights.
  - `dashboard` — visualization-ready data series.

---

## End-to-End Data Flow Summary

```
Client payload
    │
    ▼
[1] FastAPI / CLI entry point
    │  (builds request_id, validates schema)
    ▼
[2] Pre-processing — temporal chunking into 5s windows
    │
    ├──[3] Transcription (ASR) ──────────────────────┐
    │                                                 │
    ├──[4] Video extraction (frames, visual metrics)  │
    │                                                 │
    └──[5] Audio extraction (WAV clips, waveform)     │
              │                                       │
              ▼                                       │
[6] Feature extraction (parallel per chunk) ◄────────┘
    ├── Text encoder → 6 business metrics
    ├── Visual encoder → 2 delivery metrics
    └── Audio encoder → 2 vocal metrics
              │
              ▼
[7] Modality fusion → fused vector + attention weights
              │
              ▼
[8] Chunk scoring → 10 metrics + aggregate + confidence
              │
              ▼
[9] Risk flag detection → diligence flags per chunk
              │
              ▼
[10] Reporting → strengths, weaknesses, suggestions, dashboard
              │
              ▼
[11] Aggregation → final EvaluationResponse
```

---

## Key Configuration Switches

| Variable | Default | Effect |
|---|---|---|
| `SPE_CHUNK_WINDOW_SECONDS` | `5` | Duration of each timeline chunk |
| `SPE_USE_HEURISTIC_PIPELINE` | `true` | `true` = rule-based path; `false` = neural path |
| `SPE_TRANSCRIBER_BACKEND` | `auto` | `faster-whisper`, `openai`, or `auto` |
| `SPE_ENABLE_VISUAL_EXTRACTION` | `true` | Enable/disable frame extraction |
| `SPE_ENABLE_AUDIO_EXTRACTION` | `true` | Enable/disable audio chunk extraction |
| `SPE_NN_CHECKPOINT_PATH` | — | Path to trained `.pt` model for neural mode |
