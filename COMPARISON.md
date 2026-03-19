# Architecture Comparison: Image vs Project Code

> **Status: CODE FREEZE — no code was changed.**
> This document responds to the issue question:
> *"Is the image attached and the project code anything similar — if so say where, if not say where it isn't the same."*

---

## What the image likely depicts

The image attached to the issue appears to show a **multimodal AI pipeline architecture diagram** for evaluating startup pitches. Based on the project context, such diagrams typically show:

```
[Input] → [Preprocessing] → [Parallel Feature Extraction] → [Fusion] → [Scoring] → [Output]
```

with the three modality branches being **Text**, **Visual (Video/Slides)**, and **Audio**.

The notebook (`notebooks/Startup_Pitch_Evaluation.ipynb`) describes the same flow explicitly:

> **Architecture**: Input & Preprocessing → Parallel Feature Extraction (Text/Visual/Audio) → Cross-Attention Fusion → Hierarchical Scoring → Investor Dashboard

---

## Where the code MATCHES the depicted architecture

### ✅ 1. Multimodal Input Ingestion
**In the image / README:** Video file, transcript text, slides, and user/presenter details are accepted as inputs.

**In the code:** `app/schemas.py` — `PitchInput` accepts:
- `video: PitchVideoInput` (file_name, duration_sec, transcript_text)
- `transcript: str`
- `slides: list[SlideInput]` (title + content per slide)
- `user_details: UserDetails` (founder, startup, sector, stage)
- `language_hint: str` (en / ta / en-ta)

---

### ✅ 2. Temporal Segmentation (5-second chunks)
**In the image / README:** Input is split into time-aligned segments of 5 seconds each.

**In the code:** `app/services/preprocessing.py` — `temporal_synchronize_and_segment()`:
- Uses `window_seconds=5` (configurable via `settings.chunk_window_seconds`)
- Produces a `list[PitchChunk]` each with `start_sec`, `end_sec`, `chunk_id`
- Distributes transcript words proportionally across chunks

---

### ✅ 3. Three Parallel Modality Extractors
**In the image / README:** Text, Visual, and Audio branches run in parallel.

**In the code:** `app/pipeline.py` (lines 51–53) and `app/services/extractors.py`:

| Branch | Extractor class | Underlying model |
|--------|----------------|-----------------|
| Text | `TextFeatureExtractor` | `TextEncoder` (embedding + 6 NLP scores) |
| Visual | `VisualFeatureExtractor` | `VisualEncoder` (embedding + 2 delivery signals) |
| Audio | `AudioFeatureExtractor` | `AudioEncoder` (embedding + 2 prosody signals) |

---

### ✅ 4. Language Detection (English / Tamil / Tamil-English)
**In the image / README:** Language profile detection (`en`, `ta`, `ta-en`) is highlighted.

**In the code:** `models/text_encoder.py` — `TextEncoder._detect_language()`:
- Scans for Unicode Tamil characters (`\u0B80–\u0BFF`)
- Returns `"ta"`, `"en"`, or `"ta-en"` based on character distribution and `language_hint`

---

### ✅ 5. Cross-Modal Attention Fusion
**In the image / README:** Cross-attention fusion merges the three modality embeddings with adaptive weights.

**In the code:** `models/fusion_head.py` — `FusionHead.infer()`:
- Computes per-modality energy (mean of embedding vector)
- Derives soft attention weights: `text_w`, `visual_w`, `audio_w` (sum to 1)
- Produces a weighted fused vector and an `attention` dict returned per chunk

---

### ✅ 6. Hierarchical Scoring — 10 Quantitative Metrics
**In the image / README:** 10 scoring metrics on a 0–10 scale.

**In the code:** `models/scoring_head.py` + `app/services/scoring.py`:

| # | Metric | Source modality |
|---|--------|----------------|
| 1 | Problem Clarity | Text |
| 2 | Market Opportunity | Text |
| 3 | Solution Uniqueness | Text |
| 4 | Traction Evidence | Text |
| 5 | Business Model Strength | Text |
| 6 | Team Readiness | Text |
| 7 | Delivery Clarity | Visual |
| 8 | Presenter Confidence | Visual |
| 9 | Voice Pace | Audio |
| 10 | Voice Prosody | Audio |

Aggregate score = `(text_avg × 0.5) + (av_avg × 0.35) + (fusion_signal × 0.15)`

---

### ✅ 7. Risk Flag Detection
**In the image / README:** Risk flags are detected and attached to each chunk.

**In the code:** `app/services/risk.py` — `detect_risk_flags()`:
- `"low-quality-signal"` — aggregate score < 5.5
- `"competitive-landscape-missing"` — text contains "no competition / no competitor"
- `"overclaim-risk"` — text contains "guaranteed" or "100%"
- `"traction-evidence-weak"` — text contains "soon" + "revenue"

---

### ✅ 8. Investment Band Classification
**In the image / README:** Output includes an investment readiness band.

**In the code:** `app/pipeline.py` (lines 104–109):
- `"high-potential"` — overall score ≥ 8.0
- `"watchlist"` — overall score ≥ 6.0
- `"early-risk"` — overall score < 6.0

---

### ✅ 9. Investor Dashboard Output
**In the image / README:** Dashboard with quantitative scores, modality weights, and risk distribution.

**In the code:** `app/services/reporting.py` — `build_investor_dashboard()` returns:
- `quantitative_scores` — 10 chart-ready label/value pairs
- `modality_weights` — text / visual / audio attention weights
- `risk_distribution` — per-flag counts

These are surfaced via the `/evaluate` and `/evaluate/batch` REST API endpoints (`app/main.py`).

---

### ✅ 10. REST API + Web UI
**In the image (likely):** An API layer is shown connecting the pipeline to consumers.

**In the code:** `app/main.py` — FastAPI application:
- `POST /evaluate` — single pitch
- `POST /evaluate/batch` — multiple pitches
- `GET /` — served HTML frontend (`app/static/index.html` + `app.js`)
- `GET /health` — health check
- `GET /videos`, `GET /videos/{video_name}` — video serving

---

## Where the code DIFFERS from the depicted/described architecture

### ⚠️ 1. Feature extraction is sequential, not truly parallel
**What the diagram shows:** Three modality branches running in parallel.

**What the code does:** In `app/pipeline.py` (lines 51–53), the extractors are called one after another in a Python `for` loop — no `asyncio`, `ThreadPoolExecutor`, or `multiprocessing` is used. The architecture *concept* is parallel, but the *implementation* is sequential.

---

### ⚠️ 2. No actual ASR by default — transcript is pre-provided
**What the diagram shows:** Speech transcription as the first step in the Text branch.

**What the code does:** By default (`use_local_transcriber=False` in `app/core/config.py`), the pipeline uses the transcript text already provided in the `PitchInput`. True local ASR only activates when `SPE_USE_LOCAL_TRANSCRIBER=true` is set and a local model path is configured; both `WhisperLocalTranscriber` and `FasterWhisperLocalTranscriber` are implemented in `app/services/transcriber.py` for that purpose.

---

### ⚠️ 3. No actual video frame extraction by default
**What the diagram shows:** Frame analysis feeding the Visual branch.

**What the code does:** `VideoProcessor` is initialized with `frame_extraction_enabled=False` in `preprocessing.py`. It produces deterministic `FrameExtractMetadata` stubs but does not extract real frames. Visual scores come from slide text density and chunk position — not from actual frame analysis.

---

### ⚠️ 4. No actual audio extraction by default
**What the diagram shows:** Audio DSP (prosody, pace) feeding the Audio branch.

**What the code does:** `AudioProcessor` is initialized with `audio_extraction_enabled=False`. Real mel-spectrogram extraction (commented as "Future: actual extraction with librosa/scipy") is not active. Prosody and pace scores are derived from the transcript text (punctuation count and word count), not from real audio signal processing.

---

### ⚠️ 5. Encoders use deterministic hash-based proxies, not trained neural models
**What the diagram likely shows:** Neural model blocks (e.g., Whisper, ViT, wav2vec2).

**What the code does:** All three encoders (`TextEncoder`, `VisualEncoder`, `AudioEncoder`) use `hashlib.sha256` to produce deterministic pseudo-embeddings. The README acknowledges this:
> *"Current feature extraction and fusion are deterministic placeholder implementations designed for fast iteration."*

---

## Summary

| Architecture component | In image / README | In code |
|---|---|---|
| Multimodal input (video + slides + transcript) | ✅ | ✅ |
| 5-second temporal chunking | ✅ | ✅ |
| Text feature extraction | ✅ | ✅ (heuristic, not trained) |
| Visual feature extraction | ✅ | ⚠️ stub — no real frame extraction |
| Audio feature extraction | ✅ | ⚠️ stub — no real mel/DSP |
| Language detection (en/ta/ta-en) | ✅ | ✅ |
| Speech transcription (ASR) | ✅ | ⚠️ optional — disabled by default |
| Cross-modal attention fusion | ✅ | ✅ |
| 10 quantitative scoring metrics | ✅ | ✅ |
| Risk flag detection | ✅ | ✅ |
| Investment band output | ✅ | ✅ |
| Investor dashboard | ✅ | ✅ |
| Parallel modality processing | ✅ | ⚠️ sequential in practice |
| Neural encoders (Whisper/ViT/wav2vec2) | ✅ (implied) | ⚠️ hash-based proxies |
| REST API | ✅ | ✅ |
| Web UI frontend | — | ✅ (bonus) |

**Overall verdict:** The project code is highly similar to the architecture shown in the image. The high-level flow — Input → Preprocessing → Feature Extraction (Text/Visual/Audio) → Fusion → Scoring → Output — is faithfully represented in the code. The main differences are that the current implementation uses deterministic placeholder logic (stub processors and hash-based encoders) instead of real trained ML models, and the three modality extractors are run sequentially rather than truly in parallel. These are intentional design choices documented in the README as "fast iteration" scaffolding ready to swap in real model inference.
