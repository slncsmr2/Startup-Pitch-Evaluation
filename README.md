# Startup-Pitch-Evaluation

Multimodal startup pitch evaluation system that scores investor-facing pitch videos using text, audio, and visual signals.

This repository now includes:

- A FastAPI backend with single and batch evaluation endpoints.
- A shared inference service used by both API and CLI flows.
- Chunk-based multimodal preprocessing (default 5 second windows).
- Deterministic fallback behavior when media/dependencies are missing.
- Local faster-whisper and OpenAI Whisper API transcription options.
- Neural model training/evaluation scripts with Option A JSONL data support.
- Runtime benchmarking and dataset preparation tooling.

Detailed process documentation is available in [PROCESS_OVERVIEW.md](PROCESS_OVERVIEW.md).

## What is implemented

### Inference pipeline

- Input normalization for title, transcript, language hint, video metadata, slides, and user details.
- Timeline chunking of each pitch into fixed windows.
- Per-chunk extraction of text, visual, and audio features.
- Modality fusion and attention weighting.
- Scoring into 10 rubric metrics:
  - Problem Clarity
  - Market Opportunity
  - Solution Uniqueness
  - Traction Evidence
  - Business Model Strength
  - Team Readiness
  - Delivery Clarity
  - Presenter Confidence
  - Voice Pace
  - Voice Prosody
- Aggregation into overall score, confidence, investment band, risks, strengths, weaknesses, suggestions, and dashboard series.

### API service

- `GET /health` returns service status and version.
- `GET /` serves the frontend UI.
- `POST /evaluate` evaluates one pitch payload.
- `POST /evaluate/batch` evaluates multiple pitches in one request.
- `GET /videos` lists available videos from `backend/outputs/batch_input`.
- `GET /videos/{video_name}` streams a selected video with path traversal protection.

### CLI and scripts

- `scripts/infer_cli.py` for single-video or directory batch inference.
- `scripts/train.py` for neural model training from YAML-like config.
- `scripts/evaluate.py` for checkpoint evaluation.
- `scripts/benchmark_runtime.py` for CPU/GPU-profile latency baselines.

### Training stack (Phase 6)

- Configurable training profile files for CPU and GPU.
- Option A dataset strategy using JSONL rows with `video_id`, `transcript`, `slide_text`, and 10 `scores`.
- Automatic derivation of `investment_band` when missing.
- Backward compatibility for older legacy row formats.
- Synthetic data fallback when split files are missing.
- Checkpoint save/load flow using `.pt` artifacts.

### Dataset tooling

- `dataset_tools/prepare_videos_with_whisper.py` to batch transcribe local videos.
- Exports:
  - PitchInput-style JSONL for direct inference ingestion.
  - Labeling JSONL for Option A rubric scoring workflows.
- Windows CUDA runtime diagnostics and automatic CPU fallback for whisper when configured.

## Repository layout

```text
.
├── README.md
├── Commands.txt
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── pipeline.py
│   │   ├── schemas.py
│   │   ├── core/config.py
│   │   ├── services/
│   │   │   ├── inference.py
│   │   │   ├── preprocessing.py
│   │   │   ├── extractors.py
│   │   │   ├── fusion.py
│   │   │   ├── scoring.py
│   │   │   ├── risk.py
│   │   │   ├── reporting.py
│   │   │   ├── transcriber.py
│   │   │   ├── audio_processor.py
│   │   │   └── video_processor.py
│   │   └── static/
│   ├── datasets/
│   │   ├── README.md
│   │   └── splits/
│   ├── scripts/
│   ├── training/
│   ├── tests/
│   ├── models/
│   │   ├── config/
│   │   └── checkpoints/
│   ├── outputs/
│   └── Input-Output/
└── dataset_tools/
    ├── prepare_videos_with_whisper.py
    └── README.md
```

## Setup (Windows PowerShell)

Run from repository root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start API:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Useful URLs:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## Common commands

All commands below are run from `backend` unless noted.

### Inference CLI

Single video:

```powershell
python scripts/infer_cli.py --video outputs/batch_input/sample.mp4 --duration-sec 90 --language-hint en-ta --output outputs/sample_eval.json
```

Batch directory:

```powershell
python scripts/infer_cli.py --batch-dir outputs/batch_input --duration-sec 90 --batch-output-dir outputs/batch_results --output outputs/batch_summary.json
```

### Training and evaluation

CPU training:

```powershell
python scripts/train.py --config models/config/training_cpu.yaml
```

GPU training:

```powershell
python scripts/train.py --config models/config/training_gpu.yaml
```

CPU evaluation:

```powershell
python scripts/evaluate.py --config models/config/training_cpu.yaml --checkpoint models/checkpoints/phase6_cpu_nn_model.pt
```

GPU profile evaluation:

```powershell
python scripts/evaluate.py --config models/config/training_gpu.yaml
```

### Runtime benchmark

```powershell
python scripts/benchmark_runtime.py --runs 5 --duration-sec 60 --output outputs/benchmark_runtime.json
```

### Local whisper data prep (run from repository root)

Training input folder:

```powershell
python.exe dataset_tools/prepare_videos_with_whisper.py --input-dir backend/Input-Output/Training --model-size small --device cuda --compute-type float16 --debug-runtime --strict-gpu
```

Testing input folder:

```powershell
python.exe dataset_tools/prepare_videos_with_whisper.py --input-dir backend/Input-Output/Testing --model-size small --device cuda --compute-type float16 --debug-runtime --strict-gpu
```

## Request/response schema overview

### Input (`PitchInput`)

- `title`
- `transcript`
- `language_hint` (default `en-ta`)
- `presenter_profile`
- `slide_text`
- `video`:
  - `file_name`
  - `file_format`
  - `duration_sec` (minimum 5)
  - `transcript_text`
- `slides`
- `user_details`

### Output (`EvaluationResponse`)

- `request_id`
- `summary`:
  - `overall_score`
  - `confidence_score`
  - `investment_band` (`high-potential`, `watchlist`, `early-risk`)
  - `language_detected` (`en`, `ta`, `ta-en`)
  - `strengths`, `weaknesses`, `suggestions`
  - `processing_option`, `processing_notes`
- `chunk_reports[]`:
  - `chunk_id`, `start_sec`, `end_sec`
  - `text_metrics[]`, `av_metrics[]`
  - `attention`
  - `risk_flags`
  - `aggregate_score`
- `dashboard`:
  - `quantitative_scores`
  - `modality_weights`
  - `risk_distribution`

## Configuration

Environment variables use the `SPE_` prefix and are loaded from:

- repository root `.env`
- `backend/.env`

Core switches:

```text
SPE_CHUNK_WINDOW_SECONDS=5
SPE_USE_HEURISTIC_PIPELINE=true
SPE_USE_LOCAL_TRANSCRIBER=true
SPE_ENABLE_VISUAL_EXTRACTION=true
SPE_ENABLE_AUDIO_EXTRACTION=true
SPE_MEDIA_LOOKUP_DIR=outputs/batch_input
```

Transcriber controls:

```text
SPE_TRANSCRIBER_BACKEND=auto
SPE_TRANSCRIBER_MIN_AUDIO_QUALITY=0.35
SPE_FASTER_WHISPER_MODEL_SIZE=small
SPE_FASTER_WHISPER_DEVICE=cpu
SPE_FASTER_WHISPER_COMPUTE_TYPE=int8
SPE_OPENAI_API_KEY=
SPE_OPENAI_TRANSCRIBER_MODEL=whisper-1
```

Neural path controls:

```text
SPE_USE_HEURISTIC_PIPELINE=false
SPE_NN_CHECKPOINT_PATH=models/checkpoints/nn_model.pt
SPE_NN_TEXT_ENCODER=all-MiniLM-L6-v2
SPE_NN_VISUAL_BACKBONE=mobilenet_v3_small
SPE_NN_AUDIO_FEATURES=mfcc
SPE_NN_DEVICE=auto
```

### Heuristic vs neural mode

- Keep `SPE_USE_HEURISTIC_PIPELINE=true` for deterministic heuristic processing.
- Set `SPE_USE_HEURISTIC_PIPELINE=false` to activate the neural extraction/fusion/scoring path.
- In neural mode, set `SPE_NN_CHECKPOINT_PATH` to a valid checkpoint.

## Dataset format (Option A)

Use JSONL files in `backend/datasets/splits`:

- `train.jsonl`
- `val.jsonl` (optional; auto-derived from train when missing)
- `test.jsonl`

Each row example:

```json
{
  "video_id": "yc_pitch_001",
  "transcript": "Founder narration text...",
  "slide_text": "Optional slide OCR text...",
  "scores": [7.8, 8.1, 7.4, 6.9, 7.2, 8.0, 7.0, 7.5, 6.8, 7.1],
  "investment_band": "watchlist"
}
```

Notes:

- `scores` must have 10 values on a 0-10 scale.
- `investment_band` can be omitted and will be derived.
- Legacy dataset rows are still supported for compatibility.

## Tests

```powershell
cd backend
pytest -q
```

Current test coverage includes:

- API behavior
- Pipeline structure
- Fallback scoring behavior
- Language detection
- Audio/video processor fallbacks
- Transcriber behavior

## Dependency summary

Key libraries currently used include FastAPI, Pydantic v2, PyTorch, sentence-transformers, faster-whisper, OpenAI SDK, OpenCV headless, MediaPipe, and pytest.

Install all from:

```powershell
pip install -r backend/requirements.txt
```

## Contributor notes

- Keep API and CLI behavior aligned via shared inference service.
- Keep schema changes backward-compatible where possible.
- Update tests and docs when changing endpoint behavior, scoring semantics, or config defaults.
