# Startup-Pitch-Evaluation

Multimodal startup pitch evaluation backend with a FastAPI API, shared CLI inference, chunk-level scoring, and optional local/cloud transcription fallbacks.

## Current status

- FastAPI service with single and batch pitch evaluation
- Shared inference engine used by API and CLI (`InferenceService`)
- 5-second timeline chunking with synchronized text/audio/visual metadata
- Deterministic fallback behavior when AV dependencies or media files are unavailable
- Optional local faster-whisper and OpenAI Whisper API transcription paths
- Neural-ready extraction/fusion/scoring path controlled by `SPE_USE_HEURISTIC_PIPELINE`
- Phase 6 training pipeline with Option A JSONL dataset support and `.pt` checkpoint output
- Static frontend served by the API root route
- Training, evaluation, and runtime benchmark scripts

## Architecture summary

1. Input payload is normalized (title, transcript/video text, slides, user details).
2. Preprocessing creates chunk windows (`window_seconds=5` by default).
3. Per chunk, text/visual/audio features are extracted in parallel.
4. Modalities are fused into attention weights and scored into 10 quantitative metrics.
5. Risk flags, strengths/weaknesses/suggestions, and dashboard series are generated.

## Project structure

```text
.
├── backend/
│   ├── app/
│   │   ├── core/config.py
│   │   ├── main.py
│   │   ├── pipeline.py
│   │   ├── schemas.py
│   │   ├── services/
│   │   │   ├── inference.py
│   │   │   ├── preprocessing.py
│   │   │   ├── transcriber.py
│   │   │   ├── audio_processor.py
│   │   │   ├── video_processor.py
│   │   │   ├── extractors.py
│   │   │   ├── fusion.py
│   │   │   ├── scoring.py
│   │   │   ├── risk.py
│   │   │   └── reporting.py
│   │   └── static/
│   ├── models/
│   │   ├── config/
│   │   └── checkpoints/
│   ├── scripts/
│   │   ├── infer_cli.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── benchmark_runtime.py
│   ├── tests/
│   ├── training/
│   └── requirements.txt
├── LICENSE
└── README.md
```

## Quick start (Windows PowerShell)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Service URLs:

- `http://127.0.0.1:8000/` (static frontend)
- `http://127.0.0.1:8000/docs` (OpenAPI docs)
- `http://127.0.0.1:8000/health`

## API endpoints

- `GET /health` - service health/version
- `GET /` - serves frontend UI (`backend/app/static/index.html`)
- `POST /evaluate` - evaluate one pitch (`PitchInput`)
- `POST /evaluate/batch` - evaluate list of pitches (`BatchEvaluationRequest`)
- `GET /videos` - list videos from `backend/outputs/batch_input`
- `GET /videos/{video_name}` - stream one video from batch input directory

## Request/response highlights

Input model (`PitchInput`) supports:

- `title`, `transcript`, `language_hint`
- `video` (`file_name`, `file_format`, `duration_sec`, `transcript_text`)
- `slides` and/or `slide_text`
- `presenter_profile` and `user_details`

Response (`EvaluationResponse`) includes:

- `summary.overall_score`, `summary.confidence_score`
- `summary.investment_band` (`high-potential`, `watchlist`, `early-risk`)
- `summary.language_detected` (`en`, `ta`, `ta-en`)
- `summary.processing_option` and `summary.processing_notes`
- `chunk_reports[]` with metric-level scores, attention, and risk flags
- `dashboard` series for quantitative metrics, modality weights, and risk distribution

## CLI usage

All scripts run from `backend/`.

Single-video inference:

```powershell
python scripts/infer_cli.py --video outputs/batch_input/sample.mp4 --duration-sec 90 --language-hint en-ta --output outputs/sample_eval.json
```

Batch-video inference:

```powershell
python scripts/infer_cli.py --batch-dir outputs/batch_input --duration-sec 90 --batch-output-dir outputs/batch_results --output outputs/batch_summary.json
```

Train:

```powershell
python scripts/train.py --config models/config/training_cpu.yaml
```

Evaluate checkpoint:

```powershell
python scripts/evaluate.py --config models/config/training_cpu.yaml --checkpoint models/checkpoints/phase6_cpu_nn_model.pt
```

Benchmark runtime:

```powershell
python scripts/benchmark_runtime.py --runs 5 --duration-sec 60 --output outputs/benchmark_runtime.json
```

## Configuration

Environment variables use the `SPE_` prefix and are loaded from `.env` at repository root and/or `backend/.env`.

Core flags:

```text
SPE_USE_HEURISTIC_PIPELINE=true
SPE_USE_LOCAL_TRANSCRIBER=true
SPE_ENABLE_VISUAL_EXTRACTION=true
SPE_ENABLE_AUDIO_EXTRACTION=true
SPE_CHUNK_WINDOW_SECONDS=5
SPE_MEDIA_LOOKUP_DIR=outputs/batch_input
```

Neural pipeline flags (Phase 7):

```text
SPE_USE_HEURISTIC_PIPELINE=false
SPE_NN_CHECKPOINT_PATH=models/checkpoints/nn_model.pt
SPE_NN_TEXT_ENCODER=all-MiniLM-L6-v2
SPE_NN_VISUAL_BACKBONE=mobilenet_v3_small
SPE_NN_AUDIO_FEATURES=mfcc
```

### Switching between heuristic and neural paths

- Keep `SPE_USE_HEURISTIC_PIPELINE=true` for deterministic heuristic scoring fallback.
- Set `SPE_USE_HEURISTIC_PIPELINE=false` to enable the neural extraction, fusion, and scoring path.
- In neural mode, ensure `SPE_NN_CHECKPOINT_PATH` points to a valid `.pt` checkpoint file.

## Training data (Phase 6, Option A)

The recommended training data strategy is Option A: LLM-labeled rubric scores stored as JSONL.

- Put split files in `backend/datasets/splits/train.jsonl`, `backend/datasets/splits/val.jsonl`, and `backend/datasets/splits/test.jsonl`.
- Use records with `video_id`, `transcript`, `slide_text`, `scores` (10 values), and optional `investment_band`.
- Full schema and rules are documented in `backend/datasets/splits/README.md`.

When split files are absent, the trainer falls back to synthetic samples so local smoke tests can still run.

Transcriber selection:

```text
SPE_TRANSCRIBER_BACKEND=auto
SPE_TRANSCRIBER_MIN_AUDIO_QUALITY=0.35
SPE_FASTER_WHISPER_MODEL_SIZE=small
SPE_FASTER_WHISPER_DEVICE=cpu
SPE_FASTER_WHISPER_COMPUTE_TYPE=int8
SPE_OPENAI_API_KEY=
SPE_OPENAI_TRANSCRIBER_MODEL=whisper-1
```

Notes:

- `auto` transcriber mode tries local faster-whisper first, then OpenAI Whisper API.
- `SPE_MEDIA_LOOKUP_DIR` is resolved relative to `backend/` unless absolute.
- Audio extraction uses ffmpeg; if not on PATH, `imageio-ffmpeg` fallback is attempted.

## Tests

```powershell
cd backend
pytest -q
```

The current suite covers API parity with shared inference, pipeline response shape, language detection behavior, transcriber fallback behavior, and audio/video processor fallback safety.

## Notes for contributors

- Keep API schema changes backward-compatible where possible.
- Keep API and CLI behavior consistent through `app/services/inference.py`.
- Update docs and tests whenever endpoint behavior or configuration semantics change.
