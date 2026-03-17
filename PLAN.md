# Agent Work Plan

This file is the implementation backlog for coding agents.
Goal: build a fully local, self-trained multimodal video scoring system for startup pitch evaluation.

## Architecture Separation

CLI Path (Primary for ML)

- Training, evaluation, batch inference, data processing
- No HTTP dependency required
- Direct terminal commands in backend/scripts/

FastAPI Path (Optional for Frontend Only)

- Add only after Phase 6 is stable
- Thin wrapper around same core inference logic
- Both CLI and API return identical JSON outputs

## Global constraints

1. No external API calls for transcription, inference, or training.
2. Input must be video-first.
3. Transcription must run locally.
4. Final scores must come from self-trained models, not hardcoded heuristics.
5. CLI is primary for all ML development.
6. FastAPI is optional; add only for frontend integration at the end.
7. CLI and FastAPI share identical core inference code.

## Current baseline (already present)

1. Working backend scaffold and tests.
2. Heuristic extractors/fusion/scoring in service layer.
3. Structured response schema with chunk reports and dashboard payload.

## Phase-wise execution

## Phase 0: Migration controls

Tasks:

1. Add feature flag `USE_HEURISTIC_PIPELINE` (default: true) for safe migration.
2. Add feature flag `USE_LOCAL_TRANSCRIBER` (default: false until transcriber is ready).
3. Add clear logging at pipeline start to show active mode.

Definition of done:

1. Existing tests still pass.
2. Pipeline can switch modes without runtime crash.

## Phase 1: Dataset and labels foundation

Tasks:

1. Create dataset folder structure:
   - `backend/datasets/raw`
   - `backend/datasets/processed`
   - `backend/datasets/splits`
   - `backend/datasets/labels`
2. Add label schema file `backend/datasets/labels/schema.json`.
3. Add split script `backend/scripts/make_splits.py` with speaker/startup leakage checks.
4. Add label validator `backend/scripts/validate_labels.py`.

Definition of done:

1. Split files generated for train/val/test.
2. Validation script passes on sample dataset.
3. No startup/speaker overlap across splits.

## Phase 2: Local preprocessing from video

Tasks:

1. Add `backend/app/services/video_processor.py` for frame extraction per 5-second chunk.
2. Add `backend/app/services/audio_processor.py` for audio extraction + mel features.
3. Upgrade `backend/app/services/preprocessing.py` to use true timeline chunking from video duration.
4. Persist chunk metadata for text/audio/visual alignment.

Definition of done:

1. For any input video, aligned chunk artifacts are produced.
2. Chunk metadata is deterministic.

## Phase 3: Local transcription module

Tasks:

1. Add `backend/app/services/transcriber.py` abstraction interface.
2. Implement local backend A: Whisper local.
3. Implement local backend B: faster-whisper local.
4. Add confidence and fallback behavior for empty/low-quality audio chunks.

Definition of done:

1. No network usage during transcription.
2. Stable transcript output for same input and settings.

## Phase 4: Trainable multimodal model stack

Tasks:

1. Add model package:
   - `backend/models/text_encoder.py`
   - `backend/models/visual_encoder.py`
   - `backend/models/audio_encoder.py`
   - `backend/models/fusion_head.py`
   - `backend/models/scoring_head.py`
2. Replace heuristics in `backend/app/services/extractors.py` with model inference wrappers.
3. Replace fixed fusion in `backend/app/services/fusion.py` with learned fusion module.
4. Replace weighted formula in `backend/app/services/scoring.py` with learned multi-output scoring.

Definition of done:

1. Forward pass returns 10 metric scores, overall score, and confidence.
2. Works on both CPU and GPU profiles.

## Phase 5: Training system

Tasks:

1. Add training package:
   - `backend/training/dataset_loader.py`
   - `backend/training/trainer.py`
   - `backend/training/metrics.py`
   - `backend/training/losses.py`
2. Add scripts:
   - `backend/scripts/train.py`
   - `backend/scripts/evaluate.py`
3. Add config files:
   - `backend/models/config/training_cpu.yaml`
   - `backend/models/config/training_gpu.yaml`
4. Add checkpoint saving/loading under `backend/models/checkpoints`.

Definition of done:

1. Train run completes and writes checkpoints.
2. Eval run outputs MAE/RMSE and ranking metrics.

## Phase 6: CLI Inference Productization (ML development path)

Tasks:

1. Add `backend/scripts/infer_cli.py` for single video inference.
2. Add batch mode for directory input.
3. Extract core inference logic into `backend/app/services/inference.py`.
4. Run full pipeline: video chunking -> local transcription -> multimodal model -> report JSON.

Definition of done:

1. CLI produces full report JSON for unseen videos.
2. CLI is the primary way ML engineers run inference and batch scoring.
3. No FastAPI needed for ML development.

## Phase 6b: FastAPI Frontend Integration (optional, after Phase 6 stable)

Tasks:

1. Keep `backend/app/main.py` as thin wrapper.
2. POST /evaluate endpoint calls same `backend/app/services/inference.py` as CLI.
3. Verify FastAPI output matches CLI output for identical inputs.
4. This layer is optional for frontend UI integration only.

Definition of done:

1. FastAPI endpoint returns identical JSON to CLI.
2. Frontend can upload video and receive scoring report.
3. FastAPI can be omitted entirely if not needed.

## Phase 7: Quality and hardening

Tasks:

1. Expand tests:
   - unit tests for processors/transcriber/model wrappers
   - integration tests for end-to-end video -> score
2. Add edge-case tests:
   - silent audio
   - short video
   - corrupt file
   - mixed English-Tamil speech
3. Add benchmark script for CPU and GPU runtime.

Definition of done:

1. Test suite passes.
2. Failures are handled gracefully with clear error messages.
3. Latency baseline is documented.

## Agent assignments (sequential)

1. Agent A: Dataset contracts, split tooling, label validator.
2. Agent B: Video/audio preprocessing modules and alignment metadata.
3. Agent C: Local transcription backends and abstraction.
4. Agent D: Model modules (text/visual/audio/fusion/scoring).
5. Agent E: Training loop, config, checkpoints, evaluation metrics.
6. Agent F: CLI inference script and shared core inference service (backend/app/services/inference.py).
7. Agent G: Testing, edge cases, benchmark, docs sync.
8. Agent H (optional): FastAPI wrapper for frontend (only after Phase 6 complete and stable).

## Completion criteria (Phase 0-7: ML development stable)

1. Input is video.
2. Transcription is local.
3. Scores are from self-trained models.
4. No external API calls in training/inference paths.
5. End-to-end CLI run: video -> final investor-ready report JSON.
6. All ML tasks (train, eval, inference) work from CLI without FastAPI.

## Frontend Integration (Phase 6b: optional, add after ML stable)

1. FastAPI wrapper added only after Phase 6 is tested and stable.
2. Single endpoint POST /evaluate receives same inputs as CLI.
3. FastAPI output is identical to CLI output for same inputs.
4. Frontend app can upload video and display scores/charts from FastAPI response.
5. FastAPI is removable without affecting ML pipeline.
