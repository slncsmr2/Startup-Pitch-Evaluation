# Neural Network Migration Plan

Replace the rule-and-heuristic scoring system with a trainable neural network stack while keeping the existing FastAPI API, CLI, and reporting pipeline unchanged.

---

## Running Locally vs Google Colab

**This project runs entirely on a local machine — Google Colab is not required.**

| Requirement | Detail |
|---|---|
| Python | 3.9 or later |
| Install | `pip install -r backend/requirements.txt` |
| Web server | `uvicorn app.main:app --reload` (from `backend/`) |
| GPU | Optional — all defaults use CPU (`SPE_FASTER_WHISPER_DEVICE=cpu`) |
| ffmpeg | Optional — `imageio-ffmpeg` is used as fallback if ffmpeg is not on PATH |

For the neural network training phase, a GPU (local or cloud) is recommended to cut training time, but the inference path will run on CPU.

---

## Current Architecture (Heuristic)

```
PitchInput
    └── Preprocessing (5-sec chunks)
            └── Per Chunk:
                    ├── TextEncoder   → keyword rules → 6 metric scores + 24-dim hash embedding
                    ├── VisualEncoder → signal rules  → 2 metric scores + 24-dim hash embedding
                    └── AudioEncoder  → formula rules → 2 metric scores + 24-dim hash embedding
                            └── FusionHead (energy-weighted average)
                                    └── ScoringHead (weighted formula)
                                            └── 10 metric scores + overall score
```

All embeddings are SHA256 hashes of input text or feature tuples — not learned representations. All scores come from hand-crafted formulas and keyword lookups.

---

## Target Architecture (Neural)

```
PitchInput
    └── Preprocessing (5-sec chunks)
            └── Per Chunk:
                    ├── TextEncoder   → sentence-transformer → 384-dim semantic embedding
                    ├── VisualEncoder → CNN (MobileNetV3)    → 256-dim visual embedding
                    └── AudioEncoder  → MFCC + 1D-CNN        → 128-dim audio embedding
                            └── CrossModalFusionHead (learned attention)
                                    └── NeuralScoringHead (FC network)
                                            └── 10 metric scores + overall score
```

The downstream pipeline (risk detection, feedback, dashboard, API response) stays unchanged.

---

## Migration Phases

### Phase 1 — Text Encoder (Highest Impact, CPU-Friendly)

**File:** `backend/models/text_encoder.py`

**What to do:**
- Add `sentence-transformers` to `requirements.txt`.
- Load `all-MiniLM-L6-v2` (22 MB, runs on CPU in < 50 ms per chunk).
- Replace the SHA256 hash embedding with `model.encode(chunk_text)` → 384-dim float vector.
- Keep the 6 existing heuristic metric scores temporarily as the training targets (Phase 4 replaces them with real labels).
- Add a small 2-layer MLP on top of the 384-dim embedding to predict the 6 text metric scores.
- Gate behind `SPE_USE_HEURISTIC_PIPELINE=false` so existing behaviour is unchanged by default.

**New dependency:** `sentence-transformers>=2.7`

---

### Phase 2 — Audio Encoder (Low Compute, High Signal)

**File:** `backend/models/audio_encoder.py`

**What to do:**
- Add `torchaudio` to `requirements.txt`.
- Extract MFCC features (40 coefficients, 128 frames) from each audio chunk via `torchaudio.transforms.MFCC`.
- Feed MFCCs into a small 1D CNN:
  ```
  Conv1d(40→64, k=3) → ReLU → MaxPool
  Conv1d(64→128, k=3) → ReLU → AdaptiveAvgPool
  → 128-dim audio embedding
  ```
- Replace the hardcoded pace/prosody formulas with this embedding + a 2-layer MLP scoring head.

**New dependency:** `torchaudio>=2.0`, `torch>=2.0`

---

### Phase 3 — Visual Encoder (GPU Recommended for Training)

**File:** `backend/models/visual_encoder.py`

**What to do:**
- Add `torchvision` to `requirements.txt`.
- Load `MobileNetV3-Small` pretrained on ImageNet (backbone only, no classification head).
- For each frame in the chunk, run the backbone → 576-dim feature vector.
- Mean-pool across frames → 256-dim chunk-level visual embedding.
- Keep MediaPipe signals (face ratio, eye contact, pose, gesture) as auxiliary features concatenated to the backbone output before the scoring MLP.
- Replace the two visual metric heuristics with an MLP scoring head.

**New dependency:** `torchvision>=0.15`

---

### Phase 4 — Neural Fusion Head

**File:** `backend/models/fusion_head.py`

**What to do:**
- Replace the energy-ratio attention with a learned **cross-modal attention** layer:
  - Project each modality to a common dimension (e.g., 256-dim) using linear layers.
  - Compute softmax attention scores over modality keys.
  - Produce a single 256-dim fused representation.
- This replaces the simple weighted average of hash vectors.

---

### Phase 5 — Neural Scoring Head

**File:** `backend/models/scoring_head.py`

**What to do:**
- Replace the weighted formula with a fully-connected scoring network:
  ```
  fused_vector (256-dim)
      → FC(256→128) + LayerNorm + ReLU + Dropout(0.3)
      → FC(128→64)  + ReLU
      → FC(64→10)   + Sigmoid × 10      # 10 metric scores in [0, 10]
      → FC(10→1)    + Sigmoid × 10      # overall score
  ```
- Load weights from a checkpoint file configured via `SPE_NN_CHECKPOINT_PATH`.

---

### Phase 6 — Training Pipeline

**File:** `backend/training/trainer.py`

**What to do:**

1. **Data strategy — choose one:**
   - *Option A (Recommended fast path):* Use an LLM (GPT-4o or Gemini) to score a set of real pitch transcripts against the 10-metric rubric. Store as JSONL (`video_id`, `transcript`, `slide_text`, `scores[10]`).
   - *Option B:* Recruit domain experts to manually score 200+ pitches.
   - *Option C (Bootstrap):* Use the current heuristic scores as noisy labels; train NN to generalize, then correct iteratively.

2. **Replace linear regression with a PyTorch `nn.Module`** that wraps the encoders + fusion + scoring heads.

3. **Loss function:**
   - MSE on 10 individual metric scores.
   - Cross-entropy on the 3-class investment band (high-potential / watchlist / early-risk).
   - Optional Spearman rank loss to preserve ordering across pitches.

4. **Training config:**
   - Optimizer: AdamW, lr=1e-4, weight decay=1e-2.
   - LR schedule: cosine annealing with warm restarts.
   - Early stopping: patience=10 epochs on validation MAE.
   - Batch size: 16.
   - Mixed precision (fp16) if CUDA available.

5. **Output:** single checkpoint `.pt` file loaded at inference startup.

---

### Phase 7 — Configuration & Integration

**File:** `backend/app/core/config.py`

Add:
```
SPE_USE_HEURISTIC_PIPELINE=false      # Switch to neural path
SPE_NN_CHECKPOINT_PATH=models/checkpoints/nn_model.pt
SPE_NN_TEXT_ENCODER=all-MiniLM-L6-v2
SPE_NN_VISUAL_BACKBONE=mobilenet_v3_small
SPE_NN_AUDIO_FEATURES=mfcc
```

**`backend/app/pipeline.py`:**
- When `SPE_USE_HEURISTIC_PIPELINE=false`, instantiate neural encoders, fusion, and scoring head.
- Pass the same `PitchChunk` objects — no schema changes.
- Downstream risk detection, feedback generation, and dashboard remain unchanged.

---

## Dependencies to Add

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0 | Neural network runtime |
| `torchaudio` | ≥ 2.0 | MFCC feature extraction |
| `torchvision` | ≥ 0.15 | MobileNetV3 visual backbone |
| `sentence-transformers` | ≥ 2.7 | Pre-trained text encoder |

Add these to `backend/requirements.txt` only when starting Phase 1.

---

## File Change Summary

| File | Change |
|---|---|
| `backend/requirements.txt` | Add torch, torchaudio, torchvision, sentence-transformers |
| `backend/models/text_encoder.py` | Replace hash embedding with sentence-transformer + MLP |
| `backend/models/audio_encoder.py` | Replace formula scoring with MFCC + 1D CNN |
| `backend/models/visual_encoder.py` | Replace heuristic scoring with MobileNetV3 + MLP |
| `backend/models/fusion_head.py` | Replace energy weighting with cross-modal attention |
| `backend/models/scoring_head.py` | Replace formula with FC scoring network |
| `backend/training/trainer.py` | Replace linear regression with PyTorch training loop |
| `backend/training/dataset_loader.py` | Add real/LLM-labeled data loading |
| `backend/app/core/config.py` | Add NN checkpoint and encoder config flags |
| `backend/app/pipeline.py` | Route to neural path when heuristic flag is off |

---

## Recommended Migration Order

1. **Phase 1** — Text encoder (no GPU needed, biggest quality gain)
2. **Phase 6** — Build labeled dataset in parallel (most time-consuming step)
3. **Phase 2** — Audio encoder (lightweight, CPU-friendly)
4. **Phase 3** — Visual encoder (needs GPU for training)
5. **Phase 4 + 5** — Fusion + scoring heads (end-to-end joint fine-tuning)
6. **Phase 7** — Config integration and full inference path switch

Each phase is independently testable behind the `SPE_USE_HEURISTIC_PIPELINE` flag, so the existing heuristic path remains available as a fallback throughout the migration.
