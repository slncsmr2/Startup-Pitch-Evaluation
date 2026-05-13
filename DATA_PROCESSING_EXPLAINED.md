# Data Processing Pipeline - Technical Deep Dive

This document explains exactly how data flows through the system, both locally and on deployment.

---

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Video File + Metadata (JSON or Form Upload)             │
│ Example: pitch_video.mp4, transcript="...", founder_name="..."  │
└─────────────────────────────────────────────────────────────────┘
                               ↓
                    [1] PREPROCESSING
                               ↓
    ┌────────────────────────────────────────────────────────┐
    │ Resolve Media Paths, Detect Language, Split Chunks    │
    │ Output: List[PitchChunk] with 5-sec temporal windows  │
    └────────────────────────────────────────────────────────┘
                               ↓
              [2] MEDIA EXTRACTION (Parallel)
                               ↓
    ┌─────────────────┬─────────────────┬──────────────────┐
    │ Video Frames    │ Audio WAV Files │ Visual Metadata  │
    │ (PNG per chunk) │ (WAV per chunk) │ (JSON features)  │
    └─────────────────┴─────────────────┴──────────────────┘
                               ↓
           [3] FEATURE EXTRACTION (Per Modality)
                               ↓
    ┌──────────────┬──────────────┬──────────────┐
    │ Text         │ Visual       │ Audio        │
    │ Embeddings   │ Embeddings   │ Embeddings   │
    │ (768-dim)    │ (256-dim)    │ (128-dim)    │
    └──────────────┴──────────────┴──────────────┘
                               ↓
                   [4] MODALITY FUSION
                               ↓
        ┌──────────────────────────────────────┐
        │ Fused Multi-Modal Embeddings         │
        │ + Attention Weights per Modality     │
        └──────────────────────────────────────┘
                               ↓
                  [5] SCORING & RISK ANALYSIS
                               ↓
        ┌──────────────────────────────────────┐
        │ OUTPUT: Evaluation Report             │
        │ - Risk Score (0-100)                 │
        │ - Pitch Score (0-100)                │
        │ - Strength Score (0-100)             │
        │ - Detailed metrics per chunk         │
        │ - Recommendations                    │
        └──────────────────────────────────────┘
```

---

## Stage 1: Preprocessing

**File**: `backend/app/services/preprocessing.py`

### What Happens
```
Input Video File
    ↓
Detect language (English/Tamil/Mixed)
    ↓
Resolve media file path using media_lookup_dir
    ↓
Extract/use transcript (from payload or video metadata)
    ↓
Extract slide context (if provided)
    ↓
Split into 5-second chunks using SPE_CHUNK_WINDOW_SECONDS
    ↓
Output: List of PitchChunk objects
```

### Key Code Path
```python
# backend/app/services/preprocessing.py

class PreprocessingService:
    def process(self, pitch_input: PitchInput) -> ProcessedPitch:
        # 1. Resolve file path
        video_path = self._resolve_media_path(pitch_input.video.file_name)
        
        # 2. Get transcript (payload first, fallback to video)
        transcript = pitch_input.transcript or pitch_input.video.transcript_text
        
        # 3. Detect language
        language = self.language_detector.detect(transcript)
        
        # 4. Split into chunks
        chunks = self._temporal_segmentation(
            video_path=video_path,
            transcript=transcript,
            window_seconds=5  # Default
        )
        
        # 5. For each chunk, create PitchChunk object
        return ProcessedPitch(chunks=chunks, language=language)
```

### Output Example
```json
{
  "chunks": [
    {
      "chunk_id": 0,
      "start_sec": 0,
      "end_sec": 5,
      "transcript_text": "We identified a problem in the market...",
      "language": "en",
      "media_paths": {
        "video": "/outputs/batch_input/pitch.mp4",
        "audio": "will_be_created_in_extraction"
      }
    },
    {
      "chunk_id": 1,
      "start_sec": 5,
      "end_sec": 10,
      "transcript_text": "Our solution uses AI to solve this...",
      ...
    }
  ],
  "duration_sec": 60,
  "language": "en"
}
```

---

## Stage 2: Media Extraction

**Files**: 
- `backend/app/services/video_processor.py` - Video extraction
- `backend/app/services/audio_processor.py` - Audio extraction

### What Happens

#### 2A. Video Extraction
```python
# For each chunk:
# 1. Sample frames from the chunk (e.g., middle frame or every N frames)
# 2. Use OpenCV to read frame
# 3. Optional: Use MediaPipe to detect:
#    - Face landmarks (eye contact, emotion)
#    - Body pose (gesture, confidence)
#    - Hand tracking
# 4. Compute visual features:
#    - face_ratio (% of frame with visible face)
#    - motion_score (optical flow magnitude)
#    - eye_contact_score (facing camera)
#    - pose_ratio (upright posture %)
#    - gesture_energy (hand movement)
# 5. Save frame as PNG in outputs/frames/
```

#### 2B. Audio Extraction
```python
# For each chunk:
# 1. Extract chunk duration [start_sec, end_sec]
# 2. Use ffmpeg to extract WAV audio
# 3. Load with librosa, compute waveform features:
#    - silence_ratio (% of chunk that's silence)
#    - clipping_ratio (% of samples at max amplitude)
#    - speech_density (energy concentration)
#    - pitch_variation (F0 contour variability)
#    - energy_variation (loudness changes)
#    - audio_quality_score (combined quality metric)
# 4. Save WAV in outputs/audio_chunks/
```

### Key Code Path
```python
# backend/app/services/video_processor.py
video_processor = VideoProcessingService()
extracted_video = video_processor.extract_frames_and_metadata(
    pitch_chunk=chunk,
    output_dir="outputs/frames/pitch_name/"
)
# Returns: {
#   "frame_path": "outputs/frames/pitch_name/chunk_0000.png",
#   "face_ratio": 0.75,
#   "motion_score": 0.42,
#   "eye_contact_score": 0.68,
#   ...
# }

# backend/app/services/audio_processor.py
audio_processor = AudioProcessingService()
extracted_audio = audio_processor.extract_chunk_audio(
    pitch_chunk=chunk,
    output_dir="outputs/audio_chunks/pitch_name/"
)
# Returns: {
#   "audio_path": "outputs/audio_chunks/pitch_name/chunk_0000.wav",
#   "silence_ratio": 0.05,
#   "clipping_ratio": 0.0,
#   "speech_density": 0.92,
#   ...
# }
```

### File Structure Created
```
outputs/
├── frames/
│   └── pitch_name_mkv/
│       ├── chunk_0000.png  ← Middle frame of chunk 0
│       ├── chunk_0001.png
│       └── ...
└── audio_chunks/
    └── pitch_name_mkv/
        ├── chunk_0000.wav  ← 5-sec WAV for chunk 0
        ├── chunk_0001.wav
        └── ...
```

---

## Stage 3: Feature Extraction

**Files**:
- `backend/models/text_encoder.py` - Text features
- `backend/models/visual_encoder.py` - Visual features
- `backend/models/audio_encoder.py` - Audio features
- `backend/app/services/extractors.py` - Orchestration

### What Happens

#### 3A. Text Features
```python
# For each chunk's transcript:

# Mode 1: Heuristic (Fast, No Model)
# - Use keyword matching for:
#   - Problem Clarity (mentions: "problem", "pain", "challenge")
#   - Market Opportunity (mentions: "market", "TAM", "opportunity")
#   - Solution Uniqueness (mentions: "unique", "proprietary", "patent")
#   - Traction Evidence (mentions: "customers", "revenue", "growth")
#   - Business Model Strength (mentions: "pricing", "monetization")
#   - Team Readiness (mentions: "experienced", "team", "track record")
# - Output: Score 0-100 for each metric

# Mode 2: Neural (Using Model)
# - Use sentence-transformers to embed text
# - Pass embedding through MLP scoring head
# - Output: Score 0-100 for each metric
# - More accurate but slower
```

#### 3B. Visual Features
```python
# For each chunk's extracted frame:

# Mode 1: Heuristic
# - Use face_ratio from MediaPipe
# - Calculate Delivery Clarity = face_ratio * 100
# - Use gesture_energy for Presenter Confidence

# Mode 2: Neural
# - Resize frame to MobileNetV3 input (224x224)
# - Extract features from MobileNetV3 backbone
# - Pass through projection layer (512-dim → 256-dim)
# - Pass through scoring MLP
# - Output: Delivery Clarity, Presenter Confidence (0-100)
```

#### 3C. Audio Features
```python
# For each chunk's extracted WAV:

# Mode 1: Heuristic
# - Calculate Voice Pace = (num_words / duration) * 100
# - Calculate Voice Prosody = pitch_variation score

# Mode 2: Neural
# - Extract MFCC features (13 coefficients × 50 frames)
# - Pass through CNN feature extractor
# - Pass through scoring MLP
# - Output: Voice Pace, Voice Prosody (0-100)
```

### Output Example (Per Chunk)
```json
{
  "chunk_id": 0,
  "text_features": {
    "problem_clarity": 78,
    "market_opportunity": 82,
    "solution_uniqueness": 75,
    "traction_evidence": 45,
    "business_model_strength": 70,
    "team_readiness": 80
  },
  "visual_features": {
    "delivery_clarity": 85,
    "presenter_confidence": 75
  },
  "audio_features": {
    "voice_pace": 72,
    "voice_prosody": 68
  }
}
```

---

## Stage 4: Modality Fusion

**File**: `backend/models/fusion_head.py` and `backend/app/services/fusion.py`

### What Happens
```python
# Input: Text features (6-dim), Visual features (2-dim), Audio features (2-dim)

# Process:
# 1. Normalize each modality to [0, 1]
# 2. Project to embedding space:
#    text_embedding = text_proj(text_features)    # → 128-dim
#    visual_embedding = visual_proj(visual_features)  # → 128-dim
#    audio_embedding = audio_proj(audio_features)  # → 128-dim

# 3. Concatenate: [text_emb || visual_emb || audio_emb]  # → 384-dim
# 4. Pass through fusion attention layer
# 5. Get modality weights: [text_weight, visual_weight, audio_weight]
# 6. Output: Fused embedding (128-dim) + weights

# Interpretation:
# - Weights show which modalities are most important
# - Example: text_weight=0.5, visual_weight=0.3, audio_weight=0.2
#   → Text is 50% important, visual 30%, audio 20%
```

### Output Example
```json
{
  "fused_embedding": [0.12, 0.45, -0.23, ...],  # 128-dim vector
  "modality_weights": {
    "text": 0.52,
    "visual": 0.31,
    "audio": 0.17
  }
}
```

---

## Stage 5: Scoring & Risk Analysis

**Files**:
- `backend/services/scoring.py` - Pitch scoring
- `backend/services/risk.py` - Risk detection

### What Happens

#### 5A. Pitch Score (0-100)
```python
# Average of all text metrics
pitch_score = mean([
    problem_clarity,
    market_opportunity,
    solution_uniqueness,
    traction_evidence,
    business_model_strength,
    team_readiness
])

# Weighted by modality importance:
pitch_score = (
    0.60 * pitch_score +      # Text is most important
    0.20 * delivery_clarity +  # Visual delivery
    0.20 * voice_prosody       # Audio communication
)
```

#### 5B. Strength Score (0-100)
```python
# Combines multiple signals:
strength_score = weighted_average([
    presenter_confidence: 0.25,
    content_clarity: 0.25,
    traction_evidence: 0.25,
    business_model_strength: 0.25
])

# Penalized by red flags:
if any_red_flags:
    strength_score *= 0.8  # 20% penalty
```

#### 5C. Risk Score (0-100)
```python
# Detects problems in the pitch

# Red flags to detect:
flags = []

# 1. Speech quality issues
if silence_ratio > 0.3:
    flags.append("High silence (unclear speaking)")
if pitch_variation < 0.2:
    flags.append("Monotone voice (not engaging)")

# 2. Content gaps
if traction_evidence < 30:
    flags.append("Missing traction/customer evidence")
if market_opportunity < 40:
    flags.append("Unclear market opportunity")

# 3. Presenter issues
if eye_contact_score < 0.5:
    flags.append("Poor eye contact")
if gesture_energy < 0.2:
    flags.append("Minimal engagement/gestures")

# Risk score = severity_weighted_sum(flags)
risk_score = calculate_risk(flags)
```

### Final Output
```json
{
  "evaluation": {
    "title": "Sample Pitch",
    "duration_sec": 60,
    "language": "en",
    "scores": {
      "pitch_score": 78,
      "strength_score": 75,
      "risk_score": 22
    },
    "detailed_metrics": [
      {
        "chunk_id": 0,
        "text": { "problem_clarity": 78, ... },
        "visual": { "delivery_clarity": 85, ... },
        "audio": { "voice_pace": 72, ... }
      },
      ...
    ],
    "red_flags": [
      "Moderate silence in chunk 3 (need clearer speech)",
      "Limited traction evidence (show customer metrics)"
    ],
    "recommendations": [
      "Add specific customer acquisition metrics",
      "Emphasize unique competitive advantage",
      "Improve pacing - currently 72 WPM (aim for 130-150)"
    ]
  }
}
```

---

## How It Works During Deployment vs Locally

### Locally
```
┌─────────────────────────────────────────────┐
│ User runs: uvicorn app.main:app --reload    │
│ Working directory: backend/                 │
└─────────────────────────────────────────────┘
        ↓
✓ Can read from outputs/ (relative path works)
✓ Models load from models/checkpoints/ (relative)
✓ Fast iteration (no rebuilding)
✓ Can debug with print statements
✓ GPU available if configured
```

### On Render Deployment
```
┌─────────────────────────────────────────────┐
│ Render builds Docker container              │
│ Working directory: /app/backend/            │
└─────────────────────────────────────────────┘
        ↓
1. Build phase:
   ✓ pip install -r requirements.txt
   ✓ apt-get install ffmpeg libsm6 ...
   ✓ bash init_deployment.sh (creates outputs/ dirs)

2. Runtime phase:
   ✓ Resolve paths using settings.backend_root
   ✓ Relative paths work: outputs/batch_input/
   ✓ Load models from /app/backend/models/
   ✓ CPU only (no GPU on free plan)
   ✓ Read-only filesystem except /tmp

3. Differences to handle:
   ✗ No ffmpeg unless installed in buildCommand
   ✗ No GPU for neural features
   ✗ Limited memory (use heuristic mode)
   ✗ Timeout limits (30 sec for free plan)
```

---

## Environment Variables That Control Behavior

```python
# backend/app/core/config.py

# Feature extraction mode
SPE_USE_HEURISTIC_PIPELINE = True      # Fast, low-memory
SPE_USE_LOCAL_TRANSCRIBER = True       # Use faster-whisper

# Extraction toggles
SPE_ENABLE_VIDEO_EXTRACTION = True     # Extract frames
SPE_ENABLE_AUDIO_EXTRACTION = True     # Extract audio

# Whisper config
SPE_TRANSCRIBER_BACKEND = "auto"       # "auto", "faster-whisper", "openai"
SPE_FASTER_WHISPER_MODEL_SIZE = "small" # "tiny", "small", "base", "medium", "large"
SPE_FASTER_WHISPER_DEVICE = "cpu"      # "cpu" or "cuda"
SPE_FASTER_WHISPER_COMPUTE_TYPE = "int8" # "float32", "float16", "int8"

# Model paths
SPE_NN_CHECKPOINT_PATH = "models/checkpoints/phase6_gpu_nn_model.pt"
SPE_MEDIA_LOOKUP_DIR = "outputs/batch_input"
```

---

## Performance Expectations

### Locally (GPU)
- 30-60 second video: 2-3 minutes total
- Breakdown:
  - Preprocessing: 5 sec
  - Media extraction: 30 sec (video + audio)
  - Features (neural): 60 sec
  - Fusion & scoring: 10 sec

### Locally (CPU)
- 30-60 second video: 5-10 minutes total
- Breakdown: Same as above but 3-5x slower

### On Render (Free Plan)
- 30-60 second video: 15-30 minutes total
- Using heuristic mode: 2-5 minutes
- Limited CPU, no GPU
- May timeout on free plan (use Starter or disable extractions)

---

## How to Monitor Processing

### View logs locally
```bash
# Terminal 1: Start server
cd backend
uvicorn app.main:app --log-level debug --reload

# Terminal 2: Send request
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d @test_payload.json
```

### View logs on Render
1. Go to https://dashboard.render.com/
2. Click your service
3. Click "Logs" tab
4. Watch real-time processing messages

---

## Tips for Optimization

### Speed Up Processing
```
SPE_USE_HEURISTIC_PIPELINE=true        # Use heuristic instead of neural
SPE_FASTER_WHISPER_MODEL_SIZE=tiny     # Smaller whisper model
SPE_ENABLE_VIDEO_EXTRACTION=false      # Skip video frame extraction
```

### Reduce Memory
```
SPE_USE_HEURISTIC_PIPELINE=true        # Heuristic uses less memory
SPE_FASTER_WHISPER_COMPUTE_TYPE=int8   # Quantized whisper
```

### Improve Quality
```
SPE_USE_HEURISTIC_PIPELINE=false       # Use neural network
SPE_FASTER_WHISPER_MODEL_SIZE=base     # Larger whisper model
SPE_FASTER_WHISPER_DEVICE=cuda         # Use GPU if available
```

