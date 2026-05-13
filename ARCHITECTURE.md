# System Architecture & Deployment Diagram

## Current Architecture (What You Have Now)

```
┌────────────────────────────────────────────────────────────────┐
│                    LOCAL DEVELOPMENT                           │
│                   (Your Computer)                              │
└────────────────────────────────────────────────────────────────┘

┌─ Frontend Layer ────────────────────────────────────────────────┐
│                                                                 │
│  ┌────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │  index.html    │  │    app.js        │  │  styles.css     │ │
│  │                │  │  (Client Logic)  │  │  (UI Styling)   │ │
│  └────────────────┘  └──────────────────┘  └─────────────────┘ │
│           │                  │                      │            │
│           └──────────────────┴──────────────────────┘            │
│                         HTTP                                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─ Backend (FastAPI) ─────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  main.py (FastAPI Server)                                │  │
│  │  - /health          (Server status)                      │  │
│  │  - /               (Serve frontend)                      │  │
│  │  - /evaluate        (Process pitch data)                 │  │
│  │  - /evaluate/upload (Upload video)                       │  │
│  │  - /videos          (List available videos)              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Input: JSON {transcript, metadata} or Video File             │
│     │                                                           │
│     ▼                                                           │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Inference Service (Orchestrator)                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│     │                                                           │
│     ├──► Preprocessing       ──► Language detection            │
│     │                            Temporal segmentation         │
│     │                            Create PitchChunk[]            │
│     │                                                           │
│     ├──► Media Extraction    ──► Video frames + metadata       │
│     │    (Parallel)              Audio WAV + quality           │
│     │                            Visual features               │
│     │                            Audio features                │
│     │                                                           │
│     ├──► Feature Extraction  ──► Text embeddings               │
│     │    (Per modality)          Visual embeddings             │
│     │                            Audio embeddings              │
│     │                                                           │
│     ├──► Fusion              ──► Combine modalities            │
│     │                            Attention weights             │
│     │                                                           │
│     └──► Scoring             ──► Risk Score                    │
│                                  Pitch Score                   │
│                                  Strength Score                │
│                                  Recommendations               │
│                                                                 │
│  Output: JSON {scores, metrics, recommendations}              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Services Layer                                         │  │
│  │  ├─ preprocessing.py      (Split into chunks)          │  │
│  │  ├─ video_processor.py    (Extract frames)             │  │
│  │  ├─ audio_processor.py    (Extract audio)              │  │
│  │  ├─ extractors.py         (Compute features)           │  │
│  │  ├─ fusion.py             (Combine modalities)         │  │
│  │  ├─ scoring.py            (Calculate scores)           │  │
│  │  ├─ risk.py               (Detect red flags)           │  │
│  │  └─ inference.py          (Orchestrate all)            │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Models Layer                                           │  │
│  │  ├─ text_encoder.py       (Process text)               │  │
│  │  ├─ visual_encoder.py     (Process video frames)       │  │
│  │  ├─ audio_encoder.py      (Process audio WAV)          │  │
│  │  ├─ fusion_head.py        (Combine features)           │  │
│  │  ├─ scoring_head.py       (Generate scores)           │  │
│  │  └─ checkpoints/          (Pre-trained weights)        │  │
│  │      └─ phase6_gpu_nn_model.pt                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Storage Layer (Local Filesystem)                       │  │
│  │  ├─ outputs/batch_input/  (Input videos)               │  │
│  │  ├─ outputs/uploads/       (User uploads)               │  │
│  │  ├─ outputs/frames/        (Extracted frames)           │  │
│  │  └─ outputs/audio_chunks/  (Extracted audio)            │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  External Dependencies                                  │  │
│  │  ├─ OpenCV           (Video frame extraction)           │  │
│  │  ├─ ffmpeg           (Audio extraction)                 │  │
│  │  ├─ faster-whisper   (Speech-to-text)                   │  │
│  │  ├─ sentence-transformers (Text encoding)              │  │
│  │  ├─ torch            (Deep learning framework)          │  │
│  │  ├─ librosa          (Audio feature extraction)         │  │
│  │  └─ numpy/scipy      (Numerical computation)            │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─ Python Runtime (Local Machine) ─────────────────────────────┘
```

---

## Deployed Architecture (On Render.com)

```
┌─ INTERNET ──────────────────────────────────────────────────────┐
│                                                                  │
│  User Browser: https://pitch-eval-backend.onrender.com/         │
│                                                                  │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Render.com CDN/Load        │
                    │  Balancer                   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  RENDER CONTAINER          │
                    │  (Linux Debian)            │
                    └─────────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
    ┌─────────────────┐ ┌────────────────┐  ┌──────────────────┐
    │ Build Phase     │ │ Runtime Files  │  │ System Packages  │
    │ (1 time)        │ │                │  │                  │
    ├─────────────────┤ ├────────────────┤  ├──────────────────┤
    │ pip install -r  │ │ Python code    │  │ ffmpeg           │
    │ requirements.txt│ │ Models         │  │ libsm6           │
    │                 │ │ Config files   │  │ libxext6         │
    │ apt-get install │ │                │  │ libxrender-dev   │
    │ ffmpeg ...      │ │                │  │ (For video/image)│
    │                 │ │                │  │                  │
    │ bash            │ │                │  │                  │
    │ init_            │ │                │  │                  │
    │ deployment.sh   │ │                │  │                  │
    └─────────────────┘ └────────────────┘  └──────────────────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │ FastAPI Application Server          │
                    │ (Running on Port 8000)              │
                    │                                    │
                    │ ┌────────────────────────────────┐ │
                    │ │ HTTP Endpoints                 │ │
                    │ ├────────────────────────────────┤ │
                    │ │ GET  /health                   │ │
                    │ │ GET  /                         │ │
                    │ │ POST /evaluate                 │ │
                    │ │ POST /evaluate/upload          │ │
                    │ │ GET  /videos                   │ │
                    │ │ GET  /videos/{name}            │ │
                    │ └────────────────────────────────┘ │
                    │                                    │
                    │ ┌────────────────────────────────┐ │
                    │ │ Processing Pipeline            │ │
                    │ │ (Same as local)                │ │
                    │ │ • Preprocessing                │ │
                    │ │ • Media Extraction             │ │
                    │ │ • Feature Extraction           │ │
                    │ │ • Fusion                       │ │
                    │ │ • Scoring                      │ │
                    │ └────────────────────────────────┘ │
                    │                                    │
                    └──────────────┬───────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
    ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
    │ Storage      │       │ Storage      │       │ Storage      │
    │ (tmpfs)      │       │ (tmpfs)      │       │ (tmpfs)      │
    ├──────────────┤       ├──────────────┤       ├──────────────┤
    │ batch_input/ │       │ frames/      │       │ audio_chunks/│
    │ Input videos │       │ Video frames │       │ Audio files  │
    │              │       │              │       │              │
    │ uploads/     │       │ prepared/    │       │ outputs/     │
    │ User files   │       │ Processed    │       │ Results      │
    └──────────────┘       └──────────────┘       └──────────────┘
         (Ephemeral)          (Ephemeral)          (Ephemeral)

    Note: All storage is temporary (cleared on redeploy)
    For persistent storage, use Render Disk or external storage
```

---

## Data Flow: Request to Response

### Example: Upload Video for Evaluation

```
1. User Action
   └─► Browser: Upload video.mp4 to /evaluate/upload
   
2. HTTP Request
   └─► FastAPI receives FormData with:
       - video: binary file
       - title: string
       - transcript: string
       - founder_name, startup_name, sector, stage
       
3. File Storage
   └─► Save uploaded video to outputs/uploads/video_XXXX.mp4
   
4. Inference Service
   ├─► Create PitchInput object
   └─► Call inference_service.evaluate_payload()
   
5. Preprocessing
   ├─► Detect language (langdetect)
   ├─► Split into 5-second chunks
   ├─► Align transcript to chunks
   └─► Output: List[PitchChunk]
   
6. Media Extraction (Parallel)
   ├─► Video Processor
   │   ├─► OpenCV: Read frames per chunk
   │   ├─► MediaPipe: Detect face/pose/hands
   │   ├─► Compute: face_ratio, motion_score, etc.
   │   └─► Save: outputs/frames/video_name/chunk_XXXX.png
   │
   └─► Audio Processor
       ├─► ffmpeg: Extract chunk audio
       ├─► librosa: Compute quality features
       └─► Save: outputs/audio_chunks/video_name/chunk_XXXX.wav
   
7. Feature Extraction
   ├─► Text Encoder
   │   ├─► Tokenize transcript chunk
   │   ├─► sentence-transformers: Get embedding (768-dim)
   │   └─► MLP: Score 6 text dimensions (0-100 each)
   │
   ├─► Visual Encoder
   │   ├─► Resize frame to 224x224
   │   ├─► MobileNetV3: Extract features (512-dim)
   │   └─► MLP: Score 2 visual dimensions (0-100 each)
   │
   └─► Audio Encoder
       ├─► MFCC: Extract 13 coefficients
       ├─► CNN: Process coefficients
       └─► MLP: Score 2 audio dimensions (0-100 each)
   
8. Fusion
   ├─► Normalize all embeddings
   ├─► Project to common space (128-dim each)
   ├─► Concatenate: text || visual || audio (384-dim)
   └─► Attention: Compute modality weights
   
9. Scoring
   ├─► Risk Analysis: Detect red flags
   ├─► Pitch Score: Weighted average of text metrics
   ├─► Strength Score: Combine presenter + content
   └─► Generate recommendations
   
10. Response
    └─► Return JSON:
        {
          "evaluation": {
            "scores": {
              "pitch_score": 78,
              "strength_score": 75,
              "risk_score": 22
            },
            "red_flags": [...],
            "recommendations": [...]
          }
        }
        
11. User Sees
    └─► Results displayed in browser UI
        with visual charts and recommendations
```

---

## Component Dependencies

```
FastAPI (Web Server)
├─ Pydantic (Request validation)
├─ CORS Middleware
└─ Static Files (Serve frontend)

Preprocessing
├─ langdetect (Language detection)
└─ numpy (Timing calculations)

Video Processing
├─ OpenCV (cv2)
├─ MediaPipe (Optional: pose/hand detection)
└─ numpy

Audio Processing
├─ ffmpeg (System binary)
├─ librosa (Audio features)
├─ numpy
└─ scipy

Text Features
├─ sentence-transformers (Text embeddings)
├─ torch (Deep learning)
└─ numpy

Visual Features
├─ torch (PyTorch)
├─ torchvision (MobileNetV3)
├─ opencv-python
└─ pillow (Image processing)

Audio Features
├─ torch
├─ torchaudio
└─ librosa

Fusion & Scoring
├─ torch
└─ numpy
```

---

## Environment Differences

### Local Development
```
OS: Windows/Mac/Linux
Python: 3.9+
GPU: Optional (RTX/CUDA if available)
Storage: Local filesystem (unlimited)
Network: Localhost
Timeout: None
Memory: System RAM (16GB+ recommended)
CPU: Unlimited
```

### Render Deployment
```
OS: Linux (Debian)
Python: 3.9+ (specified by Render runtime)
GPU: None (free plan has CPU only)
Storage: 512MB (free plan), /tmp is ephemeral
Network: Public HTTPS only
Timeout: 30 seconds (free plan)
Memory: 512MB (free plan)
CPU: Shared resources (free plan)
```

---

## Performance Characteristics

| Metric | Local (GPU) | Local (CPU) | Render Free |
|--------|------------|-----------|------------|
| 30-sec video | 1-2 min | 5-10 min | 15-30 min |
| 60-sec video | 2-4 min | 10-20 min | 30-60 min |
| With heuristic | 30 sec | 2-3 min | 3-5 min |
| Memory usage | 2-4 GB | 1-2 GB | 512 MB limit |
| Concurrent requests | Depends on GPU | 1-2 max | 1 max (free) |

---

## Scaling Options

```
Free Plan ($0/month)
├─ 1 service
├─ 512 MB RAM
├─ 0.5 CPU
├─ 30-sec timeout
├─ Use for: Development, testing, light traffic
└─ Limitation: May timeout on long videos

Starter Plan ($7/month)
├─ Better performance
├─ 512 MB RAM (more available)
├─ 1 CPU
├─ 120-sec timeout
├─ Use for: Production with light traffic
└─ Better: Won't timeout on normal videos

Pro Plan ($12+/month)
├─ Full performance
├─ Configurable RAM
├─ Full CPU cores
├─ No timeout limit
├─ Use for: Production with heavy traffic
└─ Best: High availability & scaling
```

---

## Summary

Your system processes startup pitch videos through a sophisticated ML pipeline:

✅ **5-stage pipeline** for comprehensive analysis
✅ **3 modalities** (text, visual, audio) for holistic scoring
✅ **Works locally** for development
✅ **Deploys to cloud** with Render.com
✅ **Heuristic mode** for fast execution without GPU
✅ **Neural mode** for high-quality analysis with models
✅ **Fallback gracefully** when resources are limited

The architecture is designed to work across different compute environments, from your laptop to cloud servers!

