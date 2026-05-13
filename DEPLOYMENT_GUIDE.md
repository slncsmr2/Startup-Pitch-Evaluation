# Deployment Guide - Startup Pitch Evaluation

## Overview
This guide explains how to deploy the Startup Pitch Evaluation system to Render.com with full data processing capabilities (video/audio extraction, feature extraction, and scoring).

---

## Part 1: Current Local Process (How It Works Locally)

### Pipeline Flow
```
Input Video
    ↓
[Preprocessing] → Split into 5-second chunks, detect language
    ↓
[Extraction] → Video frames + Audio WAV files
    ↓
[Feature Extraction] → Text, Visual, Audio embeddings
    ↓
[Fusion] → Combine modalities
    ↓
[Scoring] → Risk, Pitch, Strength scores
    ↓
Output: Evaluation Report
```

### Key Local Dependencies
- **Media Processing**: ffmpeg, OpenCV, MediaPipe
- **Audio**: faster-whisper or OpenAI Whisper
- **ML Models**: 
  - Text encoder (sentence-transformers)
  - Visual backbone (MobileNetV3)
  - Audio features (MFCC)
  - Pre-trained checkpoints in `models/checkpoints/`
- **Storage**: `outputs/` directory for:
  - `batch_input/` → Input videos
  - `audio_chunks/` → Extracted audio
  - `frames/` → Extracted video frames
  - `uploads/` → User uploads

---

## Part 2: Deployment Setup (Render.com)

### Step 1: Prepare GitHub Repository

**1.1 Update `render.yaml`:**
```yaml
services:
  - type: web
    name: pitch-eval-backend
    runtime: python
    rootDir: backend
    buildCommand: |
      pip install -r requirements.txt
      apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    plan: free
    envVars:
      - key: SPE_USE_HEURISTIC_PIPELINE
        value: "true"
      - key: SPE_USE_LOCAL_TRANSCRIBER
        value: "true"
      - key: SPE_ENABLE_VISUAL_EXTRACTION
        value: "true"
      - key: SPE_ENABLE_AUDIO_EXTRACTION
        value: "true"
      - key: SPE_FASTER_WHISPER_DEVICE
        value: "cpu"
      - key: SPE_FASTER_WHISPER_COMPUTE_TYPE
        value: "int8"
      - key: SPE_MEDIA_LOOKUP_DIR
        value: "outputs/batch_input"
      - key: PORT
        value: "8000"
```

**1.2 Update `backend/requirements.txt`:**
Add system package installation:
```
fastapi==0.116.1
uvicorn==0.35.0
pydantic==2.11.7
pydantic-settings==2.10.1
python-multipart==0.0.20
httpx==0.28.1
langdetect==1.0.9
numpy==2.2.6
sentence-transformers>=2.7
torch>=2.0
torchaudio>=2.0
torchvision>=0.15
opencv-python==4.8.1.78
mediainfo==0.3.3
faster-whisper==0.10.0
```

### Step 2: Prepare Model Files & Assets

**2.1 Create `.gitignore` exceptions for large files:**
```
# In backend/.gitignore, modify to keep models:
# Allow model checkpoints
!models/checkpoints/
models/checkpoints/*.pt.backup

# Temp outputs can be ignored
outputs/audio_chunks/
outputs/frames/
outputs/prepared/

# But keep batch_input for demo
!outputs/batch_input/
```

**2.2 Add sample videos for deployment:**
Place 1-2 sample videos in `backend/outputs/batch_input/` (e.g., 30-60 seconds)
- Examples: `sample_pitch_1.mp4`, `sample_pitch_2.mp4`

**2.3 Ensure model checkpoints are committed:**
- `backend/models/checkpoints/phase6_gpu_nn_model.pt` → Version control
- `backend/models/config/training_gpu.yaml`
- All Python model files (text_encoder.py, visual_encoder.py, etc.)

### Step 3: Create Render Configuration Files

**3.1 Create `.env.render` (for local reference):**
```
SPE_APP_NAME=Startup Pitch Evaluation API
SPE_APP_VERSION=0.1.0
SPE_CHUNK_WINDOW_SECONDS=5
SPE_USE_HEURISTIC_PIPELINE=true
SPE_USE_LOCAL_TRANSCRIBER=true
SPE_ENABLE_VISUAL_EXTRACTION=true
SPE_ENABLE_AUDIO_EXTRACTION=true
SPE_TRANSCRIBER_BACKEND=auto
SPE_FASTER_WHISPER_MODEL_SIZE=small
SPE_FASTER_WHISPER_DEVICE=cpu
SPE_FASTER_WHISPER_COMPUTE_TYPE=int8
SPE_MEDIA_LOOKUP_DIR=outputs/batch_input
```

**3.2 Create `Procfile` (alternative if not using render.yaml):**
```
web: cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Step 4: Fix Path Issues for Deployment

**4.1 Update `backend/app/core/config.py`:**
Ensure paths resolve correctly:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os

_BACKEND_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_name: str = "Startup Pitch Evaluation API"
    app_version: str = "0.1.0"
    chunk_window_seconds: int = 5
    
    use_heuristic_pipeline: bool = True
    use_local_transcriber: bool = True
    enable_visual_extraction: bool = True
    enable_audio_extraction: bool = True
    transcriber_min_audio_quality: float = 0.35
    transcriber_backend: str = "auto"
    openai_api_key: str = ""
    openai_transcriber_model: str = "whisper-1"
    faster_whisper_model_size: str = "small"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute_type: str = "int8"
    media_lookup_dir: str = "outputs/batch_input"

    # Neural Network Config
    nn_checkpoint_path: str = "models/checkpoints/phase6_gpu_nn_model.pt"
    nn_text_encoder: str = "all-MiniLM-L6-v2"
    nn_visual_backbone: str = "mobilenet_v3_small"
    nn_audio_features: str = "mfcc"
    nn_device: str = "auto"

    @property
    def backend_root(self) -> Path:
        return _BACKEND_ROOT
    
    @property
    def checkpoint_full_path(self) -> Path:
        """Resolve checkpoint path from backend root"""
        path = Path(self.nn_checkpoint_path)
        if not path.is_absolute():
            return _BACKEND_ROOT / path
        return path
    
    @property
    def media_lookup_full_path(self) -> Path:
        """Resolve media directory from backend root"""
        path = Path(self.media_lookup_dir)
        if not path.is_absolute():
            return _BACKEND_ROOT / path
        return path

    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_ROOT / ".env"),
        env_prefix="SPE_",
    )

settings = Settings()
```

**4.2 Update main.py to use resolved paths:**
```python
# In backend/app/main.py, change:
batch_input_dir = Path(__file__).resolve().parent.parent / "outputs" / "batch_input"
upload_dir = Path(__file__).resolve().parent.parent / "outputs" / "uploads"

# To use settings:
from app.core.config import settings
batch_input_dir = settings.media_lookup_full_path
upload_dir = settings.backend_root / "outputs" / "uploads"
```

### Step 5: Create Backend Initialization Script

**5.1 Create `backend/init_deployment.sh`:**
```bash
#!/bin/bash
set -e

echo "🚀 Initializing Startup Pitch Evaluation deployment..."

# Create required directories
mkdir -p outputs/batch_input
mkdir -p outputs/audio_chunks
mkdir -p outputs/frames
mkdir -p outputs/uploads

# Verify model files exist
if [ ! -f models/checkpoints/phase6_gpu_nn_model.pt ]; then
    echo "⚠️  Warning: Model checkpoint not found"
fi

echo "✅ Deployment initialization complete"
```

Make executable:
```bash
chmod +x backend/init_deployment.sh
```

**5.2 Update `render.yaml` build command:**
```yaml
buildCommand: |
  pip install -r requirements.txt
  apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev
  bash init_deployment.sh
```

---

## Part 3: Deployment Steps on Render.com

### Step 1: Connect Repository
1. Go to [render.com](https://render.com)
2. Sign in with GitHub
3. Click "New +" → "Web Service"
4. Select your GitHub repository
5. Click "Connect"

### Step 2: Configure Service
- **Name**: `pitch-eval-backend`
- **Runtime**: Python
- **Build Command**: Use the command from `render.yaml`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Plan**: Free (or Starter for better performance)

### Step 3: Set Environment Variables
Add in Render dashboard:
```
SPE_USE_HEURISTIC_PIPELINE=true
SPE_USE_LOCAL_TRANSCRIBER=true
SPE_ENABLE_VISUAL_EXTRACTION=true
SPE_ENABLE_AUDIO_EXTRACTION=true
SPE_FASTER_WHISPER_DEVICE=cpu
SPE_FASTER_WHISPER_COMPUTE_TYPE=int8
SPE_MEDIA_LOOKUP_DIR=outputs/batch_input
```

### Step 4: Deploy
- Click "Create Web Service"
- Wait for build to complete (~5-10 minutes)
- Check logs for any errors

---

## Part 4: Testing Deployment

### Test 1: Health Check
```bash
curl https://your-app.onrender.com/health
```

Expected response:
```json
{
  "status": "ok",
  "service": "Startup Pitch Evaluation API",
  "version": "0.1.0",
  "scoring_mode": "heuristic"
}
```

### Test 2: List Videos
```bash
curl https://your-app.onrender.com/videos
```

Should return sample videos from `batch_input/`.

### Test 3: Evaluate Video
```bash
curl -X GET "https://your-app.onrender.com/videos/sample_pitch_1.mp4" \
  -H "Accept: video/mp4" > test_video.mp4
```

### Test 4: Upload & Evaluate
```bash
curl -X POST "https://your-app.onrender.com/evaluate/upload" \
  -F "video=@test_video.mp4" \
  -F "title=Test Pitch" \
  -F "transcript=This is a test startup pitch" \
  -F "founder_name=John Doe" \
  -F "startup_name=Test Startup"
```

---

## Part 5: What's Processing at Deployment

When a pitch is evaluated, the system performs:

1. **Preprocessing**
   - Split video into 5-second chunks
   - Detect language (English/Tamil)
   - Align transcript to timeline

2. **Media Extraction**
   - Extract video frames from each chunk
   - Extract audio WAV from each chunk
   - Compute visual metadata (face ratio, motion, eye contact)
   - Compute audio quality (silence ratio, clipping, energy)

3. **Feature Extraction**
   - Text: Sentiment, keywords, business signals
   - Visual: Confidence, delivery clarity
   - Audio: Pace, prosody quality

4. **Fusion & Scoring**
   - Combine all features
   - Calculate risk factors
   - Generate strength scores
   - Output evaluation report

---

## Part 6: Troubleshooting Deployment

### Issue: Build fails with ffmpeg not found
**Solution**: The `buildCommand` in `render.yaml` includes `apt-get install ffmpeg`. Verify it's there.

### Issue: Model checkpoint not found
**Solution**: Ensure `models/checkpoints/phase6_gpu_nn_model.pt` is committed to Git.

### Issue: Video processing times out
**Solution**: Free Render plan has 30-second timeout. Use Starter plan or disable heavy video extraction:
```
SPE_ENABLE_VIDEO_EXTRACTION=false
```

### Issue: Out of memory
**Solution**: Use heuristic pipeline (faster, less memory):
```
SPE_USE_HEURISTIC_PIPELINE=true
```

### Issue: Whisper model download fails
**Solution**: Pre-download model or use cached version:
```
SPE_FASTER_WHISPER_MODEL_SIZE=tiny
SPE_FASTER_WHISPER_COMPUTE_TYPE=int8
```

---

## Part 7: Frontend Setup

### Option A: Serve from Backend (Current Setup)
Frontend files are in `backend/app/static/`.
- `index.html` → Main UI
- `app.js` → Client logic
- `styles.css` → Styling

Accessed at: `https://your-app.onrender.com/`

### Option B: Deploy Separately (Future)
Deploy `frontend/` folder to Vercel/Netlify and update API endpoint:
```javascript
// In frontend/config.js
API_BASE_URL = "https://your-app.onrender.com"
```

---

## Part 8: Monitoring & Logs

### View Logs
In Render dashboard:
- Click your service
- Go to "Logs" tab
- Filter by:
  - Build logs (during deployment)
  - Runtime logs (during operation)

### Common Log Patterns
```
✓ Inference service initialized
✓ Video preprocessing complete
✓ Chunk extraction: 12 chunks created
✓ Feature extraction: Text, Visual, Audio done
✓ Scoring complete
```

---

## Next Steps

1. **Commit changes to Git**:
   ```bash
   git add .
   git commit -m "Add deployment configuration"
   git push origin main
   ```

2. **Deploy to Render**:
   - Connect repository in Render dashboard
   - Wait for build to complete

3. **Test endpoints**:
   - Health check
   - List videos
   - Upload and evaluate

4. **Monitor performance**:
   - Check processing times
   - Monitor error logs
   - Optimize as needed

---

## Support

For issues, check:
- Render logs: `https://dashboard.render.com/`
- FastAPI docs: `https://your-app.onrender.com/docs`
- Backend config: `backend/app/core/config.py`
