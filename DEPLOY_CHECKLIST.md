# Quick Deploy Checklist

## Pre-Deployment (Do These Locally First)

- [ ] **Check model files exist**
  ```bash
  ls -la backend/models/checkpoints/
  # Should show: phase6_gpu_nn_model.pt (or similar checkpoint)
  ```

- [ ] **Test locally** (ensure pipeline works)
  ```bash
  cd backend
  python -m pip install -r requirements.txt
  uvicorn app.main:app --reload
  # Test: curl http://localhost:8000/health
  ```

- [ ] **Prepare sample videos** for deployment
  - Add 1-2 sample videos (30-60 sec each) to `backend/outputs/batch_input/`
  - Example: `sample_pitch_1.mp4`

- [ ] **Create .env file** in backend/
  ```bash
  cp backend/.env.render backend/.env
  # Or let init_deployment.sh create it
  ```

- [ ] **Verify Git is ready**
  ```bash
  git status
  # Should show clean working directory (or staged files ready to commit)
  ```

---

## GitHub Setup

- [ ] **Commit all changes**
  ```bash
  git add .
  git commit -m "Add deployment configuration and initialization script"
  ```

- [ ] **Ensure model checkpoints are committed**
  ```bash
  git add backend/models/checkpoints/
  git commit -m "Add model checkpoints"
  ```

- [ ] **Push to main branch**
  ```bash
  git push origin main
  ```

---

## Render.com Deployment

### Step 1: Create Render Account
- [ ] Go to https://render.com
- [ ] Sign up / Log in with GitHub account
- [ ] Grant access to your repositories

### Step 2: Connect Repository
- [ ] Click "New +" → "Web Service"
- [ ] Select the repository: `Startup-Pitch-Evaluation`
- [ ] Click "Connect"

### Step 3: Configure Service
- [ ] **Name**: `pitch-eval-backend`
- [ ] **Runtime**: `Python`
- [ ] **Build Command**: Verify it shows from `render.yaml`:
  ```
  pip install -r requirements.txt && apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev && bash init_deployment.sh
  ```
- [ ] **Start Command**: Verify it shows:
  ```
  uvicorn app.main:app --host 0.0.0.0 --port $PORT
  ```
- [ ] **Plan**: `Free` (or `Starter` for better performance)

### Step 4: Add Environment Variables
In Render dashboard, add these:
```
SPE_USE_HEURISTIC_PIPELINE=true
SPE_USE_LOCAL_TRANSCRIBER=true
SPE_ENABLE_VISUAL_EXTRACTION=true
SPE_ENABLE_AUDIO_EXTRACTION=true
SPE_FASTER_WHISPER_DEVICE=cpu
SPE_FASTER_WHISPER_COMPUTE_TYPE=int8
SPE_MEDIA_LOOKUP_DIR=outputs/batch_input
```

### Step 5: Deploy
- [ ] Click "Create Web Service"
- [ ] Wait for build to complete (5-10 minutes)
- [ ] Check "Events" tab for any errors
- [ ] Check "Logs" tab for startup messages

---

## Post-Deployment Testing

### Test 1: Health Check
```bash
curl https://YOUR_SERVICE_NAME.onrender.com/health
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

### Test 2: List Available Videos
```bash
curl https://YOUR_SERVICE_NAME.onrender.com/videos
```

Expected response:
```json
{
  "videos": ["sample_pitch_1.mp4", "sample_pitch_2.mp4"]
}
```

### Test 3: Evaluate Video via API
```bash
curl -X POST "https://YOUR_SERVICE_NAME.onrender.com/evaluate/upload" \
  -F "video=@backend/outputs/batch_input/sample_pitch_1.mp4" \
  -F "title=Sample Pitch" \
  -F "transcript=This is a startup pitch about a new SaaS platform" \
  -F "founder_name=John Doe" \
  -F "startup_name=Example Startup" \
  -F "sector=SaaS" \
  -F "stage=Series A"
```

### Test 4: Access Web UI
- [ ] Open browser to: `https://YOUR_SERVICE_NAME.onrender.com/`
- [ ] Upload a video
- [ ] See evaluation results

---

## What Happens During Deployment

### Build Phase (5-10 minutes)
1. Install Python dependencies (pip install -r requirements.txt)
2. Install system packages (ffmpeg, OpenCV libs)
3. Run init_deployment.sh:
   - Create output directories
   - Create .env file
   - Verify model checkpoints

### Runtime Phase
1. FastAPI server starts on port $PORT
2. Exposes endpoints:
   - `/health` - Service health
   - `/` - Web UI
   - `/evaluate` - API evaluation
   - `/evaluate/upload` - Video upload
   - `/videos` - List available videos

### Data Processing Pipeline
When a video is evaluated:
1. **Preprocessing** - Split into 5-sec chunks, detect language
2. **Extraction** - Extract video frames & audio
3. **Features** - Compute text, visual, audio embeddings
4. **Fusion** - Combine features across modalities
5. **Scoring** - Generate risk & pitch scores
6. **Output** - Return evaluation report

---

## Common Issues & Solutions

### Build Fails with "ffmpeg not found"
- Render may not have installed system packages
- Check "Events" tab for build logs
- Try re-deploying

### "Model checkpoint not found" Warning
- This is OK if using heuristic pipeline
- Set `SPE_USE_HEURISTIC_PIPELINE=true` (already done)
- Heuristic mode doesn't need checkpoints

### Deployment times out (>30 min)
- Free Render plans have resource limits
- Try "Starter" plan for better performance
- Or disable heavy processing:
  ```
  SPE_ENABLE_VIDEO_EXTRACTION=false
  ```

### Slow video processing
- Free plans have limited CPU
- First request may be slow (model loading)
- Subsequent requests are faster

### Out of memory during processing
- Use heuristic pipeline (less memory)
- Use `faster_whisper_model_size=tiny`
- Process shorter videos

---

## Monitoring & Logs

### View Logs in Render Dashboard
1. Go to https://dashboard.render.com/
2. Click your service: `pitch-eval-backend`
3. Click "Logs" tab
4. Filter by time/keyword

### Expected Log Messages
```
✓ App started | Startup Pitch Evaluation API v0.1.0
✓ Temporal segmentation created 12 chunks
✓ Video extraction complete: 12 frames per chunk
✓ Audio extraction complete: 12 WAV files
✓ Feature extraction complete
✓ Scoring complete | Risk Score: 0.45
```

### Debug Logs
Add to environment variables:
```
PYTHONUNBUFFERED=1
```

---

## Deployment URL

Your deployed app will be available at:
```
https://YOUR_SERVICE_NAME.onrender.com/
```

Replace `YOUR_SERVICE_NAME` with the name you gave during setup (e.g., `pitch-eval-backend`).

---

## Next Steps After Deployment

1. **Monitor performance**
   - Check processing times
   - Monitor errors in logs
   - Optimize model settings if needed

2. **Add more sample videos**
   - Update `backend/outputs/batch_input/` with more examples
   - Commit and push
   - Render will auto-redeploy

3. **Customize frontend**
   - Edit `backend/app/static/index.html`
   - Edit `backend/app/static/app.js`
   - Push changes to deploy

4. **Scale up** (if needed)
   - Upgrade to "Starter" or "Pro" plan
   - Enable full neural network mode
   - Disable heuristic fallbacks

---

## Support

- **Render Docs**: https://render.com/docs
- **FastAPI Docs**: https://YOUR_SERVICE_NAME.onrender.com/docs
- **Local Testing**: `cd backend && uvicorn app.main:app --reload`

