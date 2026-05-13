# Quick Reference Card

## 🚀 Deploy in 3 Steps

### Step 1: Prepare
```bash
# Add sample videos to batch_input/
# Then commit:
git add .
git commit -m "Add deployment configuration"
git push origin main
```

### Step 2: Deploy
1. Go to https://render.com/dashboard
2. New Web Service → Select GitHub repo
3. Configure: Free plan, Python runtime
4. Add env vars (from DEPLOY_CHECKLIST.md)
5. Click "Create Web Service"

### Step 3: Test
```bash
# After 5-10 min build:
curl https://YOUR_SERVICE.onrender.com/health

# Open in browser:
https://YOUR_SERVICE.onrender.com/
```

---

## 📊 Data Pipeline (5 Stages)

```
Video → [1] Preprocess → [2] Extract → [3] Features
  ↓        Split chunks    Audio/Video   Text/Visual/Audio
  ↓        Language        Frame/WAV     Embeddings
  ↓
[4] Fuse → [5] Score
Combine     Risk/Pitch
Modalities  Strength
```

**Time: 3-5 min (heuristic) | 15-30 min (neural)**

---

## 🔧 Configuration

```
HEURISTIC MODE (Fast, no GPU needed)
├─ SPE_USE_HEURISTIC_PIPELINE=true ✓
├─ SPE_FASTER_WHISPER_DEVICE=cpu
├─ SPE_FASTER_WHISPER_COMPUTE_TYPE=int8
└─ Performance: 30-sec video = 3-5 minutes

NEURAL MODE (Accurate, GPU needed)
├─ SPE_USE_HEURISTIC_PIPELINE=false
├─ SPE_FASTER_WHISPER_DEVICE=cuda
├─ SPE_FASTER_WHISPER_COMPUTE_TYPE=float32
└─ Performance: 30-sec video = 1-2 minutes (with GPU)
```

---

## 📁 What Gets Deployed

```
pitch-eval-backend (Render Service)
├─ Web UI: https://your-service.onrender.com/
├─ API Health: /health
├─ Upload Video: /evaluate/upload (form)
├─ List Videos: /videos
├─ API Docs: /docs (OpenAPI)
└─ Data Pipeline: Fully functional
    ├─ Preprocessing
    ├─ Media Extraction
    ├─ Feature Extraction
    ├─ Fusion
    └─ Scoring
```

---

## 📚 Documentation Files

| File | Purpose | Time |
|------|---------|------|
| **QUICK_START_DEPLOYMENT.md** | Overview & 3-step start | 5 min |
| **DEPLOY_CHECKLIST.md** | Step-by-step guide with checkboxes | 10 min |
| **DEPLOYMENT_GUIDE.md** | Detailed explanations | 20 min |
| **DATA_PROCESSING_EXPLAINED.md** | Technical pipeline details | 15 min |
| **ARCHITECTURE.md** | Visual diagrams (this file) | 10 min |

---

## ⚠️ Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| Build fails | Check Render "Events" log |
| Timeout (30 sec) | Upgrade to Starter plan |
| Out of memory | Enable SPE_USE_HEURISTIC_PIPELINE=true |
| Slow processing | Use heuristic mode or Starter plan |
| Model not found warning | OK! Heuristic mode works without it |
| ffmpeg not found | Check buildCommand in render.yaml |

---

## 📊 Performance

| Scenario | Time | Plan |
|----------|------|------|
| 30-sec video, heuristic | 3-5 min | Free ✓ |
| 60-sec video, heuristic | 5-8 min | Free ✓ |
| 30-sec video, neural | 1-2 min | Requires GPU |
| 60-sec video, neural | 2-4 min | Requires GPU |

**Free plan**: CPU only, good for heuristic mode
**Starter plan**: Better CPU, recommended for faster processing

---

## 🔑 Environment Variables

```bash
# In Render Dashboard → Environment Variables

# Mode
SPE_USE_HEURISTIC_PIPELINE=true

# Features
SPE_ENABLE_VISUAL_EXTRACTION=true
SPE_ENABLE_AUDIO_EXTRACTION=true

# Transcription
SPE_TRANSCRIBER_BACKEND=auto
SPE_FASTER_WHISPER_MODEL_SIZE=small
SPE_FASTER_WHISPER_DEVICE=cpu
SPE_FASTER_WHISPER_COMPUTE_TYPE=int8

# Paths
SPE_MEDIA_LOOKUP_DIR=outputs/batch_input
SPE_CHUNK_WINDOW_SECONDS=5
```

---

## 🧪 Test Endpoints

```bash
# Health check (should return 200)
curl https://your-service.onrender.com/health

# List videos
curl https://your-service.onrender.com/videos

# Upload & evaluate
curl -X POST "https://your-service.onrender.com/evaluate/upload" \
  -F "video=@video.mp4" \
  -F "title=My Pitch" \
  -F "transcript=We solve X problem" \
  -F "founder_name=John" \
  -F "startup_name=Startup Inc"

# View API docs
https://your-service.onrender.com/docs
```

---

## 📋 Pre-Deployment Checklist

- [ ] Sample videos added to backend/outputs/batch_input/
- [ ] All files committed to Git
- [ ] Pushed to main branch
- [ ] Render account created
- [ ] GitHub connected to Render

---

## 🎯 Key Files Modified

```
✓ render.yaml              (Build config)
✓ backend/requirements.txt (Dependencies)
✓ backend/app/core/config.py (Path resolution)
✓ backend/init_deployment.sh (New - init script)
```

---

## 📍 Your Deployment URL

```
https://YOUR_SERVICE_NAME.onrender.com/
```

Replace `YOUR_SERVICE_NAME` with your chosen service name
(default: `pitch-eval-backend`)

---

## 🎬 Next Steps

1. **Read** DEPLOY_CHECKLIST.md
2. **Add sample videos** to backend/outputs/batch_input/
3. **Commit & push** to GitHub
4. **Go to** render.com/dashboard
5. **Deploy** following the checklist
6. **Test** endpoints
7. **Monitor** logs

---

## 💡 Pro Tips

✓ Use **heuristic mode** on free plan (fast, works well)
✓ Start with **1-2 sample videos** to test
✓ Monitor **Render logs** to see processing steps
✓ Use **Starter plan** if processing times are too long
✓ Keep **sample videos short** (30-60 sec) for testing

---

## 🆘 Get Help

- **Render Logs**: Dashboard → Service → Logs tab
- **API Docs**: https://your-service.onrender.com/docs
- **Local Testing**: `cd backend && uvicorn app.main:app`
- **Config**: backend/app/core/config.py

---

*Setup completed! Follow DEPLOY_CHECKLIST.md to deploy →*

