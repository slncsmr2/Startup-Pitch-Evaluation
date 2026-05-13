# Deployment Setup - Summary

## What Was Created

I've set up your project for deployment to Render.com with full data processing capabilities. Here's what was done:

### 1. **Configuration Files**
- ✅ Updated `render.yaml` - Render.com deployment config
- ✅ Updated `backend/requirements.txt` - Added media processing dependencies
- ✅ Updated `backend/app/core/config.py` - Added path resolution for deployment
- ✅ Created `backend/init_deployment.sh` - Initialize directories on deployment

### 2. **Documentation Files**
- ✅ **DEPLOYMENT_GUIDE.md** - Complete 8-part guide for deployment
- ✅ **DEPLOY_CHECKLIST.md** - Step-by-step checklist to follow
- ✅ **DATA_PROCESSING_EXPLAINED.md** - Technical deep-dive on data pipeline

---

## Quick Start (3 Steps)

### Step 1: Prepare Your Repository
```bash
cd ~/OneDrive/Documents/GitHub/Startup-Pitch-Evaluation

# 1. Add sample videos to batch_input (important!)
# Put 1-2 sample pitch videos (30-60 sec each) in:
# backend/outputs/batch_input/

# 2. Commit all changes
git add .
git commit -m "Add deployment configuration"
git push origin main
```

### Step 2: Deploy to Render
1. Go to https://render.com/dashboard
2. Click **New +** → **Web Service**
3. Select your repository
4. Configure:
   - **Runtime**: Python
   - **Root Directory**: backend
   - **Build Command**: (will auto-read from render.yaml)
   - **Start Command**: (will auto-read from render.yaml)
   - **Plan**: Free (recommended to start)
5. Add Environment Variables (from DEPLOY_CHECKLIST.md)
6. Click **Create Web Service**
7. Wait 5-10 minutes for build

### Step 3: Test Your Deployment
```bash
# Wait for build to complete, then:

# Test health check
curl https://YOUR_SERVICE_NAME.onrender.com/health

# Test listing videos
curl https://YOUR_SERVICE_NAME.onrender.com/videos

# Open in browser
https://YOUR_SERVICE_NAME.onrender.com/
```

---

## How Data Processing Works (Simple Version)

When you upload or evaluate a pitch video:

```
Video Input
    ↓
[Split into 5-second chunks]
    ↓
[Extract audio & video frames]
    ↓
[Analyze text, visual, and audio]
    ↓
[Combine insights from all 3 modalities]
    ↓
[Generate Risk Score, Pitch Score, Strength Score]
    ↓
Output: Evaluation Report + Recommendations
```

**Locally**: Uses GPU, fast (2-5 minutes for 60-sec video)
**On Render**: Uses CPU, slower (5-30 minutes, use Starter plan for better speed)

---

## Files Modified

| File | Changes |
|------|---------|
| `render.yaml` | Added build steps, system packages, environment variables |
| `backend/requirements.txt` | Added: opencv, faster-whisper, librosa, imageio-ffmpeg |
| `backend/app/core/config.py` | Added path resolution properties for deployment |
| `backend/init_deployment.sh` | New file - initializes directories on deploy |

---

## Files Created

| File | Purpose |
|------|---------|
| **DEPLOYMENT_GUIDE.md** | Comprehensive 8-part deployment guide (READ THIS FIRST) |
| **DEPLOY_CHECKLIST.md** | Checkboxes for pre-deployment, GitHub, Render, testing |
| **DATA_PROCESSING_EXPLAINED.md** | Technical details on 5-stage data pipeline |
| **This file** | Quick reference summary |

---

## Important: What You Need to Do

### ⚠️ BEFORE DEPLOYING

1. **Add sample videos** to `backend/outputs/batch_input/`
   - Add 1-2 sample pitch videos (30-60 seconds each)
   - Example filenames: `sample_1.mp4`, `sample_2.mp4`
   - This is important so deployment can show it works!

2. **Commit everything to Git**
   ```bash
   git add .
   git commit -m "Add deployment configuration"
   git push origin main
   ```

3. **Create Render account** at https://render.com (if not already done)
   - Sign up with GitHub

### 🚀 DURING DEPLOYMENT

Follow the **DEPLOY_CHECKLIST.md** step by step

### ✅ AFTER DEPLOYMENT

Test the endpoints and monitor the logs

---

## What Gets Deployed?

Your Render.com instance will have:

- ✅ FastAPI backend server
- ✅ Web UI (at root path `/`)
- ✅ Video evaluation API (`/evaluate/upload`)
- ✅ Batch evaluation API (`/evaluate/batch`)
- ✅ Video listing (`/videos`)
- ✅ Full data processing pipeline
- ✅ Heuristic-mode scoring (fast, no GPU needed)

---

## Troubleshooting

### Build fails?
→ Check Render logs in dashboard → Events tab

### Video processing takes too long?
→ Use Free plan for testing
→ Upgrade to Starter plan for faster CPU
→ Disable `SPE_ENABLE_VIDEO_EXTRACTION` to speed up

### Out of memory?
→ Set `SPE_USE_HEURISTIC_PIPELINE=true` (already done)
→ Use `faster_whisper_model_size=tiny`

### Model checkpoint warnings?
→ This is OK! Heuristic mode works without it
→ Neural features are optional

---

## Next Reading

1. **For deployment steps**: Read `DEPLOY_CHECKLIST.md`
2. **For detailed guide**: Read `DEPLOYMENT_GUIDE.md`
3. **To understand data flow**: Read `DATA_PROCESSING_EXPLAINED.md`

---

## Your Deployed URL

Once deployed, your app will be available at:
```
https://pitch-eval-backend.onrender.com/
```
(Or whatever service name you chose)

Visit this URL in your browser to see the web interface!

---

## Summary

✅ Deployment config ready
✅ Data processing pipeline ready
✅ Documentation complete
⏳ Next: Follow DEPLOY_CHECKLIST.md to deploy!

