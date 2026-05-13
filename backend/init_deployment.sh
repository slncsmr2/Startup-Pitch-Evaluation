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
    echo "⚠️  Warning: Model checkpoint not found at models/checkpoints/phase6_gpu_nn_model.pt"
    echo "   The heuristic pipeline will still work, but neural network scoring will be disabled"
fi

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating default .env file"
    cat > .env << 'EOF'
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
EOF
fi

echo "✅ Deployment initialization complete"
echo "📁 Directories created:"
echo "   - outputs/batch_input (for input videos)"
echo "   - outputs/audio_chunks (for extracted audio)"
echo "   - outputs/frames (for extracted video frames)"
echo "   - outputs/uploads (for user uploads)"
