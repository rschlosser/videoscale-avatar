#!/bin/bash
# Startup script for Hallo3 RunPod Serverless worker.
#
# 1. Check if models exist on network volume (/runpod-volume/hallo3_models)
# 2. If not, download them (~52GB, takes ~20-30 min first time)
# 3. Symlink models into the expected location
# 4. Launch the RunPod handler

set -e

MODEL_DIR="/runpod-volume/hallo3_models"
TARGET_DIR="/app/hallo3/pretrained_models"
MARKER="$MODEL_DIR/.download_complete"

echo "=== Hallo3 Startup ==="
echo "Checking for models at $MODEL_DIR..."

if [ -f "$MARKER" ]; then
    echo "Models found on network volume!"
else
    echo "Models NOT found — downloading (~52GB, this may take 20-30 minutes)..."
    echo "This only happens once per network volume."

    # If /runpod-volume doesn't exist, fall back to local storage
    if [ ! -d "/runpod-volume" ]; then
        echo "WARNING: No network volume mounted at /runpod-volume!"
        echo "Models will be downloaded to local storage (lost on restart)."
        MODEL_DIR="/app/hallo3/pretrained_models"
        MARKER="$MODEL_DIR/.download_complete"
    fi

    mkdir -p "$MODEL_DIR"

    # Set download target for download_models.py
    export PRETRAINED_DIR="$MODEL_DIR"
    python -u download_models.py

    # Mark download complete
    touch "$MARKER"
    echo "Download complete!"
fi

# Symlink network volume models to where Hallo3 expects them
if [ "$MODEL_DIR" != "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"
    ln -sf "$MODEL_DIR" "$TARGET_DIR"
    echo "Symlinked $MODEL_DIR -> $TARGET_DIR"
fi

echo "Starting RunPod handler..."
exec python -u runpod_handler.py
