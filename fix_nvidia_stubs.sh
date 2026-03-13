#!/bin/bash
# Fix NVIDIA stub libraries in the base image that shadow the real driver.
#
# The shaoguo/faster_liveportrait:v3 image ships empty/stub .so files from the
# build-time CUDA 11.7 toolkit in /lib/x86_64-linux-gnu/. At runtime on RunPod,
# the real driver libraries (e.g. libcuda.so.550.127.05) are bind-mounted into
# the same directory, but the stale .so.1 symlinks and old versioned stubs
# (e.g. .515.105.01) take precedence and cause "file too short" or CUDA init
# failures.
#
# This script detects the real driver version and fixes the symlinks.

LIB_DIR="/lib/x86_64-linux-gnu"

# Detect the real driver version from the bind-mounted libcuda
DRIVER_VER=$(ls "$LIB_DIR"/libcuda.so.*.*.* 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1)

if [ -z "$DRIVER_VER" ]; then
    echo "[fix_nvidia_stubs] No NVIDIA driver detected, skipping stub fix"
    exit 0
fi

echo "[fix_nvidia_stubs] Detected NVIDIA driver $DRIVER_VER"

# Libraries that need their .so.1 symlinks fixed
LIBS=(
    "libcuda"
    "libnvidia-ml"
    "libnvidia-ptxjitcompiler"
    "libnvidia-nvvm"
    "libcudadebugger"
)

for lib in "${LIBS[@]}"; do
    real="$LIB_DIR/${lib}.so.${DRIVER_VER}"
    if [ -f "$real" ]; then
        # Remove stale stubs (empty files / old versions) — skip if bind-mounted
        for f in "$LIB_DIR"/${lib}.so*; do
            [ "$f" = "$real" ] && continue
            rm -f "$f" 2>/dev/null || true
        done
        # Create proper symlinks
        ln -sf "${lib}.so.${DRIVER_VER}" "$LIB_DIR/${lib}.so.1"
        ln -sf "${lib}.so.1" "$LIB_DIR/${lib}.so"
        echo "[fix_nvidia_stubs]   Fixed $lib -> $DRIVER_VER"
    fi
done

ldconfig 2>/dev/null || true
echo "[fix_nvidia_stubs] Done"
