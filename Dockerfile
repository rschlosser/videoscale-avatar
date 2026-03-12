# VideoScale Avatar — GPU inference server
# Base: FasterLivePortrait with JoyVASA audio-driven mode
#
# Build:  docker build --platform linux/amd64 -t videoscale-avatar .
# Run:    docker run --gpus all -p 8000:8000 videoscale-avatar
# RunPod: docker run --gpus all videoscale-avatar python runpod_handler.py

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    ffmpeg git wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone FasterLivePortrait (includes JoyVASA support)
RUN git clone --depth 1 https://github.com/warmshao/FasterLivePortrait.git /app/FasterLivePortrait \
    && cd /app/FasterLivePortrait && pip install --no-cache-dir -e .

# Download model checkpoints (cached in Docker layer)
COPY download_models.py .
RUN python download_models.py

# App code
COPY app/ app/
COPY runpod_handler.py .

EXPOSE 8000

# Default: HTTP server mode. Override CMD for RunPod handler.
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
