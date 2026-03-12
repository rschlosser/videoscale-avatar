# VideoScale Avatar

Self-hosted talking-head avatar generation using LivePortrait + JoyVASA.

Replaces VEED Fabric ($0.08/s) with self-hosted inference (~$0.001/s).

## Architecture

```
Image + Audio → JoyVASA (audio → motion) → LivePortrait (motion → video) → MP4
```

## Deployment Options

### 1. RunPod Serverless (recommended for production)

```bash
docker build --platform linux/amd64 -f Dockerfile.runpod -t videoscale-avatar:runpod .
docker push <registry>/videoscale-avatar:runpod
# Create endpoint in RunPod dashboard → point to image
```

### 2. HTTP Server (Railway, any Docker host)

```bash
docker build -t videoscale-avatar .
docker run --gpus all -p 8000:8000 videoscale-avatar
```

### 3. Local Dev

```bash
pip install -r requirements.txt
python download_models.py
uvicorn app.server:app --reload --port 8000
```

## API

### POST /generate

Multipart form: `image` (file) + `audio` (file) + `resolution` (480p|720p)

Returns: MP4 video bytes

```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@avatar.jpg" \
  -F "audio=@speech.mp3" \
  -F "resolution=480p" \
  -o output.mp4
```

### RunPod Serverless

```python
import runpod, base64

runpod.api_key = "your_key"
endpoint = runpod.Endpoint("your_endpoint_id")

result = endpoint.run_sync({
    "image_base64": base64.b64encode(open("avatar.jpg", "rb").read()).decode(),
    "audio_base64": base64.b64encode(open("speech.mp3", "rb").read()).decode(),
    "resolution": "480p",
})

with open("output.mp4", "wb") as f:
    f.write(base64.b64decode(result["video_base64"]))
```

## Cost

| Provider | 10s video | 30s video |
|----------|-----------|-----------|
| VEED Fabric (fal.ai) | $0.80 | $2.40 |
| Self-hosted (RunPod T4) | $0.003 | $0.008 |
| Self-hosted (RunPod A100) | $0.006 | $0.015 |

## Integration with VideoScale

In the main `videoscale` repo, add a `LivePortraitClient` in `shared/` that calls this service. The existing `avatar_provider` routing in `pipeline.py` already supports switching between providers.
