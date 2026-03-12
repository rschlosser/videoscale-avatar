"""
FastAPI server for avatar video generation.

Accepts an image + audio file, returns an MP4 talking-head video.
Wraps FasterLivePortrait + JoyVASA for audio-driven portrait animation.

Can run standalone (uvicorn) or as a RunPod serverless handler.
"""

import io
import logging
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response

from app.engine import AvatarEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VideoScale Avatar", version="0.1.0")

# Singleton engine — loads models once on startup
engine: AvatarEngine | None = None


@app.on_event("startup")
async def startup():
    global engine
    logger.info("Loading avatar engine...")
    engine = AvatarEngine()
    engine.load_models()
    logger.info("Avatar engine ready")


@app.get("/health")
async def health():
    return {"status": "ok", "engine_loaded": engine is not None}


@app.post("/generate")
async def generate(
    image: UploadFile = File(..., description="Avatar portrait image (JPG/PNG)"),
    audio: UploadFile = File(..., description="Audio file (MP3/WAV)"),
    resolution: str = Form("480p", description="Output resolution: 480p or 720p"),
):
    """Generate a talking-head video from image + audio.

    Returns the MP4 video bytes directly.
    """
    if engine is None:
        raise HTTPException(503, "Engine not loaded yet")

    if resolution not in ("480p", "720p"):
        raise HTTPException(400, f"Invalid resolution: {resolution}")

    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="avatar_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Save uploads to disk
        img_ext = Path(image.filename or "avatar.jpg").suffix or ".jpg"
        audio_ext = Path(audio.filename or "audio.mp3").suffix or ".mp3"

        img_path = tmpdir / f"input{img_ext}"
        audio_path = tmpdir / f"input{audio_ext}"
        output_path = tmpdir / "output.mp4"

        img_path.write_bytes(await image.read())
        audio_path.write_bytes(await audio.read())

        logger.info(
            "Generating avatar video: image=%s (%d bytes), audio=%s (%d bytes), resolution=%s",
            image.filename, img_path.stat().st_size,
            audio.filename, audio_path.stat().st_size,
            resolution,
        )

        # Generate
        await engine.generate(
            image_path=str(img_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            resolution=resolution,
        )

        if not output_path.exists():
            raise HTTPException(500, "Generation failed — no output produced")

        video_bytes = output_path.read_bytes()

    elapsed = time.time() - t0
    logger.info("Generated avatar video: %.1fs, %d bytes", elapsed, len(video_bytes))

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={
            "X-Generation-Time": f"{elapsed:.2f}",
            "Content-Disposition": 'attachment; filename="avatar.mp4"',
        },
    )
