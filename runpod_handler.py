"""
RunPod Serverless Handler for avatar video generation.

This is the entry point when deployed on RunPod Serverless.
Accepts base64-encoded image + audio, returns base64-encoded MP4.

Deploy: build Docker image → push to registry → create RunPod endpoint.
"""

import base64
import logging
import os
import tempfile
import time
from pathlib import Path

import runpod

from app.engine import AvatarEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models once at cold start
engine = AvatarEngine()
try:
    engine.load_models()
    logger.info("RunPod handler ready")
except Exception as e:
    logger.error("Failed to load models: %s", e, exc_info=True)
    raise


def handler(job):
    """RunPod serverless handler.

    Input (job["input"]):
        image_base64: str   — base64-encoded portrait image
        audio_base64: str   — base64-encoded audio file
        resolution: str     — "480p" or "720p" (default: "480p")

    Output:
        video_base64: str   — base64-encoded MP4
        generation_time: float
    """
    input_data = job["input"]

    image_b64 = input_data.get("image_base64")
    audio_b64 = input_data.get("audio_base64")
    resolution = input_data.get("resolution", "480p")

    if not image_b64 or not audio_b64:
        return {"error": "image_base64 and audio_base64 are required"}

    if resolution not in ("480p", "720p"):
        return {"error": f"Invalid resolution: {resolution}"}

    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="avatar_") as tmpdir:
        tmpdir = Path(tmpdir)

        img_path = tmpdir / "input.jpg"
        audio_path = tmpdir / "input.mp3"
        output_path = tmpdir / "output.mp4"

        img_path.write_bytes(base64.b64decode(image_b64))
        audio_path.write_bytes(base64.b64decode(audio_b64))

        logger.info(
            "Generating: image=%d bytes, audio=%d bytes, resolution=%s",
            img_path.stat().st_size,
            audio_path.stat().st_size,
            resolution,
        )

        # Call sync method directly (avoid asyncio.run conflicts with RunPod's event loop)
        try:
            engine._generate_sync(
                image_path=str(img_path),
                audio_path=str(audio_path),
                output_path=str(output_path),
                resolution=resolution,
            )
        except Exception as e:
            logger.error("Generation failed: %s", e, exc_info=True)
            return {"error": str(e)}

        if not output_path.exists():
            return {"error": "Generation failed — no output produced"}

        video_bytes = output_path.read_bytes()

    elapsed = time.time() - t0
    logger.info("Generated: %.1fs, %d bytes", elapsed, len(video_bytes))

    return {
        "video_base64": base64.b64encode(video_bytes).decode("ascii"),
        "generation_time": round(elapsed, 2),
    }


runpod.serverless.start({"handler": handler})
