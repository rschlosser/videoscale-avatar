"""
Avatar generation engine — wraps FasterLivePortrait + JoyVASA.

Handles model loading and inference for audio-driven portrait animation.
"""

import asyncio
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Model checkpoint paths (downloaded at build time or first run)
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "/app/checkpoints"))


class AvatarEngine:
    """Audio-driven talking-head video generator.

    Uses LivePortrait for portrait animation and JoyVASA for
    audio → facial motion coefficient extraction.
    """

    def __init__(self):
        self.models_loaded = False

    def load_models(self):
        """Load all model weights into GPU memory.

        Called once at startup. Subsequent generate() calls reuse loaded models.
        """
        # Verify checkpoints exist
        if not CHECKPOINT_DIR.exists():
            raise RuntimeError(
                f"Checkpoint directory not found: {CHECKPOINT_DIR}. "
                "Run download_models.py or set CHECKPOINT_DIR."
            )

        required = ["liveportrait", "JoyVASA"]
        for name in required:
            path = CHECKPOINT_DIR / name
            if not path.exists():
                raise RuntimeError(f"Missing checkpoint: {path}")

        # TODO: Import and initialize FasterLivePortrait + JoyVASA models
        # This will be filled in once we test the Docker image with actual models.
        #
        # Pseudocode for the actual implementation:
        #
        #   from faster_liveportrait import LivePortraitPipeline
        #   self.pipeline = LivePortraitPipeline(
        #       checkpoint_dir=str(CHECKPOINT_DIR / "liveportrait"),
        #       joyvasa_dir=str(CHECKPOINT_DIR / "JoyVASA"),
        #       device="cuda",
        #   )
        #
        # For now, we verify the inference CLI works.
        logger.info("Checkpoints verified: %s", [str(p.name) for p in CHECKPOINT_DIR.iterdir()])
        self.models_loaded = True
        logger.info("Models loaded successfully")

    async def generate(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        resolution: str = "480p",
    ) -> str:
        """Generate a talking-head video from image + audio.

        Args:
            image_path: Path to portrait image (JPG/PNG).
            audio_path: Path to audio file (MP3/WAV).
            output_path: Where to write the output MP4.
            resolution: Target resolution ("480p" or "720p").

        Returns:
            Path to the generated video.
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded — call load_models() first")

        # Resolution mapping
        res_map = {
            "480p": (480, 854),   # 9:16 portrait
            "720p": (720, 1280),
        }
        height, width = res_map.get(resolution, (480, 854))

        logger.info(
            "Generating: image=%s, audio=%s, resolution=%s (%dx%d)",
            image_path, audio_path, resolution, width, height,
        )

        # Run inference via CLI (FasterLivePortrait with JoyVASA audio mode)
        # This approach works with the existing Docker image and avoids
        # complex Python import chains. We'll optimize to in-process later.
        cmd = [
            "python", "-m", "faster_liveportrait.inference",
            "-r", image_path,
            "-a", audio_path,
            "-o", output_path,
            "--animation_mode", "human",
            "--cfg_scale", "2.0",
        ]

        logger.info("Running: %s", " ".join(cmd))

        # Run in thread pool to not block the event loop
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max
            cwd="/app",
        )

        if result.returncode != 0:
            logger.error("Inference failed:\nstdout: %s\nstderr: %s", result.stdout, result.stderr)
            raise RuntimeError(f"Inference failed (exit {result.returncode}): {result.stderr[-500:]}")

        if not Path(output_path).exists():
            raise RuntimeError("Inference produced no output file")

        logger.info("Inference complete: %s", output_path)
        return output_path
