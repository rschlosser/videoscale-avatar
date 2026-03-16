"""
Avatar generation engine — wraps FasterLivePortrait + JoyVASA.

Uses the in-process Python API (not CLI) for audio-driven portrait animation.
Two-stage pipeline:
  1. JoyVASA: audio → facial motion coefficients (diffusion transformer)
  2. LivePortrait: motion coefficients + source image → video frames
"""

import logging
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "/app/checkpoints"))
FLPROOT = Path(os.getenv("FLPROOT", "/app/FasterLivePortrait"))
CONFIG_PATH = FLPROOT / "configs" / "onnx_infer.yaml"


class AvatarEngine:
    """Audio-driven talking-head video generator.

    Uses FasterLivePortrait for portrait animation and JoyVASA for
    audio → facial motion coefficient extraction.
    """

    def __init__(self):
        self.models_loaded = False
        self.lp_pipeline = None
        self.joyvasa_pipeline = None
        self.infer_cfg = None

    def load_models(self):
        """Load all model weights into GPU memory.

        Called once at startup. Subsequent generate() calls reuse loaded models.
        """
        # Verify checkpoints exist
        required = {
            "liveportrait_onnx": CHECKPOINT_DIR / "liveportrait_onnx",
            "JoyVASA": CHECKPOINT_DIR / "JoyVASA",
            "chinese-hubert-base": CHECKPOINT_DIR / "chinese-hubert-base",
        }
        for name, path in required.items():
            if not path.exists():
                raise RuntimeError(f"Missing checkpoint: {path}")

        # Load FasterLivePortrait config
        self.infer_cfg = OmegaConf.load(str(CONFIG_PATH))
        self.infer_cfg.infer_params.flag_pasteback = True

        # Override checkpoint paths to use our download location
        joyvasa_cfg = self.infer_cfg.joyvasa_models
        joyvasa_cfg.motion_model_path = str(
            CHECKPOINT_DIR / "JoyVASA" / "motion_generator" / "motion_generator_hubert_chinese.pt"
        )
        joyvasa_cfg.audio_model_path = str(CHECKPOINT_DIR / "chinese-hubert-base")
        joyvasa_cfg.motion_template_path = str(
            CHECKPOINT_DIR / "JoyVASA" / "motion_template" / "motion_template.pkl"
        )

        # Initialize JoyVASA audio-to-motion pipeline
        from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline

        self.joyvasa_pipeline = JoyVASAAudio2MotionPipeline(
            motion_model_path=joyvasa_cfg.motion_model_path,
            audio_model_path=joyvasa_cfg.audio_model_path,
            motion_template_path=joyvasa_cfg.motion_template_path,
            cfg_mode=self.infer_cfg.infer_params.get("cfg_mode", "incremental"),
            cfg_scale=self.infer_cfg.infer_params.get("cfg_scale", 4.0),
        )
        logger.info("JoyVASA pipeline loaded")

        # Initialize LivePortrait pipeline
        from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

        self.lp_pipeline = FasterLivePortraitPipeline(cfg=self.infer_cfg, is_animal=False)
        logger.info("LivePortrait pipeline loaded")

        self.models_loaded = True
        logger.info("All models loaded successfully")

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

        import asyncio

        return await asyncio.to_thread(
            self._generate_sync, image_path, audio_path, output_path, resolution
        )

    def _generate_sync(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        resolution: str,
    ) -> str:
        """Synchronous generation — runs in a thread."""
        res_map = {
            "480p": (480, 854),
            "720p": (720, 1280),
        }
        target_h, target_w = res_map.get(resolution, (480, 854))

        logger.info(
            "Generating: image=%s, audio=%s, resolution=%s (%dx%d)",
            image_path, audio_path, resolution, target_w, target_h,
        )

        # Step 1: Audio → motion sequence via JoyVASA
        import time as _time
        logger.info("Step 1/3: Extracting motion from audio...")
        _t1 = _time.time()
        tgt_motion = self.joyvasa_pipeline.gen_motion_sequence(audio_path)
        n_frames = tgt_motion["n_frames"]
        fps = tgt_motion["output_fps"]
        logger.info("Motion extracted: %d frames at %d fps (%.1fs)", n_frames, fps, _time.time() - _t1)

        # Step 2: Prepare source image
        logger.info("Step 2/3: Preparing source image...")
        _t2 = _time.time()
        try:
            ret = self.lp_pipeline.prepare_source(image_path, realtime=False)
        except Exception as _e:
            logger.error("prepare_source exception: %s", _e, exc_info=True)
            raise RuntimeError(f"Face detection crashed: {_e}")
        logger.info(
            "prepare_source returned=%s in %.1fs, src_imgs=%d, src_infos=%d",
            ret, _time.time() - _t2,
            len(getattr(self.lp_pipeline, 'src_imgs', [])),
            len(getattr(self.lp_pipeline, 'src_infos', [])),
        )
        if not ret:
            # Log image details for debugging
            import cv2 as _cv2
            _img = _cv2.imread(image_path)
            _img_info = f"shape={_img.shape}, dtype={_img.dtype}" if _img is not None else "imread returned None"
            raise RuntimeError(f"Failed to detect face in source image: {image_path} ({_img_info})")

        # Step 3: Render each frame
        logger.info("Step 3/3: Rendering %d frames...", n_frames)
        frames = []
        for i in range(n_frames):
            motion_info = [
                tgt_motion["motion"][i],
                tgt_motion["c_eyes_lst"][i] if tgt_motion["c_eyes_lst"] else None,
                tgt_motion["c_lip_lst"][i] if tgt_motion["c_lip_lst"] else None,
            ]
            out_crop, out_org = self.lp_pipeline.run_with_pkl(
                motion_info,
                self.lp_pipeline.src_imgs[0],
                self.lp_pipeline.src_infos[0],
                first_frame=(i == 0),
            )
            # out_org is RGB, convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)

            # Resize to target resolution if needed
            h, w = frame_bgr.shape[:2]
            if h != target_h or w != target_w:
                frame_bgr = cv2.resize(frame_bgr, (target_w, target_h))

            frames.append(frame_bgr)

            if (i + 1) % 50 == 0:
                logger.info("Rendered %d/%d frames", i + 1, n_frames)

        # Write video
        logger.info("Writing video: %s", output_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
        for frame in frames:
            writer.write(frame)
        writer.release()

        # Remux with ffmpeg for proper H.264 encoding + audio
        final_path = output_path
        tmp_raw = output_path + ".raw.mp4"
        os.rename(output_path, tmp_raw)

        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_raw,
            "-i", audio_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            final_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        os.unlink(tmp_raw)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

        logger.info("Generation complete: %s (%d frames)", final_path, n_frames)
        return final_path
