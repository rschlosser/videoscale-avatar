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


def _debug_handler(input_data):
    """Run diagnostics on model pipeline — returns dict with results."""
    import traceback
    import cv2
    import numpy as np

    results = {"models_loaded": engine.models_loaded}
    pipeline = engine.lp_pipeline

    # List loaded models
    results["model_dict_keys"] = list(pipeline.model_dict.keys())
    for key, model in pipeline.model_dict.items():
        results[f"model_{key}_type"] = type(model).__name__

    # Test with provided image if available
    image_b64 = input_data.get("image_base64")
    if image_b64:
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "test.jpg")
            Path(img_path).write_bytes(base64.b64decode(image_b64))

            # Test face detection
            img_bgr = cv2.imread(img_path)
            results["image_shape"] = list(img_bgr.shape) if img_bgr is not None else None

            t0 = time.time()
            faces = pipeline.model_dict["face_analysis"].predict(img_bgr)
            results["face_detection_time"] = round(time.time() - t0, 2)
            results["faces_detected"] = len(faces)
            if len(faces) > 0:
                results["face_0_shape"] = list(faces[0].shape)

            # Test crop_image
            if len(faces) > 0:
                from src.utils.crop import crop_image
                try:
                    t0 = time.time()
                    ret_dct = crop_image(
                        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                        faces[0],
                        dsize=pipeline.cfg.crop_params.src_dsize,
                        scale=pipeline.cfg.crop_params.src_scale,
                        vx_ratio=pipeline.cfg.crop_params.src_vx_ratio,
                        vy_ratio=pipeline.cfg.crop_params.src_vy_ratio,
                    )
                    results["crop_image_time"] = round(time.time() - t0, 2)
                    results["crop_image_keys"] = list(ret_dct.keys())
                    results["crop_img_shape"] = list(ret_dct["img_crop"].shape)
                except Exception as e:
                    results["crop_image_error"] = f"{e}\n{traceback.format_exc()}"

            # Test landmark
            if len(faces) > 0 and "crop_image_error" not in results:
                try:
                    t0 = time.time()
                    lmk = pipeline.model_dict["landmark"].predict(
                        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), faces[0]
                    )
                    results["landmark_time"] = round(time.time() - t0, 2)
                    results["landmark_shape"] = list(lmk.shape) if hasattr(lmk, 'shape') else str(type(lmk))
                except Exception as e:
                    results["landmark_error"] = f"{e}\n{traceback.format_exc()}"

            # Test motion_extractor with 256x256 crop
            if "crop_image_error" not in results and len(faces) > 0:
                try:
                    img_crop_256 = cv2.resize(ret_dct["img_crop"], (256, 256))
                    t0 = time.time()
                    pitch, yaw, roll, t, exp, scale, kp = pipeline.model_dict["motion_extractor"].predict(img_crop_256)
                    results["motion_extractor_time"] = round(time.time() - t0, 2)
                    results["motion_extractor_ok"] = True
                except Exception as e:
                    results["motion_extractor_error"] = f"{e}\n{traceback.format_exc()}"

            # Test app_feat_extractor
            if "crop_image_error" not in results and len(faces) > 0:
                try:
                    img_crop_256 = cv2.resize(ret_dct["img_crop"], (256, 256))
                    t0 = time.time()
                    f_s = pipeline.model_dict["app_feat_extractor"].predict(img_crop_256)
                    results["app_feat_extractor_time"] = round(time.time() - t0, 2)
                    results["app_feat_extractor_ok"] = True
                except Exception as e:
                    results["app_feat_extractor_error"] = f"{e}\n{traceback.format_exc()}"

            # Now test full prepare_source
            try:
                t0 = time.time()
                ret = pipeline.prepare_source(img_path, realtime=False)
                results["prepare_source_returned"] = ret
                results["prepare_source_time"] = round(time.time() - t0, 2)
                results["src_imgs_count"] = len(pipeline.src_imgs)
                results["src_infos_count"] = len(pipeline.src_infos)
            except Exception as e:
                results["prepare_source_error"] = f"{e}\n{traceback.format_exc()}"

    return results


def handler(job):
    """RunPod serverless handler.

    Input (job["input"]):
        image_base64: str   — base64-encoded portrait image
        audio_base64: str   — base64-encoded audio file
        resolution: str     — "480p" or "720p" (default: "480p")
        debug: bool         — if true, run diagnostics instead of generation

    Output:
        video_base64: str   — base64-encoded MP4
        generation_time: float
    """
    input_data = job["input"]

    # Debug mode: run diagnostics
    if input_data.get("debug"):
        try:
            return _debug_handler(input_data)
        except Exception as e:
            import traceback
            return {"debug_error": str(e), "traceback": traceback.format_exc()}

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
