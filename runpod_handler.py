"""
RunPod Serverless Handler for avatar video generation.
"""

import logging
import sys
import time

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

logger.info("Handler module loading...")
t_start = time.time()

import runpod

logger.info("runpod imported in %.1fs", time.time() - t_start)

# Lazy model loading
engine = None
load_error = None


def _ensure_models():
    global engine, load_error
    if engine is not None and engine.models_loaded:
        return
    if load_error is not None:
        raise RuntimeError(f"Model loading failed previously: {load_error}")
    try:
        import os
        import traceback
        from app.engine import AvatarEngine
        logger.info("Loading models...")
        t0 = time.time()
        engine = AvatarEngine()
        engine.load_models()
        logger.info("Models loaded in %.1fs", time.time() - t0)
    except Exception as e:
        load_error = f"{e}\n{traceback.format_exc()}"
        logger.error("Failed to load models: %s", e, exc_info=True)
        raise


def handler(job):
    """RunPod serverless handler."""
    import base64
    import os
    import tempfile
    import traceback
    from pathlib import Path

    input_data = job["input"]
    logger.info("Job received: keys=%s", list(input_data.keys()))

    # Ping mode: instant return (no model loading)
    if input_data.get("ping"):
        return {
            "pong": True,
            "models_loaded": engine.models_loaded if engine else False,
            "python": sys.version,
            "uptime": round(time.time() - t_start, 1),
        }

    # Debug mode: test model loading + individual model steps
    if input_data.get("debug"):
        results = {}
        try:
            _ensure_models()
            results["models_loaded"] = True
            results["load_error"] = None
        except Exception as e:
            results["models_loaded"] = False
            results["load_error"] = str(load_error)
            return results

        import cv2

        pipeline = engine.lp_pipeline
        results["model_dict_keys"] = list(pipeline.model_dict.keys())

        image_b64 = input_data.get("image_base64")
        if image_b64:
            with tempfile.TemporaryDirectory() as td:
                img_path = os.path.join(td, "test.jpg")
                Path(img_path).write_bytes(base64.b64decode(image_b64))
                img_bgr = cv2.imread(img_path)
                results["image_shape"] = list(img_bgr.shape) if img_bgr is not None else None

                # Face detection
                t0 = time.time()
                faces = pipeline.model_dict["face_analysis"].predict(img_bgr)
                results["face_detection_time"] = round(time.time() - t0, 2)
                results["faces_detected"] = len(faces)

                if len(faces) > 0:
                    from src.utils.crop import crop_image
                    # Crop
                    try:
                        ret_dct = crop_image(
                            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), faces[0],
                            dsize=pipeline.cfg.crop_params.src_dsize,
                            scale=pipeline.cfg.crop_params.src_scale,
                            vx_ratio=pipeline.cfg.crop_params.src_vx_ratio,
                            vy_ratio=pipeline.cfg.crop_params.src_vy_ratio,
                        )
                        results["crop_ok"] = True
                    except Exception as e:
                        results["crop_error"] = str(e)

                    # Landmark
                    try:
                        t0 = time.time()
                        lmk = pipeline.model_dict["landmark"].predict(
                            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), faces[0])
                        results["landmark_time"] = round(time.time() - t0, 2)
                        results["landmark_ok"] = True
                    except Exception as e:
                        results["landmark_error"] = f"{e}\n{traceback.format_exc()}"

                    # Motion extractor
                    if "crop_error" not in results:
                        try:
                            img_crop_256 = cv2.resize(ret_dct["img_crop"], (256, 256))
                            t0 = time.time()
                            pipeline.model_dict["motion_extractor"].predict(img_crop_256)
                            results["motion_extractor_time"] = round(time.time() - t0, 2)
                            results["motion_extractor_ok"] = True
                        except Exception as e:
                            results["motion_extractor_error"] = f"{e}\n{traceback.format_exc()}"

                    # Full prepare_source
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

    # Normal mode: generate video
    try:
        _ensure_models()
    except Exception as e:
        return {"error": f"Model loading failed: {e}"}

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

        logger.info("Generating: image=%d bytes, audio=%d bytes, res=%s",
                     img_path.stat().st_size, audio_path.stat().st_size, resolution)
        try:
            engine._generate_sync(
                image_path=str(img_path), audio_path=str(audio_path),
                output_path=str(output_path), resolution=resolution)
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


logger.info("Starting RunPod handler (%.1fs since module load)...", time.time() - t_start)
runpod.serverless.start({"handler": handler})
