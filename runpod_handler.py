"""
RunPod Serverless Handler for avatar video generation.
"""

import base64
import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

logger.info("Starting handler module...")
t_module_start = time.time()

# Log system info for debugging
try:
    import subprocess
    gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10)
    logger.info("GPU: %s", gpu_info.stdout.strip() if gpu_info.returncode == 0 else f"nvidia-smi failed: {gpu_info.stderr}")
except Exception as e:
    logger.info("GPU info unavailable: %s", e)

try:
    import torch
    logger.info("PyTorch %s, CUDA available: %s, CUDA version: %s",
                torch.__version__, torch.cuda.is_available(),
                torch.version.cuda if torch.cuda.is_available() else "N/A")
    if torch.cuda.is_available():
        logger.info("GPU device: %s, memory: %.1f GB",
                     torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_mem / 1e9)
except Exception as e:
    logger.info("PyTorch info unavailable: %s", e)

logger.info("Disk usage: %s", subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout.strip())
logger.info("Python: %s", sys.version)
logger.info("CWD: %s", os.getcwd())
logger.info("PYTHONPATH: %s", os.environ.get("PYTHONPATH", "not set"))

import runpod

logger.info("runpod %s imported (%.1fs)", getattr(runpod, '__version__', '?'), time.time() - t_module_start)

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
        from app.engine import AvatarEngine
        logger.info("Loading models...")
        t0 = time.time()
        engine = AvatarEngine()
        engine.load_models()
        logger.info("Models loaded in %.1fs", time.time() - t0)
    except Exception as e:
        load_error = f"{e}\n{traceback.format_exc()}"
        logger.error("Model loading failed: %s", e, exc_info=True)
        raise


def handler(job):
    """RunPod serverless handler."""
    input_data = job["input"]
    logger.info("Job received: keys=%s", list(input_data.keys()))

    # Ping: instant return
    if input_data.get("ping"):
        return {
            "pong": True,
            "models_loaded": engine.models_loaded if engine else False,
            "uptime": round(time.time() - t_module_start, 1),
            "runpod_version": getattr(runpod, '__version__', '?'),
        }

    # Debug: test model loading
    if input_data.get("debug"):
        results = {"uptime": round(time.time() - t_module_start, 1)}
        try:
            _ensure_models()
            results["models_loaded"] = True
            results["load_error"] = None
        except Exception as e:
            results["models_loaded"] = False
            results["load_error"] = str(load_error)
            return results

        # If image provided, test face detection
        import cv2
        pipeline = engine.lp_pipeline
        results["model_keys"] = list(pipeline.model_dict.keys())

        image_b64 = input_data.get("image_base64")
        if image_b64:
            with tempfile.TemporaryDirectory() as td:
                img_path = os.path.join(td, "test.jpg")
                Path(img_path).write_bytes(base64.b64decode(image_b64))
                img_bgr = cv2.imread(img_path)
                results["image_shape"] = list(img_bgr.shape) if img_bgr is not None else None

                t0 = time.time()
                faces = pipeline.model_dict["face_analysis"].predict(img_bgr)
                results["face_detection"] = {
                    "faces": len(faces),
                    "time": round(time.time() - t0, 2),
                }

                # Full prepare_source test
                try:
                    t0 = time.time()
                    ret = pipeline.prepare_source(img_path, realtime=False)
                    results["prepare_source"] = {
                        "returned": ret,
                        "time": round(time.time() - t0, 2),
                        "src_imgs": len(pipeline.src_imgs),
                        "src_infos": len(pipeline.src_infos),
                    }
                except Exception as e:
                    results["prepare_source_error"] = f"{e}\n{traceback.format_exc()}"

        return results

    # Normal: generate video
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


logger.info("Registering handler (%.1fs since module load)...", time.time() - t_module_start)
try:
    runpod.serverless.start({"handler": handler})
except Exception as e:
    logger.critical("runpod.serverless.start() CRASHED: %s", e, exc_info=True)
    sys.exit(1)
