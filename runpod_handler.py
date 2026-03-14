"""
RunPod Serverless Handler for avatar video generation.
"""

import base64
import logging
import os
import ssl
import sys
import tempfile
import time
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

logger.info("Starting handler module...")
t_module_start = time.time()

# Fix: base image may have stale SSL certificates that prevent aiohttp
# from posting results back to RunPod's webhook endpoint.
# Use certifi's CA bundle if available, and patch the SDK session factory.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    logger.info("Set SSL_CERT_FILE=%s", certifi.where())
except ImportError:
    logger.info("certifi not available")

import runpod

# Patch the AsyncClientSession factory to disable SSL verification
# This ensures result delivery works even with stale system CA certs
import runpod.http_client as _rp_http
_orig_async_session = _rp_http.AsyncClientSession

def _patched_async_session(*args, **kwargs):
    import aiohttp
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0, ssl=False),
        headers=_rp_http.get_auth_header(),
        timeout=aiohttp.ClientTimeout(600, ceil_threshold=400),
        *args, **kwargs,
    )

_rp_http.AsyncClientSession = _patched_async_session
try:
    import runpod.serverless.modules.rp_scale as _rp_scale
    _rp_scale.AsyncClientSession = _patched_async_session
except Exception:
    pass
logger.info("Patched AsyncClientSession to disable SSL verification")

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
runpod.serverless.start({"handler": handler})
