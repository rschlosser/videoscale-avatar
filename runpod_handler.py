"""
RunPod Serverless Handler for avatar video generation.

Minimal startup to ensure handler registers quickly with RunPod.
Heavy imports (torch, cv2, etc.) are deferred to first job.
"""

import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

print("=== Handler module starting ===", flush=True)
t_module_start = time.time()

print(f"Python: {sys.version}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)

import runpod

# SSL is now handled via env vars (SSL_CERT_FILE, REQUESTS_CA_BUNDLE) set in Dockerfile.
# Verify they're set at startup for debugging.
print(f"SSL_CERT_FILE={os.environ.get('SSL_CERT_FILE', 'NOT SET')}", flush=True)
print(f"REQUESTS_CA_BUNDLE={os.environ.get('REQUESTS_CA_BUNDLE', 'NOT SET')}", flush=True)

# Print OpenSSL version for diagnostics
try:
    import ssl
    print(f"OpenSSL: {ssl.OPENSSL_VERSION}", flush=True)
    print(f"Verify paths: {ssl.get_default_verify_paths()}", flush=True)
except Exception as e:
    print(f"SSL info error: {e}", flush=True)

print(f"runpod {getattr(runpod, '__version__', '?')} imported ({time.time() - t_module_start:.1f}s)", flush=True)

# Diagnostic: print all RUNPOD_ env vars (webhook URLs, worker config)
print("=== RunPod env vars ===", flush=True)
for k, v in sorted(os.environ.items()):
    if k.startswith("RUNPOD"):
        # Mask API keys
        if "KEY" in k or "SECRET" in k:
            print(f"  {k}={v[:8]}...", flush=True)
        else:
            print(f"  {k}={v}", flush=True)
print("=== End RunPod env vars ===", flush=True)

# Diagnostic: test HTTPS connectivity at startup
try:
    import urllib.request
    t0 = time.time()
    resp = urllib.request.urlopen("https://api.runpod.ai/", timeout=10)
    print(f"HTTPS test (urllib): {resp.status} in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"HTTPS test (urllib) FAILED: {e}", flush=True)

try:
    import requests as req_lib
    t0 = time.time()
    resp = req_lib.get("https://api.runpod.ai/", timeout=10)
    print(f"HTTPS test (requests): {resp.status_code} in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"HTTPS test (requests) FAILED: {e}", flush=True)

try:
    import asyncio
    import aiohttp
    async def _test_aiohttp():
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.runpod.ai/", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                return resp.status
    t0 = time.time()
    status = asyncio.get_event_loop().run_until_complete(_test_aiohttp())
    print(f"HTTPS test (aiohttp): {status} in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"HTTPS test (aiohttp) FAILED: {e}", flush=True)

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
        import traceback as tb
        load_error = f"{e}\n{tb.format_exc()}"
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
        import base64
        import tempfile
        from pathlib import Path
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
                    import traceback as tb
                    results["prepare_source_error"] = f"{e}\n{tb.format_exc()}"

        return results

    # Normal: generate video
    import base64
    import tempfile
    from pathlib import Path

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


print(f"Registering handler ({time.time() - t_module_start:.1f}s since module load)...", flush=True)
try:
    runpod.serverless.start({"handler": handler})
except Exception as e:
    print(f"CRITICAL: runpod.serverless.start() CRASHED: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
