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

print(f"runpod {getattr(runpod, '__version__', '?')} imported ({time.time() - t_module_start:.1f}s)", flush=True)

# --- Print all RunPod env vars for debugging ---
print("=== RunPod environment ===", flush=True)
for key in sorted(os.environ):
    if key.startswith("RUNPOD"):
        val = os.environ[key]
        # Mask API keys
        if "KEY" in key or "SECRET" in key:
            val = val[:8] + "..." if len(val) > 8 else "***"
        print(f"  {key}={val}", flush=True)
print("=== End RunPod env ===", flush=True)

# --- MONKEY-PATCH: Replace aiohttp POST with requests POST for result delivery ---
# The RunPod SDK uses aiohttp to POST job results to webhook URLs. On this base image
# (conda OpenSSL), aiohttp POST requests consistently fail/timeout while requests
# (used for heartbeat) works fine. Patch _transmit() to use requests instead.
try:
    import json as _json
    import requests as req_lib
    import runpod.serverless.modules.rp_http as rp_http

    _original_transmit = rp_http._transmit
    print(f"Original _transmit: {_original_transmit}", flush=True)
    print(f"JOB_DONE_URL from rp_http: {getattr(rp_http, 'JOB_DONE_URL', 'NOT SET')}", flush=True)

    async def _requests_transmit(client_session, url, job_data):
        """Replace aiohttp POST with requests.post for result delivery."""
        print(f"  [_requests_transmit CALLED] url={url}", flush=True)
        print(f"  [_requests_transmit] data_len={len(job_data) if job_data else 0}", flush=True)
        try:
            headers = {
                "charset": "utf-8",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            # Copy auth header from the aiohttp session
            if hasattr(client_session, '_default_headers'):
                auth = client_session._default_headers.get("Authorization")
                if auth:
                    headers["Authorization"] = auth
                    print(f"  [_requests_transmit] auth from session: {auth[:12]}...", flush=True)
            elif os.environ.get("RUNPOD_AI_API_KEY"):
                headers["Authorization"] = os.environ["RUNPOD_AI_API_KEY"]
                print(f"  [_requests_transmit] auth from env", flush=True)
            else:
                print(f"  [_requests_transmit] WARNING: no auth found!", flush=True)

            t0 = time.time()
            resp = req_lib.post(url, data=job_data, headers=headers, timeout=30)
            elapsed = time.time() - t0
            print(f"  [_requests_transmit] -> {resp.status_code} ({elapsed:.1f}s)", flush=True)
            print(f"  [_requests_transmit] response body: {resp.text[:200]}", flush=True)
            resp.raise_for_status()
            print(f"  [_requests_transmit] SUCCESS", flush=True)
        except Exception as e:
            print(f"  [_requests_transmit] FAILED: {type(e).__name__}: {e}", flush=True)
            # Don't re-raise — _handle_result only catches ClientError/TypeError/RuntimeError.
            # If we raise a requests exception, it propagates uncaught and may crash the worker.
            # Instead, just log and return silently — the job result was attempted.

    rp_http._transmit = _requests_transmit
    # Verify patch stuck
    print(f"Patched _transmit: {rp_http._transmit}", flush=True)
    print(f"Patch verified: {rp_http._transmit is _requests_transmit}", flush=True)
except Exception as e:
    import traceback as _tb
    print(f"WARNING: Failed to patch _transmit: {e}", flush=True)
    _tb.print_exc()

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

    # Ping: instant return + diagnostics
    if input_data.get("ping"):
        import requests as _req
        import runpod.serverless.modules.rp_http as _rp_http
        diag = {
            "pong": True,
            "models_loaded": engine.models_loaded if engine else False,
            "uptime": round(time.time() - t_module_start, 1),
            "runpod_version": getattr(runpod, '__version__', '?'),
            "transmit_patched": "requests_transmit" in str(_rp_http._transmit),
            "job_done_url": getattr(_rp_http, 'JOB_DONE_URL', 'NOT SET'),
            "webhook_post_output": os.environ.get('RUNPOD_WEBHOOK_POST_OUTPUT', 'NOT SET'),
        }
        # Test outbound POST
        try:
            t0 = time.time()
            r = _req.post("https://httpbin.org/post", data='{"test":true}',
                          headers={"Content-Type": "application/json"}, timeout=10)
            diag["outbound_post_test"] = {"status": r.status_code, "time": round(time.time() - t0, 2)}
        except Exception as e:
            diag["outbound_post_test"] = {"error": str(e)}
        print(f"  [PING] diag={diag}", flush=True)
        return diag

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
