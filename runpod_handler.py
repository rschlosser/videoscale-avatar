"""
RunPod Serverless Handler for avatar video generation.

Minimal startup to ensure handler registers quickly with RunPod.
Heavy imports (torch, cv2, etc.) are deferred to first job.
"""

import logging
import os
import sys
import time
import urllib.request

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

print("=== Handler module starting ===", flush=True)
t_module_start = time.time()

print(f"Python: {sys.version}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)

# --- External diagnostic logging via ntfy.sh (stdlib only) ---
NTFY_TOPIC = "videoscale-avatar-debug-9f3k2x"


def _ntfy(msg):
    """Fire-and-forget POST to ntfy.sh for external diagnostics. Uses only stdlib."""
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=str(msg)[:4000].encode("utf-8"),
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


# --- FIX SSL FOR AIOHTTP ---
# The base image (conda OpenSSL) has stale CA certs. Patch ssl.create_default_context
# to always load our cert bundle so aiohttp's SSL context works.
import ssl

_orig_create_default_context = ssl.create_default_context


def _patched_create_default_context(
    purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None
):
    ctx = _orig_create_default_context(
        purpose, cafile=cafile, capath=capath, cadata=cadata
    )
    cert_file = os.environ.get(
        "SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt"
    )
    try:
        ctx.load_verify_locations(cert_file)
    except Exception:
        pass
    return ctx


ssl.create_default_context = _patched_create_default_context
print(
    f"Patched ssl.create_default_context to load "
    f"{os.environ.get('SSL_CERT_FILE', '/etc/ssl/certs/ca-certificates.crt')}",
    flush=True,
)

import runpod

print(
    f"runpod {getattr(runpod, '__version__', '?')} imported "
    f"({time.time() - t_module_start:.1f}s)",
    flush=True,
)

# --- Print RunPod env vars + send to ntfy ---
env_lines = []
print("=== RunPod environment ===", flush=True)
for key in sorted(os.environ):
    if key.startswith("RUNPOD"):
        val = os.environ[key]
        if "KEY" in key or "SECRET" in key:
            val = val[:8] + "..." if len(val) > 8 else "***"
        env_lines.append(f"  {key}={val}")
        print(f"  {key}={val}", flush=True)
print("=== End RunPod env ===", flush=True)

# Check JOB_DONE_URL
try:
    import runpod.serverless.modules.rp_http as rp_http

    _job_done_url = getattr(rp_http, "JOB_DONE_URL", "NOT SET")
    print(f"JOB_DONE_URL: {_job_done_url}", flush=True)
    _ntfy(
        f"STARTUP: runpod={getattr(runpod, '__version__', '?')}, "
        f"JOB_DONE_URL={_job_done_url}\n" + "\n".join(env_lines)
    )
except Exception as e:
    print(f"WARNING: Could not read JOB_DONE_URL: {e}", flush=True)
    _ntfy(f"STARTUP: error reading JOB_DONE_URL: {e}")

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


def _real_handler(job):
    """RunPod serverless handler — actual logic."""
    input_data = job["input"]

    # Ping: instant return + diagnostics
    if input_data.get("ping"):
        import runpod.serverless.modules.rp_http as _rp_http

        diag = {
            "pong": True,
            "models_loaded": engine.models_loaded if engine else False,
            "uptime": round(time.time() - t_module_start, 1),
            "runpod_version": getattr(runpod, "__version__", "?"),
            "job_done_url": getattr(_rp_http, "JOB_DONE_URL", "NOT SET"),
            "webhook_post_output": os.environ.get(
                "RUNPOD_WEBHOOK_POST_OUTPUT", "NOT SET"
            ),
        }
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
        except Exception:
            results["models_loaded"] = False
            results["load_error"] = str(load_error)
            return results

        import cv2

        pipeline = engine.lp_pipeline
        results["model_keys"] = list(pipeline.model_dict.keys())

        image_b64 = input_data.get("image_base64")
        if image_b64:
            with tempfile.TemporaryDirectory() as td:
                img_path = os.path.join(td, "test.jpg")
                Path(img_path).write_bytes(base64.b64decode(image_b64))
                img_bgr = cv2.imread(img_path)
                results["image_shape"] = (
                    list(img_bgr.shape) if img_bgr is not None else None
                )

                t0 = time.time()
                faces = pipeline.model_dict["face_analysis"].predict(img_bgr)
                results["face_detection"] = {
                    "faces": len(faces),
                    "time": round(time.time() - t0, 2),
                }

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

        logger.info(
            "Generating: image=%d bytes, audio=%d bytes, res=%s",
            img_path.stat().st_size,
            audio_path.stat().st_size,
            resolution,
        )
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


def handler(job):
    """Wrapper handler with ntfy diagnostics."""
    job_id = job.get("id", "unknown")
    _ntfy(f"HANDLER CALLED: id={job_id}, keys={list(job.get('input', {}).keys())}")
    try:
        t0 = time.time()
        result = _real_handler(job)
        elapsed = round(time.time() - t0, 2)
        # Truncate result for ntfy (avoid sending huge base64)
        result_summary = {k: type(v).__name__ for k, v in result.items()} if isinstance(result, dict) else str(type(result))
        _ntfy(f"HANDLER OK: id={job_id}, elapsed={elapsed}s, result_keys={result_summary}")
        return result
    except Exception as e:
        _ntfy(f"HANDLER ERROR: id={job_id}, err={type(e).__name__}: {e}")
        raise


print(
    f"Registering handler ({time.time() - t_module_start:.1f}s since module load)...",
    flush=True,
)
try:
    runpod.serverless.start({"handler": handler})
except Exception as e:
    print(f"CRITICAL: runpod.serverless.start() CRASHED: {e}", flush=True)
    import traceback

    traceback.print_exc()
    sys.exit(1)
