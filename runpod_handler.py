"""
RunPod handler with generator-based output for debugging.
"""

import sys
import time
import traceback

print(f"[STARTUP] Python {sys.version}", flush=True)
t0 = time.time()

try:
    import runpod
    print(f"[STARTUP] runpod {getattr(runpod, '__version__', '?')} loaded ({time.time()-t0:.1f}s)", flush=True)
except Exception as e:
    print(f"[STARTUP] FATAL: {e}", flush=True)
    sys.exit(1)


def handler(job):
    """Generator handler — yields result as stream."""
    try:
        print(f"[HANDLER] Job received", flush=True)
        result = {
            "ok": True,
            "timestamp": time.time(),
            "uptime": round(time.time() - t0, 1),
            "runpod_version": getattr(runpod, '__version__', '?'),
        }
        print(f"[HANDLER] Yielding result", flush=True)
        yield result
        print(f"[HANDLER] Done", flush=True)
    except Exception as e:
        print(f"[HANDLER] ERROR: {e}", flush=True)
        yield {"error": str(e)}


print(f"[STARTUP] Starting handler ({time.time()-t0:.1f}s)...", flush=True)
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
