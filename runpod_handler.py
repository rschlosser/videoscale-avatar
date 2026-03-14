"""
Minimal RunPod handler with maximum error catching.
"""

import sys
import time
import json
import traceback

print(f"[STARTUP] Python {sys.version}", flush=True)
print(f"[STARTUP] sys.path: {sys.path[:5]}", flush=True)
t0 = time.time()

try:
    import runpod
    print(f"[STARTUP] runpod {runpod.__version__} imported in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"[STARTUP] FATAL: cannot import runpod: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print(f"[STARTUP] Total init: {time.time()-t0:.1f}s", flush=True)


def handler(job):
    """Ultra-simple handler that logs everything."""
    try:
        print(f"[HANDLER] Called! job type={type(job).__name__}", flush=True)
        print(f"[HANDLER] job keys={list(job.keys()) if isinstance(job, dict) else 'NOT_DICT'}", flush=True)

        if isinstance(job, dict):
            input_data = job.get("input", {})
            print(f"[HANDLER] input keys={list(input_data.keys()) if isinstance(input_data, dict) else input_data}", flush=True)
        else:
            input_data = {}

        result = {
            "ok": True,
            "timestamp": time.time(),
            "uptime": round(time.time() - t0, 1),
            "runpod_version": getattr(runpod, '__version__', 'unknown'),
        }
        print(f"[HANDLER] Returning: {json.dumps(result)}", flush=True)
        return result

    except Exception as e:
        err = f"Handler error: {e}\n{traceback.format_exc()}"
        print(f"[HANDLER] ERROR: {err}", flush=True)
        return {"error": err}


print(f"[STARTUP] Starting runpod.serverless.start()...", flush=True)
runpod.serverless.start({"handler": handler})
print(f"[STARTUP] runpod.serverless.start() returned (unexpected!)", flush=True)
