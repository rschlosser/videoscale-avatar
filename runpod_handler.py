"""Minimal RunPod handler for debugging."""

import sys
import time

print(f"[handler] Starting... Python {sys.version}", flush=True)
t0 = time.time()

try:
    import runpod
    print(f"[handler] runpod imported in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"[handler] FAILED to import runpod: {e}", flush=True)
    raise


def handler(job):
    print(f"[handler] Job received: {list(job.get('input', {}).keys())}", flush=True)
    return {"ok": True, "time": time.time()}


print(f"[handler] Registering handler ({time.time()-t0:.1f}s since start)...", flush=True)
runpod.serverless.start({"handler": handler})
