#!/usr/bin/env python3
"""Quick test for the RunPod serverless endpoint.

Usage:
    python test_runpod.py --api-key YOUR_RUNPOD_KEY

Or with env vars:
    RUNPOD_API_KEY=xxx python test_runpod.py
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error

ENDPOINT_ID = "2q0b5zdxy2kufz"
MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
DEFAULT_IMAGE = os.path.join(MEDIA_DIR, "ai_4ab86e0c.png")
DEFAULT_AUDIO = os.path.join(MEDIA_DIR, "ElevenLabs_2026-03-11T16_20_57_Bella - Professional, Bright, Warm_pre_sp100_s40_sb70_se55_b_m2.mp3")

MAX_POLL_TIME = 300  # 5 minutes max
POLL_INTERVAL = 5


def main():
    parser = argparse.ArgumentParser(description="Test RunPod avatar endpoint")
    parser.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"), help="RunPod API key")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to portrait image")
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="Path to audio file")
    parser.add_argument("--resolution", default="480p", choices=["480p", "720p"])
    parser.add_argument("--prompt", default="A person talking", help="Text prompt for T5 conditioning")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--endpoint", default=ENDPOINT_ID, help="RunPod endpoint ID")
    parser.add_argument("--timeout", type=int, default=MAX_POLL_TIME, help="Max seconds to wait for job")
    parser.add_argument("--sync", action="store_true", help="Use /runsync instead of /run + polling")
    parser.add_argument("--ping", action="store_true", help="Send a ping job instead of video generation")
    parser.add_argument("--debug", action="store_true", help="Send a debug job (loads models, tests face detection)")
    parser.add_argument("--health", action="store_true", help="Check endpoint health and exit")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set RUNPOD_API_KEY env var")
        sys.exit(1)

    # Health check mode
    if args.health:
        health_url = f"https://api.runpod.ai/v2/{args.endpoint}/health"
        req = urllib.request.Request(health_url, headers={"Authorization": f"Bearer {args.api_key}"})
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())
            print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Health check failed: {e}")
            sys.exit(1)
        return

    # Build payload based on mode
    if args.ping:
        payload = json.dumps({"input": {"ping": True}}).encode("utf-8")
        print("Sending ping job...")
    elif args.debug:
        input_data = {"debug": True}
        if os.path.exists(args.image):
            with open(args.image, "rb") as f:
                input_data["image_base64"] = base64.b64encode(f.read()).decode("ascii")
            print(f"Debug with image: {args.image}")
        payload = json.dumps({"input": input_data}).encode("utf-8")
        print("Sending debug job...")
    else:
        # Encode inputs
        with open(args.image, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
        with open(args.audio, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")

        print(f"Image: {args.image} ({len(image_b64) // 1024} KB base64)")
        print(f"Audio: {args.audio} ({len(audio_b64) // 1024} KB base64)")

        payload = json.dumps({
            "input": {
                "image_base64": image_b64,
                "audio_base64": audio_b64,
                "resolution": args.resolution,
                "prompt": args.prompt,
            }
        }).encode("utf-8")

    mode = "runsync" if args.sync else "run"
    url = f"https://api.runpod.ai/v2/{args.endpoint}/{mode}"

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
    )

    print(f"\nSubmitting job to {url}...")
    t0 = time.time()
    try:
        timeout = args.timeout if args.sync else 30
        resp = urllib.request.urlopen(req, timeout=timeout)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"Error submitting job: {e}")
        sys.exit(1)

    result = json.loads(resp.read())
    job_id = result.get("id")
    status = result.get("status")
    print(f"Job submitted: {job_id} (status: {status})")

    is_simple = args.ping or args.debug

    # If runsync, the result comes back directly
    if args.sync:
        if status == "COMPLETED":
            if is_simple:
                print(f"\nResult ({time.time() - t0:.0f}s):")
                print(json.dumps(result.get("output", result), indent=2))
            else:
                handle_completed(result, args, t0)
        elif status == "FAILED":
            print(f"\nJob failed: {json.dumps(result, indent=2)}")
            sys.exit(1)
        elif status == "IN_QUEUE" or status == "IN_PROGRESS":
            print(f"\nJob didn't complete within runsync timeout. Falling back to polling...")
            poll_for_result(args, job_id, t0, simple_output=is_simple)
        else:
            print(f"\nUnexpected status: {status}")
            print(json.dumps(result, indent=2))
            sys.exit(1)
        return

    # Async mode: poll for completion
    poll_for_result(args, job_id, t0, simple_output=is_simple)


def handle_completed(result, args, t0):
    output = result.get("output", {})
    if "error" in output:
        print(f"\nError: {output['error']}")
        sys.exit(1)
    video_b64 = output.get("video_base64")
    gen_time = output.get("generation_time")
    elapsed = time.time() - t0
    if video_b64:
        video_bytes = base64.b64decode(video_b64)
        with open(args.output, "wb") as f:
            f.write(video_bytes)
        print(f"\nSuccess! Output: {args.output} ({len(video_bytes) / 1e6:.2f} MB)")
        print(f"Generation time: {gen_time}s (total: {elapsed:.0f}s)")
    else:
        print("\nNo video in response")
        print(json.dumps(output, indent=2))


def poll_for_result(args, job_id, t0, simple_output=False):
    status_url = f"https://api.runpod.ai/v2/{args.endpoint}/status/{job_id}"
    while True:
        elapsed = time.time() - t0
        if elapsed > args.timeout:
            print(f"\nTimed out after {elapsed:.0f}s. Job {job_id} may still be running.")
            print(f"Check manually: {status_url}")
            sys.exit(2)

        time.sleep(POLL_INTERVAL)
        req = urllib.request.Request(
            status_url,
            headers={"Authorization": f"Bearer {args.api_key}"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=15)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            elapsed = time.time() - t0
            print(f"  [{elapsed:.0f}s] Poll error: {e}, retrying...")
            continue

        result = json.loads(resp.read())
        status = result.get("status")
        elapsed = time.time() - t0
        print(f"  [{elapsed:.0f}s] {status}")

        if status == "COMPLETED":
            if simple_output:
                print(f"\nResult ({time.time() - t0:.0f}s):")
                print(json.dumps(result.get("output", result), indent=2))
            else:
                handle_completed(result, args, t0)
            break
        elif status == "FAILED":
            print(f"\nJob failed: {json.dumps(result, indent=2)}")
            sys.exit(1)
        elif status == "CANCELLED":
            print(f"\nJob was cancelled: {job_id}")
            sys.exit(1)


if __name__ == "__main__":
    main()
