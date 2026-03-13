#!/usr/bin/env python3
"""
End-to-end smoke test for the avatar generation pipeline.

Run inside the Docker container:
    python test_e2e.py

What it does:
    1. Generates a synthetic test image (solid color + circle "face")
    2. Generates a 2-second sine-wave audio file
    3. Tries to load models and run the full pipeline
    4. Reports success/failure at each stage

For a real test, drop a portrait photo and audio clip into /app/test_assets/
and run:
    python test_e2e.py --image /app/test_assets/photo.jpg --audio /app/test_assets/audio.wav
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("test_e2e")

TESTS_DIR = Path("/tmp/avatar_test")


def generate_test_image(path: Path):
    """Create a simple test image with a face-like circle."""
    import cv2
    import numpy as np

    img = np.full((512, 512, 3), (200, 180, 160), dtype=np.uint8)
    # Draw a "face"
    cv2.circle(img, (256, 256), 150, (220, 200, 180), -1)
    # Eyes
    cv2.circle(img, (210, 220), 15, (60, 40, 30), -1)
    cv2.circle(img, (300, 220), 15, (60, 40, 30), -1)
    # Mouth
    cv2.ellipse(img, (256, 310), (40, 20), 0, 0, 180, (60, 40, 30), 2)
    cv2.imwrite(str(path), img)
    logger.info("Test image written: %s", path)


def generate_test_audio(path: Path, duration: float = 2.0, sr: int = 16000):
    """Create a 2-second sine wave WAV file."""
    import numpy as np
    import soundfile as sf

    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), audio, sr)
    logger.info("Test audio written: %s (%.1fs)", path, duration)


def test_imports():
    """Test that all required imports work."""
    logger.info("--- Test: imports ---")
    errors = []

    for module in [
        "cv2",
        "numpy",
        "torch",
        "torchaudio",
        "onnxruntime",
        "omegaconf",
        "soundfile",
    ]:
        try:
            __import__(module)
            logger.info("  ✓ %s", module)
        except ImportError as e:
            logger.error("  ✗ %s: %s", module, e)
            errors.append(module)

    # FasterLivePortrait-specific imports
    for module in [
        "src.pipelines.faster_live_portrait_pipeline",
        "src.pipelines.joyvasa_audio_to_motion_pipeline",
    ]:
        try:
            __import__(module)
            logger.info("  ✓ %s", module)
        except ImportError as e:
            logger.error("  ✗ %s: %s", module, e)
            errors.append(module)

    if errors:
        logger.error("Missing imports: %s", errors)
        return False
    logger.info("All imports OK")
    return True


def test_checkpoints():
    """Verify checkpoint files exist."""
    logger.info("--- Test: checkpoints ---")
    from app.engine import CHECKPOINT_DIR

    required = {
        "liveportrait_onnx": CHECKPOINT_DIR / "liveportrait_onnx",
        "JoyVASA motion model": CHECKPOINT_DIR / "JoyVASA" / "motion_generator" / "motion_generator_hubert_chinese.pt",
        "JoyVASA motion template": CHECKPOINT_DIR / "JoyVASA" / "motion_template" / "motion_template.pkl",
        "HuBERT": CHECKPOINT_DIR / "chinese-hubert-base",
    }

    ok = True
    for name, path in required.items():
        if path.exists():
            logger.info("  ✓ %s: %s", name, path)
        else:
            logger.error("  ✗ %s: MISSING %s", name, path)
            ok = False
    return ok


def test_gpu():
    """Check CUDA availability."""
    logger.info("--- Test: GPU ---")
    import torch

    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("  ✓ CUDA available: %s (%.1f GB)", dev, mem)
        return True
    else:
        logger.error("  ✗ CUDA not available — inference will fail")
        return False


def test_model_load():
    """Try loading the engine."""
    logger.info("--- Test: model loading ---")
    from app.engine import AvatarEngine

    engine = AvatarEngine()
    t0 = time.time()
    engine.load_models()
    elapsed = time.time() - t0
    logger.info("  ✓ Models loaded in %.1fs", elapsed)
    return engine


def test_generation(engine, image_path: str, audio_path: str):
    """Run a full generation."""
    logger.info("--- Test: generation ---")
    output_path = str(TESTS_DIR / "output.mp4")

    t0 = time.time()
    import asyncio
    asyncio.run(engine.generate(
        image_path=image_path,
        audio_path=audio_path,
        output_path=output_path,
        resolution="480p",
    ))
    elapsed = time.time() - t0

    out = Path(output_path)
    if out.exists():
        size_mb = out.stat().st_size / 1e6
        logger.info("  ✓ Output: %s (%.2f MB, %.1fs)", output_path, size_mb, elapsed)
        return True
    else:
        logger.error("  ✗ No output file produced")
        return False


def main():
    parser = argparse.ArgumentParser(description="E2E smoke test")
    parser.add_argument("--image", help="Path to portrait image (default: synthetic)")
    parser.add_argument("--audio", help="Path to audio file (default: synthetic sine wave)")
    args = parser.parse_args()

    TESTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # Stage 1: Imports
    results["imports"] = test_imports()
    if not results["imports"]:
        logger.error("STOP: fix imports before continuing")
        sys.exit(1)

    # Stage 2: GPU
    results["gpu"] = test_gpu()

    # Stage 3: Checkpoints
    results["checkpoints"] = test_checkpoints()
    if not results["checkpoints"]:
        logger.error("STOP: fix checkpoints before continuing")
        sys.exit(1)

    # Stage 4: Model load
    try:
        engine = test_model_load()
        results["model_load"] = True
    except Exception as e:
        logger.error("  ✗ Model load failed: %s", e, exc_info=True)
        results["model_load"] = False
        logger.error("STOP: fix model loading before continuing")
        sys.exit(1)

    # Stage 5: Generation
    image_path = args.image
    audio_path = args.audio

    if not image_path:
        image_path = str(TESTS_DIR / "test_face.jpg")
        generate_test_image(Path(image_path))
        logger.info("NOTE: using synthetic face — may fail face detection. "
                     "Use --image with a real portrait for a proper test.")

    if not audio_path:
        audio_path = str(TESTS_DIR / "test_audio.wav")
        generate_test_audio(Path(audio_path))

    try:
        results["generation"] = test_generation(engine, image_path, audio_path)
    except Exception as e:
        logger.error("  ✗ Generation failed: %s", e, exc_info=True)
        results["generation"] = False

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS:")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
    print("=" * 50)

    if all(results.values()):
        print("\nAll tests passed! Ready to deploy.")
    else:
        print("\nSome tests failed — see logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
