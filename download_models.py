#!/usr/bin/env python3
"""Download Hallo3 pretrained models from HuggingFace.

Run this during Docker build. Downloads ~52GB of model weights:
- Hallo3 fine-tuned checkpoint (CogVideoX-5B + audio conditioning)
- CogVideoX-5B-I2V base model (3D VAE + transformer)
- T5-v1_1-xxl text encoder
- wav2vec2-base-960h audio encoder
- InsightFace face analysis models
- Audio separator (MDX-Net vocal isolation)
"""

import os

from huggingface_hub import snapshot_download

PRETRAINED_DIR = os.getenv("PRETRAINED_DIR", "/app/hallo3/pretrained_models")


def download():
    os.makedirs(PRETRAINED_DIR, exist_ok=True)

    # Main Hallo3 model bundle — contains all checkpoints in the expected directory structure:
    #   hallo3/1/mp_rank_00_model_states.pt
    #   cogvideox-5b-i2v-sat/transformer/1/mp_rank_00_model_states.pt
    #   cogvideox-5b-i2v-sat/vae/3d-vae.pt
    #   t5-v1_1-xxl/ (text encoder)
    #   wav2vec/wav2vec2-base-960h/ (audio encoder)
    #   audio_separator/Kim_Vocal_2.onnx
    #   face_analysis/models/ (InsightFace + MediaPipe)
    print("\n--- Downloading fudan-generative-ai/hallo3 (all models) ---")
    snapshot_download(
        repo_id="fudan-generative-ai/hallo3",
        local_dir=PRETRAINED_DIR,
    )
    print(f"    Done: {PRETRAINED_DIR}")

    # Verify key files exist
    required_files = [
        "hallo3/1/mp_rank_00_model_states.pt",
        "cogvideox-5b-i2v-sat/vae/3d-vae.pt",
    ]
    for f in required_files:
        path = os.path.join(PRETRAINED_DIR, f)
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024**3)
            print(f"    ✓ {f} ({size_gb:.1f} GB)")
        else:
            print(f"    ✗ MISSING: {f}")

    print(f"\n✅ All Hallo3 models downloaded to {PRETRAINED_DIR}")


if __name__ == "__main__":
    download()
