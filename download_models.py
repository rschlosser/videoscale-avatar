#!/usr/bin/env python3
"""Download model checkpoints from HuggingFace.

Run this during Docker build or on first startup.
Downloads FasterLivePortrait ONNX models + JoyVASA + HuBERT checkpoints.
"""

from huggingface_hub import snapshot_download

CHECKPOINT_DIR = "/app/checkpoints"

MODELS = [
    # FasterLivePortrait ONNX models (NOT the raw KwaiVGI/LivePortrait PyTorch weights)
    {
        "repo": "warmshao/FasterLivePortrait",
        "local_dir": f"{CHECKPOINT_DIR}",
    },
    # JoyVASA audio-driven motion model
    {
        "repo": "jdh-algo/JoyVASA",
        "local_dir": f"{CHECKPOINT_DIR}/JoyVASA",
    },
    # HuBERT audio encoder (used by JoyVASA for audio feature extraction)
    {
        "repo": "TencentGameMate/chinese-hubert-base",
        "local_dir": f"{CHECKPOINT_DIR}/chinese-hubert-base",
    },
]


def download():
    for model in MODELS:
        print(f"\n--- Downloading {model['repo']} ---")
        snapshot_download(
            repo_id=model["repo"],
            local_dir=model["local_dir"],
        )
        print(f"    Done: {model['local_dir']}")

    print(f"\n✅ All models downloaded to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    download()
