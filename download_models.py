#!/usr/bin/env python3
"""Download model checkpoints from HuggingFace.

Run this during Docker build or on first startup.
Downloads LivePortrait + JoyVASA + HuBERT checkpoints.
"""

import subprocess
import sys

CHECKPOINT_DIR = "/app/checkpoints"

MODELS = [
    # LivePortrait base models
    {
        "repo": "KwaiVGI/LivePortrait",
        "local_dir": f"{CHECKPOINT_DIR}/liveportrait",
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
        cmd = [
            sys.executable, "-m", "huggingface_hub", "download",
            model["repo"],
            "--local-dir", model["local_dir"],
            "--local-dir-use-symlinks", "False",
        ]
        # Use huggingface-cli if available
        cli_cmd = [
            "huggingface-cli", "download",
            model["repo"],
            "--local-dir", model["local_dir"],
        ]
        try:
            subprocess.run(cli_cmd, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            subprocess.run(cmd, check=True)

    print(f"\n✅ All models downloaded to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    download()
