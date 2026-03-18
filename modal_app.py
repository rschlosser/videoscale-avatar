"""
Hallo3 avatar generation on Modal.

Deploy:  modal deploy modal_app.py
Test:    modal run modal_app.py
Dev:     modal serve modal_app.py
"""

import modal

app = modal.App("hallo3-avatar")

# Container image with all Hallo3 dependencies
hallo3_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
    )
    .entrypoint([])
    .apt_install([
        "git", "ffmpeg", "wget", "curl",
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
        "build-essential", "clang",
    ])
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands("git clone --depth 1 https://github.com/fudan-generative-vision/hallo3.git /app/hallo3")
    .run_commands("sed -i '/^pyav==/d; /^jax==/d; /^jaxlib==/d; /^nvidia-/d; /^torch==/d; /^torchvision==/d; /^torchaudio==/d; /^triton==/d' /app/hallo3/requirements.txt && pip install -r /app/hallo3/requirements.txt")
    .pip_install("fastapi[standard]")
    .env({
        "PYTHONPATH": "/app/hallo3:/app/hallo3/hallo3",
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",
        "MEDIAPIPE_DISABLE_GPU": "1",
    })
)

# Persistent volume for model weights (~52GB)
model_volume = modal.Volume.from_name("hallo3-weights", create_if_missing=True)

MODELS_PATH = "/models"
HALLO3_ROOT = "/app/hallo3"


def download_models():
    """Download Hallo3 pretrained models to the volume."""
    import os
    from huggingface_hub import snapshot_download

    marker = os.path.join(MODELS_PATH, ".download_complete")
    if os.path.exists(marker):
        print("Models already downloaded!")
        return

    print("Downloading Hallo3 models (~52GB)...")
    snapshot_download(
        repo_id="fudan-generative-ai/hallo3",
        local_dir=MODELS_PATH,
    )

    # Verify key files
    for f in [
        "hallo3/1/mp_rank_00_model_states.pt",
        "cogvideox-5b-i2v-sat/vae/3d-vae.pt",
    ]:
        path = os.path.join(MODELS_PATH, f)
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024**3)
            print(f"  OK: {f} ({size_gb:.1f} GB)")
        else:
            print(f"  MISSING: {f}")

    open(marker, "w").close()
    print("Download complete!")


@app.cls(
    image=hallo3_image,
    gpu="H100",
    timeout=1800,
    scaledown_window=300,
    volumes={MODELS_PATH: model_volume},
)
class Hallo3:
    @modal.enter()
    def load_models(self):
        """Called once when container starts — download models if needed, load into GPU."""
        import os
        import sys
        import time

        sys.path.insert(0, HALLO3_ROOT)
        sys.path.insert(0, os.path.join(HALLO3_ROOT, "hallo3"))

        # Ensure models are downloaded
        download_models()
        model_volume.commit()

        # Hallo3 configs use relative paths like ./pretrained_models/...
        os.chdir(HALLO3_ROOT)

        # Symlink models to where Hallo3 expects them
        pretrained_dir = os.path.join(HALLO3_ROOT, "pretrained_models")
        if not os.path.exists(pretrained_dir):
            os.symlink(MODELS_PATH, pretrained_dir)

        # Load Hallo3 model
        t0 = time.time()
        from arguments import get_args
        from diffusion_video import SATVideoDiffusionEngine
        from sat.model.base_model import get_model
        from sat.training.model_io import load_checkpoint

        # configs may be at repo root or inside hallo3/ subdir
        configs_dir = os.path.join(HALLO3_ROOT, "configs")
        if not os.path.exists(configs_dir):
            configs_dir = os.path.join(HALLO3_ROOT, "hallo3", "configs")
        config_base = os.path.join(configs_dir, "cogvideox_5b_i2v_s2.yaml")
        config_infer = os.path.join(configs_dir, "inference.yaml")

        self.args = get_args(["--base", config_base, config_infer])
        self.args.model_config.first_stage_config.params.cp_size = 1
        self.args.model_config.network_config.params.transformer_args.model_parallel_size = 1
        self.args.model_config.network_config.params.transformer_args.checkpoint_activations = False
        self.args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

        import torch
        self.model = get_model(self.args, SATVideoDiffusionEngine)
        load_checkpoint(self.model, self.args)
        self.model.eval()
        self.model = self.model.to("cuda")
        print(f"Hallo3 model loaded ({time.time() - t0:.1f}s)")

        # Audio processor (matches sample_video.py init)
        from sgm.utils.audio_processor import AudioProcessor
        audio_sep_model_file = getattr(self.args, 'audio_separator_model_path',
            os.path.join(MODELS_PATH, "audio_separator", "Kim_Vocal_2.onnx"))
        wav2vec_path = getattr(self.args, 'wav2vec_model_path',
            os.path.join(MODELS_PATH, "wav2vec", "wav2vec2-base-960h"))
        wav2vec_only_last = getattr(self.args, 'wav2vec_features', 'all') == 'last'
        sample_rate = getattr(self.args, 'sample_rate', 16000)
        self.audio_processor = AudioProcessor(
            sample_rate,
            wav2vec_path,
            wav2vec_only_last,
            os.path.dirname(audio_sep_model_file),
            os.path.basename(audio_sep_model_file),
            os.path.join("/tmp", "audio_preprocess"),
        )

        # Image processor
        from sgm.utils.image_processor import ImageProcessor
        face_model_path = getattr(self.args, 'face_analysis_model_path',
            os.path.join(MODELS_PATH, "face_analysis"))
        self.image_processor = ImageProcessor(face_model_path)
        print("All models loaded!")

    @modal.method()
    def generate(self, image_bytes: bytes, audio_bytes: bytes, prompt: str = "A person talking") -> bytes:
        """Generate a talking-head video from image + audio bytes.

        Returns MP4 video bytes.
        """
        import math
        import os
        import subprocess
        import tempfile
        import time

        import imageio
        import numpy as np
        import torch
        import torch.nn.functional as F
        import torchvision.transforms as TT
        import torchvision.transforms as transforms
        from PIL import Image
        from torchvision.transforms.functional import center_crop, resize
        from torchvision.transforms import InterpolationMode

        t_start = time.time()

        # --- Local helper functions (from hallo3/sample_video.py) ---
        def process_audio_emb(audio_emb):
            concatenated_tensors = []
            for i in range(audio_emb.shape[0]):
                vectors_to_concat = [
                    audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)] for j in range(-2, 3)
                ]
                concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))
            return torch.stack(concatenated_tensors, dim=0)

        def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
            if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
                arr = resize(arr, size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                             interpolation=InterpolationMode.BICUBIC)
            else:
                arr = resize(arr, size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                             interpolation=InterpolationMode.BICUBIC)
            h, w = arr.shape[2], arr.shape[3]
            arr = arr.squeeze(0)
            delta_h = h - image_size[0]
            delta_w = w - image_size[1]
            if reshape_mode == "center":
                top, left = delta_h // 2, delta_w // 2
            else:
                top = np.random.randint(0, delta_h + 1)
                left = np.random.randint(0, delta_w + 1)
            arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
            return arr

        def resize_for_square_padding(arr, image_size):
            arr = transforms.Resize(size=[image_size[0], image_size[0]])(arr)
            t, c, h, w = arr.shape
            padding_width = image_size[1] - w
            pad_left = padding_width // 2
            pad_right = padding_width - pad_left
            arr = F.pad(arr, (pad_left, pad_right, 0, 0), mode='constant', value=0)
            return arr

        def add_mask_to_first_frame(image, mask_rate=0.25):
            b, c, f, h, w = image.shape
            image = image.permute(0, 2, 1, 3, 4).contiguous()
            rand_mask = torch.rand(h, w).to(dtype=image.dtype, device=image.device)
            mask = rand_mask > mask_rate
            mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            mask = mask.expand(b, f, c, h, w)
            image = image * mask
            image = image.permute(0, 2, 1, 3, 4).contiguous()
            return image

        # --- Write inputs to temp files ---
        tmpdir = tempfile.mkdtemp(prefix="hallo3_")
        img_ext = ".png" if image_bytes[:4] == b'\x89PNG' else ".jpg"
        image_path = os.path.join(tmpdir, f"input{img_ext}")
        audio_path = os.path.join(tmpdir, "input.wav")
        output_path = os.path.join(tmpdir, "output.mp4")

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        raw_audio_path = os.path.join(tmpdir, "input_raw")
        with open(raw_audio_path, "wb") as f:
            f.write(audio_bytes)
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_audio_path, "-ar", "16000", "-ac", "1", audio_path],
            capture_output=True, timeout=30,
        )

        # --- Config ---
        T = self.args.sampling_num_frames
        L = (T - 1) * 4 + 1
        fps = self.args.sampling_fps
        image_size = [480, 720]
        H, W = image_size
        C = self.args.latent_channels
        F = 8
        n_motion_frame = 2
        mask_rate = 0.1

        # --- Step 1: Audio ---
        audio_emb, length = self.audio_processor.preprocess(audio_path, L)
        audio_emb = process_audio_emb(audio_emb)

        # --- Step 2: Image + Face ---
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        face_emb, face_mask_path = self.image_processor.preprocess(image_path, cache_dir, 1.2)
        face_emb = torch.tensor(face_emb.reshape(1, -1)).to("cuda")

        transform = TT.Compose([TT.ToTensor()])
        image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to("cuda")
        face_mask = transform(Image.open(face_mask_path).convert("RGB")).unsqueeze(0).to("cuda")
        ref_image = image * face_mask

        _, _, h, w = image.shape
        is_padding = (h == w)

        if is_padding:
            image = resize_for_square_padding(image, image_size).clamp(0, 1)
            ref_image = resize_for_square_padding(ref_image, image_size).clamp(0, 1)
        else:
            image = resize_for_rectangle_crop(image, image_size, reshape_mode="center").unsqueeze(0)
            ref_image = resize_for_rectangle_crop(ref_image, image_size, reshape_mode="center").unsqueeze(0)

        image = image * 2.0 - 1.0
        ref_image = ref_image * 2.0 - 1.0

        # Encode reference image and initial motion
        motion_image = image.unsqueeze(2).to(torch.bfloat16)
        ref_image_pixel = image.unsqueeze(2).to(torch.bfloat16)
        ref_image_latent = ref_image.unsqueeze(2).to(torch.bfloat16)

        motion_image = torch.cat([motion_image] * n_motion_frame, dim=2)
        mask_image = add_mask_to_first_frame(motion_image, mask_rate=mask_rate)
        mask_image = torch.cat([ref_image_pixel, mask_image], dim=2)
        mask_image = self.model.encode_first_stage(mask_image, None)
        mask_image = mask_image.permute(0, 2, 1, 3, 4).contiguous()

        ref_image_enc = self.model.encode_first_stage(ref_image_latent, None)
        ref_image_enc = ref_image_enc.permute(0, 2, 1, 3, 4).contiguous()

        pad_shape = (mask_image.shape[0], T - 1, C, H // F, W // F)
        mask_image = torch.cat([mask_image, torch.zeros(pad_shape).to(mask_image.device).to(mask_image.dtype)], dim=1)

        # --- Step 3: Text conditioning ---
        from sample_video import get_batch, get_unique_embedder_keys_from_conditioner
        value_dict = {"prompt": prompt, "negative_prompt": "", "num_frames": torch.tensor(T).unsqueeze(0)}
        batch, batch_uc = get_batch(
            get_unique_embedder_keys_from_conditioner(self.model.conditioner), value_dict, [1]
        )
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to("cuda")
                batch_uc[key] = batch_uc[key].to("cuda")
        c, uc = self.model.conditioner.get_unconditional_conditioning(
            batch, batch_uc=batch_uc, force_uc_zero_embeddings=["txt"],
        )
        for k in c:
            if k != "crossattn":
                c[k], uc[k] = map(lambda y: y[k][:1].to("cuda"), (c, uc))

        # --- Step 4: Chunked generation ---
        times = audio_emb.shape[0] // (L - n_motion_frame)
        if times * (L - n_motion_frame) < audio_emb.shape[0]:
            times += 1

        video = []
        pre_fix = torch.zeros_like(audio_emb[:n_motion_frame])

        with torch.no_grad():
            for t in range(times):
                print(f"[{t+1}/{times}]")

                c["concat"] = mask_image
                uc["concat"] = mask_image

                audio_tensor = audio_emb[t * (L - n_motion_frame): min((t + 1) * (L - n_motion_frame), audio_emb.shape[0])]
                audio_tensor = torch.cat([pre_fix, audio_tensor], dim=0)
                pre_fix = audio_tensor[-n_motion_frame:]

                if audio_tensor.shape[0] != L:
                    pad = L - audio_tensor.shape[0]
                    padding = pre_fix[-1:].repeat(pad, *([1] * (pre_fix.dim() - 1)))
                    audio_tensor = torch.cat([audio_tensor, padding], dim=0)

                audio_tensor = audio_tensor.unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)

                samples_z = self.model.sample(
                    c, uc=uc, batch_size=1, shape=(T, C, H // F, W // F),
                    audio_emb=audio_tensor, ref_image=ref_image_enc, face_emb=face_emb,
                )

                samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()
                torch.cuda.empty_cache()
                latent = 1.0 / self.model.scale_factor * samples_z

                # Serial VAE decode to save GPU memory
                recons = []
                loop_num = (T - 1) // 2
                for i in range(loop_num):
                    if i == 0:
                        start_frame, end_frame = 0, 3
                    else:
                        start_frame, end_frame = i * 2 + 1, i * 2 + 3
                    clear_fake_cp_cache = (i == loop_num - 1)
                    recon = self.model.first_stage_model.decode(
                        latent[:, :, start_frame:end_frame].contiguous(),
                        clear_fake_cp_cache=clear_fake_cp_cache,
                    )
                    recons.append(recon)

                recon = torch.cat(recons, dim=2).to(torch.float32)
                samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

                # Update motion for next chunk
                motion_image = samples[:, -n_motion_frame:].permute(0, 2, 1, 3, 4).contiguous().to(dtype=torch.bfloat16, device="cuda")
                motion_image = motion_image * 2 - 1
                mask_image = add_mask_to_first_frame(motion_image, mask_rate=mask_rate)
                mask_image = torch.cat([ref_image_pixel, mask_image], dim=2)
                mask_image = self.model.encode_first_stage(mask_image, None)
                mask_image = mask_image.permute(0, 2, 1, 3, 4).contiguous()
                mask_image = torch.cat([mask_image, torch.zeros(pad_shape).to(mask_image.device).to(mask_image.dtype)], dim=1)

                video.append(samples[:, n_motion_frame:])

        video = torch.cat(video, dim=1)
        video = video[:, :length]
        video_np = (video[0].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

        if is_padding:
            h, w = video_np.shape[1], video_np.shape[2]
            crop_start = (w - h) // 2
            video_np = video_np[:, :, crop_start:crop_start + h, :]

        # Write video
        tmp_video = output_path + ".tmp.mp4"
        writer = imageio.get_writer(tmp_video, fps=fps, codec="libx264", quality=8)
        for frame in video_np:
            writer.append_data(frame)
        writer.close()

        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_video, "-i", audio_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart",
            output_path,
        ], capture_output=True, timeout=120)

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        elapsed = time.time() - t_start
        print(f"Generated: {len(video_bytes) / 1e6:.1f} MB, {video_np.shape[0]} frames, {elapsed:.1f}s")

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

        return video_bytes

    @modal.fastapi_endpoint(method="POST")
    def api_generate(self, request: dict):
        """HTTP endpoint: POST with {image_base64, audio_base64, prompt}."""
        import base64

        image_b64 = request.get("image_base64", "")
        audio_b64 = request.get("audio_base64", "")
        prompt = request.get("prompt", "A person talking")

        if not image_b64 or not audio_b64:
            return {"error": "image_base64 and audio_base64 are required"}

        image_bytes = base64.b64decode(image_b64)
        audio_bytes = base64.b64decode(audio_b64)

        video_bytes = self.generate.local(image_bytes, audio_bytes, prompt)

        return {
            "video_base64": base64.b64encode(video_bytes).decode("ascii"),
            "size_mb": round(len(video_bytes) / 1e6, 2),
        }


# One-time model download (run with: modal run modal_app.py::download)
@app.function(
    image=hallo3_image,
    volumes={MODELS_PATH: model_volume},
    timeout=3600,
)
def download():
    """Download models to the volume. Run once: modal run modal_app.py::download"""
    download_models()
    model_volume.commit()
    print("Models downloaded and committed to volume!")


@app.function(
    image=hallo3_image,
    gpu="H100",
    timeout=600,
    volumes={MODELS_PATH: model_volume},
)
def test_imports():
    """Quick test that all imports work. Run: modal run modal_app.py::test_imports"""
    import sys
    import os
    sys.path.insert(0, HALLO3_ROOT)
    sys.path.insert(0, os.path.join(HALLO3_ROOT, "hallo3"))
    print(f"sys.path includes: {[p for p in sys.path if 'hallo3' in p]}")
    print(f"Files in /app/hallo3/hallo3/: {os.listdir('/app/hallo3/hallo3/')[:10]}")

    from arguments import get_args
    print("OK: arguments imported")
    from diffusion_video import SATVideoDiffusionEngine
    print("OK: diffusion_video imported")
    from sat.model.base_model import get_model
    print("OK: sat imported")
    print("All imports OK!")


@app.local_entrypoint()
def test_generate():
    """Test full generation pipeline. Run: modal run modal_app.py"""
    import base64
    import os

    media_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
    image_path = os.path.join(media_dir, "ai_4ab86e0c.png")
    audio_path = os.path.join(media_dir, "ElevenLabs_2026-03-11T16_20_57_Bella - Professional, Bright, Warm_pre_sp100_s40_sb70_se55_b_m2.mp3")

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    print(f"Image: {len(image_bytes)//1024} KB, Audio: {len(audio_bytes)//1024} KB")
    print("Calling Hallo3.generate (cold start will load models)...")

    hallo3 = Hallo3()
    video_bytes = hallo3.generate.remote(image_bytes, audio_bytes, "A person talking")

    output_path = "output_modal.mp4"
    with open(output_path, "wb") as f:
        f.write(video_bytes)
    print(f"Success! Output: {output_path} ({len(video_bytes)/1e6:.2f} MB)")
