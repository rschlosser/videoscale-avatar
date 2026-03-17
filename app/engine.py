"""
Avatar generation engine — wraps Hallo3 (CogVideoX-5B DiT).

State-of-the-art audio-driven portrait animation using a Video Diffusion
Transformer architecture. Generates talking-head videos from a single
portrait image + audio clip.

Pipeline:
  1. AudioProcessor: audio → wav2vec2 embeddings (vocal isolation + feature extraction)
  2. ImageProcessor: image → face embedding (InsightFace) + face mask (MediaPipe)
  3. SATVideoDiffusionEngine: chunked diffusion sampling → video frames
  4. FFmpeg: mux video + audio → final MP4
"""

import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

HALLO3_ROOT = Path(os.getenv("HALLO3_ROOT", "/app/hallo3"))
PRETRAINED_DIR = Path(os.getenv("PRETRAINED_DIR", str(HALLO3_ROOT / "pretrained_models")))

# Ensure hallo3 is importable
if str(HALLO3_ROOT) not in sys.path:
    sys.path.insert(0, str(HALLO3_ROOT))


class AvatarEngine:
    """Hallo3-based audio-driven talking-head video generator.

    Uses CogVideoX-5B DiT architecture with audio conditioning for
    high-quality portrait animation.
    """

    def __init__(self):
        self.models_loaded = False
        self.model = None
        self.audio_processor = None
        self.image_processor = None
        self.args = None

    def load_models(self):
        """Load all model weights into GPU memory.

        Called once at startup. Subsequent generate() calls reuse loaded models.
        Takes ~60-90s on H100.
        """
        t0 = time.time()

        # Verify pretrained models exist
        required = [
            PRETRAINED_DIR / "hallo3" / "1" / "mp_rank_00_model_states.pt",
            PRETRAINED_DIR / "cogvideox-5b-i2v-sat" / "vae" / "3d-vae.pt",
        ]
        for path in required:
            if not path.exists():
                raise RuntimeError(f"Missing checkpoint: {path}")

        # Load configs via Hallo3's argument system
        from arguments import get_args
        from diffusion_video import SATVideoDiffusionEngine
        from sat.model.base_model import get_model
        from sat.training.model_io import load_checkpoint

        config_base = str(HALLO3_ROOT / "configs" / "cogvideox_5b_i2v_s2.yaml")
        config_infer = str(HALLO3_ROOT / "configs" / "inference.yaml")

        self.args = get_args(["--base", config_base, config_infer])

        # Override for single-GPU inference
        self.args.model_config.first_stage_config.params.cp_size = 1
        self.args.model_config.network_config.params.transformer_args.model_parallel_size = 1
        self.args.model_config.network_config.params.transformer_args.checkpoint_activations = False
        self.args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

        logger.info("Building Hallo3 model...")
        self.model = get_model(self.args, SATVideoDiffusionEngine)
        load_checkpoint(self.model, self.args)
        self.model.eval()
        self.model = self.model.to("cuda")
        logger.info("Hallo3 model loaded (%.1fs)", time.time() - t0)

        # Audio processor (wav2vec2 + vocal separator)
        t1 = time.time()
        from hallo3.sgm.utils.audio_processor import AudioProcessor

        infer_cfg = self.args.model_config.get("inference_config", {})
        sample_rate = infer_cfg.get("sample_rate", 16000)
        wav2vec_features = infer_cfg.get("wav2vec_features", "all")

        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            wav2vec_model_path=str(PRETRAINED_DIR / "wav2vec" / "wav2vec2-base-960h"),
            only_last_features=(wav2vec_features != "all"),
            audio_separator_model_path=str(PRETRAINED_DIR / "audio_separator"),
            audio_separator_model_name="Kim_Vocal_2.onnx",
            cache_dir=str(Path(tempfile.gettempdir()) / "hallo3_audio_cache"),
        )
        logger.info("AudioProcessor loaded (%.1fs)", time.time() - t1)

        # Image processor (InsightFace + MediaPipe)
        t2 = time.time()
        from hallo3.sgm.utils.image_processor import ImageProcessor

        self.image_processor = ImageProcessor(
            face_analysis_model_path=str(PRETRAINED_DIR / "face_analysis"),
        )
        logger.info("ImageProcessor loaded (%.1fs)", time.time() - t2)

        self.models_loaded = True
        logger.info("All Hallo3 models loaded in %.1fs", time.time() - t0)

    async def generate(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        prompt: str = "A person talking",
        resolution: str = "480p",
    ) -> str:
        """Generate a talking-head video from image + audio.

        Args:
            image_path: Path to portrait image (JPG/PNG).
            audio_path: Path to audio file (WAV, 16kHz preferred).
            output_path: Where to write the output MP4.
            prompt: Text description for T5 conditioning.
            resolution: Target resolution (currently only "480p" — Hallo3 native).

        Returns:
            Path to the generated video.
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded — call load_models() first")

        import asyncio

        return await asyncio.to_thread(
            self._generate_sync, image_path, audio_path, output_path, prompt, resolution
        )

    def _generate_sync(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        prompt: str,
        resolution: str,
    ) -> str:
        """Synchronous generation — runs in a thread."""
        import imageio
        import torchvision.transforms as TT
        from PIL import Image

        from hallo3.sgm.utils.util import (
            add_mask_to_first_frame,
            process_audio_emb,
            resize_for_rectangle_crop,
            resize_for_square_padding,
        )

        logger.info(
            "Generating: image=%s, audio=%s, prompt='%s'",
            image_path, audio_path, prompt,
        )

        t_start = time.time()

        # --- Config ---
        T = self.args.sampling_num_frames  # 13 latent frames
        L = (T - 1) * 4 + 1  # 49 video frames per chunk
        fps = self.args.sampling_fps  # 25
        image_size = [480, 720]  # H, W — Hallo3 native resolution
        n_motion_frame = 2
        mask_rate = 0.1
        latent_channels = self.args.latent_channels  # 16

        # --- Step 1: Audio processing ---
        logger.info("Step 1/4: Processing audio...")
        t1 = time.time()
        audio_emb, audio_length = self.audio_processor.preprocess(
            audio_path, clip_length=L, fps=fps,
        )
        logger.info("Audio processed: %d frames at %d fps (%.1fs)", audio_length, fps, time.time() - t1)

        # Apply 5-frame temporal windowing
        audio_emb = process_audio_emb(audio_emb)

        # --- Step 2: Image processing ---
        logger.info("Step 2/4: Processing image...")
        t2 = time.time()

        cache_dir = Path(tempfile.gettempdir()) / "hallo3_image_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        face_emb, face_mask_path = self.image_processor.preprocess(
            image_path, str(cache_dir), face_region_ratio=1.2,
        )
        face_emb = torch.tensor(face_emb.reshape(1, -1)).to("cuda")
        logger.info("Image processed (%.1fs)", time.time() - t2)

        # Load and prepare image tensors
        transform = TT.Compose([TT.ToTensor()])
        image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to("cuda")
        face_mask = transform(Image.open(face_mask_path).convert("RGB")).unsqueeze(0).to("cuda")
        ref_image = image * face_mask

        # Detect aspect ratio and resize
        _, _, orig_h, orig_w = image.shape
        is_square = abs(orig_h - orig_w) < min(orig_h, orig_w) * 0.1
        is_padding = False

        if is_square:
            image = resize_for_square_padding(image, image_size)
            ref_image = resize_for_square_padding(ref_image, image_size)
            is_padding = True
        else:
            image = resize_for_rectangle_crop(image, image_size, reshape_mode="center")
            ref_image = resize_for_rectangle_crop(ref_image, image_size, reshape_mode="center")

        # Normalize to [-1, 1]
        image = image * 2.0 - 1.0
        ref_image = ref_image * 2.0 - 1.0

        # --- Step 3: Text conditioning ---
        logger.info("Step 3/4: Setting up conditioning...")

        # Build conditioning batch — get text embeddings via T5
        value_dict = {
            "prompt": prompt,
            "negative_prompt": "",
            "num_frames": torch.tensor(T).unsqueeze(0),
        }
        batch, batch_uc = self.model.conditioner.get_batch(value_dict)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to("cuda")
                batch_uc[key] = batch_uc[key].to("cuda")

        c, uc = self.model.conditioner.get_unconditional_conditioning(
            batch, batch_uc=batch_uc, force_uc_zero_embeddings=["txt"],
        )

        # --- Step 4: Chunked diffusion generation ---
        n_chunks = math.ceil(audio_emb.shape[0] / (L - n_motion_frame))
        logger.info(
            "Step 4/4: Generating %d chunks (%d total frames)...",
            n_chunks, audio_length,
        )

        # Encode reference image through VAE
        ref_image_pixel = ref_image.unsqueeze(2).to(torch.bfloat16)  # [B, C, 1, H, W]
        ref_image_encoded = self.model.encode_first_stage(ref_image_pixel, None)

        # First frame conditioning (image encoded through VAE)
        image_encoded = self.model.encode_first_stage(
            image.unsqueeze(2).to(torch.bfloat16), None
        )

        all_frames = []
        motion_image = None
        pre_fix = torch.zeros_like(audio_emb[:n_motion_frame])

        for chunk_idx in range(n_chunks):
            t_chunk = time.time()

            # Build motion conditioning from previous chunk's last frames
            if motion_image is not None:
                # Apply random masking
                masked_motion = add_mask_to_first_frame(motion_image, mask_rate)
                # Concatenate with ref image, encode, zero-pad to T frames
                ref_expanded = ref_image.unsqueeze(2).expand_as(masked_motion)
                motion_with_ref = torch.cat([masked_motion, ref_expanded], dim=1)
                motion_encoded = self.model.encode_first_stage(
                    motion_with_ref.to(torch.bfloat16), None
                )
                # Pad to full T length
                pad_size = T - motion_encoded.shape[2]
                if pad_size > 0:
                    padding = torch.zeros(
                        *motion_encoded.shape[:2], pad_size, *motion_encoded.shape[3:],
                        device=motion_encoded.device, dtype=motion_encoded.dtype,
                    )
                    mask_concat = torch.cat([motion_encoded, padding], dim=2)
                else:
                    mask_concat = motion_encoded
            else:
                # First chunk — use encoded source image
                pad_size = T - image_encoded.shape[2]
                if pad_size > 0:
                    padding = torch.zeros(
                        *image_encoded.shape[:2], pad_size, *image_encoded.shape[3:],
                        device=image_encoded.device, dtype=image_encoded.dtype,
                    )
                    mask_concat = torch.cat([image_encoded, padding], dim=2)
                else:
                    mask_concat = image_encoded

            c["concat"] = mask_concat
            uc["concat"] = mask_concat

            # Slice audio for this chunk (with overlap)
            start = chunk_idx * (L - n_motion_frame)
            end = start + (L - n_motion_frame)
            audio_chunk = audio_emb[start:end]

            # Pad if last chunk is short
            if audio_chunk.shape[0] < (L - n_motion_frame):
                pad_len = (L - n_motion_frame) - audio_chunk.shape[0]
                audio_chunk = torch.cat([
                    audio_chunk,
                    torch.zeros(pad_len, *audio_chunk.shape[1:], device=audio_chunk.device, dtype=audio_chunk.dtype),
                ], dim=0)

            audio_tensor = torch.cat([pre_fix, audio_chunk], dim=0)
            pre_fix = audio_tensor[-n_motion_frame:]

            # Run diffusion sampling
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                shape = (T, latent_channels, image_size[0] // 8, image_size[1] // 8)
                samples_z = self.model.sample(
                    c, uc=uc, batch_size=1, shape=shape,
                    audio_emb=audio_tensor.unsqueeze(0).to("cuda"),
                    ref_image=ref_image_encoded,
                    face_emb=face_emb,
                )

            # Decode latent → pixels (serial for VRAM savings)
            latent = samples_z / self.model.scale_factor
            recons = []
            loop_num = (T - 1) // 2
            for i in range(loop_num):
                start_frame = i * 2
                end_frame = min((i + 1) * 2 + 1, latent.shape[2])
                recon = self.model.first_stage_model.decode(
                    latent[:, :, start_frame:end_frame]
                )
                recons.append(recon)
            samples = torch.cat(recons, dim=2)
            samples = torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0)

            # Save last frames as motion seed for next chunk
            motion_image = samples[:, :, -n_motion_frame:]

            # Append non-overlapping frames
            if chunk_idx == 0:
                all_frames.append(samples)
            else:
                all_frames.append(samples[:, :, n_motion_frame:])

            logger.info(
                "Chunk %d/%d done (%.1fs)",
                chunk_idx + 1, n_chunks, time.time() - t_chunk,
            )

        # Concatenate all chunks and trim to exact audio length
        video = torch.cat(all_frames, dim=2)  # [B, C, T_total, H, W]
        video = video[:, :, :audio_length]

        # Convert to numpy frames [T, H, W, 3] (uint8 RGB)
        video_np = (video[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)

        # If input was square (padded), crop back to square
        if is_padding:
            h, w = video_np.shape[1], video_np.shape[2]
            crop_start = (w - h) // 2
            video_np = video_np[:, :, crop_start:crop_start + h, :]

        # Write video frames
        logger.info("Writing %d frames to video...", video_np.shape[0])
        tmp_video = output_path + ".tmp.mp4"
        writer = imageio.get_writer(tmp_video, fps=fps, codec="libx264", quality=8)
        for frame in video_np:
            writer.append_data(frame)
        writer.close()

        # Mux audio with ffmpeg
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", tmp_video,
                "-i", audio_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-shortest",
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True, text=True, timeout=120,
        )
        os.unlink(tmp_video)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

        elapsed = time.time() - t_start
        logger.info(
            "Generation complete: %s (%d frames, %.1fs)",
            output_path, video_np.shape[0], elapsed,
        )
        return output_path
