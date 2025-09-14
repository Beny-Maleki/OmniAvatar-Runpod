import os
import sys
import uuid
import random
import math
import librosa
import numpy as np

import importlib, site, sys
from huggingface_hub import snapshot_download
for sitedir in site.getsitepackages():
    site.addsitedir(sitedir)

# Clear caches so importlib will pick up new modules
importlib.invalidate_caches()

import torch
print(f"Torch version: {torch.__version__}")
import torch.nn as nn
import torchvision.transforms as TT
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from functools import partial
from argparse import Namespace
from transformers import Wav2Vec2FeatureExtractor
import runpod
import requests
import tempfile

from peft import LoraConfig, inject_adapter_in_model
from omegaconf import OmegaConf

_args_cfg = OmegaConf.load("args_config.yaml")
args = Namespace(**OmegaConf.to_container(_args_cfg, resolve=True))

from OmniAvatar.utils.args_config import set_global_args

set_global_args(args)

from OmniAvatar.utils.io_utils import load_state_dict, save_video_as_grid_and_mp4
from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.schedulers.flow_match import FlowMatchScheduler
from OmniAvatar.wan_video import WanVideoPipeline
from OmniAvatar.utils.audio_preprocess import add_silence_to_audio_ffmpeg

# --- Helper Functions (copied from app.py) ---
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def match_size(image_size, h, w):
    ratio_ = 9999
    size_ = 9999
    select_size = None
    for image_s in image_size:
        ratio_tmp = abs(image_s[0] / image_s[1] - h / w)
        size_tmp = abs(max(image_s) - max(w, h))
        if ratio_tmp < ratio_:
            ratio_ = ratio_tmp
            size_ = size_tmp
            select_size = image_s
        if ratio_ == ratio_tmp:
            if size_ == size_tmp:
                select_size = image_s
    return select_size

def resize_pad(image, ori_size, tgt_size):
    h, w = ori_size
    scale_ratio = max(tgt_size[0] / h, tgt_size[1] / w)
    scale_h = int(h * scale_ratio)
    scale_w = int(w * scale_ratio)
    image = transforms.Resize(size=[scale_h, scale_w])(image)
    padding_h = tgt_size[0] - scale_h
    padding_w = tgt_size[1] - scale_w
    pad_top = padding_h // 2
    pad_bottom = padding_h - pad_top
    pad_left = padding_w // 2
    pad_right = padding_w - pad_left
    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return image

class WanInferencePipeline(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.pipe = self.load_model()
        self.transform = TT.Compose([TT.ToTensor()])

        if self.args.use_audio:
            from OmniAvatar.models.wav2vec import Wav2VecModel
            self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.args.wav2vec_path
            )
            self.audio_encoder = Wav2VecModel.from_pretrained(self.args.wav2vec_path, local_files_only=True).to(device=self.device, dtype=self.dtype)
            self.audio_encoder.feature_extractor._freeze_parameters()

    def load_model(self):
        ckpt_path = f'{self.args.exp_path}/pytorch_model.pt'
        assert os.path.exists(ckpt_path), f"pytorch_model.pt not found in {self.args.exp_path}"
        if self.args.train_architecture == 'lora':
            self.args.pretrained_lora_path = pretrained_lora_path = ckpt_path
        else:
            resume_path = ckpt_path
        
        model_manager = ModelManager(device="cuda", infer=True)
        model_manager.load_models(
            [self.args.dit_path.split(","), self.args.vae_path, self.args.text_encoder_path],
            torch_dtype=self.dtype, device='cuda'
        )
        pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=self.dtype, device="cuda", use_usp=False, infer=True)

        if self.args.train_architecture == "lora":
            self.add_lora_to_model(pipe.denoising_model())
        else:
            pipe.denoising_model().load_state_dict(load_state_dict(resume_path), strict=True)
        
        pipe.requires_grad_(False)
        pipe.eval()
        return pipe
    
    def add_lora_to_model(self, model):
        lora_config = LoraConfig(
            r=self.args.lora_rank, lora_alpha=self.args.lora_alpha,
            init_lora_weights="kaiming", target_modules=self.args.lora_target_modules.split(",")
        )
        model = inject_adapter_in_model(lora_config, model)
        if self.args.pretrained_lora_path is not None:
            state_dict = load_state_dict(self.args.pretrained_lora_path, torch_dtype=self.dtype)
            model.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def forward(self, prompt, image_path, audio_path, num_steps, guidance_scale, audio_scale, negative_prompt):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device, dtype=self.dtype)
        _, _, h, w = image_tensor.shape
        # Assuming 1:1 aspect ratio for simplicity, you can adjust this
        select_size = match_size(self.args.image_sizes_720, h, w) 
        image_tensor = resize_pad(image_tensor, (h, w), select_size)
        image_tensor = image_tensor * 2.0 - 1.0
        image_tensor = image_tensor[:, :, None]

        num = self.args.max_tokens * 16 * 16 * 4
        den = select_size[0] * select_size[1]
        L0 = num // den
        diff = (L0 - 1) % 4
        L = L0 - diff
        if L < 1: L = 1
        T = (L + 3) // 4

        fixed_frame = self.args.overlap_frame if self.args.random_prefix_frames else 1
        prefix_lat_frame = (3 + fixed_frame) // 4
        first_fixed_frame = 1

        audio, sr = librosa.load(audio_path, sr=self.args.sample_rate)
        input_values = np.squeeze(self.wav_feature_extractor(audio, sampling_rate=16000).input_values)
        input_values = torch.from_numpy(input_values).float().to(device=self.device, dtype=self.dtype)
        ori_audio_len = audio_len = math.ceil(len(input_values) / self.args.sample_rate * self.args.fps)
        input_values = input_values.unsqueeze(0)

        if audio_len < L - first_fixed_frame:
            audio_len += ((L - first_fixed_frame) - audio_len % (L - first_fixed_frame))
        elif (audio_len - (L - first_fixed_frame)) % (L - fixed_frame) != 0:
            audio_len += ((L - fixed_frame) - (audio_len - (L - first_fixed_frame)) % (L - fixed_frame))
        
        input_values = F.pad(input_values, (0, audio_len * int(self.args.sample_rate / self.args.fps) - input_values.shape[1]), mode='constant', value=0)
        
        with torch.no_grad():
            hidden_states = self.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
            audio_embeddings = hidden_states.last_hidden_state
            for mid_hidden_states in hidden_states.hidden_states:
                audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
        
        seq_len = audio_len
        audio_embeddings = audio_embeddings.squeeze(0)
        audio_prefix = torch.zeros_like(audio_embeddings[:first_fixed_frame])

        times = (seq_len - L + first_fixed_frame) // (L - fixed_frame) + 1
        if times * (L - fixed_frame) + fixed_frame < seq_len:
            times += 1
        
        video = []
        img_lat = self.pipe.encode_video(image_tensor.to(dtype=self.dtype)).to(self.device, dtype=self.dtype)
        
        for t in tqdm(range(times), desc="Generating video chunks"):
            overlap = first_fixed_frame if t == 0 else fixed_frame
            prefix_overlap = (3 + overlap) // 4
            
            audio_start = 0 if t == 0 else (L - first_fixed_frame + (t - 1) * (L - overlap))
            audio_end = min(audio_start + L - overlap, audio_embeddings.shape[0])
            audio_tensor_chunk = audio_embeddings[audio_start:audio_end]
            audio_tensor = torch.cat([audio_prefix, audio_tensor_chunk], dim=0).unsqueeze(0).to(device=self.device, dtype=self.dtype)
            audio_prefix = audio_tensor[0, -fixed_frame:]
            
            current_lat = torch.cat([img_lat, torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T - prefix_overlap, 1, 1), dtype=self.dtype)], dim=2)
            
            image_emb = {"y": torch.cat([current_lat.repeat(1, 1, T, 1, 1), torch.ones_like(current_lat.repeat(1, 1, T, 1, 1)[:,:1])], dim=1)}
            if t > 0:
                image_emb["y"][:, -1:, :prefix_lat_frame] = 0

            frames, _, _ = self.pipe.log_video(
                current_lat, prompt, prefix_overlap, image_emb, {"audio_emb": audio_tensor},
                negative_prompt, num_inference_steps=num_steps,
                cfg_scale=guidance_scale, audio_cfg_scale=audio_scale,
                return_latent=True,
            )
            
            torch.cuda.empty_cache()
            
            image_tensor_next = (frames[:, -fixed_frame:].clip(0, 1) * 2.0 - 1.0).permute(0, 2, 1, 3, 4).contiguous()
            img_lat = self.pipe.encode_video(image_tensor_next.to(dtype=self.dtype)).to(self.device, dtype=self.dtype)

            video.append(frames[:, overlap:] if t > 0 else frames)

        video = torch.cat(video, dim=1)[:, :ori_audio_len + 1]
        return video

def download_and_cache_models():
    """
    Checks for models in the persistent /workspace volume and downloads them if they don't exist.
    """
    model_list = {
        "Wan-AI/Wan2.1-T2V-14B": "/runpod-volume/pretrained_models/Wan2.1-T2V-14B",
        "facebook/wav2vec2-base-960h": "/runpod-volume/pretrained_models/wav2vec2-base-960h",
        "OmniAvatar/OmniAvatar-14B": "/runpod-volume/pretrained_models/OmniAvatar-14B"
    }

    for repo_id, local_dir in model_list.items():
        if not os.path.exists(local_dir):
            print(f"Models not found in {local_dir}. Downloading from Hugging Face Hub...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
        else:
            print(f"Models already exist in {local_dir}. Skipping download.")
    print("All models are available in the persistent volume.")

def download_file(url):
    """Downloads a file from a URL to a temporary file."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(url)[1]) as tmp_file:
            for chunk in r.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            return tmp_file.name

set_seed(args.seed)

download_and_cache_models()

# Load model once when the worker starts
pipeline = WanInferencePipeline(args)

def handler(job):
    job_input = job['input']
    
    # --- Input Validation and Defaults ---
    prompt = job_input.get("prompt", "A realistic video of a person speaking.")
    num_steps = job_input.get("num_steps", args.num_steps)
    guidance_scale = job_input.get("guidance_scale", args.guidance_scale)
    audio_scale = job_input.get("audio_scale", args.audio_scale if args.audio_scale is not None else guidance_scale)
    negative_prompt = job_input.get("negative_prompt", args.negative_prompt)

    image_path = None
    audio_path = None

    try:
        # --- Handle Image Input ---
        if "image_file" in job_input:
            image_path = job_input["image_file"]
            print(f"Using uploaded image file: {image_path}")
        elif "image_url" in job_input:
            print(f"Downloading image from URL: {job_input['image_url']}")
            image_path = download_file(job_input["image_url"])
        else:
            return {"error": "Missing input. Provide 'image_file' for direct upload or 'image_url'."}

        # --- Handle Audio Input ---
        if "audio_file" in job_input:
            audio_path = job_input["audio_file"]
            print(f"Using uploaded audio file: {audio_path}")
        elif "audio_url" in job_input:
            print(f"Downloading audio from URL: {job_input['audio_url']}")
            audio_path = download_file(job_input["audio_url"])
        else:
            return {"error": "Missing input. Provide 'audio_file' for direct upload or 'audio_url'."}

        # Create a unique directory for outputs
        output_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        os.makedirs(output_dir, exist_ok=True)
        
        # Run inference
        video_tensor = pipeline(
            prompt=prompt,
            image_path=image_path,
            audio_path=audio_path,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            audio_scale=audio_scale,
            negative_prompt=negative_prompt
        )

        torch.cuda.empty_cache()
        
        # Save the output video using the original audio path for muxing
        video_paths = save_video_as_grid_and_mp4(
            video_tensor,
            output_dir,
            args.fps,
            prompt=prompt,
            audio_path=audio_path,
            prefix='result'
        )
        
        output_video_path = video_paths[0]

        # Note: RunPod automatically handles uploading the output file if it's a path.
        # Alternatively, you can return a URL if you upload it to your own storage.
        return {"video_path": output_video_path}

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Clean up files that were downloaded from URLs.
        # Files from direct uploads are managed by the RunPod environment.
        if "image_url" in job_input and image_path and os.path.exists(image_path):
            os.remove(image_path)
        if "audio_url" in job_input and audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})