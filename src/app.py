import os
import sys
import uuid
import shutil
import base64
import requests # For downloading files from URLs
import subprocess

# --- Original project imports ---
import importlib, site
from glob import glob
from datetime import datetime
import math
import random
import librosa
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial
from omegaconf import OmegaConf
from argparse import Namespace

# This part of the code runs only once when the worker starts.
print("Attempting to install FlashAttention...")
try:
    flash_attention_wheel = hf_hub_download(
        repo_id="alexnasa/flash-attn-3",
        repo_type="model",
        filename="128/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl",
    )
    sh(f"pip install {flash_attention_wheel}")
    importlib.invalidate_caches()
    print("FlashAttention installed successfully.")
except Exception as e:
    print(f"⚠️ Could not install FlashAttention: {e}")

print("Loading args_config.yaml...")
_args_cfg = OmegaConf.load("args_config.yaml")
args = Namespace(**OmegaConf.to_container(_args_cfg, resolve=True))
set_global_args(args)
set_seed(args.seed)
os.environ["PROCESSED_RESULTS"] = f"{os.getcwd()}/runpod_results"
print("Config loaded.")


import torchaudio
import torchvision.transforms as TT
from transformers import Wav2Vec2FeatureExtractor
import torchvision.transforms as transforms
import torch.nn.functional as F

from huggingface_hub import hf_hub_download, snapshot_download

# --- Re-discover packages ---
for sitedir in site.getsitepackages():
    site.addsitedir(sitedir)
importlib.invalidate_caches()

from OmniAvatar.utils.args_config import set_global_args
from OmniAvatar.utils.io_utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.schedulers.flow_match import FlowMatchScheduler
from OmniAvatar.wan_video import WanVideoPipeline
from OmniAvatar.utils.io_utils import save_video_as_grid_and_mp4
from OmniAvatar.utils.audio_preprocess import add_silence_to_audio_ffmpeg

# --- Helper Functions ---

def sh(cmd):
    """Executes a shell command."""
    subprocess.check_call(cmd, shell=True)

def download_file(url, output_dir, file_extension):
    """Downloads a file from a URL and saves it locally."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"input.{file_extension}")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    return file_path

def file_to_base64(file_path):
    """Reads a file and encodes it into a base64 string."""
    with open(file_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode('utf-8')
    return encoded_string

def tensor_to_pil(tensor):
    if tensor.dim() > 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    tensor = tensor.squeeze()
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3 dims after squeeze, got {tensor.dim()}")
    tensor = tensor.cpu().float()
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)
    np_img = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(np_img)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_from_file(p):
    with open(p, "r") as fin:
        for l in fin:
            yield l.strip()

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

# --- Main Inference Class ---

class WanInferencePipeline(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(f"cuda")
        self.dtype = torch.bfloat16
        self.pipe = self.load_model()
        chained_trainsforms = []
        chained_trainsforms.append(TT.ToTensor())
        self.transform = TT.Compose(chained_trainsforms)
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
        self.step = 0
        model_manager = ModelManager(device="cuda", infer=True)
        model_manager.load_models(
            [
                self.args.dit_path.split(","),
                self.args.vae_path,
                self.args.text_encoder_path
            ],
            torch_dtype=self.dtype,
            device='cuda',
        )
        pipe = WanVideoPipeline.from_model_manager(model_manager,
                                                   torch_dtype=self.dtype,
                                                   device="cuda",
                                                   use_usp=False,
                                                   infer=True)
        if self.args.train_architecture == "lora":
            print(f'Use LoRA: lora rank: {self.args.lora_rank}, lora alpha: {self.args.lora_alpha}')
            self.add_lora_to_model(
                pipe.denoising_model(),
                lora_rank=self.args.lora_rank,
                lora_alpha=self.args.lora_alpha,
                lora_target_modules=self.args.lora_target_modules,
                init_lora_weights=self.args.init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
            print(next(pipe.denoising_model().parameters()).device)
        else:
            missing_keys, unexpected_keys = pipe.denoising_model().load_state_dict(load_state_dict(resume_path), strict=True)
            print(f"load from {resume_path}, {len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys")
        pipe.requires_grad_(False)
        pipe.eval()
        return pipe

    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path, torch_dtype=self.dtype)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")

    @torch.no_grad()
    def forward(self, prompt, image_path=None, audio_path=None, orientation_state = None,
                seq_len=101, height=720, width=720, overlap_frame=None, num_steps=None,
                negative_prompt=None, guidance_scale=None, audio_scale=None):
        overlap_frame = overlap_frame if overlap_frame is not None else self.args.overlap_frame
        num_steps = num_steps if num_steps is not None else self.args.num_steps
        negative_prompt = negative_prompt if negative_prompt is not None else self.args.negative_prompt
        guidance_scale = guidance_scale if guidance_scale is not None else self.args.guidance_scale
        audio_scale = audio_scale if audio_scale is not None else self.args.audio_scale

        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device, dtype=self.dtype)
            _, _, h, w = image.shape
            select_size = match_size(orientation_state, h, w)
            image = resize_pad(image, (h, w), select_size)
            image = image * 2.0 - 1.0
            image = image[:, :, None]
        else:
            image = None
            select_size = [height, width]

        num = self.args.max_tokens * 16 * 16 * 4
        den = select_size[0] * select_size[1]
        L0 = num // den
        diff = (L0 - 1) % 4
        L = L0 - diff
        if L < 1:
            L = 1
        T = (L + 3) // 4

        if self.args.i2v:
            if self.args.random_prefix_frames:
                fixed_frame = overlap_frame
                assert fixed_frame % 4 == 1
            else:
                fixed_frame = 1
            prefix_lat_frame = (3 + fixed_frame) // 4
            first_fixed_frame = 1
        else:
            fixed_frame = 0
            prefix_lat_frame = 0
            first_fixed_frame = 0

        if audio_path is not None and self.args.use_audio:
            audio, sr = librosa.load(audio_path, sr=self.args.sample_rate)
            input_values = np.squeeze(
                self.wav_feature_extractor(audio, sampling_rate=16000).input_values
            )
            input_values = torch.from_numpy(input_values).float().to(device=self.device, dtype=self.dtype)
            ori_audio_len = audio_len = math.ceil(len(input_values) / self.args.sample_rate * self.args.fps)
            input_values = input_values.unsqueeze(0)
            if audio_len < L - first_fixed_frame:
                audio_len = audio_len + ((L - first_fixed_frame) - audio_len % (L - first_fixed_frame))
            elif (audio_len - (L - first_fixed_frame)) % (L - fixed_frame) != 0:
                audio_len = audio_len + ((L - fixed_frame) - (audio_len - (L - first_fixed_frame)) % (L - fixed_frame))
            input_values = F.pad(input_values, (0, audio_len * int(self.args.sample_rate / self.args.fps) - input_values.shape[1]), mode='constant', value=0)
            with torch.no_grad():
                hidden_states = self.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
                audio_embeddings = hidden_states.last_hidden_state
                for mid_hidden_states in hidden_states.hidden_states:
                    audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
            seq_len = audio_len
            audio_embeddings = audio_embeddings.squeeze(0)
            audio_prefix = torch.zeros_like(audio_embeddings[:first_fixed_frame])
        else:
            audio_embeddings = None

        times = (seq_len - L + first_fixed_frame) // (L - fixed_frame) + 1
        if times * (L - fixed_frame) + fixed_frame < seq_len:
            times += 1
        video = []
        image_emb = {}
        img_lat = None
        if self.args.i2v:
            self.pipe.load_models_to_device(['vae'])
            img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device, dtype=self.dtype)
            msk = torch.zeros_like(img_lat.repeat(1, 1, T, 1, 1)[:, :1], dtype=self.dtype)
            image_cat = img_lat.repeat(1, 1, T, 1, 1)
            msk[:, :, 1:] = 1
            image_emb["y"] = torch.cat([image_cat, msk], dim=1)
        
        with tqdm(total=times * num_steps) as pbar:
            for t in range(times):
                print(f"[{t+1}/{times}]")
                audio_emb = {}
                if t == 0:
                    overlap = first_fixed_frame
                else:
                    overlap = fixed_frame
                    image_emb["y"][:, -1:, :prefix_lat_frame] = 0
                prefix_overlap = (3 + overlap) // 4
                if audio_embeddings is not None:
                    if t == 0:
                        audio_tensor = audio_embeddings[:min(L - overlap, audio_embeddings.shape[0])]
                    else:
                        audio_start = L - first_fixed_frame + (t - 1) * (L - overlap)
                        audio_tensor = audio_embeddings[audio_start: min(audio_start + L - overlap, audio_embeddings.shape[0])]
                    audio_tensor = torch.cat([audio_prefix, audio_tensor], dim=0)
                    audio_prefix = audio_tensor[-fixed_frame:]
                    audio_tensor = audio_tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype)
                    audio_emb["audio_emb"] = audio_tensor
                else:
                    audio_prefix = None
                
                if image is not None and img_lat is None:
                    self.pipe.load_models_to_device(['vae'])
                    img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device, dtype=self.dtype)
                    assert img_lat.shape[2] == prefix_overlap
                img_lat = torch.cat([img_lat, torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T - prefix_overlap, 1, 1), dtype=self.dtype)], dim=2)
                
                frames, _, latents = self.pipe.log_video(img_lat, prompt, prefix_overlap, image_emb, audio_emb,
                                                         negative_prompt, num_inference_steps=num_steps,
                                                         cfg_scale=guidance_scale, audio_cfg_scale=audio_scale if audio_scale is not None else guidance_scale,
                                                         return_latent=True,
                                                         tea_cache_l1_thresh=self.args.tea_cache_l1_thresh, tea_cache_model_id="Wan2.1-T2V-14B", progress_bar_cmd=pbar)
                
                torch.cuda.empty_cache()
                img_lat = None
                image = (frames[:, -fixed_frame:].clip(0, 1) * 2.0 - 1.0).permute(0, 2, 1, 3, 4).contiguous()
                
                if t == 0:
                    video.append(frames)
                else:
                    video.append(frames[:, overlap:])
        
        video = torch.cat(video, dim=1)
        video = video[:, :ori_audio_len + 1]
        return video


# --- Global Model Initialization ---

print("Downloading models...")
snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-14B", local_dir="./pretrained_models/Wan2.1-T2V-14B")
snapshot_download(repo_id="facebook/wav2vec2-base-960h", local_dir="./pretrained_models/wav2vec2-base-960h")
snapshot_download(repo_id="OmniAvatar/OmniAvatar-14B", local_dir="./pretrained_models/OmniAvatar-14B")
print("Model download complete.")

print("Initializing WanInferencePipeline...")
inferpipe = WanInferencePipeline(args)
print("✅ Inference pipeline ready.")


# --- RunPod Handler ---

def handler(job):
    """
    This is the main handler function for the RunPod serverless worker.
    """
    job_input = job['input']
    session_id = uuid.uuid4().hex
    output_dir = os.path.join(os.environ["PROCESSED_RESULTS"], session_id)
    
    try:
        # --- 1. Parse and Validate Input ---
        image_url = job_input.get("image_url")
        audio_url = job_input.get("audio_url")

        if not image_url or not audio_url:
            return {"error": "Missing 'image_url' or 'audio_url' in input."}
            
        text = job_input.get("prompt", "A realistic video of a person speaking.")
        num_steps = job_input.get("num_steps", 8)
        
        aspect_ratio = job_input.get("aspect_ratio", "9:16")
        if aspect_ratio == "9:16":
            orientation_state = [[720, 400]]
        elif aspect_ratio == "1:1":
            orientation_state = [[720, 720]]
        elif aspect_ratio == "16:9":
            orientation_state = [[400, 720]]
        else:
            orientation_state = [[720, 400]] # Default

        # --- 2. Download Input Files ---
        print(f"Downloading files for job {session_id}...")
        # Determine file extension from URL if possible, otherwise guess
        img_ext = os.path.splitext(image_url.split('?')[0])[-1][1:] or 'png'
        audio_ext = os.path.splitext(audio_url.split('?')[0])[-1][1:] or 'wav'
        
        image_path = download_file(image_url, output_dir, img_ext)
        audio_path = download_file(audio_url, output_dir, audio_ext)

        # --- 3. Preprocess Inputs ---
        input_audio_path = audio_path
        if args.silence_duration_s > 0:
            audio_dir = os.path.join(output_dir, 'audio')
            os.makedirs(audio_dir, exist_ok=True)
            input_audio_path = os.path.join(audio_dir, "audio_input.wav")
            add_silence_to_audio_ffmpeg(audio_path, input_audio_path, args.silence_duration_s)
        
        # --- 4. Run Inference ---
        print(f"Starting inference for job {session_id} with {num_steps} steps...")
        video_tensor = inferpipe(
            prompt=text,
            image_path=image_path,
            audio_path=input_audio_path,
            orientation_state=orientation_state,
            seq_len=args.seq_len,
            num_steps=num_steps
        )
        torch.cuda.empty_cache()

        # --- 5. Post-process and Save Output Video ---
        final_audio_path = os.path.join(output_dir, "audio_out.wav")
        add_silence_to_audio_ffmpeg(audio_path, final_audio_path, 1.0 / args.fps + args.silence_duration_s)

        video_paths = save_video_as_grid_and_mp4(
            video_tensor,
            output_dir,
            args.fps,
            prompt=text,
            prompt_path=os.path.join(output_dir, "prompt.txt"),
            audio_path=final_audio_path if args.use_audio else None,
            prefix='result'
        )
        output_video_path = video_paths[0]
        
        # --- 6. Encode Output for JSON Response ---
        video_base64 = file_to_base64(output_video_path)

        return {
            "video_base64": video_base64
        }

    except Exception as e:
        # It's good practice to log the full traceback for debugging
        import traceback
        print(traceback.format_exc())
        return {"error": f"An error occurred: {str(e)}"}
    
    finally:
        # --- 7. Cleanup ---
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)