import spaces
import subprocess
import gradio as gr

import os, sys
from glob import glob
from datetime import datetime
import math
import random
import librosa
import numpy as np
import uuid
import shutil
from tqdm import tqdm

import importlib, site, sys
from huggingface_hub import hf_hub_download, snapshot_download

# Re-discover all .pth/.egg-link files
for sitedir in site.getsitepackages():
    site.addsitedir(sitedir)

# Clear caches so importlib will pick up new modules
importlib.invalidate_caches()

def sh(cmd): subprocess.check_call(cmd, shell=True)

flash_attention_installed = False

try:
    print("Attempting to download and install FlashAttention wheel...")
    flash_attention_wheel = hf_hub_download(
            repo_id="alexnasa/flash-attn-3",
            repo_type="model",
            filename="128/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl",
        )

    sh(f"pip install {flash_attention_wheel}")

    # tell Python to re-scan site-packages now that the egg-link exists
    import importlib, site; site.addsitedir(site.getsitepackages()[0]); importlib.invalidate_caches()

    flash_attention_installed = True
    print("FlashAttention installed successfully.")

except Exception as e:
    print(f"⚠️ Could not install FlashAttention: {e}")
    print("Continuing without FlashAttention...")

import torch
print(f"Torch version: {torch.__version__}")
print(f"FlashAttention available: {flash_attention_installed}")


import torch.nn as nn
from tqdm import tqdm
from functools import partial
from omegaconf import OmegaConf
from argparse import Namespace
from gradio_extendedaudio import ExtendedAudio
from gradio_extendedimage import extendedimage

import torchaudio

# load the one true config you dumped
_args_cfg = OmegaConf.load("args_config.yaml")
args = Namespace(**OmegaConf.to_container(_args_cfg, resolve=True))

from OmniAvatar.utils.args_config import set_global_args

set_global_args(args)
# args = parse_args()

from OmniAvatar.utils.io_utils import load_state_dict 
from peft import LoraConfig, inject_adapter_in_model
from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.schedulers.flow_match import FlowMatchScheduler
from OmniAvatar.wan_video import WanVideoPipeline
from OmniAvatar.utils.io_utils import save_video_as_grid_and_mp4
import torchvision.transforms as TT
from transformers import Wav2Vec2FeatureExtractor
import torchvision.transforms as transforms
import torch.nn.functional as F
from OmniAvatar.utils.audio_preprocess import add_silence_to_audio_ffmpeg
from higgs_audio_utils import text_to_speech, initialize_engine


DEFAULT_TTS_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
DEFAULT_AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
engine = initialize_engine(DEFAULT_TTS_MODEL_PATH, DEFAULT_AUDIO_TOKENIZER_PATH)

os.environ["PROCESSED_RESULTS"] = f"{os.getcwd()}/proprocess_results"

@spaces.GPU
def tts_from_text(text):
    _, output = text_to_speech(engine, text)
    return output

def speak_to_me(session_id, evt: gr.EventData):
    detail = getattr(evt, "data", None) or getattr(evt, "_data", {}) or {}

    current_text = detail.get("text", "")

    output = tts_from_text(current_text)

    if session_id is None:
        session_id = uuid.uuid4().hex
        
    output_dir = os.path.join(os.environ["PROCESSED_RESULTS"], session_id)

    tts_dir = output_dir + '/tts'
    os.makedirs(tts_dir, exist_ok=True)
    speech_to_text_path = os.path.join(tts_dir, f"speech_to_text.wav")

    sampling_rate = output[0]
    audio_data = output[1]
    
    torchaudio.save(speech_to_text_path, torch.from_numpy(audio_data)[None, :], output[0])
    
    return speech_to_text_path

def tensor_to_pil(tensor):
    """
    Args:
        tensor: torch.Tensor with shape like
                (1, C, H, W), (1, C, 1, H, W), (C, H, W), etc.
                values in [-1, 1], on any device.
    Returns:
        A PIL.Image in RGB mode.
    """
    # 1) Remove batch dim if it exists
    if tensor.dim() > 3 and tensor.shape[0] == 1:
        tensor = tensor[0]

    # 2) Squeeze out any other singleton dims (e.g. that extra frame axis)
    tensor = tensor.squeeze()

    # Now we should have exactly 3 dims: (C, H, W)
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3 dims after squeeze, got {tensor.dim()}")

    # 3) Move to CPU float32
    tensor = tensor.cpu().float()

    # 4) Undo normalization from [-1,1] -> [0,1]
    tensor = (tensor + 1.0) / 2.0

    # 5) Clamp to [0,1]
    tensor = torch.clamp(tensor, 0.0, 1.0)

    # 6) To NumPy H×W×C in [0,255]
    np_img = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")

    # 7) Build PIL Image
    return Image.fromarray(np_img)
    
    
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 设置当前GPU
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU

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
        
        # Load models
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
        # pipe.enable_vram_management(num_persistent_param_in_dit=args.num_persistent_param_in_dit)
        return pipe
    
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet

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
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path, torch_dtype=self.dtype)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)

            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")

    def get_times(self, prompt, 
            image_path=None, 
            audio_path=None, 
            orientation_state = None,
            seq_len=101, # not used while audio_path is not None
            height=720, 
            width=720,
            overlap_frame=None,
            num_steps=None,
            negative_prompt=None,
            guidance_scale=None,
            audio_scale=None):
    
        overlap_frame = overlap_frame if overlap_frame is not None else  self.args.overlap_frame
        num_steps = num_steps if num_steps is not None else  self.args.num_steps
        negative_prompt = negative_prompt if negative_prompt is not None else  self.args.negative_prompt
        guidance_scale = guidance_scale if guidance_scale is not None else  self.args.guidance_scale
        audio_scale = audio_scale if audio_scale is not None else  self.args.audio_scale
    
        if image_path is not None:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
    
            image = self.transform(image).unsqueeze(0).to(dtype=self.dtype)
            
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
        L  = L0 - diff
        if L < 1:
            L = 1  
        T = (L + 3) // 4
    
    
        if  self.args.random_prefix_frames:
            fixed_frame = overlap_frame
            assert fixed_frame % 4 == 1
        else:
            fixed_frame = 1
        prefix_lat_frame = (3 + fixed_frame) // 4
        first_fixed_frame = 1
    
    
        audio, sr = librosa.load(audio_path, sr= self.args.sample_rate)

        input_values = np.squeeze(
            self.wav_feature_extractor(audio, sampling_rate=16000).input_values
        )
        input_values = torch.from_numpy(input_values).float().to(dtype=self.dtype)
        audio_len = math.ceil(len(input_values) / self.args.sample_rate * self.args.fps)
    
        if audio_len < L - first_fixed_frame:
            audio_len = audio_len + ((L - first_fixed_frame) - audio_len % (L - first_fixed_frame))
        elif (audio_len - (L - first_fixed_frame)) % (L - fixed_frame) != 0:
            audio_len = audio_len + ((L - fixed_frame) - (audio_len - (L - first_fixed_frame)) % (L - fixed_frame))
    
        seq_len = audio_len
        
        times = (seq_len - L + first_fixed_frame) // (L-fixed_frame) + 1
        if times * (L-fixed_frame) + fixed_frame < seq_len:
            times += 1
    
        return times
        
    @torch.no_grad()
    def forward(self, prompt, 
                image_path=None, 
                audio_path=None, 
                orientation_state = None,
                seq_len=101, # not used while audio_path is not None
                height=720, 
                width=720,
                overlap_frame=None,
                num_steps=None,
                negative_prompt=None,
                guidance_scale=None,
                audio_scale=None):
        overlap_frame = overlap_frame if overlap_frame is not None else self.args.overlap_frame
        num_steps = num_steps if num_steps is not None else self.args.num_steps
        negative_prompt = negative_prompt if negative_prompt is not None else self.args.negative_prompt
        guidance_scale = guidance_scale if guidance_scale is not None else self.args.guidance_scale
        audio_scale = audio_scale if audio_scale is not None else self.args.audio_scale

        if image_path is not None:
            from PIL import Image
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

        # step 1: numerator and denominator as ints
        num = args.max_tokens * 16 * 16 * 4
        den = select_size[0] * select_size[1]
        
        # step 2: integer division
        L0 = num // den  # exact floor division, no float in sight
        
        # step 3: make it ≡ 1 mod 4
        diff = (L0 - 1) % 4
        L  = L0 - diff
        if L < 1:
            L = 1  # or whatever your minimal frame count is
        
        # step 4: latent frames
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
            # padding audio
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

        # loop
        times = (seq_len - L + first_fixed_frame) // (L-fixed_frame) + 1
        if times * (L-fixed_frame) + fixed_frame < seq_len:
            times += 1
        video = []
        image_emb = {}
        img_lat = None
        if self.args.i2v:
            self.pipe.load_models_to_device(['vae'])
            img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device, dtype=self.dtype)

            msk = torch.zeros_like(img_lat.repeat(1, 1, T, 1, 1)[:,:1], dtype=self.dtype)
            image_cat = img_lat.repeat(1, 1, T, 1, 1)
            msk[:, :, 1:] = 1
            image_emb["y"] = torch.cat([image_cat, msk], dim=1)

        total_iterations = times * num_steps

        with tqdm(total=total_iterations) as pbar:
            for t in range(times):
                print(f"[{t+1}/{times}]")
                audio_emb = {}
                if t == 0:
                    overlap = first_fixed_frame
                else:
                    overlap = fixed_frame
                    image_emb["y"][:, -1:, :prefix_lat_frame] = 0 # 第一次推理是mask只有1，往后都是mask overlap
                prefix_overlap = (3 + overlap) // 4
                if audio_embeddings is not None:
                    if t == 0:
                        audio_tensor = audio_embeddings[
                                :min(L - overlap, audio_embeddings.shape[0])
                            ]
                    else:
                        audio_start = L - first_fixed_frame + (t - 1) * (L - overlap)
                        audio_tensor = audio_embeddings[
                            audio_start: min(audio_start + L - overlap, audio_embeddings.shape[0])
                        ]
                        
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
                                                     tea_cache_l1_thresh=self.args.tea_cache_l1_thresh,tea_cache_model_id="Wan2.1-T2V-14B", progress_bar_cmd=pbar)
    
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
        

snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-14B", local_dir="./pretrained_models/Wan2.1-T2V-14B")
snapshot_download(repo_id="facebook/wav2vec2-base-960h", local_dir="./pretrained_models/wav2vec2-base-960h")
snapshot_download(repo_id="OmniAvatar/OmniAvatar-14B", local_dir="./pretrained_models/OmniAvatar-14B")


import tempfile

from PIL import Image


set_seed(args.seed)
seq_len = args.seq_len
inferpipe = WanInferencePipeline(args)

ADAPTIVE_PROMPT_TEMPLATES = [
    "A realistic video of a person speaking and moving their head accordingly but without moving their hands.",
    "A realistic video of a person speaking and moving their head and eyes accordingly, sometimes looking at the camera and sometimes looking away but with subtle hands movement that complements their speech.",
    "A realistic video of a person speaking and sometimes looking directly to the camera and moving their eyes and pupils and head accordingly and turning and looking at the camera and looking away from the camera based on their movements with dynamic and rhythmic and extensive hand gestures that complement their speech. Their hands are clearly visible, independent, and unobstructed. Their facial expressions are expressive and full of emotion, enhancing the delivery. The camera remains steady, capturing sharp, clear movements and a focused, engaging presence."
]

def slider_value_change(image_path, audio_path, orientation_state, text, num_steps, session_state, adaptive_text):

    if adaptive_text:

        if not args.image_sizes_720 == [[720, 720]]:
            if num_steps < 8:
                text = ADAPTIVE_PROMPT_TEMPLATES[1]
            elif num_steps < 10:
                text = ADAPTIVE_PROMPT_TEMPLATES[1]
            else:
                text = ADAPTIVE_PROMPT_TEMPLATES[2]
        else:
            text = ADAPTIVE_PROMPT_TEMPLATES[1]
            
    return update_generate_button(image_path, audio_path, orientation_state, text, num_steps, session_state), text

    
def update_generate_button(image_path, audio_path, orientation_state, text, num_steps, session_state):

    if image_path is None or audio_path is None:
        return gr.update(value="⌚ Zero GPU Required: --")

    duration_s = get_duration(image_path, audio_path, text, orientation_state, num_steps, session_state, None)
    duration_m = duration_s / 60
    
    return gr.update(value=f"⌚ Zero GPU Required: ~{duration_s}.0s ({duration_m:.1f} mins)")

def get_duration(image_path, audio_path, text, orientation_state, num_steps, session_id, progress):
  
    if image_path is None:
        gr.Info("Step1: Please Provide an Image or Choose from Image Samples")
        print("Step1: Please Provide an Image or Choose from Image Samples")

        return 0
        
    if audio_path is None:
        gr.Info("Step2: Please Provide an Audio or Choose from Audio Samples")
        print("Step2: Please Provide an Audio or Choose from Audio Samples")

        return 0


    audio_chunks = inferpipe.get_times(
                prompt=text,
                image_path=image_path,
                audio_path=audio_path,
                orientation_state= orientation_state,
                seq_len=args.seq_len,
                num_steps=num_steps
            )
    
    warmup_s = 30
    duration_s = (20 * num_steps) + warmup_s

    if audio_chunks > 1:
        duration_s =  (20 * num_steps * audio_chunks) + warmup_s

    print(f'for {audio_chunks} times and {num_steps} steps, {session_id} is preparing for {duration_s}')

    return int(duration_s)

def preprocess_img(input_image_path, raw_image_path, orientation_state, session_id = None):

    if session_id is None:
        session_id = uuid.uuid4().hex

    if input_image_path is None:
        return None, None

    if raw_image_path == '':
        raw_image_path = input_image_path
        
    image = Image.open(raw_image_path).convert("RGB")

    image = inferpipe.transform(image).unsqueeze(0).to(dtype=inferpipe.dtype)
    
    _, _, h, w = image.shape
    select_size = match_size(orientation_state, h, w)
    image = resize_pad(image, (h, w), select_size)
    image = image * 2.0 - 1.0
    image = image[:, :, None]

    output_dir = os.path.join(os.environ["PROCESSED_RESULTS"], session_id)

    img_dir = output_dir + '/image'
    os.makedirs(img_dir, exist_ok=True)
    input_img_path = os.path.join(img_dir, f"img_input.jpg")
    
    image = tensor_to_pil(image)
    image.save(input_img_path)

    return input_img_path, raw_image_path

def infer_example(image_path, audio_path, text, num_steps, raw_image_path, session_id = None, progress=gr.Progress(track_tqdm=True),):

    if session_id is None:
        session_id = uuid.uuid4().hex

    image_path, _ = preprocess_img(image_path, image_path, [[720, 400]], session_id)
    result = infer(image_path, audio_path, text, [[720, 400]], num_steps, session_id, progress)

    return result
    
@spaces.GPU(duration=get_duration)
def infer(image_path, audio_path, text, orientation_state, num_steps, session_id = None, progress=gr.Progress(track_tqdm=True),):

    if image_path is None:

        return None
        
    if audio_path is None:

        return None
        
    if session_id is None:
        session_id = uuid.uuid4().hex

        
    output_dir = os.path.join(os.environ["PROCESSED_RESULTS"], session_id)

    audio_dir = output_dir + '/audio'
    os.makedirs(audio_dir, exist_ok=True)
    if args.silence_duration_s > 0:
        input_audio_path = os.path.join(audio_dir, f"audio_input.wav")
    else:
        input_audio_path = audio_path
    prompt_dir = output_dir + '/prompt'
    os.makedirs(prompt_dir, exist_ok=True)

    if args.silence_duration_s > 0:
        add_silence_to_audio_ffmpeg(audio_path, input_audio_path, args.silence_duration_s)

    tmp2_audio_path = os.path.join(audio_dir, f"audio_out.wav")
    prompt_path = os.path.join(prompt_dir, f"prompt.txt") 
    
    video = inferpipe(
                prompt=text,
                image_path=image_path,
                audio_path=input_audio_path,
                orientation_state=orientation_state,
                seq_len=args.seq_len,
                num_steps=num_steps
            )
    
    torch.cuda.empty_cache()

    add_silence_to_audio_ffmpeg(audio_path, tmp2_audio_path, 1.0 / args.fps + args.silence_duration_s)
    video_paths = save_video_as_grid_and_mp4(video, 
                            output_dir, 
                            args.fps, 
                            prompt=text,
                            prompt_path = prompt_path,
                            audio_path=tmp2_audio_path if args.use_audio else None, 
                            prefix=f'result')

    return video_paths[0]

def apply_image(request):
    print('image applied')

    return request, request

def apply_audio(request):
    print('audio applied')
    return request

def cleanup(request: gr.Request):

    sid = request.session_hash
    if sid:
        d1 = os.path.join(os.environ["PROCESSED_RESULTS"], sid)
        shutil.rmtree(d1, ignore_errors=True)
        
def start_session(request: gr.Request):

    return request.session_hash

def check_box_clicked(adapative_tick):
    print("checkbox clicked")
    return gr.update(interactive=not adapative_tick)

def orientation_changed(session_id, evt: gr.EventData):

    detail = getattr(evt, "data", None) or getattr(evt, "_data", {}) or {}

    if detail['value'] == "9:16":
        orientation_state = [[720, 400]]
    elif detail['value'] == "1:1":
        orientation_state = [[720, 720]]
    elif detail['value'] == "16:9":
        orientation_state = [[400, 720]]

    print(f'{session_id} has {orientation_state} orientation')

    return orientation_state

def clear_raw_image():
    return ''

def preprocess_audio_first_5s_librosa(audio_path, limit_on, session_id=None):
    """
    If the uploaded audio is < 5s, return it unchanged.
    If it's >= 5s, trim to the first 5s and return the trimmed WAV path.
    """

    if not limit_on:
        print(f'the limit has been ignored for {session_id}')
        return audio_path
    if not audio_path:
        return None

    # Robust duration check (librosa changed arg name across versions)
    try:
        dur = librosa.get_duration(path=audio_path)
    except TypeError:
        dur = librosa.get_duration(filename=audio_path)

    # Small tolerance to avoid re-encoding 4.9999s files
    if dur < 5.0 - 1e-3:
        return audio_path

    if session_id is None:
        session_id = uuid.uuid4().hex

    # Where we'll store per-session processed audio
    output_dir = os.path.join(os.environ["PROCESSED_RESULTS"], session_id)
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    trimmed_path = os.path.join(audio_dir, "audio_input_5s.wav")
    sr = getattr(args, "sample_rate", 16000)

    # Load exactly the first 5s as mono at target sample rate
    y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=5.0)

    # Save as 16-bit PCM mono WAV
    waveform = torch.from_numpy(y).unsqueeze(0)  # [1, num_samples]
    torchaudio.save(
        trimmed_path,
        waveform,
        sr,
        encoding="PCM_S",
        bits_per_sample=16,
        format="wav",
    )

    return trimmed_path

    
css = """
    #col-container {
        margin: 0 auto;
        max-width: 1560px;
    }

    /* editable vs locked, reusing theme variables that adapt to dark/light */
    .stateful textarea:not(:disabled):not([readonly]) {
      color: var(--color-text) !important;            /* accent in both modes */
    }
    .stateful textarea:disabled,
    .stateful textarea[readonly]{
      color: var(--body-text-color-subdued) !important; /* subdued in both modes */
    }
    """

with gr.Blocks(css=css) as demo:

    session_state = gr.State()
    orientation_state = gr.State([[720, 400]])
    demo.load(start_session, outputs=[session_state])


    with gr.Column(elem_id="col-container"):
        gr.HTML(
            """
            <div style="text-align: left;">
                <p style="font-size:16px; display: inline; margin: 0;">
                    <strong>OmniAvatar</strong> – Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation
                </p>
                <a href="https://huggingface.co/OmniAvatar/OmniAvatar-14B" style="display: inline-block; vertical-align: middle; margin-left: 0.5em;">
                    [model]
                </a>
            </div>
            <div style="text-align: left;">
                <strong>HF Space by:</strong>
                <a href="https://twitter.com/alexandernasa/" style="display: inline-block; vertical-align: middle; margin-left: 0.5em;">
                    <img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow Me" alt="GitHub Repo">
                </a>
            </div>

            <div style="text-align: left;">
                <p style="font-size:16px; display: inline; margin: 0;">
                    For a fast clay variation please check out this HF Space:
                </p>
                <a href="https://huggingface.co/spaces/alexnasa/OmniAvatar-Clay-Fast" style="display: inline-block; vertical-align: middle; margin-left: 0.5em;">
                    <img src="https://img.shields.io/badge/🤗-HF Demo-yellow.svg">
                </a>
            </div>

            """
        )

        with gr.Row():

            with gr.Column():

                image_input = extendedimage(label="Reference Image", type="filepath", height=512)
                audio_input = ExtendedAudio(label="Input Audio", type="filepath", options=["EMPTY"], show_download_button=True)
                gr.Markdown("*A 5-second limit is applied to audio files to shorten generation time. You can turn this off in Advanced Settings*")


            with gr.Column():
                
                output_video = gr.Video(label="Avatar", height=512)
                num_steps = gr.Slider(4, 50, value=8, step=1, label="Steps")
                
                time_required = gr.Text(value="⌚ Zero GPU Required: --", show_label=False)
                infer_btn = gr.Button("🦜 Avatar Me", variant="primary")
                with gr.Accordion("Advanced Settings", open=False):
                    raw_img_text = gr.Text(show_label=False, label="", value='', visible=False)
                    limit_on = gr.Checkbox(label="Limit Audio files to 5 seconds", value=True)
                    adaptive_text = gr.Checkbox(label="Adaptive Video Prompt", value=True)
                    text_input = gr.Textbox(show_label=False, lines=6, elem_classes=["stateful"], interactive=False, value= ADAPTIVE_PROMPT_TEMPLATES[1])

            with gr.Column():
                
                cached_examples = gr.Examples(                    
                    examples=[ 
        
                        [
                            "examples/images/creature-001.png",
                            "examples/audios/keen.wav",
                            ADAPTIVE_PROMPT_TEMPLATES[2],
                            20,
                            ''
                        ],
                        
                        [
                            "examples/images/female-001.png",
                            "examples/audios/script.wav",
                            ADAPTIVE_PROMPT_TEMPLATES[2],
                            14,
                            ''
                        ],
                        
                        [
                            "examples/images/male-001.png",
                            "examples/audios/denial.wav",
                            ADAPTIVE_PROMPT_TEMPLATES[2],
                            12,
                            ''
                        ],

                        [
                            "examples/images/female-007.png",
                            "examples/audios/listen.wav",
                            ADAPTIVE_PROMPT_TEMPLATES[1],
                            8,
                            ''
                        ],

                    ],
                    label="Cached Examples",
                    inputs=[image_input, audio_input, text_input, num_steps, raw_img_text],
                    outputs=[output_video],
                    fn=infer_example,
                    cache_examples=True
                    )

                image_examples = gr.Examples(        
                     examples=[ 
                        [
                            "examples/images/female-009.png",
                        ],
                        [
                            "examples/images/male-005.png",
                        ],
                        [
                            "examples/images/female-003.png",
                        ],
                        [
                            "examples/images/female-002.png",
                        ],
                    ],
                    label="Image Samples",
                    inputs=[image_input],
                    outputs=[image_input, raw_img_text],
                    fn=apply_image,
                    cache_examples=True
                    )

                audio_examples = gr.Examples(        
                     examples=[ 
                        [
                            "examples/audios/londoners.wav",
                        ],

                        [
                            "examples/audios/keen.wav",
                        ],

                        [
                            "examples/audios/matcha.wav",
                        ],

                        [
                            "examples/audios/nature.wav",
                        ],
                    ],
                    label="Audio Samples",
                    inputs=[audio_input],
                    cache_examples=False
                    )

    infer_btn.click(
        fn=infer,
        inputs=[image_input, audio_input, text_input, orientation_state, num_steps, session_state],
        outputs=[output_video]
    )

    audio_input.generate(
        fn=speak_to_me, 
        inputs=[session_state],
        outputs=[audio_input]
    ).then(
        fn=apply_audio, 
        inputs=[audio_input], 
        outputs=[audio_input]
    ).then(
        fn=preprocess_audio_first_5s_librosa,
        inputs=[audio_input, limit_on, session_state],
        outputs=[audio_input],
    )
    image_input.orientation(fn=orientation_changed, inputs=[session_state], outputs=[orientation_state]).then(fn=preprocess_img, inputs=[image_input, raw_img_text, orientation_state, session_state], outputs=[image_input, raw_img_text])
    image_input.clear(fn=clear_raw_image, outputs=[raw_img_text])
    image_input.upload(fn=preprocess_img, inputs=[image_input, raw_img_text, orientation_state, session_state], outputs=[image_input, raw_img_text])
    image_input.change(fn=update_generate_button, inputs=[image_input, audio_input, orientation_state, text_input, num_steps, session_state], outputs=[time_required])
    audio_input.change(fn=update_generate_button, inputs=[image_input, audio_input, orientation_state, text_input, num_steps, session_state], outputs=[time_required])
    num_steps.change(fn=slider_value_change, inputs=[image_input, audio_input, orientation_state, text_input, num_steps, session_state, adaptive_text], outputs=[time_required, text_input])
    adaptive_text.change(fn=check_box_clicked, inputs=[adaptive_text], outputs=[text_input])
    audio_input.upload(fn=apply_audio, inputs=[audio_input], outputs=[audio_input]
    ).then(
        fn=preprocess_audio_first_5s_librosa,
        inputs=[audio_input, limit_on, session_state],
        outputs=[audio_input],
    )

if __name__ == "__main__":
    demo.unload(cleanup)
    demo.queue()
    demo.launch(ssr_mode=False)