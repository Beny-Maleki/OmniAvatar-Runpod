from typing import Optional

# Import HiggsAudio components
from higgs_audio.serve.serve_engine import HiggsAudioServeEngine
from higgs_audio.data_types import ChatMLSample, AudioContent, Message

import base64
from functools import lru_cache
from loguru import logger
import os
import json
import uuid
import time
import numpy as np
import re

def process_text_output(text_output: str):
    # remove all the continuous <|AUDIO_OUT|> tokens with a single <|AUDIO_OUT|>
    text_output = re.sub(r"(<\|AUDIO_OUT\|>)+", r"<|AUDIO_OUT|>", text_output)
    return text_output


def check_return_audio(audio_wv: np.ndarray):
    # check if the audio returned is all silent
    if np.all(audio_wv == 0):
        logger.warning("Audio is silent, returning None")


def load_voice_presets():
    """Load the voice presets from the voice_examples directory."""
    try:
        with open(
            os.path.join(os.path.dirname(__file__), "examples", "audios", "config.json"),
            "r",
        ) as f:
            voice_dict = json.load(f)
        voice_presets = {k: v for k, v in voice_dict.items()}
        voice_presets["EMPTY"] = "No reference voice"
        logger.info(f"Loaded voice presets: {list(voice_presets.keys())}")
        return voice_presets
    except FileNotFoundError:
        logger.warning("Voice examples config file not found. Using empty voice presets.")
        return {"EMPTY": "No reference voice"}
    except Exception as e:
        logger.error(f"Error loading voice presets: {e}")
        return {"EMPTY": "No reference voice"}


SAMPLE_RATE = 24000
DEFAULT_STOP_STRINGS = ["<|end_of_text|>", "<|eot_id|>"]
VOICE_PRESETS = load_voice_presets()


def initialize_engine(model_path, audio_tokenizer_path) -> bool:
    engine = HiggsAudioServeEngine(
        model_name_or_path=model_path,
        audio_tokenizer_name_or_path=audio_tokenizer_path,
        device="cuda",
    )
    return engine

def get_voice_preset(voice_preset):
    """Get the voice path and text for a given voice preset."""

    preset_dir = os.path.join(os.path.dirname(__file__), "examples", "audios")
    voice_path = os.path.join(preset_dir, VOICE_PRESETS[voice_preset]["audio_file"])
    
    if not os.path.exists(voice_path):
        logger.warning(f"Voice preset file not found: {voice_path}")
        return None, "Voice preset not found"

    text = VOICE_PRESETS[voice_preset]["transcript"]
    return voice_path, text


def normalize_chinese_punctuation(text):
    """
    Convert Chinese (full-width) punctuation marks to English (half-width) equivalents.
    """
    # Mapping of Chinese punctuation to English punctuation
    chinese_to_english_punct = {
        "，": ", ",  # comma
        "。": ".",  # period
        "：": ":",  # colon
        "；": ";",  # semicolon
        "？": "?",  # question mark
        "！": "!",  # exclamation mark
        "（": "(",  # left parenthesis
        "）": ")",  # right parenthesis
        "【": "[",  # left square bracket
        "】": "]",  # right square bracket
        "《": "<",  # left angle quote
        "》": ">",  # right angle quote
        "“": '"',  # left double quotation
        "”": '"',  # right double quotation
        "‘": "'",  # left single quotation
        "’": "'",  # right single quotation
        "、": ",",  # enumeration comma
        "—": "-",  # em dash
        "…": "...",  # ellipsis
        "·": ".",  # middle dot
        "「": '"',  # left corner bracket
        "」": '"',  # right corner bracket
        "『": '"',  # left double corner bracket
        "』": '"',  # right double corner bracket
    }

    # Replace each Chinese punctuation with its English counterpart
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)

    return text


def normalize_text(transcript: str):
    transcript = normalize_chinese_punctuation(transcript)
    # Other normalizations (e.g., parentheses and other symbols. Will be improved in the future)
    transcript = transcript.replace("(", " ")
    transcript = transcript.replace(")", " ")
    transcript = transcript.replace("°F", " degrees Fahrenheit")
    transcript = transcript.replace("°C", " degrees Celsius")

    for tag, replacement in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE>[Humming]</SE>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        transcript = transcript.replace(tag, replacement)

    lines = transcript.split("\n")
    transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    transcript = transcript.strip()

    if not any([transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        transcript += "."

    return transcript

@lru_cache(maxsize=20)
def encode_audio_file(file_path):
    """Encode an audio file to base64."""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


def prepare_chatml_sample(
    voice_preset: str,
    text: str,
    reference_audio: Optional[str] = None,
    reference_text: Optional[str] = None,
    system_prompt: str = "",
):
    """Prepare a ChatMLSample for the HiggsAudioServeEngine."""
    messages = []

    # Add system message if provided
    if len(system_prompt) > 0:
        messages.append(Message(role="system", content=system_prompt))

    # Add reference audio if provided
    audio_base64 = None
    ref_text = ""

    if reference_audio:
        # Custom reference audio
        audio_base64 = encode_audio_file(reference_audio)
        ref_text = reference_text or ""
    elif voice_preset != "EMPTY":
        # Voice preset
        voice_path, ref_text = get_voice_preset(voice_preset)
        if voice_path is None:
            logger.warning(f"Voice preset {voice_preset} not found, skipping reference audio")
        else:
            audio_base64 = encode_audio_file(voice_path)

    # Only add reference audio if we have it
    if audio_base64 is not None:
        # Add user message with reference text
        messages.append(Message(role="user", content=ref_text))

        # Add assistant message with audio content
        audio_content = AudioContent(raw_audio=audio_base64, audio_url="")
        messages.append(Message(role="assistant", content=[audio_content]))

    # Add the main user message
    text = normalize_text(text)
    messages.append(Message(role="user", content=text))

    return ChatMLSample(messages=messages)



def text_to_speech(
    engine,
    text,
    system_prompt="",
    voice_preset="EMPTY",
    reference_audio=None,
    reference_text=None,
    max_completion_tokens=1024,
    temperature=1.0,
    top_p=0.95,
    top_k=50,
    stop_strings=None,
    ras_win_len=7,
    ras_win_max_num_repeat=2,
):
    """
    Convert text to speech using HiggsAudioServeEngine.
    
    Args:
        text: The text to convert to speech
        voice_preset: The voice preset to use (or "EMPTY" for no preset)
        reference_audio: Optional path to reference audio file
        reference_text: Optional transcript of the reference audio
        max_completion_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature for generation
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        system_prompt: System prompt to guide the model
        stop_strings: Dataframe containing stop strings
        ras_win_len: Window length for repetition avoidance sampling
        ras_win_max_num_repeat: Maximum number of repetitions allowed in the window
        
    Returns:
        Tuple of (generated_text, (sample_rate, audio_data)) where audio_data is int16 numpy array
    """

    try:
        # Prepare ChatML sample
        chatml_sample = prepare_chatml_sample(voice_preset, text, reference_audio, reference_text, system_prompt)

        # Convert stop strings format
        if stop_strings is None:
            stop_list = DEFAULT_STOP_STRINGS
        else:
            stop_list = [s for s in stop_strings["stops"] if s.strip()]

        request_id = f"tts-playground-{str(uuid.uuid4())}"

        start_time = time.time()

        # Generate using the engine
        response = engine.generate(
            chat_ml_sample=chatml_sample,
            max_new_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
            stop_strings=stop_list,
            ras_win_len=ras_win_len if ras_win_len > 0 else None,
            ras_win_max_num_repeat=max(ras_win_len, ras_win_max_num_repeat),
        )

        generation_time = time.time() - start_time

        # Process the response
        text_output = process_text_output(response.generated_text)

        if response.audio is not None:
            # Convert to int16 for Gradio
            audio_data = (response.audio * 32767).astype(np.int16)
            check_return_audio(audio_data)
            return text_output, (response.sampling_rate, audio_data)
        else:
            logger.warning("No audio generated")
            return text_output, None

    except Exception as e:
        error_msg = f"Error generating speech: {e}"
        logger.error(error_msg)
        return f"❌ {error_msg}", None