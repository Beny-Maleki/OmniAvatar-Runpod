from huggingface_hub import snapshot_download, hf_hub_download

if __name__ == "__main__":
    print("Downloading models from Hugging Face Hub...")

    # Download main model repositories
    snapshot_download(
        repo_id="Wan-AI/Wan2.1-T2V-14B",
        local_dir="./pretrained_models/Wan2.1-T2V-14B",
        local_dir_use_symlinks=False
    )
    snapshot_download(
        repo_id="facebook/wav2vec2-base-960h",
        local_dir="./pretrained_models/wav2vec2-base-960h",
        local_dir_use_symlinks=False
    )
    snapshot_download(
        repo_id="OmniAvatar/OmniAvatar-14B",
        local_dir="./pretrained_models/OmniAvatar-14B",
        local_dir_use_symlinks=False
    )

    # Download the specific FlashAttention wheel
    hf_hub_download(
        repo_id="alexnasa/flash-attn-3",
        repo_type="model",
        filename="128/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl",
        local_dir="./flash_attn_wheel",
        local_dir_use_symlinks=False
    )

    print("All models downloaded successfully.")