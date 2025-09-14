from huggingface_hub import snapshot_download

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

    print("All models downloaded successfully.")