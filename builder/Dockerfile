FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download models from Hugging Face
COPY download_models.py .
RUN python download_models.py

# Install the pre-downloaded FlashAttention wheel
# Note: The wheel is for Python 3.9, but pip might handle it.
# If this fails, switch the base image to a py3.9 version.
RUN pip install /app/flash_attn_wheel/128/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl --no-deps

# Copy the source code for the handler and configs
COPY ../src /app

# Set the command to run the worker
CMD ["python", "-u", "/runpod-wrapper.py"]