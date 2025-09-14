FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY builder/requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download models from Hugging Face
COPY builder/download_models.py .
RUN python download_models.py

# Copy the source code for the handler and configs
COPY ../src /app

# Set the command to run the worker
CMD ["python", "-u", "/runpod-wrapper.py"]