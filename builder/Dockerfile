# Use a leaner base image for a smaller final size
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker's layer caching
COPY builder/requirements.txt .

# Install Python dependencies without caching
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir -r requirements.txt

# Copy the source code for the handler and configs
COPY src/ /app/

CMD ["python", "-u", "/app/app.py"]
