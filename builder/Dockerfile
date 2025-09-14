# --- STAGE 1: Builder ---
# Use the development image to build dependencies
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04 AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python requirements into the virtual environment
COPY builder/requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# --- STAGE 2: Final Image ---
# Use a smaller runtime image for the final product
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-runtime-ubuntu22.04

WORKDIR /app

# Install only necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the source code for the handler and configs
COPY src/ /app/

# Activate the virtual environment and set the command
ENV PATH="/opt/venv/bin:$PATH"
CMD ["python", "-u", "/runpod-wrapper.py"]