# Use the official NVIDIA CUDA runtime as the base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# CRITICAL: pia_prod lives at <repo>/packages/pia_prod (per setup.py:
# `package_dir={'': 'packages'}`), so PYTHONPATH points at the `packages` dir.
ENV PYTHONPATH="/opt/mono/packages"

# Set the working directory for the API server
WORKDIR /app

# Install dependencies for adding PPAs and downloading pip
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install pip specifically for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Symlink python and python3 to point to python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Copy dependency list and install them
# (Doing this before copying the code caches the pip install step!)
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy the actual server script into the container
COPY server.py .

# Expose the default port
EXPOSE 9002

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9002"]