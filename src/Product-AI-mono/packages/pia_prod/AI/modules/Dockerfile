# =============================================================================
# Shared Base Dockerfile for Module-Specific Test Containers
# =============================================================================
# Location: packages/pia_prod/AI/modules/Dockerfile
#
# Purpose:
#   Provides the common system-level and ML runtime dependencies shared across
#   ALL inference module test containers. Module-specific images inherit from
#   this base and add only what they need.
#
# What belongs here (truly common to ALL modules):
#   - OS + CUDA runtime
#   - Python interpreter and pip
#   - torch + torchvision (used by service_base.py, event_base.py, all modules)
#   - numpy (universal array operations)
#   - opencv-python (image I/O for all vision modules)
#
# What does NOT belong here:
#   - Module-specific packages (tensorrt, ultralytics model configs, etc.)
#   - The pia or pia_prod packages (version conflicts should surface per-module)
#   - Infrastructure clients (redis, pika, boto3) until proven common at stable versions
#
# Build (from Product-AI-mono root):
#   docker build -t pia-test-base:latest \
#       -f packages/pia_prod/AI/modules/Dockerfile .
#
# NOTE on CUDA 13.0:
#   Verify that the exact image tag exists on Docker Hub / NGC before building.
#   If nvidia/cuda:13.0.0-devel-ubuntu24.04 is not yet published, substitute
#   the closest available tag (e.g., 13.0.1, or a different variant).
# =============================================================================

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

# -- Environment variables -----------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Allow pip install without venv in container (PEP 668 / Ubuntu 24.04)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# -- System dependencies -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        git \
        wget \
        curl \
        # OpenCV runtime dependencies
        # libgl1-mesa-glx is obsoleted in Ubuntu 24.04; libgl1 replaces it
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# -- pip / setuptools -----------------------------------------------------------
# --ignore-installed is needed because the Debian-packaged pip lacks a RECORD
# file, which causes a normal --upgrade to fail in Ubuntu 24.04 containers.
RUN python3 -m pip install --no-cache-dir --ignore-installed pip setuptools wheel

# -- Common ML dependencies (shared across ALL inference modules) ---------------
# torch + torchvision : every module's service, event, and base classes use torch
# numpy              : universal numerical operations
# opencv-python      : image I/O used by all vision modules
#
# NOTE: ultralytics is NOT here — it belongs in each YOLO-based module's own
#       requirements.txt (e.g. khonkaen_helmet, gangnam_falldown, etc.)
#
# NOTE: Adjust the --index-url if a CUDA-13.0-specific PyTorch wheel index is
#       required. Check https://download.pytorch.org/whl/ for available builds.
RUN pip install --no-cache-dir \
        torch \
        torchvision \
        "numpy<2.3" \
        "opencv-python==4.10.0.84"

WORKDIR /workspace
