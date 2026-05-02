# ComfyUI Wan2.2 Worker for RunPod Serverless
# Ubuntu 24.04 base ships Python 3.12 natively, so we don't need the
# deadsnakes PPA (which kept failing during builds — dbus chain hang,
# keyserver timeout, then 503 from Launchpad's CDN). NVIDIA hasn't
# published cudnn-runtime images for Ubuntu 24.04 below CUDA 12.6, so we
# pick the lowest stable 12.6.x. PyTorch's cu124 wheel still works against
# a 12.6 runtime — driver-level forward compat handles the difference, and
# PyTorch bundles its own CUDA libs.
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# ComfyUI startup wait defaults
#
# handler.py's check_server() falls back to a bounded retry loop when it
# can't determine whether ComfyUI is still alive from the PID file. With the
# upstream defaults (50ms interval × 500 attempts = 25s) flaky RunPod workers
# with slow cold starts routinely miss the window and fail with "ComfyUI
# server (127.0.0.1:8188) not reachable after multiple retries."
#
# Bumping the interval to 100ms and the retry cap to 2000 gives a ~200s
# fallback window, which covers every real-world cold start we've observed
# while still failing fast on a truly broken worker. Operators can override
# either value at the RunPod endpoint level without rebuilding this image.
# ---------------------------------------------------------------------------
ENV COMFY_API_AVAILABLE_INTERVAL_MS=100
ENV COMFY_API_AVAILABLE_MAX_RETRIES=2000

# Prevent apt post-install hooks from trying to start services during the
# Docker build. Without this, packages like dbus/packagekit/networkd-dispatcher
# hang on `Processing triggers for dbus` waiting on a system bus socket that
# doesn't exist in the build container. policy-rc.d returning exit 101 tells
# `invoke-rc.d` "deny start, but exit success" so apt continues.
RUN printf '#!/bin/sh\nexit 101\n' > /usr/sbin/policy-rc.d \
    && chmod +x /usr/sbin/policy-rc.d

# Install Python 3.12 + system libs. On Ubuntu 24.04, `python3` already IS
# 3.12, so there's no PPA setup needed. The `python` -> `python3.12` symlink
# is added because some downstream tooling expects bare `python` to exist.
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    openssh-server \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Single virtual environment for everything
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ComfyUI via git clone (not comfy-cli)
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui && pip install -r requirements.txt

# Install ComfyUI-Manager
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/ltdrdata/ComfyUI-Manager.git \
    && cd ComfyUI-Manager && pip install -r requirements.txt || true

# Install Impact-Pack for FaceDetailer/DetailerForEach
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git \
    && cd ComfyUI-Impact-Pack && pip install -r requirements.txt || true

# Install Impact-Subpack — provides UltralyticsDetectorProvider (moved out of main pack in V8+)
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git \
    && cd ComfyUI-Impact-Subpack && pip install -r requirements.txt || true

# Install WanVideoWrapper — Wan2.2 video generation nodes
# (Bundles WanVideoLoraSelect + WanVideoTeaCache custom nodes used by the
# cost-optimized base workflow.)
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
    && cd ComfyUI-WanVideoWrapper && pip install -r requirements.txt || true

# Install sage-attention so WanVideoModelLoader.attention_mode="sageattn" works.
# Triton ships with the cu124 PyTorch wheel; the sageattention wheel pulls a
# matching prebuilt kernel — no build step required.
RUN pip install sageattention

# Install VideoHelperSuite — video I/O utilities (combine frames, load video, etc.)
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt || true

# Install ComfyUI-Frame-Interpolation — RIFE / FILM / GMFSS frame interpolation nodes.
# Used by the refinement workflow to double 16fps drafts to 32fps.
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git \
    && cd ComfyUI-Frame-Interpolation && pip install -r requirements-no-cupy.txt || true

# Install facerestore_cf — provides FaceRestoreCFWithModel + FaceRestoreModelLoader.
# Lightweight per-frame face restore (no diffusion sampler), suitable for video
# refinement where Impact-Pack's FaceDetailer would be 77× too expensive per clip.
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/mav-rik/facerestore_cf.git \
    && cd facerestore_cf && pip install -r requirements.txt || true

# Ensure ultralytics is installed (required for UltralyticsDetectorProvider YOLO loading)
RUN pip install ultralytics

# Install handler dependencies
RUN pip install runpod requests websocket-client

# Download YOLO detection models (bbox + segmentation)
RUN mkdir -p /comfyui/models/ultralytics/bbox /comfyui/models/ultralytics/segm \
    && wget -q -O /comfyui/models/ultralytics/bbox/face_yolov8n.pt \
       "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt" \
    && wget -q -O /comfyui/models/ultralytics/bbox/hand_yolov8s.pt \
       "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt"

# Copy nipple segmentation model (ADetailer Nipples v2.0 YOLO11s-seg, from CivitAI #490259)
COPY assets/nipples_v2_yolov11s-seg.pt /comfyui/models/ultralytics/segm/nipples_v2_yolov11s-seg.pt

# Download upscale models
# RealESRGAN_x4plus is used by the video refinement workflow for the 480p→1920p
# pre-downscale; the others are kept from the upstream image for completeness.
RUN mkdir -p /comfyui/models/upscale_models \
    && wget -q -O /comfyui/models/upscale_models/4x-UltraSharp.pth \
       "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth" \
    && wget -q -O /comfyui/models/upscale_models/4x_foolhardy_Remacri.pth \
       "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth" \
    && wget -q -O /comfyui/models/upscale_models/RealESRGAN_x2plus.pth \
       "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
    && wget -q -O /comfyui/models/upscale_models/RealESRGAN_x4plus.pth \
       "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# Download RIFE model used by ComfyUI-Frame-Interpolation. The Frame-Interpolation
# node looks under /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/
# by default — pre-stage the checkpoint so the first cold start doesn't burn a
# minute downloading it.
RUN mkdir -p /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife \
    && wget -q -O /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife47.pth \
       "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/rife47.pth"

# Download SAM model for precise segmentation masking in FaceDetailer
RUN mkdir -p /comfyui/models/sams \
    && wget -q -O /comfyui/models/sams/sam_vit_b_01ec64.pth \
       "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# Download CodeFormer face restoration model (final face cleanup after upscaling)
RUN mkdir -p /comfyui/models/facerestore_models \
    && wget -q -O /comfyui/models/facerestore_models/codeformer-v0.1.0.pth \
       "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"

# Add extra model paths for network volume
WORKDIR /comfyui
COPY src/extra_model_paths.yaml ./

# Add handler and startup scripts
WORKDIR /
COPY src/start.sh src/network_volume.py handler.py test_input.json ./
RUN chmod +x /start.sh
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode || true

# Prevent pip from asking for confirmation
ENV PIP_NO_INPUT=1

WORKDIR /comfyui
CMD ["/start.sh"]
