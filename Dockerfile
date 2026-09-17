# =============================================================================
# IFD-PART2 — GPU image (NVIDIA CUDA 12.1)
# -----------------------------------------------------------------------------
# Tuned for: NVIDIA RTX 3060 12GB (Ampere, SM 8.6) · 16GB RAM · i7 12th gen.
# CUDA 12.1 + PyTorch cu121 wheels run natively on Ampere GPUs.
#
# Host requirements:
#   - NVIDIA driver >= 530 (ships with CUDA 12.1 support)
#   - nvidia-container-toolkit installed and configured
#       Linux : https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
#       Windows: Docker Desktop 4.x+ with the WSL2 backend (GPU passthrough is
#                automatic once your Windows NVIDIA driver is installed)
#
# Build:  docker build -t ifd-part2:gpu .
# Verify GPU is visible inside the container:
#         docker run --rm --gpus all ifd-part2:gpu python -c "import torch; print(torch.cuda.get_device_name(0))"
# Run:    docker run --rm --gpus all \
#           -v "$(pwd)/data/raw:/app/data/raw:ro" \
#           -v "$(pwd)/results:/app/results" \
#           ifd-part2:gpu
# =============================================================================

# CUDA 12.1 runtime + cuDNN 8 on Ubuntu 22.04.
# uv installs Python 3.14 on top (not yet present in CUDA base images).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
        procps \
    && rm -rf /var/lib/apt/lists/*

# uv binary (brings its own Python version manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# --- Dependency install (cached layer — only re-runs when deps change) --------
COPY pyproject.toml uv.lock ./

ENV UV_HTTP_TIMEOUT=1200


# 1. Install PyTorch 2.13 built for CUDA 12.1 from the official PyTorch index.
# 2. Sync the remaining locked dependencies from PyPI (no dev extras, project
#    installed later once source is copied).
RUN uv python install 3.14 && \
    uv venv /app/.venv && \
    uv pip install --python /app/.venv \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple \
        "torch>=2.13.0" && \
    uv sync --frozen --no-dev --no-install-project

# --- Source code (separate layer — only re-runs on source changes) -----------
COPY . .

# Mount points: CSVs go in (read-only), results come out
VOLUME ["/app/data/raw", "/app/results"]

# Default run. Override any train.py args at runtime, e.g.
#   docker run --rm --gpus all ifd-part2:gpu \
#     python train.py --num-adversaries 2 --attack-type sign_flip
#
# NOTE for 16GB RAM: the full IEEE-CIS dataset + Ray simulation is memory-heavy.
# If you hit OOM, cap rows with --nrows (e.g. --nrows 100000) or lower
# --num-clients / --batch-size. The 3060's 12GB VRAM handles batch-size 512 fine.
CMD ["python", "train.py", \
     "--num-clients", "10", \
     "--num-rounds", "50", \
     "--batch-size", "512", \
     "--epochs-per-round", "2", \
     "--results-dir", "/app/results"]
