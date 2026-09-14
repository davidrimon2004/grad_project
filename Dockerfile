FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# System deps + Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip git wget \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /workspace

# Clone the repo (or COPY it in if you already have it locally)
RUN git clone https://github.com/PKU-YuanGroup/Video-LLaVA.git
WORKDIR /workspace/Video-LLaVA

# Install pinned dependencies exactly as the repo expects
RUN pip install --upgrade pip
RUN pip install -e .
RUN pip install einops

# (Optional) uncomment if you need training deps too
# RUN pip install -e ".[train]"

CMD ["/bin/bash"]