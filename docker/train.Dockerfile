# Training environment for pipeline.train / pipeline.evaluate (CUDA PyTorch; the PyPI Linux torch wheels bundle CUDA).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# Triton (GPU kernels: flash-linear-attention, torch.compile) compiles a small C helper on first use
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /repo
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
