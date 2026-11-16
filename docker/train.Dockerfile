# Training environment for pipeline.train / pipeline.evaluate (CUDA PyTorch; the PyPI Linux torch wheels bundle CUDA).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /repo
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
