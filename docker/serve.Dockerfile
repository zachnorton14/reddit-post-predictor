# Inference image: pipeline/serve.py on CPU with the deployed model baked in at /model.
#
#   python scripts/hf_models.py fetch <path> <revision> model --name <name> --type <type>   (deploy.yml does this)
#   docker build -f docker/serve.Dockerfile --build-arg MODEL_VERSION=model-<n> -t serve .
#   docker run -p 8000:8000 serve
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MODEL_DIR=/model
WORKDIR /repo
# CPU wheels: the default Linux torch wheels bundle CUDA (several GB) that a CPU instance can't use.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.14"
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY *.py ./
COPY pipeline/ pipeline/
COPY model/ /model/
# Load the model and score one post now, so a broken model fails the build. This also downloads the base encoder's
# config and weights into the image's Hugging Face cache, so the container needs no network at start.
RUN python -m pipeline.serve --check /model
ENV HF_HUB_OFFLINE=1
ARG MODEL_VERSION=""
ENV MODEL_VERSION=$MODEL_VERSION
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "pipeline.serve:app", "--host", "0.0.0.0", "--port", "8000"]
