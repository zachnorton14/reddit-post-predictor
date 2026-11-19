"""Train one GLiDE-9B run (glide_models.py) on a Modal H100. Paid: use when VCL is too slow for the deadline.

Runs land in the Modal volume `glide-runs` (weights, best.json with the validation scores). Needs HF_TOKEN:

    modal run --detach infra/modal/glide.py --model glide_score10 --train-per-sub 5000 --val-per-sub 3000
    modal volume get glide-runs runs/glide_score10_s42 data/glide/runs/
"""

import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
REPO = "/repo"
DATASET = "jbduran/textonly-subreddit-predictors"
REVISION = "0d0f857fb0811889ec29bd3dcaae738774165065"  # the revision d1_omni_readouts was scored on

image = modal.Image.from_dockerfile(ROOT / "docker" / "train.Dockerfile", context_dir=ROOT).add_local_dir(
    ROOT, REPO, ignore=[".git", ".venv", "data", "results", "artifacts", "**/__pycache__", "*.ipynb", "*.docx"]
)
app = modal.App("reddit-post-predictor-glide")
runs = modal.Volume.from_name("glide-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("glide-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="H100", timeout=6 * 60 * 60, volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_local_environ(["HF_TOKEN"])])
def train(model: str, seed: int, train_per_sub: int, val_per_sub: int, epochs: int):
    try:
        subprocess.run(["python", "glide_models.py", "--models", model, "--seeds", str(seed), "--output", "/runs/runs",
                        "--results", f"/runs/results/{model}_s{seed}", "--dataset", DATASET, "--revision", REVISION,
                        "--max-train", str(train_per_sub), "--max-val", str(val_per_sub), "--epochs", str(epochs)],
                       cwd=REPO, check=True, env={**os.environ, "EVAL_CACHE": "/runs/eval_cache"})
    finally:
        runs.commit()
        hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "glide_score10", seed: int = 42, train_per_sub: int = 5000, val_per_sub: int = 3000, epochs: int = 1):
    train.remote(model, seed, train_per_sub, val_per_sub, epochs)
    print(f"done: modal volume get glide-runs runs/{model}_s{seed} data/glide/runs/")
