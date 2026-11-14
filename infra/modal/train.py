"""Train and evaluate a candidate model on a Modal GPU. A fallback for when VCL is unavailable.

Same contract as the VCL path: runs `pipeline.train` then `pipeline.evaluate` and writes the model and
`metrics.json` to `--out`. Run from the repo root (needs MODAL_TOKEN_ID, MODAL_TOKEN_SECRET and HF_TOKEN):

    modal run infra/modal/train.py --dataset <hf repo> --revision <sha> --out artifacts --gpu L40S
"""

import io
import subprocess
import tarfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
REPO = "/repo"

image = modal.Image.from_dockerfile(ROOT / "docker" / "train.Dockerfile", context_dir=ROOT).add_local_dir(
    ROOT, REPO, ignore=[".git", ".venv", "data", "results", "artifacts", "**/__pycache__", "*.ipynb", "*.docx"]
)
app = modal.App("reddit-post-predictor-train")


@app.function(image=image, gpu="L40S", timeout=10 * 60 * 60,
              secrets=[modal.Secret.from_local_environ(["HF_TOKEN"])])
def train_eval(dataset: str, revision: str) -> bytes:
    """Return the model directory and metrics.json as a gzipped tar."""
    out = Path("/tmp/artifacts")
    model = out / "model"
    subprocess.run(["python", "-m", "pipeline.train", "--dataset", dataset, "--revision", revision,
                    "--out", str(model)], cwd=REPO, check=True)
    subprocess.run(["python", "-m", "pipeline.evaluate", "--model", str(model),
                    "--out", str(out / "metrics.json")], cwd=REPO, check=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=".")
    return buf.getvalue()


@app.local_entrypoint()
def main(dataset: str, revision: str, out: str = "artifacts", gpu: str = "L40S"):
    data = train_eval.with_options(gpu=gpu).remote(dataset, revision)
    Path(out).mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(out, filter="data")
    print(f"wrote {out}/model and {out}/metrics.json")
