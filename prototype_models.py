"""Posting-time neural prototypes and validation-only ensemble selection.

No training/selection function accepts test outcomes. Checkpoints contain the
encoder, tokenizer, head, preprocessing, configuration, and row fingerprints.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from dadjokes_models import META
import training_recovery as recovery

# Audited predecessor: same architectures, loss and preprocessing, without recovery.
LEGACY_TRAINING_HASHES = {"700e4bb82addee670e703643256e00b5e4d356397b87c122a619cff6a3750698"}


@dataclass(frozen=True)
class Spec:
    name: str
    architecture: str = "joint"
    source: str = "data/pretrained/minilm"
    max_length: int = 96
    hidden: int = 64
    dropout: float = 0.1
    epochs: int = 3
    batch_size: int = 64
    accumulation: int = 2
    learning_rate: float = 3e-5
    seed: int = 42
    checkpointing: bool = False
    use_fast_tokenizer: bool = True
    body_chars: int | None = 1000
    # "mean": masked mean of the token vectors. "attention": `queries` learned queries each take a weighted
    # average of the token vectors; the head reads all of them. Queries start at zero (= mean pooling).
    pooling: str = "mean"
    queries: int = 1
    # Learned embeddings for local hour, weekday and month instead of their standardized raw values.
    time_embed: bool = False


# Spec fields added after runs were saved; left out of run identity while at their defaults.
NEW_SPEC_DEFAULTS = {"pooling": "mean", "queries": 1, "time_embed": False}
TIME_EMBED = {"local_hour": (24, 8), "local_dow": (8, 4), "local_month": (13, 4)}  # (table size, dim)
N_TIME = len(TIME_EMBED)
if META[:N_TIME] != list(TIME_EMBED):
    raise ImportError("prototype_models expects the timing columns first in dadjokes_models.META")


def spec_identity(spec):
    return {k: v for k, v in asdict(spec).items() if NEW_SPEC_DEFAULTS.get(k, object()) != v}


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(frame):
    """Includes row order, features and outcomes; never use row counts as identity."""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def source_fingerprint(path):
    result = {}
    for file in sorted(Path(path).glob("*")):
        if file.is_file() and file.suffix in (".json", ".bin", ".safetensors", ".txt", ".model"):
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(2**20), b""):
                    digest.update(chunk)
            result[file.name] = digest.hexdigest()
    if not result:
        raise ValueError(f"No local pretrained files found: {path}")
    return result


def metrics(frame, pred):
    y = frame.label.to_numpy()
    pred = np.asarray(pred)
    if pred.shape != y.shape or not np.isfinite(pred).all():
        raise ValueError("Predictions must be finite and aligned with the frame")
    order = np.argsort(-pred, kind="stable")
    return {
        "ap": float(average_precision_score(y, pred)),
        "spearman": float(spearmanr(frame.metric_pct, pred).statistic),
        **{f"precision_top{int(f * 100)}": float(y[order[:max(1, int(len(y) * f))]].mean())
           for f in (0.01, 0.05, 0.1)},
    }


def fit_preprocessing(train):
    x = train[META].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Metadata must be finite")
    scale = x.std(0)
    return dict(mean=x.mean(0), scale=np.where(scale < 1e-6, 1., scale),
                mu=float(train.metric.mean()), sd=max(float(train.metric.std(ddof=0)), 1e-6))


class Network(nn.Module):
    def __init__(self, spec, source=None):
        super().__init__()
        if spec.architecture not in ("joint", "interaction"):
            raise ValueError(f"Unknown architecture: {spec.architecture}")
        if spec.pooling not in ("mean", "attention") or spec.queries < 1:
            raise ValueError(f"Unknown pooling: {spec.pooling} with {spec.queries} queries")
        self.spec = spec
        # float32 master weights: transformers 5 keeps a checkpoint's stored dtype, and DeBERTa-v3 is stored in fp16,
        # which overflows to a non-finite loss under training. Autocast still runs the forward pass in bf16 on CUDA.
        self.encoder = AutoModel.from_pretrained(source or spec.source, local_files_only=True, dtype=torch.float32)
        if spec.checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        width = self.encoder.config.hidden_size
        if spec.pooling == "attention":
            self.queries = nn.Parameter(torch.zeros(spec.queries, width))
        pooled = width * (spec.queries if spec.pooling == "attention" else 1)
        dim = pooled * (4 if spec.architecture == "interaction" else 1)
        meta_dim = len(META)
        if spec.time_embed:
            self.time = nn.ModuleList(nn.Embedding(n, d) for n, d in TIME_EMBED.values())
            meta_dim += sum(d for _, d in TIME_EMBED.values()) - N_TIME
        # Interaction features explicitly compare setup and punchline semantics.
        self.head = nn.Sequential(nn.Linear(dim + meta_dim, spec.hidden), nn.ReLU(),
                                  nn.Dropout(spec.dropout), nn.Linear(spec.hidden, 1))

    def pool(self, tokens):
        h = self.encoder(**tokens).last_hidden_state
        keep = tokens["attention_mask"].bool()
        if self.spec.pooling == "mean":
            mask = keep.unsqueeze(-1).to(h.dtype)
            return (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        # (batch, queries, tokens): each query's weights over the real tokens sum to 1
        scores = torch.einsum("btd,qd->bqt", h, self.queries.to(h.dtype)) / math.sqrt(h.shape[-1])
        scores = scores.masked_fill(~keep.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = scores.float().softmax(-1).to(h.dtype)
        return torch.einsum("bqt,btd->bqd", weights, h).flatten(1)

    def forward(self, first, meta, second=None, times=None):
        a = self.pool(first)
        if self.spec.architecture == "interaction":
            b = self.pool(second)
            a = torch.cat((a, b, (a - b).abs(), a * b), dim=-1)
        meta = meta.to(a.dtype)
        if self.spec.time_embed:
            # Replace the standardized timing columns with their learned vectors
            meta = torch.cat([meta[:, N_TIME:], *(e(times[:, i]).to(a.dtype) for i, e in enumerate(self.time))], -1)
        return self.head(torch.cat((a, meta), dim=-1)).squeeze(-1)


def time_tensor(frame, device):
    """Integer local hour, weekday and month for the timing embeddings."""
    times = frame[list(TIME_EMBED)].to_numpy()
    if not np.array_equal(times, np.round(times)) or (times < 0).any() \
            or (times >= [n for n, _ in TIME_EMBED.values()]).any():
        raise ValueError("Timing columns must be integers within the embedding tables")
    return torch.tensor(times.astype(np.int64), device=device)


def encode(frame, tokenizer, spec):
    title = frame.title.fillna("").astype(str)
    body = frame.body.fillna("").astype(str)
    if spec.body_chars is not None:
        body = body.str.slice(stop=spec.body_chars)
    kwargs = dict(truncation=True, max_length=spec.max_length, padding=False,
                  return_token_type_ids=False)
    if spec.architecture == "interaction":
        # Reserve a separate budget for the punchline. Keep its end if truncated.
        first = tokenizer(title.tolist(), **kwargs)
        original = tokenizer.truncation_side
        try:
            tokenizer.truncation_side = "left"
            second = tokenizer(body.tolist(), **kwargs)
        finally:
            tokenizer.truncation_side = original
        return first, second
    return tokenizer((title + " \n " + body).tolist(), **kwargs), None


def batch_order(lengths, size, rng=None):
    """Length buckets reduce padding; every sample appears exactly once."""
    order = np.arange(len(lengths)) if rng is None else rng.permutation(len(lengths))
    if rng is None:
        return [part for part in np.array_split(order[np.argsort(lengths, kind="stable")],
                                                math.ceil(len(order) / size)) if len(part)]
    batches = []
    for start in range(0, len(order), size * 50):
        part = order[start:start + size * 50]
        part = part[np.argsort(np.asarray(lengths)[part], kind="stable")]
        batches.extend(part[i:i + size] for i in range(0, len(part), size))
    rng.shuffle(batches)
    return batches


def collate(tokens, indices, tokenizer, device):
    if tokens is None:
        return None
    items = [{key: value[int(i)] for key, value in tokens.items()} for i in indices]
    return tokenizer.pad(items, padding=True, return_tensors="pt").to(device)


def amp_context(device):
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return nullcontext()


def export_best(net, output, saved, val_frame):
    """Rebuild inference files from one durable best-checkpoint transaction."""
    state = saved["model"]
    net.encoder.save_pretrained(output / "encoder", safe_serialization=True,
        state_dict={k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")})
    recovery.atomic_save({k.removeprefix("head."): v for k, v in state.items() if k.startswith("head.")},
                         output / "head.pt")
    # Pooling queries and timing embeddings (absent for mean pooling without timing embeddings)
    extra = {k: v for k, v in state.items() if not k.startswith(("encoder.", "head."))}
    if extra:
        recovery.atomic_save(extra, output / "extra.pt")
    np.save(output / "val_predictions.npy", saved["predictions"])
    val_frame[["id", "created_at"]].to_parquet(output / "val_rows.parquet", index=False)
    write_json(output / "best.json", saved["row"])


@torch.no_grad()
def infer(net, tokenizer, frame, prep, device, encoded=None):
    net.eval()
    first, second = encoded if encoded is not None else encode(frame, tokenizer, net.spec)
    lengths = [len(x) for x in first["input_ids"]]
    if second is not None:
        lengths = np.maximum(lengths, [len(x) for x in second["input_ids"]])
    meta = torch.tensor((frame[META].to_numpy(np.float32) - prep["mean"]) / prep["scale"],
                        dtype=torch.float32, device=device)
    times = time_tensor(frame, device) if net.spec.time_embed else None
    pred = np.empty(len(frame), dtype=np.float32)
    # DeBERTa has substantially larger attention activations than MiniLM.
    for indices in batch_order(lengths, net.spec.batch_size * 2):
        with amp_context(device):
            p = net(collate(first, indices, tokenizer, device), meta[indices],
                    collate(second, indices, tokenizer, device), None if times is None else times[indices])
        pred[indices] = p.float().cpu().numpy()
    return pred * prep["sd"] + prep["mu"]


def train(spec, train_frame, val_frame, output, device=None, resume=False, checkpoint_seconds=300):
    """Train all epochs; select checkpoint by validation average precision only."""
    if train_frame.created_at.max() >= val_frame.created_at.min():
        raise ValueError("Training must strictly precede validation")
    if min(spec.epochs, spec.batch_size, spec.accumulation) < 1:
        raise ValueError("Epochs and batch sizes must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(spec=spec_identity(spec), train_fingerprint=fingerprint(train_frame),
                    val_fingerprint=fingerprint(val_frame),
                    pretrained_sha256=source_fingerprint(spec.source),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    recovery_sha256=hashlib.sha256(Path(recovery.__file__).read_bytes()).hexdigest())
    legacy_warm_start = False
    if (output / "identity.json").exists():
        old = json.loads((output / "identity.json").read_text())
        ignored = {"code_sha256", "recovery_sha256"}
        legacy = (old.get("code_sha256") in LEGACY_TRAINING_HASHES
                  and {k: v for k, v in old.items() if k not in ignored}
                      == {k: v for k, v in identity.items() if k not in ignored})
        if old != identity and not (legacy and (resume or (output / "complete.json").exists())):
            raise ValueError(f"Run identity changed: use a new output directory: {output}")
        if (output / "complete.json").exists():
            print(f"Reusing {spec.name}", flush=True)
            return output
        if not resume:
            raise ValueError(f"Incomplete run at {output}: use --resume to recover it, or a new output directory")
        if legacy:
            needed = ["encoder/config.json", "head.pt", "best.json", "history.json", "val_predictions.npy", "val_rows.parquet"]
            if not all((output / name).exists() for name in needed):
                raise ValueError("Legacy run has no complete saved best epoch to recover; use a new output directory")
            write_json(output / "identity.before_recovery.json", old)
            legacy_warm_start = True
    if checkpoint_seconds < 0:
        raise ValueError("Checkpoint interval must be nonnegative")
    if resume and not legacy_warm_start and (output / "identity.json").exists() and not (output / "resume.pt").exists():
        raise ValueError("No full recovery snapshot found; use a new output directory")
    write_json(output / "identity.json", identity)
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)
    device = device or default_device()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(spec.source, local_files_only=True, use_fast=spec.use_fast_tokenizer)
    tokenizer.save_pretrained(output / "tokenizer")
    train_tokens = encode(train_frame, tokenizer, spec)
    val_tokens = encode(val_frame, tokenizer, spec)
    prep = fit_preprocessing(train_frame)
    np.savez(output / "preprocessing.npz", **prep)
    write_json(output / "spec.json", asdict(spec))
    if legacy_warm_start:
        net = Network(spec, output / "encoder").to(device)
        net.head.load_state_dict(torch.load(output / "head.pt", map_location=device, weights_only=True))
    else:
        net = Network(spec).to(device)
    meta = torch.tensor((train_frame[META].to_numpy(np.float32) - prep["mean"]) / prep["scale"],
                        device=device)
    times = time_tensor(train_frame, device) if spec.time_embed else None
    y = torch.tensor((train_frame.metric.to_numpy(np.float32) - prep["mu"]) / prep["sd"], device=device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=spec.learning_rate, weight_decay=.01, foreach=False)
    steps = math.ceil(math.ceil(len(train_frame) / spec.batch_size) / spec.accumulation)
    history = []
    start_epoch, start_offset, total_loss, seen, batches = 0, 0, 0., 0, None
    elapsed_before, warm_start_epoch = 0., 0
    best = -np.inf
    if legacy_warm_start:
        row = json.loads((output / "best.json").read_text())
        start_epoch = warm_start_epoch = row["epoch"]
        if start_epoch >= spec.epochs:
            raise ValueError("No training epochs remain after the saved best checkpoint")
        history = [r for r in json.loads((output / "history.json").read_text()) if r["epoch"] <= start_epoch]
        best, elapsed_before = row["ap"], row["seconds"]
        write_json(output / "recovery.json", {"mode": "legacy_weights_warm_start", "from_epoch": start_epoch,
                   "optimizer": "reset; predecessor did not save optimizer state",
                   "schedule": f"new warmup/decay over the remaining {spec.epochs - start_epoch} epochs",
                   "lost_work": "unsaved partial epoch after the best checkpoint"})
        print(f"Recovering {spec.name} from saved epoch {start_epoch}; optimizer reset, "
              f"{spec.epochs - start_epoch} epochs remain", flush=True)
    saved = None
    if resume and not legacy_warm_start and (output / "resume.pt").exists():
        # Locally generated optimizer/RNG state; never load an untrusted downloaded pickle.
        saved = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
        if saved["identity"] != identity or saved["device"] != device:
            raise ValueError("Recovery snapshot identity/device does not match this run")
        warm_start_epoch = saved["warm_start_epoch"]
    total_steps = steps * (spec.epochs - warm_start_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(total_steps * .06)), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and not torch.cuda.is_bf16_supported())
    lengths = np.array([len(x) for x in train_tokens[0]["input_ids"]])
    if train_tokens[1] is not None:
        lengths = np.maximum(lengths, [len(x) for x in train_tokens[1]["input_ids"]])
    rng = np.random.default_rng(spec.seed)
    if legacy_warm_start:
        for _ in range(start_epoch):
            batch_order(lengths, spec.batch_size, rng)
    elif saved is not None:
        net.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch, start_offset = saved["epoch"], saved["offset"]
        batches, total_loss, seen = saved["batches"], saved["total_loss"], saved["seen"]
        history, best, elapsed_before = saved["history"], saved["best"], saved["elapsed"]
        rng.bit_generator.state = saved["order_rng"]
        recovery.restore_random_state(saved["random"], device)
        del saved
        print(f"Resuming {spec.name} at epoch {start_epoch + 1}, {seen:,} posts completed in that epoch", flush=True)

    if legacy_warm_start:
        recovery.atomic_save({"model": net.state_dict(), "row": row,
                              "predictions": np.load(output / "val_predictions.npy")}, output / "best_checkpoint.pt")
    elif (output / "best_checkpoint.pt").exists():
        best_saved = torch.load(output / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        export_best(net, output, best_saved, val_frame)
        del best_saved

    # net and optimizer are deleted at the end of train(), after the last snapshot() call
    def snapshot(epoch, offset, batches, total_loss, seen):
        recovery.atomic_save({"identity": identity, "device": device, "model": net.state_dict(),  # noqa: F821
            "optimizer": optimizer.state_dict(),  # noqa: F821
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "offset": offset, "batches": batches, "total_loss": total_loss, "seen": seen,
            "history": history, "best": best, "elapsed": elapsed_before + time.monotonic() - start,
            "order_rng": rng.bit_generator.state, "random": recovery.random_state(device),
            "warm_start_epoch": warm_start_epoch}, output / "resume.pt")

    last_save = time.monotonic()
    # Establish a recoverable starting point before the first optimizer update.
    if legacy_warm_start or not (output / "resume.pt").exists():
        snapshot(start_epoch, 0, None, 0., 0)
    for epoch in range(start_epoch, spec.epochs):
        net.train()
        if batches is None:
            batches = batch_order(lengths, spec.batch_size, rng)
            total_loss, seen, start_offset = 0., 0, 0
        last_log = time.monotonic()
        for offset in range(start_offset, len(batches), spec.accumulation):
            group = batches[offset:offset + spec.accumulation]
            count = sum(map(len, group))
            optimizer.zero_grad(set_to_none=True)
            for indices in group:
                with amp_context(device):
                    pred = net(collate(train_tokens[0], indices, tokenizer, device), meta[indices],
                               collate(train_tokens[1], indices, tokenizer, device),
                               None if times is None else times[indices])
                    loss = nn.functional.mse_loss(pred.float(), y[indices])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss for {spec.name}")
                scaler.scale(loss * len(indices) / count).backward()
                total_loss += loss.item() * len(indices)
                seen += len(indices)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if time.monotonic() - last_save >= checkpoint_seconds:
                snapshot(epoch, offset + len(group), batches, total_loss, seen)
                print(f"Saved recovery snapshot: epoch {epoch + 1}, {seen:,} posts", flush=True)
                last_save = time.monotonic()
            if time.monotonic() - last_log > 40:
                print(f"{spec.name} epoch {epoch + 1}: {seen:,}/{len(train_frame):,} posts, "
                      f"MSE {total_loss / seen:.4f}", flush=True)
                last_log = time.monotonic()
        pv = infer(net, tokenizer, val_frame, prep, device, val_tokens)
        row = dict(epoch=epoch + 1, train_loss=total_loss / seen,
                   seconds=elapsed_before + time.monotonic() - start, **metrics(val_frame, pv))
        history.append(row)
        print(f"{spec.name}: {row}", flush=True)
        if row["ap"] > best:
            best = row["ap"]
            best_saved = recovery.cpu_tree({"model": net.state_dict(), "row": row, "predictions": pv})
            recovery.atomic_save(best_saved, output / "best_checkpoint.pt")
            export_best(net, output, best_saved, val_frame)
            del best_saved
        write_json(output / "history.json", history)
        snapshot(epoch + 1, 0, None, 0., 0)
        last_save = time.monotonic()
        batches, start_offset = None, 0
    write_json(output / "complete.json", dict(seconds=elapsed_before + time.monotonic() - start, train_posts=len(train_frame),
               val_posts=len(val_frame), device=device, torch=torch.__version__,
               peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0,
               best_val_ap=best, recovery=json.loads((output / "recovery.json").read_text())
               if (output / "recovery.json").exists() else None))
    del net, optimizer, meta, y, times
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return output


def predict_checkpoint(path, frame, device=None, legacy=False):
    path = Path(path)
    if legacy:
        old = json.loads((path / "complete.json").read_text())["spec"]
        if old["objective"] != "regression" or not old["metadata"] or old["structured_text"]:
            raise ValueError("Legacy adapter supports the saved joint regression finalists only")
        spec = Spec(name=old["name"], max_length=old["max_length"], body_chars=old["body_chars"], batch_size=128)
    else:
        spec = Spec(**json.loads((path / "spec.json").read_text()))
    device = device or default_device()
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True, use_fast=spec.use_fast_tokenizer)
    net = Network(spec, path / "encoder").to(device)
    net.head.load_state_dict(torch.load(path / "head.pt", map_location=device, weights_only=True))
    expected = {k for k in net.state_dict() if not k.startswith(("encoder.", "head."))}
    if expected:
        extra = torch.load(path / "extra.pt", map_location=device, weights_only=True)
        if set(extra) != expected:
            raise ValueError(f"extra.pt has {sorted(extra)}, the network needs {sorted(expected)}")
        net.load_state_dict(extra, strict=False)
    prep = dict(np.load(path / "preprocessing.npz"))
    result = infer(net, tokenizer, frame, prep, device)
    del net
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def empirical_ranks(reference, values):
    """Fixed validation reference CDF: inference does not depend on batch peers."""
    reference = np.sort(np.asarray(reference))
    if not len(reference):
        raise ValueError("Empty rank reference")
    return np.searchsorted(reference, values, side="right") / len(reference)


def choose_blend(labels, first, second, weights=None):
    """Convex grid blend on a supplied selection partition only; simpler wins ties."""
    weights = np.arange(0, 1.001, .05) if weights is None else weights
    rows = [{"weight_second": float(w),
             "ap": float(average_precision_score(labels, (1 - w) * first + w * second))} for w in weights]
    best = max(rows, key=lambda r: (r["ap"], -r["weight_second"]))
    return best["weight_second"], rows


def bootstrap_delta(frame, reference, candidate, repetitions=500, seed=42):
    """Paired calendar-week bootstrap. Exploratory: does not cover selection bias."""
    weeks = frame.created_at.dt.to_period("W").astype(str).to_numpy()
    groups = [np.flatnonzero(weeks == week) for week in np.unique(weeks)]
    rng, samples = np.random.default_rng(seed), []
    for _ in range(repetitions):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        part = frame.iloc[indices]
        a, b = metrics(part, reference[indices]), metrics(part, candidate[indices])
        samples.append([b["ap"] - a["ap"], b["precision_top10"] - a["precision_top10"]])
    bounds = np.quantile(samples, [.025, .975], axis=0)
    return {"ap_delta_95ci": bounds[:, 0].tolist(), "precision_top10_delta_95ci": bounds[:, 1].tolist(),
            "repetitions": repetitions, "resampling": "paired calendar weeks"}
