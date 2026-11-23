"""Fine-tune Liquid AI's d1-omni-600M to rank text posts by engagement on every subreddit. Experiment runner + loader.

    python d1_models.py --models d1_score10 --seeds 42 --output data/d1/runs --results data/d1/results \\
        --dataset jbduran/textonly-subreddit-predictors --revision <sha> [--epochs N]
    python d1_models.py ... --max-train 64 --max-val 64 --epochs 1          # smoke run on a laptop

d1 reads a post as a "state" and answers a question about it with its own decision head, so the posting context
(subreddit, local time, how busy the subreddit was, the author's history here) is written into the state text.
Readouts (MODELS), all fine-tuned end to end:

    d1_noul      yes/no "in the top 10% of r/<sub> this quarter?"; cross-entropy on the label
    d1_score10   10 levels, the post's engagement decile within its subreddit and quarter; cross-entropy
    d1_reg       mean of the trunk's token states -> new MLP -> engagement percentile within subreddit and quarter; MSE

Trains on every subreddit before 2024 (up to Spec.train_per_sub posts each). The checkpoint, and for the native
readouts the ranking score (RANKERS), are chosen by macro-average AP on a 2024 validation sample. Test is never loaded.
The experiment workflow then scores the saved run on all of 2024 validation with pipeline.evaluate (type `d1`).
"""

import argparse
import importlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

BASE = "LiquidAI/d1-omni-600M"
BASE_REVISION = "02b55d7076f15129e59ab3f94783f32c4b088674"  # pinned: the model runs remote code
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
LEVELS = 10


@dataclass(frozen=True)
class Spec:
    readout: str                 # noul, score10 or reg
    max_len: int = 320           # tokens: question and options first, then the state, cut on the right
    epochs: int = 2
    batch_size: int = 32
    lr: float = 2e-5             # trunk (pretrained)
    head_lr: float = 1e-4        # decision head (pretrained) or the new regression MLP
    weight_decay: float = 0.01
    warmup: float = 0.06
    train_per_sub: int = 25_000  # training posts per subreddit (labels use all posts)
    val_per_sub: int = 3_000     # validation posts per subreddit for choosing the checkpoint
    seed: int = 42


MODELS = {
    "d1_noul": Spec("noul"),
    "d1_score10": Spec("score10"),
    "d1_reg": Spec("reg", head_lr=1e-3),
}
# Ranking scores computed from a readout's outputs; the best on validation AP is saved as `rank_by`
RANKERS = {"noul": ("yes_margin",), "score10": ("expected_level", "top_level"), "reg": ("prediction",)}


# ---------------------------------------------------------------- text


def state_text(r) -> str:
    """One post with its posting context, as d1 reads it. `r` is a row of the labeled frame."""
    prev_hour, prev_day = round(math.expm1(r.log_posts_prev_hour)), round(math.expm1(r.log_posts_prev_day))
    author = round(math.expm1(r.log_author_prev_posts))
    lines = [f"Subreddit: r/{r.subreddit}",
             f"Posted: {DAYS[int(r.local_dow) - 1]} {int(r.local_hour):02d}:00 New York time, "
             f"{MONTHS[int(r.local_month) - 1]}",
             f"Subreddit activity: {prev_hour} posts in the previous hour, {prev_day} in the previous day",
             f"Author: {author} earlier posts in this subreddit" + (", has a user flair" if r.has_author_flair else ""),
             *(["The same title was posted here before."] if r.is_repost else []),
             f"Title: {r.title}"]
    if isinstance(r.body, str) and r.body.strip():
        lines.append(f"Body: {r.body}")
    return "\n".join(lines)


def question(subreddit, readout) -> dict:
    """The question asked of every post (d1's Decision Index schema). The regression readout reuses the noul one."""
    if readout == "score10":
        return {"type": "score", "instructions": f"Compared with the other text posts on r/{subreddit} in the same "
                                                 f"quarter, how much engagement (upvotes plus comments) will this "
                                                 f"post get?",
                "criteria": [f"in the {name} tenth of posts" for name in
                             ("lowest", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
                              "highest")]}
    return {"type": "noul", "instructions": f"Will this post be in the top 10% of text posts on r/{subreddit} by "
                                            f"engagement (upvotes plus comments) this quarter?",
            "criteria": {"true": "yes, top 10%", "false": "no"}}


def targets(frame, readout) -> np.ndarray:
    if readout == "noul":
        return frame.label.to_numpy(np.int64)  # option order is [false, true]
    if readout == "score10":
        return np.minimum((frame.metric_pct.to_numpy() * LEVELS).astype(np.int64), LEVELS - 1)
    return frame.metric_pct.to_numpy(np.float32)


def rank_scores(readout, out) -> dict:
    """Candidate ranking scores (higher = more engagement) from the raw outputs of a batch of posts."""
    out = np.asarray(out, dtype=np.float64)
    if readout == "noul":
        return {"yes_margin": out[:, 1] - out[:, 0]}
    if readout == "score10":
        z = out - out.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return {"expected_level": p @ np.arange(LEVELS), "top_level": np.log(p[:, -1] + 1e-12)}
    return {"prediction": out[:, 0]}


# ---------------------------------------------------------------- model


def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_base(dev):
    """d1 in float32 (as trained; the model card advises against bfloat16), its tokenizer and its prompt module."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(BASE, revision=BASE_REVISION, trust_remote_code=True, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION, trust_remote_code=True)
    prompt = importlib.import_module(type(model).__module__.rsplit(".", 1)[0] + ".prompt")
    for part in (model.vision, model.audio):  # text only
        part.requires_grad_(False)
    return model.to(dev), tok, prompt


class Readout:
    """d1's trunk and decision head (or a new regression MLP on the trunk), differentiable."""

    def __init__(self, model, readout):
        import torch
        from torch import nn

        self.model, self.readout = model, readout
        width = model.config.text_config["hidden_size"]
        self.reg = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 1)).to(model.device) \
            if readout == "reg" else None
        self.qtype = 1 if readout == "score10" else 2  # prompt.QTYPES: score 1, noul 2
        self.torch = torch

    def parameters(self):
        """(pretrained trunk params, head params): the head is d1's decision head or the new MLP."""
        head = self.reg if self.reg is not None else self.model.head
        return list(self.model.encoder.parameters()), list(head.parameters())

    def train(self, mode=True):
        self.model.train(mode)
        if self.reg is not None:
            self.reg.train(mode)

    def __call__(self, ids, markers):
        """Raw outputs: option logits (noul [false, true], score levels) or the regression prediction."""
        torch, dev = self.torch, self.model.device
        lengths = torch.tensor([len(x) for x in ids], device=dev)
        width = int(lengths.max())
        tokens = torch.zeros(len(ids), width, dtype=torch.long, device=dev)
        for i, x in enumerate(ids):
            tokens[i, :len(x)] = torch.tensor(x, device=dev)
        pad = torch.arange(width, device=dev)[None] < lengths[:, None]
        h = self.model.encoder.embed_tokens(tokens) * pad[..., None]  # as d1 pads: zero vectors
        h = self.model.encoder(h, pad, torch.zeros(len(ids), dtype=torch.long, device=dev))
        if self.reg is not None:
            pooled = (h * pad[..., None]).sum(1) / lengths[:, None]
            return self.reg(pooled)
        k = max(len(m) for m in markers)
        pos = torch.zeros(len(ids), k, dtype=torch.long, device=dev)
        for i, m in enumerate(markers):
            pos[i, :len(m)] = torch.tensor(m, device=dev)
        mask = torch.ones(len(ids), k, dtype=torch.bool, device=dev)
        qtype = torch.full((len(ids),), self.qtype, dtype=torch.long, device=dev)
        return self.model.head(h, pad, pos, mask, qtype)

    def state_dict(self):
        state = {f"encoder.{k}": v for k, v in self.model.encoder.state_dict().items()}
        head = self.reg if self.reg is not None else self.model.head
        return state | {f"{'reg' if self.reg is not None else 'head'}.{k}": v for k, v in head.state_dict().items()}

    def load_state_dict(self, state):
        self.model.encoder.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("encoder.")})
        head, prefix = (self.reg, "reg.") if self.reg is not None else (self.model.head, "head.")
        head.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})


def encode_frame(frame, tok, prompt, spec):
    """Token ids and option-marker positions for every post."""
    ids, markers = [], []
    for r in frame.itertuples(index=False):
        q = prompt.as_question(question(r.subreddit, spec.readout))
        i, m = prompt.encode(tok, state_text(r), q, spec.max_len)
        ids.append(i)
        markers.append(m)
    return ids, markers


def batches(lengths, size, rng=None):
    """Indices in batches of similar length (less padding). Shuffled batches when `rng` is given."""
    order = np.argsort(lengths, kind="stable") if rng is None else rng.permutation(len(lengths))
    if rng is not None:  # sort within chunks of 50 batches, so batches are similar in length but still random
        order = np.concatenate([c[np.argsort(np.asarray(lengths)[c], kind="stable")]
                                for c in np.array_split(order, max(1, len(order) // (size * 50)))])
    out = [order[i:i + size] for i in range(0, len(order), size)]
    if rng is not None:
        rng.shuffle(out)
    return out


def infer(net, ids, markers, batch_size=64):
    import torch

    net.train(False)
    out = [None] * len(ids)
    with torch.no_grad():
        for b in batches([len(x) for x in ids], batch_size):
            y = net([ids[i] for i in b], [markers[i] for i in b]).float().cpu().numpy()
            for j, i in enumerate(b):
                out[i] = y[j]
    return np.stack(out)


def macro_scores(frame, p):
    from pipeline.evaluate import MIN_POSTS, metrics

    # a smoke run's tiny sample still gets a macro average
    m = metrics(frame, p, "val", min_posts=min(MIN_POSTS, frame.subreddit.value_counts().max()))
    return {"ap": m["pr_auc"], "precision_top10": m["top10_hit_rate"], "spearman": m["spearman"]}


def train(spec, train_frame, val_frame, out: Path):
    """Fine-tune one readout. Saves weights.safetensors, spec.json, best.json and history.json in `out`."""
    import torch
    from safetensors.torch import save_file
    from transformers import get_linear_schedule_with_warmup

    if train_frame.created_at.max() >= val_frame.created_at.min():
        raise ValueError("Training must strictly precede validation")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    dev = device()
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # float32 weights, TF32 matmuls
        torch.backends.cudnn.allow_tf32 = True
    model, tok, prompt = load_base(dev)
    net = Readout(model, spec.readout)
    start = time.monotonic()
    tr_ids, tr_markers = encode_frame(train_frame, tok, prompt, spec)
    va_ids, va_markers = encode_frame(val_frame, tok, prompt, spec)
    y = torch.tensor(targets(train_frame, spec.readout), device=dev)
    print(f"encoded {len(tr_ids):,} train and {len(va_ids):,} validation posts in {time.monotonic() - start:.0f}s; "
          f"mean length {np.mean([len(x) for x in tr_ids]):.0f} tokens", flush=True)

    trunk, head = net.parameters()
    decay = lambda ps: [p for p in ps if p.ndim > 1]  # noqa: E731
    no_decay = lambda ps: [p for p in ps if p.ndim <= 1]  # noqa: E731
    optimizer = torch.optim.AdamW([
        {"params": decay(trunk), "lr": spec.lr, "weight_decay": spec.weight_decay},
        {"params": no_decay(trunk), "lr": spec.lr, "weight_decay": 0.0},
        {"params": decay(head), "lr": spec.head_lr, "weight_decay": spec.weight_decay},
        {"params": no_decay(head), "lr": spec.head_lr, "weight_decay": 0.0}])
    steps = spec.epochs * math.ceil(len(tr_ids) / spec.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * spec.warmup), steps)
    lengths = [len(x) for x in tr_ids]
    history, best = [], None
    for epoch in range(1, spec.epochs + 1):
        net.train(True)
        total, seen, last = 0.0, 0, time.monotonic()
        for b in batches(lengths, spec.batch_size, rng):
            outputs = net([tr_ids[i] for i in b], [tr_markers[i] for i in b])
            if spec.readout == "reg":
                loss = torch.nn.functional.mse_loss(outputs[:, 0].float(), y[b])
            else:
                loss = torch.nn.functional.cross_entropy(outputs.float(), y[b])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trunk + head, 1.0)
            optimizer.step()
            scheduler.step()
            total, seen = total + loss.item() * len(b), seen + len(b)
            if time.monotonic() - last > 60:
                print(f"epoch {epoch}: {seen:,}/{len(tr_ids):,} posts, loss {total / seen:.4f}, "
                      f"{seen / (time.monotonic() - start):.0f} posts/s overall", flush=True)
                last = time.monotonic()
        candidates = rank_scores(spec.readout, infer(net, va_ids, va_markers))
        scored = {name: macro_scores(val_frame, p) for name, p in candidates.items()}
        rank_by = max(scored, key=lambda n: scored[n]["ap"])
        row = {"epoch": epoch, "train_loss": total / seen, "seconds": time.monotonic() - start, "rank_by": rank_by,
               **scored[rank_by], "rankers": scored}
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or row["ap"] > best["ap"]:
            best = row
            save_file({k: v.detach().contiguous().cpu() for k, v in net.state_dict().items()},
                      str(out / "weights.safetensors"))
            (out / "best.json").write_text(json.dumps(best, indent=2))
        (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "spec.json").write_text(json.dumps(asdict(spec), indent=2))
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
    (out / "complete.json").write_text(json.dumps({"seconds": time.monotonic() - start, "device": dev,
                                                   "train_posts": len(tr_ids), "val_posts": len(va_ids),
                                                   "peak_gpu_gib": peak, "base": BASE, "base_revision": BASE_REVISION},
                                                  indent=2))
    return best


def predict(model_dir, frame):
    """Ranking scores for `frame` from a run saved by train() (the pipeline/evaluate.py `d1` loader)."""
    from safetensors.torch import load_file

    model_dir = Path(model_dir)
    spec = Spec(**json.loads((model_dir / "spec.json").read_text()))
    rank_by = json.loads((model_dir / "best.json").read_text())["rank_by"]
    model, tok, prompt = load_base(device())
    net = Readout(model, spec.readout)
    net.load_state_dict(load_file(str(model_dir / "weights.safetensors"), device=str(model.device)))
    ids, markers = encode_frame(frame, tok, prompt, spec)
    return rank_scores(spec.readout, infer(net, ids, markers))[rank_by]


# ---------------------------------------------------------------- runner


def main(argv=None):
    """The experiment runner contract (pipeline/experiment.py): one model and seed per call, validation only."""
    import pandas as pd

    from pipeline.data import load_eval

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output", type=Path, default=Path("data/d1/runs"))
    parser.add_argument("--results", type=Path, default=Path("data/d1/results"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true", help="skip runs that already finished")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset repo")
    parser.add_argument("--revision", required=True, help="dataset commit")
    parser.add_argument("--max-train", type=int, help="smoke runs only: posts per subreddit")
    parser.add_argument("--max-val", type=int, help="smoke runs only: posts per subreddit")
    args = parser.parse_args(argv)

    rows = []
    for name in args.models:
        for seed in args.seeds:
            spec = replace(MODELS[name], seed=seed, **({"epochs": args.epochs} if args.epochs else {}),
                           **({"train_per_sub": args.max_train} if args.max_train else {}),
                           **({"val_per_sub": args.max_val} if args.max_val else {}))
            out = args.output / f"{name}_s{seed}"
            # shortcut: --resume skips finished runs but restarts unfinished ones from scratch; add mid-run
            # checkpoints if runs grow past a few hours
            if args.resume and (out / "complete.json").exists():
                print(f"{name} seed {seed}: already finished", flush=True)
                best = json.loads((out / "best.json").read_text())
            else:
                data = load_eval(args.dataset, args.revision, body_chars=None, per_sub=spec.val_per_sub,
                                 validation_only=True, train_per_sub=spec.train_per_sub)
                train_frame = data.train[data.train.created_at < pd.Timestamp("2024-01-01")].reset_index(drop=True)
                print(f"{name} seed {seed}: {len(train_frame):,} train posts, {len(data.val):,} validation posts, "
                      f"{train_frame.subreddit.nunique()} subreddits", flush=True)
                best = train(spec, train_frame, data.val, out)
            rows.append({"model": name, "seed": seed, **{k: v for k, v in best.items() if k != "rankers"}})
    args.results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.results / "val_metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
