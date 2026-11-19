"""LoRA fine-tune of Fastino's GLiDE-9B (Qwen3.5-9B decision model) on every subreddit. Experiment runner + loader.

    python glide_models.py --models glide_noul --seeds 42 --output data/glide/runs --results data/glide/results \\
        --dataset jbduran/textonly-subreddit-predictors --revision <sha> [--epochs N]
    python glide_models.py ... --max-train 8 --max-val 8 --epochs 1          # smoke run

The scaling test for d1_models.py: the same posts, the same state text (d1_models.state_text) and the same questions,
asked of a 9B model instead of a 0.6B one. GLiDE reads its whole prompt (full-attention layers see every token) and
answers with a 255-way readout on the last token; the first n logits are the question's options. LoRA adapters on every
linear layer of the backbone, plus the readout, are trained in bf16 with gradient checkpointing; it fits a 48 GB L40S.

    glide_noul     yes/no "top 10% of r/<sub> this quarter?"; cross-entropy on the label
    glide_score10  10 levels, the engagement decile within subreddit and quarter; cross-entropy

The checkpoint and the ranking score are chosen by macro AP on a 2024 validation sample. Test is never loaded.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

import d1_models as d1

BASE = "fastino/GLiDE-9B"
BASE_REVISION = "ab54df5acba0eebf7df45560ffdaacd64e32df3b"  # pinned: the model's own package (glide/) is imported from it
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
           "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Spec:
    readout: str                 # noul or score10
    epochs: int = 1
    batch_size: int = 32         # posts per optimizer step
    token_budget: int = 6144     # padded tokens per micro-batch (gradient accumulation fills batch_size)
    lr: float = 1e-4             # LoRA adapters
    head_lr: float = 1e-4        # the readout
    rank: int = 16
    alpha: int = 32
    weight_decay: float = 0.0
    warmup: float = 0.06
    max_len: int = 768           # tokens per prompt; longer states are cut (bodies first)
    state_chars: int = 2000      # characters of state text kept
    train_per_sub: int = 5_000
    val_per_sub: int = 1_000     # picks the ranking score; the experiment's eval_per_sub sample is the reported one
    seed: int = 42
    clean_train: bool = False    # drop likely silent removals (score 0 but commented) from training; eval is unchanged


MODELS = {"glide_noul": Spec("noul"), "glide_score10": Spec("score10")}
RANKERS = {"noul": ("yes_margin",), "score10": ("expected_level", "top_level")}


def rank_scores(readout, out):
    """GLiDE lists a noul's options as [true, false] (d1: [false, true]); score levels are in the same order."""
    out = np.asarray(out, dtype=np.float64)
    return d1.rank_scores(readout, out[:, ::-1] if readout == "noul" else out)


def silent_removals_dropped(frame):
    """Training posts without score-0-but-commented ones: mods (often AutoModerator) removed most of those without a
    marker, so their label says nothing about the text. Labels were set on all posts, so thresholds don't move."""
    return frame[~((frame.score == 0) & (frame.num_comments > 0))]


def targets(frame, readout):
    y = d1.targets(frame, readout)
    return 1 - y if readout == "noul" else y  # option 0 is "true"


# ---------------------------------------------------------------- model


def load_base(dev):
    """GLiDE in bf16 (as stored and served), with its own package from the pinned snapshot. Vision is not loaded."""
    import torch
    from huggingface_hub import snapshot_download

    path = snapshot_download(BASE, revision=BASE_REVISION, allow_patterns=["*.json", "*.safetensors", "*.jinja",
                                                                           "tokenizer*", "glide/*.py"])
    sys.path.insert(0, path)
    from glide import Glide
    from glide.prompt import options_for

    fast = dev == "cuda" and not os.environ.get("GLIDE_ALLOW_SLOW")  # GLIDE_ALLOW_SLOW=1 accepts the PyTorch path
    if fast:  # fail now, not hours in: without Triton the linear-attention layers fall back to slow paths
        import triton

        triton.runtime.driver.active.get_current_target()
        import fla.ops.gated_delta_rule  # noqa: F401  (raises the real error if the kernels can't load)
    model = Glide.from_pretrained(path, device=dev, dtype=torch.bfloat16, vision=False)
    if fast:
        import inspect

        import transformers.models.qwen3_5.modeling_qwen3_5 as QM

        if not inspect.getclosurevars(QM.torch_chunk_gated_delta_rule).nonlocals.get("is_new_implementation"):
            raise RuntimeError("transformers chose the slow PyTorch gated-delta-rule path: check the fla-core install")
    return model, options_for


class LoRALinear:
    """y = W x + (alpha / r) * B(A(x)) on a frozen nn.Linear, with A and B in float32. B starts at zero."""

    def __init__(self, base, rank, alpha):
        import torch
        from torch import nn

        self.base, self.scale = base, alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        base.lora = self
        base.register_parameter("lora_A", self.A)
        base.register_parameter("lora_B", self.B)
        original = base.forward

        def forward(x):
            return original(x) + ((x.float() @ self.A.T) @ self.B.T * self.scale).to(x.dtype)

        base.forward = forward


def add_lora(backbone, rank, alpha):
    from torch import nn

    wrapped = [name for name, mod in backbone.named_modules()
               if isinstance(mod, nn.Linear) and name.rsplit(".", 1)[-1] in TARGETS]
    for name in wrapped:
        LoRALinear(backbone.get_submodule(name), rank, alpha)
    return wrapped


class Readout:
    """GLiDE's backbone with LoRA adapters and its 255-way readout (in float32), differentiable."""

    def __init__(self, glide, spec):
        import torch

        self.glide, self.spec, self.torch = glide, spec, torch
        net = glide.network
        self.wrapped = add_lora(net.backbone, spec.rank, spec.alpha)
        self.readout = torch.nn.Linear(net.readout.in_features, net.readout.out_features, bias=False,
                                       device=net.readout.weight.device)
        with torch.no_grad():
            self.readout.weight.copy_(net.readout.weight.float())

    def parameters(self):
        lora = [p for n, p in self.glide.network.backbone.named_parameters() if n.endswith(("lora_A", "lora_B"))]
        return lora, list(self.readout.parameters())

    def train(self, mode=True):
        backbone = self.glide.network.backbone
        backbone.train(mode)
        if mode:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            backbone.gradient_checkpointing_disable()

    def __call__(self, seqs):
        """Logits over the question's options for token-id lists (left-padded, as GLiDE batches)."""
        from glide.model import attention_masks, left_pad, position_ids

        x, m = left_pad(seqs, self.glide.pad_id, self.glide.device)
        h = self.glide.network.backbone(input_ids=x, attention_mask=attention_masks(m), position_ids=position_ids(m),
                                        use_cache=False).last_hidden_state
        n = d1.LEVELS if self.spec.readout == "score10" else 2
        return self.readout(h[:, -1].float())[:, :n]

    def state_dict(self):
        backbone = self.glide.network.backbone
        lora = {k: v for k, v in backbone.state_dict().items() if k.endswith(("lora_A", "lora_B"))}
        return {f"backbone.{k}": v for k, v in lora.items()} | {"readout.weight": self.readout.weight}

    def load_state_dict(self, state):
        backbone = self.glide.network.backbone
        missing = backbone.load_state_dict({k[9:]: v for k, v in state.items() if k.startswith("backbone.")}, strict=False)
        if missing.unexpected_keys or any(k.endswith(("lora_A", "lora_B")) for k in missing.missing_keys):
            raise ValueError(f"LoRA weights do not match: {missing}")
        self.readout.weight.data.copy_(state["readout.weight"])


def encode_frame(frame, glide, options_for, spec):
    """GLiDE prompt token ids for every post. The state text is cut to spec.state_chars, and further for the few posts
    (emoji, non-Latin scripts) whose prompt still exceeds spec.max_len tokens."""
    rows = list(frame.itertuples(index=False))
    chars = [spec.state_chars] * len(rows)

    def encode(idx):
        texts = []
        for i in idx:
            q = d1.question(rows[i].subreddit, spec.readout)
            texts.append(glide.prompter.text(d1.state_text(rows[i])[:chars[i]], q, options_for(q)))
        return glide.tokenizer(texts, add_special_tokens=False)["input_ids"]

    ids = encode(range(len(rows)))
    for _ in range(8):
        long = [i for i, x in enumerate(ids) if len(x) > spec.max_len]
        if not long:
            return ids
        for i in long:  # shrink in proportion to the overflow, with a margin
            chars[i] = int(chars[i] * spec.max_len / len(ids[i]) * 0.9)
        for i, x in zip(long, encode(long)):
            ids[i] = x
    raise ValueError(f"{len(long)} prompts still exceed {spec.max_len} tokens")


def micro_batches(indices, lengths, budget):
    """Split a batch (sorted by length) into micro-batches of at most `budget` padded tokens."""
    order = sorted(indices, key=lambda i: lengths[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[i] > budget:
            out.append(cur)
            cur = []
        cur.append(i)
    return out + [cur] if cur else out


def infer(net, ids, budget):
    import torch

    net.train(False)
    out = [None] * len(ids)
    lengths = [len(x) for x in ids]
    with torch.no_grad():
        for mb in micro_batches(range(len(ids)), lengths, budget * 2):
            y = net([ids[i] for i in mb]).float().cpu().numpy()
            for j, i in enumerate(mb):
                out[i] = y[j]
    return np.stack(out)


def train(spec, train_frame, val_frame, out: Path):
    """LoRA fine-tune. Saves weights.safetensors (adapters + readout), spec.json, best.json and history.json."""
    import torch
    from safetensors.torch import save_file
    from transformers import get_linear_schedule_with_warmup

    if train_frame.created_at.max() >= val_frame.created_at.min():
        raise ValueError("Training must strictly precede validation")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    dev = d1.device()
    start = time.monotonic()
    glide, options_for = load_base(dev)
    net = Readout(glide, spec)
    tr_ids, va_ids = encode_frame(train_frame, glide, options_for, spec), encode_frame(val_frame, glide, options_for, spec)
    y = torch.tensor(targets(train_frame, spec.readout), device=dev)
    lengths = [len(x) for x in tr_ids]
    print(f"loaded and encoded {len(tr_ids):,} train / {len(va_ids):,} validation posts in "
          f"{time.monotonic() - start:.0f}s; mean prompt {np.mean(lengths):.0f} tokens; LoRA on {len(net.wrapped)} layers",
          flush=True)
    lora, head = net.parameters()
    optimizer = torch.optim.AdamW([{"params": lora, "lr": spec.lr}, {"params": head, "lr": spec.head_lr}],
                                  weight_decay=spec.weight_decay)
    steps = spec.epochs * math.ceil(len(tr_ids) / spec.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * spec.warmup), steps)
    history, best = [], None
    for epoch in range(1, spec.epochs + 1):
        net.train(True)
        total, seen, last = 0.0, 0, time.monotonic()
        for batch in d1.batches(lengths, spec.batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            for mb in micro_batches(batch, lengths, spec.token_budget):
                loss = torch.nn.functional.cross_entropy(net([tr_ids[i] for i in mb]), y[mb])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                (loss * len(mb) / len(batch)).backward()
                total += loss.item() * len(mb)
            torch.nn.utils.clip_grad_norm_(lora + head, 1.0)
            optimizer.step()
            scheduler.step()
            seen += len(batch)
            if time.monotonic() - last > 60:
                print(f"epoch {epoch}: {seen:,}/{len(tr_ids):,} posts, loss {total / seen:.4f}, "
                      f"{seen / (time.monotonic() - start):.1f} posts/s overall", flush=True)
                last = time.monotonic()
        candidates = rank_scores(spec.readout, infer(net, va_ids, spec.token_budget))
        scored = {name: d1.macro_scores(val_frame, p) for name, p in candidates.items()}
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
    """Ranking scores for `frame` from a run saved by train() (the pipeline/evaluate.py `glide` loader)."""
    from safetensors.torch import load_file

    model_dir = Path(model_dir)
    spec = Spec(**json.loads((model_dir / "spec.json").read_text()))
    rank_by = json.loads((model_dir / "best.json").read_text())["rank_by"]
    glide, options_for = load_base(d1.device())
    net = Readout(glide, spec)
    net.load_state_dict(load_file(str(model_dir / "weights.safetensors"), device=str(glide.device)))
    return rank_scores(spec.readout, infer(net, encode_frame(frame, glide, options_for, spec), spec.token_budget))[rank_by]


def main(argv=None):
    """The experiment runner contract (pipeline/experiment.py): one model and seed per call, validation only."""
    import pandas as pd

    from pipeline.data import load_eval

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output", type=Path, default=Path("data/glide/runs"))
    parser.add_argument("--results", type=Path, default=Path("data/glide/results"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true", help="skip runs that already finished")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-train", type=int, help="smoke runs only: posts per subreddit")
    parser.add_argument("--max-val", type=int, help="smoke runs only: posts per subreddit")
    parser.add_argument("--clean-train", action="store_true", help="drop likely silent removals from training")
    args = parser.parse_args(argv)

    rows = []
    for name in args.models:
        for seed in args.seeds:
            spec = replace(MODELS[name], seed=seed, **({"epochs": args.epochs} if args.epochs else {}),
                           **({"train_per_sub": args.max_train} if args.max_train else {}),
                           **({"val_per_sub": args.max_val} if args.max_val else {}),
                           **({"clean_train": True} if args.clean_train else {}))
            out = args.output / f"{name}_s{seed}"
            # shortcut: --resume skips finished runs but restarts unfinished ones from scratch
            if args.resume and (out / "complete.json").exists():
                best = json.loads((out / "best.json").read_text())
            else:
                data = load_eval(args.dataset, args.revision, body_chars=None, per_sub=spec.val_per_sub,
                                 validation_only=True, train_per_sub=spec.train_per_sub)
                train_frame = silent_removals_dropped(data.train) if spec.clean_train else data.train
                print(f"{name} seed {seed}: {len(train_frame):,} train posts, {len(data.val):,} validation posts", flush=True)
                best = train(spec, train_frame, data.val, out)
            rows.append({"model": name, "seed": seed, **{k: v for k, v in best.items() if k != "rankers"}})
    args.results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.results / "val_metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
