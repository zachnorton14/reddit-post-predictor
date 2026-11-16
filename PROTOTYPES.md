# CSC422 model prototypes

This experiment adds two trainable neural architectures and a ranking-oriented
revision of the saved three-seed MiniLM + metadata/HGB ensemble. It concerns the
dadjokes prediction project only.

## Current handoff

The setup/punchline model completed all three epochs: best validation AP 0.2506.
Its checkpoint is ready in `data/prototypes/full/interaction`. The standalone
comparison so far is in `results/prototypes/interaction_review/`.

DeBERTa completed epoch one (validation AP 0.2470), then failed with a CUDA
unknown error early in epoch two. Its epoch-one encoder, head and validation
predictions remain intact. The original trainer did not save optimizer state.
Use the recovery command below to keep those weights and train the remaining
two epochs with a fresh optimizer/warmup schedule. This is a recorded warm start,
so it differs from a continuous three-epoch run.

A 512-example GPU check with
every input padded to the maximum 96-token length passed at about 6.0 GiB peak
allocated memory. That check is for execution/memory only, not model quality.
An earlier, slower checkpointed pilot was interrupted and preserved under
`data/prototypes/checkpointed_pilot/`; its incomplete DeBERTa weights are not used.

The recovery command reuses the completed interaction checkpoint and the saved
DeBERTa epoch-one weights. It writes the full comparison after training finishes.
Future interruptions recover full training state from the last successfully
written snapshot. No long training run is launched by the recovery fix.

## Architectures

| Candidate | Architecture | Question |
|---|---|---|
| Ranking blend | Existing three-seed MiniLM average and metadata HGB; fixed-reference percentile scores; convex blend selected for average precision | Does selecting blend weights for ranking work better than the old least-squares blend? |
| Setup/punchline MiniLM | Shared fine-tuned MiniLM encodes title and cleaned body separately; concatenate title, body, absolute difference, elementwise product and 16 metadata features; 128-unit ReLU, 10% dropout, scalar head | Does an explicit comparison of setup and punchline help? |
| DeBERTa + metadata | Fine-tuned DeBERTa-v3-base, masked mean pooling, 16 metadata features, 128-unit ReLU, 10% dropout, scalar head | Does a larger language encoder learn signal that MiniLM misses? |

MiniLM uses a 384-dimensional text representation. DeBERTa-v3-base has 12 layers
and a 768-dimensional representation (about 184M total parameters including its
large vocabulary embeddings). Sources: [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
and [Microsoft DeBERTa model card](https://huggingface.co/microsoft/deberta-v3-base).

Both new networks regress standardized log engagement with MSE. The metadata is
the posting-time allowlist in `dadjokes_models.META`; score, comments, label and
engagement percentile are never input features. Standardization uses training
rows only. A title/body split approximates setup/punchline; some jokes put both
in the title, so the interpretation is imperfect.

The interaction model gives each branch 64 tokens and keeps the body's ending
when it exceeds its budget. DeBERTa keeps the existing 96-token combined input
and 1,000-character body cap. Thus these are architecture packages, not isolated
causal ablations of a single change. DeBERTa uses its native SentencePiece
tokenizer rather than a fast conversion that drops byte-fallback behavior.

## Evaluation protocol

- Train on all 174,043 posts from 2019 through 2023; validate on all 29,760
  posts in 2024. Test is the same historical 32,826 posts from Jan 2025–Feb 2026.
- Each prototype trains for three epochs, seed 42, effective batch size 128.
  Checkpoint selection uses full-2024 average precision. MiniLM learning rate is
  3e-5; DeBERTa is 2e-5. Both use AdamW, 6% linear warmup, linear decay, gradient
  clipping and mixed precision. DeBERTa uses a microbatch of 32 with four gradient
  accumulation steps; gradient checkpointing is available but disabled after a
  worst-case sequence-length memory check. Four CPU threads feed the GPU. The
  interaction run used 16 CPU threads and microbatches of 64. Its two passes share
  MiniLM weights.
- Tune each blend on Jan–Jun 2024 using a coarse convex-weight grid. The existing
  text/metadata blend allows 0–50% HGB; each prototype/incumbent blend allows
  0–100% prototype. Compute rank transformations against fixed Jan–Jun reference
  distributions, so prediction does not change with inference-batch composition.
- Gate replacement on Jul–Dec 2024 AP, requiring a gain of at least 0.002 over the
  incumbent. Keep all standalone candidates in the report. Freeze selection
  before generating test predictions. Test performance never chooses the winner.
- The validation gate is **not untouched**: checkpoint selection and the prior
  ensemble already used 2024. Test was also inspected in prior project work.
  Treat results as exploratory; a new prospective period is needed for a clean
  final estimate. Weekly bootstrap intervals do not cover selection uncertainty.
- The reference averages three seeds; these prototypes initially use one seed
  each. A gain should be replicated before being described as robust.

AP is sklearn average precision, called PR-AUC elsewhere in this repository.
Top-10% success is precision among the highest-ranked tenth, not classification
accuracy. No change to the quarter-based label definition is made.

## Run locally

From the repository root, with the existing `.venv` and local pretrained models:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TOKENIZERS_PARALLELISM='false'
.\.venv\Scripts\python.exe run_prototypes.py --report
```

For the existing interrupted run, use this instead:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TOKENIZERS_PARALLELISM='false'
.\.venv\Scripts\python.exe -u run_prototypes.py --report --threads 4 --resume
```

Keep the laptop plugged in with its lid open. The runner requests that Windows
prevent **idle** sleep during training/report generation and restores the previous
request on exit. It does not override lid closure, explicit sleep, shutdown or
driver resets. It leaves the display free to turn off and does not edit power
settings. See [Microsoft's execution-state documentation](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate).

The new trainer atomically saves `resume.pt` about every five minutes (after a
complete optimizer update), initially, and at each epoch boundary. Rerun the same
command with `--resume` after an interruption. A failed CUDA process exits; recovery
starts in a fresh process. Snapshots include optimizer, scheduler, scaler, exact
batch position/order, metrics/history and random state. An interrupted snapshot
write preserves the prior file. `best_checkpoint.pt` separately preserves the
validation-selected model and rebuilds inference files during recovery.

Full DeBERTa snapshots require several GB of disk space, temporarily more during
atomic replacement. Use `--checkpoint-seconds 120` for more frequent saves. The
first recovery from the old run logs its optimizer reset in `recovery.json` and
keeps the original identity in `identity.before_recovery.json`.

The input loader reuses a single manifested, full-body dadjokes Parquet cache
from `data/models/`, or rebuilds from `data/textonly/data/dadjokes/*/*.parquet`.
Use `--data PATH` if multiple caches match. The full comparison requires the
saved reference at `data/improvements/full/`: its selection manifest, three
finalist checkpoints, classical predictions/blend and ID-aligned test scores.
These ignored data files are local prerequisites, not included in Git. Training
the new prototypes without `--report` does not require the saved reference.

Pretrained checkpoints, if absent, can be downloaded as ordinary files on Windows:

```powershell
.\.venv\Scripts\hf.exe download sentence-transformers/all-MiniLM-L6-v2 --local-dir data/pretrained/minilm --include config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt
.\.venv\Scripts\hf.exe download microsoft/deberta-v3-base --local-dir data/pretrained/deberta --include config.json pytorch_model.bin spm.model tokenizer_config.json
```

Offline smoke test and integration tests:

```powershell
.\.venv\Scripts\python.exe run_prototypes.py --output data/prototypes/smoke_new --max-train 128 --max-val 128 --epochs 1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Sample caps are explicitly prohibited for full-data reporting. Completed runs
can be reused only when configuration, ordered training/validation data,
pretrained file hashes and implementation hashes match. An explicitly allowlisted
recovery upgrade accepts this checkout's predecessor without invalidating its
completed MiniLM model; data and configuration checks still apply. The original
protocol remains in place, and `execution.json` records the new runner and recovery
settings. Other implementation/configuration changes require a new `--output`.
Incomplete runs require `--resume`; they are never silently restarted. GPU seeds are set,
but bitwise reproducibility across CUDA/hardware versions is not guaranteed.

Rebuild the report without retraining:

```powershell
.\.venv\Scripts\python.exe run_prototypes.py --report-only
.\.venv\Scripts\python.exe analyze_prototypes.py
```

## Artifacts and inference

- `data/prototypes/full/<model>/`: encoder, tokenizer, head, preprocessing,
  spec, input/code/pretrained fingerprints, epoch history and best checkpoint.
- `data/prototypes/full/selection.json` and `rank_references.npz`: frozen ensemble
  rule and reference distributions.
- `results/prototypes/full/`: metrics, summary, standalone PNG/SVG comparison,
  paired weekly-bootstrap intervals and run manifest.
- `analyze_prototypes.py` adds quarterly and seen/novel-title breakdowns,
  a quarter stability chart, changes in top-10% picks, and concrete error examples.
  It only analyzes frozen predictions; it never changes model selection.
- `prototype_models.predict_checkpoint(path, frame)` scores a new prototype.
  The frame must contain title, cleaned body, and the same 16 derived posting-time
  metadata columns. It does not require engagement outcomes or labels.
- `run_prototypes.apply_pipeline(...)` applies the frozen ensemble to aligned
  component predictions; it never fits on the inference batch.

Score a Parquet table of derived features without any labels or outcomes:

```powershell
.\.venv\Scripts\python.exe predict_prototypes.py data/new_post_features.parquet data/new_predictions.parquet --model selected
```

Use `--model interaction` or `--model deberta` for a standalone prototype. The
output is a ranking score, **not** a calibrated success probability. Creating
posting-time history features for a brand-new raw post still requires the same
historical corpus used by the existing feature pipeline.

The previous experiment's scripts were absent in this checkout, while its data
and saved models remained. This workflow is self-contained and checks reference
validation scores against reloaded checkpoints plus test rows against stored
IDs, rather than depending on the missing scripts.
