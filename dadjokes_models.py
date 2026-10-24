"""Models for one subreddit (dadjokes by default) that predict log engagement. `dadjokes_models.ipynb` runs them.

The target is log engagement, ln(1 + score) + ln(1 + comments). Evaluation asks the project question: do the
posts that the model ranks highest end up in the top 10% of their quarter? Models, from simple to sophisticated:

- ridge on TF-IDF + metadata (the reference from the loss comparison)
- gradient boosting on metadata + the out-of-fold prediction of a text model (stacking), which can learn
  interactions such as weekday x hour
- ridge on pretrained sentence embeddings (MiniLM), alone and with TF-IDF
- a fine-tuned transformer (MiniLM) on the text, stacked with metadata in gradient boosting

Typical use:
    data = load(Config())
    X = Features(data)
    res = Results(data)
    res.add("ridge (tf-idf + meta)", *fit_ridge(data, X.sparse(["meta", "tfidf"])))
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

import textonly_baselines as tb

SPLITS = tb.SPLITS
CACHE_DIR = Path("data/models")

# Metadata known at posting time. Categorical columns go to gradient boosting as categories.
CATEGORICAL = ["local_hour", "local_dow", "local_month"]
NUMERIC = ["log_posts_prev_hour", "log_posts_prev_day", "log_title_length", "title_words", "log_body_length",
           "has_body", "title_has_question", "title_has_number", "title_all_caps", "body_has_url",
           "has_author_flair", "log_author_prev_posts", "is_repost"]
META = CATEGORICAL + NUMERIC


@dataclass
class Config:
    subreddit: str = "dadjokes"
    train_start: str = "2011-01-01"
    train_end: str = "2024-01-01"
    val_end: str = "2025-01-01"
    target: str = "metric"           # "metric" (log engagement) or "metric_pct" (percentile in quarter)
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    smoothing: int = 20              # pseudo-posts for the weekday x hour success rates (textonly_baselines)
    seed: int = 42


# ---------------------------------------------------------------- data

def load(cfg):
    """All labeled text posts of one subreddit, split by time. Cached as Parquet per subreddit."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{cfg.subreddit}.parquet"
    if not path.exists():
        bcfg = tb.Config(source=f"data/textonly/data/{cfg.subreddit}/*/*.parquet", train_end=cfg.train_end,
                         val_end=cfg.val_end, max_train_per_sub=10**9, max_eval_per_sub=10**9)
        d = tb.load_data(bcfg)
        pd.concat([d.train.assign(split="train"), d.val.assign(split="val"), d.test.assign(split="test")]) \
            .to_parquet(path)
    df = pd.read_parquet(path)
    df = df[(df.split != "train") | (df.created_at >= pd.Timestamp(cfg.train_start))]
    parts = {s: df[df.split == s].sort_values("created_at").reset_index(drop=True) for s in SPLITS}
    data = tb.Data(cfg, None, parts["train"], parts["val"], parts["test"])
    print({s: len(parts[s]) for s in SPLITS})
    return data


def text_of(d):
    return (d.title + " \n " + d.body).tolist()


# ---------------------------------------------------------------- features

class Features:
    """Feature blocks per split: "meta" (one-hot timing + scaled numeric), "tfidf", and "embed" once added."""

    def __init__(self, data, max_features=50_000):
        tb.GROUPS = {k: v for k, v in tb.GROUPS.items() if k != "subreddit"}
        f = tb.build_features(data, text=False)
        self.blocks = {"meta": {s: f.combine(["timing", "structure", "history"], s) for s in SPLITS}}
        t0 = time.time()
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=max_features, sublinear_tf=True,
                                strip_accents="unicode", dtype=np.float32)
        self.blocks["tfidf"] = {"train": tfidf.fit_transform(text_of(data.train)),
                                "val": tfidf.transform(text_of(data.val)), "test": tfidf.transform(text_of(data.test))}
        self.tfidf = tfidf
        print(f"TF-IDF: {len(tfidf.vocabulary_):,} terms in {time.time() - t0:.0f}s")

    def add(self, name, by_split):
        self.blocks[name] = by_split

    def sparse(self, names):
        return {s: sparse.hstack([sparse.csr_matrix(self.blocks[n][s]) for n in names]).tocsr() for s in SPLITS}


def embed(data, model_name, batch_size=256):
    """Mean-pooled sentence embeddings of title + body, cached as .npy per split."""
    from sentence_transformers import SentenceTransformer

    tag = model_name.split("/")[-1]
    paths = {s: CACHE_DIR / f"{data.cfg.subreddit}_{tag}_{s}_{len(getattr(data, s))}.npy" for s in SPLITS}
    if not all(p.exists() for p in paths.values()):
        model = SentenceTransformer(model_name, device=device())
        for s in SPLITS:
            t0 = time.time()
            e = model.encode(text_of(getattr(data, s)), batch_size=batch_size, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
            np.save(paths[s], e.astype(np.float32))
            print(f"embedded {s}: {e.shape} in {time.time() - t0:.0f}s")
    return {s: np.load(paths[s]) for s in SPLITS}


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- evaluation

def scores(d, p):
    y, pct = d.label.to_numpy(), d.metric_pct.to_numpy()
    top = np.argsort(-p)[: len(y) // 10]
    return {"spearman": spearmanr(p, pct).statistic, "pr_auc": average_precision_score(y, p),
            "top10_hit": y[top].mean()}


class Results:
    def __init__(self, data):
        self.data, self.rows, self.preds = data, {}, {}

    def add(self, name, p_val, p_test, seconds=None):
        row = {f"val_{k}": v for k, v in scores(self.data.val, p_val).items()}
        row |= {f"test_{k}": v for k, v in scores(self.data.test, p_test).items()}
        row["fit_seconds"] = seconds
        self.rows[name], self.preds[name] = row, {"val": p_val, "test": p_test}
        print(f"{name}: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items() if k.startswith("test")), flush=True)
        return self.table().loc[[name]]

    def table(self):
        return pd.DataFrame(self.rows).T.round(4)


# ---------------------------------------------------------------- models

def target(data, split="train"):
    return getattr(data, split)[data.cfg.target].to_numpy()


def fit_ridge(data, X, alphas=(0.3, 1, 3, 10), return_model=False):
    """Ridge with alpha chosen on validation Spearman. Returns (p_val, p_test, seconds[, model])."""
    t0, best = time.time(), None
    for a in alphas:
        m = Ridge(alpha=a).fit(X["train"], target(data))
        v = spearmanr(m.predict(X["val"]), data.val.metric_pct).statistic
        if best is None or v > best[0]:
            best = (v, a, m)
    _, a, m = best
    print(f"  ridge alpha={a}")
    out = (m.predict(X["val"]), m.predict(X["test"]), time.time() - t0)
    return out + (m,) if return_model else out


def oof_ridge(data, X, alpha, folds=5):
    """Out-of-fold ridge predictions on train, and full-train predictions on val/test (for stacking).

    Folds are contiguous blocks of time, so a fold's prediction never comes from a model that saw its neighbors.
    """
    n = X["train"].shape[0]
    oof = np.zeros(n)
    for tr, te in KFold(folds, shuffle=False).split(np.arange(n)):
        oof[te] = Ridge(alpha=alpha).fit(X["train"][tr], target(data)[tr]).predict(X["train"][te])
    m = Ridge(alpha=alpha).fit(X["train"], target(data))
    return {"train": oof, "val": m.predict(X["val"]), "test": m.predict(X["test"])}


def meta_frame(data, split, extra=None):
    d = getattr(data, split)[META].copy()
    for c in CATEGORICAL:
        d[c] = d[c].astype("category")
    for name, by_split in (extra or {}).items():
        d[name] = by_split[split]
    return d


def fit_hgb(data, extra=None, max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200):
    """Gradient boosting on metadata (+ stacked text predictions in `extra`), early-stopped on a time holdout."""
    t0 = time.time()
    m = HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=1.0, categorical_features="from_dtype",
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=data.cfg.seed)
    m.fit(meta_frame(data, "train", extra), target(data))
    print(f"  hgb: {m.n_iter_} trees")
    return (m.predict(meta_frame(data, "val", extra)), m.predict(meta_frame(data, "test", extra)),
            time.time() - t0, m)


def finetune(data, model_name="sentence-transformers/all-MiniLM-L6-v2", max_len=96, batch_size=128, epochs=3,
             lr=5e-5, patience=1, folds=None):
    """Fine-tune a transformer encoder with a linear head to predict the target (squared error).

    Early stopping on validation Spearman. Returns predictions for train/val/test from the best epoch. With
    `folds`, the train predictions are out-of-fold (one model per fold), for stacking; that costs `folds` x time.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = device()
    tok = AutoTokenizer.from_pretrained(model_name)
    texts = {s: text_of(getattr(data, s)) for s in SPLITS}
    y_all = target(data).astype(np.float32)
    mu, sd = y_all.mean(), y_all.std()

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(model_name)
            self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

        def forward(self, ids, mask):
            h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
            pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            return self.head(pooled).squeeze(-1)

    def batches(txt, size):
        for i in range(0, len(txt), size):
            b = tok(txt[i:i + size], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            yield b["input_ids"].to(dev), b["attention_mask"].to(dev)

    @torch.no_grad()
    def predict(net, txt):
        net.eval()
        return np.concatenate([net(i, m).float().cpu().numpy() for i, m in batches(txt, 512)]) * sd + mu

    def train_one(tr_idx):
        torch.manual_seed(data.cfg.seed)
        net = Net().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * (len(tr_idx) // batch_size + 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.06)
        rng = np.random.default_rng(data.cfg.seed)
        tr_txt = [texts["train"][i] for i in tr_idx]
        tr_y = (y_all[tr_idx] - mu) / sd
        best, bad = (-np.inf, None), 0
        for ep in range(epochs):
            net.train()
            t0 = time.time()
            order = rng.permutation(len(tr_idx))
            for i in range(0, len(order), batch_size):
                j = order[i:i + batch_size]
                b = tok([tr_txt[k] for k in j], truncation=True, max_length=max_len, padding=True,
                        return_tensors="pt")
                pred = net(b["input_ids"].to(dev), b["attention_mask"].to(dev))
                loss = torch.nn.functional.mse_loss(pred, torch.tensor(tr_y[j], device=dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            pv = predict(net, texts["val"])
            v = spearmanr(pv, data.val.metric_pct).statistic
            print(f"  epoch {ep + 1}: val spearman {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if v > best[0]:
                best, bad = (v, {k: t.detach().clone() for k, t in net.state_dict().items()}), 0
            else:
                bad += 1
                if bad > patience:
                    break
        net.load_state_dict(best[1])
        return net

    t0 = time.time()
    n = len(texts["train"])
    if folds:
        oof = np.zeros(n)
        for f, (tr, te) in enumerate(KFold(folds, shuffle=False).split(np.arange(n))):
            print(f" fold {f + 1}/{folds}")
            oof[te] = predict(train_one(tr), [texts["train"][i] for i in te])
    net = train_one(np.arange(n))
    out = {"val": predict(net, texts["val"]), "test": predict(net, texts["test"])}
    out["train"] = oof if folds else predict(net, texts["train"])
    out["seconds"] = time.time() - t0
    return out
