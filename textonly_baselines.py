"""Baseline models for the text-only subreddit dataset. `textonly_baselines.ipynb` runs and plots them.

Features are built in named groups (subreddit, timing, structure, history, text), so that an ablation can
measure what each group adds, and new groups (for example text embeddings) or larger models plug in later.

Typical use:
    cfg = Config()
    data = load_data(cfg)
    feats = build_features(data)
    res = Results(data)
    run_ladder(data, feats, res)
    run_ablation(data, feats)
"""

import copy
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, precision_recall_curve, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import textonly_data as td

SPLITS = ("train", "val", "test")


@dataclass
class Config:
    source: str = "local"            # "local", "hf", or a Parquet glob
    label_metric: str = "engagement"  # "engagement", "score", or "comments"
    success_quantile: float = 0.9     # success = metric above this quantile of its subreddit and period
    period: str = "quarter"           # "month", "quarter", or "year"
    min_period_posts: int = 100       # subreddit-periods with fewer text posts are dropped
    self_only: bool = True            # keep only text (self) posts
    train_end: str = "2024-01-01"     # train: posts before this date
    val_end: str = "2025-01-01"       # validation: train_end to val_end; test: val_end and later
    max_train_per_sub: int = 150_000  # cap per subreddit, so AskReddit does not dominate the pooled model
    max_eval_per_sub: int = 30_000    # cap per subreddit in validation and in test
    body_chars: int = 1000            # characters of the body (before any EDIT marker) used as text
    tfidf_max_features: int = 50_000
    tfidf_min_df: int = 5
    smoothing: int = 20               # pseudo-posts for the subreddit x time lookup rates
    seed: int = 42


# ---------------------------------------------------------------- data

@dataclass
class Data:
    cfg: Config
    con: object
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def y(self, split):
        return getattr(self, split).label.to_numpy()


def load_data(cfg):
    """Label all posts, then load a capped random sample per subreddit and split into pandas."""
    con = td.connect(cfg.source)
    td.add_labels(con, cfg.label_metric, cfg.success_quantile, cfg.period, cfg.min_period_posts, cfg.self_only)
    split_sql = (f"CASE WHEN created_at < TIMESTAMP '{cfg.train_end}' THEN 'train' "
                 f"WHEN created_at < TIMESTAMP '{cfg.val_end}' THEN 'val' ELSE 'test' END")
    df = con.execute(f"""
    WITH s AS (SELECT *, {split_sql} AS split FROM labeled)
    SELECT split, label, subreddit, period, created_at,
           local_hour, local_dow, local_month,
           title, left(body_clean, {cfg.body_chars})       AS body,
           ln(1 + title_length)                             AS log_title_length,
           title_words,
           ln(1 + body_length)                              AS log_body_length,
           has_body, title_has_question, title_has_number, title_all_caps, body_has_url,
           has_link_flair, has_author_flair,
           ln(1 + posts_prev_hour)                          AS log_posts_prev_hour,
           ln(1 + posts_prev_day)                           AS log_posts_prev_day,
           ln(1 + author_prev_posts)                        AS log_author_prev_posts,
           (title_prev_count > 0)::INT                      AS is_repost
    FROM s
    QUALIFY row_number() OVER (PARTITION BY split, subreddit ORDER BY hash(id || '{cfg.seed}'))
            <= CASE WHEN split = 'train' THEN {cfg.max_train_per_sub} ELSE {cfg.max_eval_per_sub} END
    """).df()
    parts = {s: df[df.split == s].sort_values("created_at").reset_index(drop=True) for s in SPLITS}
    for s, d in parts.items():
        if d.empty:
            raise ValueError(f"The {s} split is empty. Check train_end/val_end against the extracted months.")
    return Data(cfg, con, parts["train"], parts["val"], parts["test"])


def summary(data):
    """Posts and success rate per split and subreddit."""
    rows = []
    for s in SPLITS:
        d = getattr(data, s)
        g = d.groupby("subreddit").agg(posts=("label", "size"), success_rate=("label", "mean"),
                                       first=("created_at", "min"), last=("created_at", "max"))
        g["split"] = s
        rows.append(g.reset_index())
    return pd.concat(rows).set_index(["split", "subreddit"])


# ---------------------------------------------------------------- features

GROUPS = {
    "subreddit": {"categorical": ["subreddit"]},
    "timing": {"categorical": ["local_hour", "local_dow", "local_month"],
               "numeric": ["log_posts_prev_hour", "log_posts_prev_day", "sub_hour_rate", "sub_dow_hour_rate"]},
    # No link flair: the archived flair is the final one, and moderators change it after posting
    # (Showerthoughts flairs posts "Pending" until they are reviewed)
    "structure": {"numeric": ["log_title_length", "title_words", "log_body_length"],
                  "binary": ["has_body", "title_has_question", "title_has_number", "title_all_caps",
                             "body_has_url", "has_author_flair"]},
    "history": {"numeric": ["log_author_prev_posts"], "binary": ["is_repost"]},
}
TABULAR = list(GROUPS)


def add_time_rates(data):
    """Add smoothed training success rates per subreddit x hour and subreddit x weekday x hour.

    These let a linear model use a different daily pattern in each subreddit. They come from the training
    sample only. Training posts count toward their own cell's rate; the smoothing keeps that effect small.
    """
    k = data.cfg.smoothing
    tr = data.train
    sub_rate = tr.groupby("subreddit").label.mean().rename("sub_rate")
    for name, keys in [("sub_hour_rate", ["subreddit", "local_hour"]),
                       ("sub_dow_hour_rate", ["subreddit", "local_dow", "local_hour"])]:
        cell = tr.groupby(keys).label.agg(["sum", "count"]).reset_index().join(sub_rate, on="subreddit")
        cell[name] = (cell["sum"] + k * cell.sub_rate) / (cell["count"] + k)
        for s in SPLITS:
            d = getattr(data, s).drop(columns=name, errors="ignore")
            d = d.merge(cell[keys + [name]], on=keys, how="left")
            d[name] = d[name].fillna(d.subreddit.map(sub_rate)).fillna(tr.label.mean())
            setattr(data, s, d)


class Features:
    """Feature matrices per group and split: feats[group][split] -> sparse CSR matrix."""

    def __init__(self):
        self.matrices, self.names, self.transformers = {}, {}, {}

    def __getitem__(self, group):
        return self.matrices[group]

    def combine(self, groups, split):
        return sparse.hstack([self.matrices[g][split] for g in groups]).tocsr()

    def feature_names(self, groups):
        return [n for g in groups for n in self.names[g]]


def build_features(data, text=True):
    """Fit each feature group on the training split and transform all splits."""
    add_time_rates(data)
    feats = Features()
    for group, spec in GROUPS.items():
        parts = []
        if spec.get("numeric"):
            parts.append(("num", StandardScaler(), spec["numeric"]))
        if spec.get("binary"):
            parts.append(("bin", "passthrough", spec["binary"]))
        if spec.get("categorical"):
            parts.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), spec["categorical"]))
        ct = ColumnTransformer(parts, sparse_threshold=1.0)
        feats.matrices[group] = {
            s: sparse.csr_matrix(ct.fit_transform(data.train) if s == "train" else ct.transform(getattr(data, s)),
                                 dtype=np.float32)
            for s in SPLITS}
        feats.names[group] = [n.split("__", 1)[1] for n in ct.get_feature_names_out()]
        feats.transformers[group] = ct
    if text:
        t0 = time.time()
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=data.cfg.tfidf_min_df,
                                max_features=data.cfg.tfidf_max_features, sublinear_tf=True,
                                strip_accents="unicode", dtype=np.float32)
        text_of = lambda d: (d.title + " \n " + d.body).to_numpy()
        feats.matrices["text"] = {"train": tfidf.fit_transform(text_of(data.train)),
                                  "val": tfidf.transform(text_of(data.val)),
                                  "test": tfidf.transform(text_of(data.test))}
        feats.names["text"] = list(tfidf.get_feature_names_out())
        feats.transformers["text"] = tfidf
        print(f"TF-IDF: {len(feats.names['text']):,} terms in {time.time() - t0:.0f}s")
    print({g: feats[g]["train"].shape[1] for g in feats.matrices})
    return feats


# ---------------------------------------------------------------- evaluation

def _scores(y, p):
    if y.min() == y.max():
        return {"pr_auc": np.nan, "roc_auc": np.nan, "log_loss": np.nan}
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "log_loss": log_loss(y, p, labels=[0, 1])}


class Results:
    """Pooled and per-subreddit metrics for each model, on validation and test."""

    def __init__(self, data):
        self.data = data
        self.rows, self.sub_rows, self.preds = {}, [], {}

    def evaluate(self, name, p_val, p_test, seconds=None):
        p = {"val": np.clip(p_val, 1e-6, 1 - 1e-6), "test": np.clip(p_test, 1e-6, 1 - 1e-6)}
        row = {}
        self.sub_rows = [r for r in self.sub_rows if r["model"] != name]
        for split in ("val", "test"):
            y, d = self.data.y(split), getattr(self.data, split)
            row.update({f"{split}_{k}": v for k, v in _scores(y, p[split]).items()})
            for sub, idx in d.groupby("subreddit").indices.items():
                sc = _scores(y[idx], p[split][idx])
                self.sub_rows.append({"model": name, "split": split, "subreddit": sub, "posts": len(idx),
                                      "base_rate": y[idx].mean(), **sc})
        row["fit_seconds"] = seconds
        self.rows[name] = row
        self.preds[name] = p
        return self.table().loc[[name]]

    def table(self):
        return pd.DataFrame(self.rows).T

    def per_subreddit(self, split="test", metric="pr_auc"):
        """Subreddit x model table of `metric`. `lift` divides PR-AUC by the subreddit's base rate."""
        d = pd.DataFrame(self.sub_rows)
        d = d[d.split == split]
        if metric == "lift":
            d = d.assign(lift=d.pr_auc / d.base_rate)
        t = d.pivot(index="subreddit", columns="model", values=metric)
        return t[[m for m in self.rows if m in t.columns]]


def fit_timed(model, X, y):
    t0 = time.time()
    model.fit(X, y)
    return model, time.time() - t0


def logreg(C=1.0):
    return LogisticRegression(C=C, max_iter=2000, solver="lbfgs")


# ---------------------------------------------------------------- models

META = ["subreddit", "timing", "structure", "history"]


def run_ladder(data, feats, res, mlp=True, mlp_units=64, mlp_max_epochs=8, mlp_patience=2):
    """Fit the baseline ladder. Returns the fitted models by name."""
    y = data.y("train")
    models = {}

    base_rate = y.mean()
    res.evaluate("0 constant", np.full(len(data.val), base_rate), np.full(len(data.test), base_rate))
    res.evaluate("1 subreddit x weekday x hour rate", data.val.sub_dow_hour_rate, data.test.sub_dow_hour_rate)

    m, secs = fit_timed(MultinomialNB(alpha=0.1), feats["text"]["train"], y)
    res.evaluate("2 naive bayes (text)", m.predict_proba(feats["text"]["val"])[:, 1],
                 m.predict_proba(feats["text"]["test"])[:, 1], secs)
    models["nb"] = m

    for key, name, groups in [("lr_meta", "3 logreg (metadata)", META),
                              ("lr_text", "4 logreg (text)", ["text"]),
                              ("lr_all", "5 logreg (metadata + text)", META + ["text"])]:
        X = {s: feats.combine(groups, s) for s in SPLITS}
        m, secs = fit_timed(logreg(), X["train"], y)
        res.evaluate(name, m.predict_proba(X["val"])[:, 1], m.predict_proba(X["test"])[:, 1], secs)
        models[key] = m

    if mlp:
        X = {s: feats.combine(META + ["text"], s) for s in SPLITS}
        m, history, secs = fit_mlp(X, y, data.y("val"), mlp_units, mlp_max_epochs, mlp_patience, data.cfg.seed)
        res.evaluate("6 mlp, 1 hidden layer (metadata + text)", m.predict_proba(X["val"])[:, 1],
                     m.predict_proba(X["test"])[:, 1], secs)
        models["mlp"], models["mlp_history"] = m, history
    return models


def fit_mlp(X, y, y_val, units=64, max_epochs=8, patience=2, seed=42):
    """One epoch per partial_fit call; keep the epoch with the best validation PR-AUC."""
    mlp = MLPClassifier(hidden_layer_sizes=(units,), activation="relu", alpha=1e-4,
                        batch_size=1024, learning_rate_init=1e-3, random_state=seed)
    best_ap, best, history, bad = -1, None, [], 0
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        perm = rng.permutation(X["train"].shape[0])
        mlp.partial_fit(X["train"][perm], y[perm], classes=[0, 1])
        ap = average_precision_score(y_val, mlp.predict_proba(X["val"])[:, 1])
        history.append({"epoch": epoch, "train_loss": mlp.loss_, "val_pr_auc": ap})
        print(f"epoch {epoch}: train loss {mlp.loss_:.4f}, val PR-AUC {ap:.4f} ({time.time() - t0:.0f}s)")
        if ap > best_ap:
            best_ap, best, bad = ap, copy.deepcopy(mlp), 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best, pd.DataFrame(history), time.time() - t0


ABLATION_SETS = {
    "subreddit only": ["subreddit"],
    "+ timing": ["subreddit", "timing"],
    "+ structure": ["subreddit", "structure"],
    "+ history": ["subreddit", "history"],
    "+ text": ["subreddit", "text"],
    "content (structure + history + text)": ["subreddit", "structure", "history", "text"],
    "content + timing (all)": ["subreddit", "structure", "history", "text", "timing"],
}


def run_ablation(data, feats, sets=ABLATION_SETS, C=1.0):
    """Logistic regression on each feature set. Returns (pooled table, Results with per-subreddit metrics)."""
    res = Results(data)
    y = data.y("train")
    for name, groups in sets.items():
        X = {s: feats.combine(groups, s) for s in SPLITS}
        m, secs = fit_timed(logreg(C), X["train"], y)
        res.evaluate(name, m.predict_proba(X["val"])[:, 1], m.predict_proba(X["test"])[:, 1], secs)
    return res.table(), res


def coefficients(model, feats, groups, top=10):
    """Largest negative and positive coefficients of a linear model over `groups`."""
    coef = pd.Series(model.coef_[0], index=feats.feature_names(groups)).sort_values()
    return pd.concat([coef.head(top), coef.tail(top)]).to_frame("coefficient")


def top_terms(model, feats, n=25):
    names = np.array(feats.names["text"])
    order = np.argsort(model.coef_[0])
    return pd.DataFrame({"most negative": names[order[:n]], "most positive": names[order[::-1][:n]]})


# ---------------------------------------------------------------- plots

def plot_pr_auc(res):
    t = res.table()
    names = t.index[::-1]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(names) + 1.5))
    ax.barh(y + 0.2, t.loc[names, "val_pr_auc"], height=0.38, color=td.BLUE, label="validation")
    ax.barh(y - 0.2, t.loc[names, "test_pr_auc"], height=0.38, color=td.ORANGE, label="test")
    for yi, v in zip(y, t.loc[names, "test_pr_auc"]):
        ax.text(v, yi - 0.2, f" {v:.3f}", va="center", fontsize=8, color=td.INK)
    ax.set_yticks(y, names)
    ax.axvline(res.data.y("train").mean(), color=td.MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel("PR-AUC (dashed line = training base rate)")
    ax.set_title("PR-AUC by model, all subreddits pooled")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    return fig


def plot_per_subreddit(res, split="test", metric="lift"):
    """Heatmap of subreddit x model, with the value written in every cell."""
    t = res.per_subreddit(split, metric)
    fig, ax = plt.subplots(figsize=(1.3 * len(t.columns) + 3, 0.5 * len(t) + 1.8))
    im = ax.imshow(t.to_numpy(dtype=float), cmap="Blues", aspect="auto")
    vmax = np.nanmax(t.to_numpy(dtype=float))
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 * vmax else td.INK)
    ax.set_xticks(range(len(t.columns)), t.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(t)), t.index)
    ax.grid(False)
    label = "PR-AUC / base rate (1 = no better than random)" if metric == "lift" else metric
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(f"{label}, by subreddit ({split})")
    plt.tight_layout()
    return fig


def plot_pr_curves(res, names, split="test"):
    colors = [td.BLUE, td.ORANGE, td.AQUA, td.YELLOW, td.MAGENTA]
    y = res.data.y(split)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, color in zip(names, colors):
        prec, rec, _ = precision_recall_curve(y, res.preds[name][split])
        ax.plot(rec, prec, color=color, linewidth=2, label=name)
    ax.axhline(y.mean(), color=td.MUTED, linewidth=1, linestyle="--", label="base rate")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Precision-recall curves ({split})")
    ax.legend(frameon=False)
    return fig


def plot_timing_gain(ablation_res, split="test", without="content (structure + history + text)",
                     with_="content + timing (all)"):
    """PR-AUC gained per subreddit by adding the timing features to the content features."""
    t = ablation_res.per_subreddit(split, "pr_auc")
    gain = (t[with_] - t[without]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(gain) + 1.5))
    ax.barh(gain.index, gain.values, color=[td.COLORS.get(s, td.BLUE) for s in gain.index], height=0.6)
    for yi, v in enumerate(gain.values):  # labels right of the bar end or of zero, clear of the axis labels
        ax.text(max(v, 0), yi, f" {v:+.3f}", va="center", fontsize=8, color=td.INK, ha="left")
    ax.axvline(0, color=td.MUTED, linewidth=1)
    ax.set_xlabel("PR-AUC with timing − PR-AUC without timing")
    ax.set_title(f"What timing adds beyond content, by subreddit ({split})")
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    return fig
