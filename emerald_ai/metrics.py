"""Phase 3 metric panel — the only metrics allowed on this dataset.

Raw accuracy is deliberately absent (a constant predictor scores 99.64%). Everything here is
defined against the rare event (default = 1) and reported with an across-fold uncertainty band,
never as a bare point estimate.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def pr_auc(y_true, y_score) -> float:
    """Area under the precision-recall curve (a.k.a. average precision). Primary RQ1 metric."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def recall_at_top_decile(y_true, y_score, frac: float = 0.10) -> float:
    """Operational metric: of all true defaults, what share lands in the top-``frac`` riskiest?

    This is how a lending desk would actually use the score — review the riskiest decile.
    """
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return np.nan
    k = max(1, int(np.ceil(len(y_true) * frac)))
    top_idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_idx].sum() / n_pos)


def within_minority_ece(y_true, y_score, n_bins: int = 5) -> float:
    """Expected calibration error computed ONLY on the minority (event) cases.

    For true-positive rows the empirical accuracy is 1, so this reduces to how far the model's
    confidence sits below 1 on actual defaults — the calibration failure that matters for
    adverse-action decisioning. Marginal ECE hides it behind the 98%+ favourable class.
    High variance at this N — always reported with the fold band.
    """
    y_true = np.asarray(y_true)
    mask = y_true == 1
    if mask.sum() == 0:
        return np.nan
    conf = np.asarray(y_score)[mask]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(conf[sel].mean() - 1.0)  # accuracy = 1 for all positives
    return float(ece)


def fold_band(values) -> dict:
    """Median and 2.5–97.5 percentile interval across folds — the honest small-N uncertainty."""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if v.size == 0:
        return {"median": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    return {
        "median": float(np.median(v)),
        "lo": float(np.percentile(v, 2.5)),
        "hi": float(np.percentile(v, 97.5)),
        "n": int(v.size),
    }
