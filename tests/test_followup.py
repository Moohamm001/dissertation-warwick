"""Tests for the review-round-1 follow-up checks (emerald_ai.followup).

Pure-function tests on synthetic inputs — no dataset load, no model fitting — locking in the
statistical contracts: paired differences computed on shared folds, stability metrics bounded
and order-invariant, and the distribution check's flag-share arithmetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from emerald_ai import followup as F


def _synthetic_per_fold(n_folds: int = 25, gap: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Two combos on shared folds; challenger sits `gap` below the baseline on average."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.05, 0.25, n_folds)
    rows = []
    for f in range(n_folds):
        rows.append({"combo": "logreg+class_weight", "fold": f, "pr_auc": base[f]})
        rows.append({"combo": "xgboost+class_weight", "fold": f,
                     "pr_auc": base[f] - gap + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


def test_paired_comparison_detects_a_real_gap():
    out = F.paired_comparison(_synthetic_per_fold(gap=0.05))
    row = out.iloc[0]
    assert row["challenger"] == "xgboost+class_weight"
    assert row["median_diff"] > 0.03            # baseline clearly ahead
    assert row["sign_test_p"] < 0.01            # and the sign test sees it


def test_paired_comparison_null_gap_is_not_significant():
    out = F.paired_comparison(_synthetic_per_fold(gap=0.0, seed=1))
    assert out.iloc[0]["sign_test_p"] > 0.05
    assert abs(out.iloc[0]["median_diff"]) < 0.02


def test_paired_comparison_excludes_dummy_and_baseline():
    df = _synthetic_per_fold()
    df = pd.concat([df, pd.DataFrame(
        [{"combo": "dummy+none", "fold": f, "pr_auc": 0.013} for f in range(25)])])
    out = F.paired_comparison(df)
    assert set(out["challenger"]) == {"xgboost+class_weight"}


def test_stability_summary_perfectly_stable_ranks():
    imp = pd.DataFrame([[5.0, 3.0, 1.0, 0.5]] * 4,
                       columns=["Revenue", "Average Monthly Sales", "Credit Score", "Tier"])
    s = F.stability_summary(imp)
    assert s["mean_spearman"] == 1.0 and s["min_spearman"] == 1.0
    assert s["cluster_rank1_share"] == 1.0        # Revenue is rank 1 in every fold
    assert s["top3_freq"]["Revenue"] == 1.0
    assert "Tier" not in s["top3_freq"].index      # never in top-3


def test_stability_summary_detects_instability():
    rng = np.random.default_rng(0)
    imp = pd.DataFrame(rng.uniform(0, 1, (5, 6)),
                       columns=[f"f{i}" for i in range(6)])
    s = F.stability_summary(imp)
    assert -1.0 <= s["min_spearman"] <= s["mean_spearman"] <= 1.0
    assert s["mean_spearman"] < 0.9               # random ranks should not look stable
