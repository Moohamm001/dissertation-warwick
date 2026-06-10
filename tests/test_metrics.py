"""Tests for the Phase 3 metric panel — small, exact, no model training."""
from __future__ import annotations

import numpy as np

from emerald_ai import metrics as M


def test_recall_at_top_decile_captures_top_ranked_positive():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    score = np.arange(10) / 10.0  # the single positive has the highest score
    assert M.recall_at_top_decile(y, score) == 1.0


def test_pr_auc_nan_without_positives():
    assert np.isnan(M.pr_auc(np.zeros(10), np.random.rand(10)))


def test_within_minority_ece_bounds():
    y = np.array([1, 1, 1, 1])
    assert M.within_minority_ece(y, np.ones(4)) == 0.0   # perfectly confident on defaults
    assert M.within_minority_ece(y, np.zeros(4)) == 1.0  # maximally under-confident


def test_fold_band_ignores_nan():
    b = M.fold_band([0.1, 0.2, 0.3, np.nan])
    assert b["n"] == 3
    assert b["median"] == 0.2
