"""Tests for the cost-sensitive decision layer (emerald_ai.decision).

Lock in the two things that make the policy defensible: (1) it is genuinely cost-optimal (never
dearer than the naive cut-offs it replaces), and (2) it behaves monotonically in the cost ratio —
a costlier missed default must lower the threshold and raise recall, never the reverse.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from emerald_ai import decision as DEC


@lru_cache(maxsize=1)
def _oof():
    from emerald_ai.experiments import oof_predictions
    y, p = oof_predictions("logreg", "class_weight")
    return y, p


def test_optimal_is_never_dearer_than_naive_policies():
    y, p = _oof()
    for R in (5, 20, 100):
        opt = DEC.optimal_policy(y, p, R)
        base = DEC.baselines_cost(y, p, R)
        # the cost-minimiser must not cost more than any fixed policy
        assert opt["cost"] <= base["threshold_0.5"] + 1e-6
        assert opt["cost"] <= base["review_nothing"] + 1e-6
        assert opt["cost"] <= base["review_all"] + 1e-6


def test_higher_cost_ratio_lowers_threshold_and_raises_recall():
    y, p = _oof()
    pols = [DEC.optimal_policy(y, p, R) for R in (5, 20, 100)]
    thr = [q["threshold"] for q in pols]
    rec = [q["recall"] for q in pols]
    flagged = [q["n_flagged"] for q in pols]
    assert thr[0] >= thr[1] >= thr[2]          # costlier FN -> lower cut
    assert rec[0] <= rec[1] <= rec[2]          # ... -> catch more defaults
    assert flagged[0] <= flagged[1] <= flagged[2]


def test_recall_and_precision_are_consistent():
    y, p = _oof()
    opt = DEC.optimal_policy(y, p, 20)
    assert opt["defaults_caught"] <= opt["defaults_total"] == int(y.sum())
    assert 0.0 <= opt["recall"] <= 1.0
    assert opt["n_flagged"] == 0 or 0.0 <= opt["precision"] <= 1.0


def test_bootstrap_saving_returns_a_proper_interval():
    y, p = _oof()
    b = DEC.bootstrap_saving(y, p, 20, n_boot=80)
    assert b["lo"] <= b["median"] <= b["hi"]
    assert b["n"] > 0
