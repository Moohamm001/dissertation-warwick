"""Tests for the RQ1-follow-up improvement experiment (emerald_ai.improve).

These guard the two things that make the experiment defensible: the affordability ratios are
leakage-safe and finite, and the configurations under test stay inside the permitted feature set.
The headline claims (null result, events projection) are not asserted here — they are empirical
outputs reported with their uncertainty, not invariants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from emerald_ai import data as D
from emerald_ai import feature_audit as FA
from emerald_ai import improve as I


def test_affordability_ratios_are_finite_and_leakage_safe():
    df = D.build_target(D.load_raw(), "paidoff_only").reset_index(drop=True)
    out = I.affordability_features(df)
    for r in I.RATIOS:
        assert r in out.columns
        # no infinities leaked through (zero denominators must become NaN, not inf)
        assert not np.isinf(out[r].to_numpy(dtype=float)).any()
    # ratios are built only from permitted pre-funding numerics
    assert set(I.BASE_NUMERICS) <= set(FA.permitted_columns())


def test_ratio_values_match_their_definition():
    df = pd.DataFrame({
        "Revenue": [1000.0, 0.0], "Average Monthly Sales": [2000.0, 5000.0],
        "Amount Sought": [50000.0, 10000.0],
    })
    out = I.affordability_features(df)
    assert out.loc[0, "loan_to_revenue"] == 50.0          # 50000 / 1000
    assert out.loc[0, "loan_to_sales"] == 25.0            # 50000 / 2000
    assert np.isnan(out.loc[1, "loan_to_revenue"])        # zero revenue -> NaN, not inf


def test_configs_only_reference_known_columns():
    permitted_or_ratio = set(FA.permitted_columns()) | set(I.RATIOS)
    for _, penalty, feats in I.CONFIGS:
        assert penalty in {"l1", "l2", "elasticnet"}
        assert set(feats) <= permitted_or_ratio
