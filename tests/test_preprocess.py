"""Phase 2 tests — the leakage guard is the point of this file.

If any of these fail, a post-funding field could reach the model and every downstream number is
invalid. This is the automated assertion the roadmap promises at the Phase 2 gate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emerald_ai import feature_audit as FA
from emerald_ai import preprocess as P


def test_permitted_and_forbidden_are_disjoint():
    cat = FA.classify()
    permitted = set(cat[cat.role == "permitted"]["column"])
    forbidden = set(cat[cat.role == "forbidden"]["column"])
    assert permitted.isdisjoint(forbidden)
    assert FA.C.LABEL_COL not in permitted  # label is never an input


def test_known_leakage_columns_never_permitted():
    permitted = set(FA.permitted_columns())
    leaked = FA.KNOWN_LEAKAGE & permitted
    assert not leaked, f"leakage columns slipped into the allowlist: {leaked}"


def test_every_column_classified_once():
    cat = FA.classify()
    assert cat["column"].is_unique
    assert set(cat["role"]) <= {"label", "permitted", "forbidden"}


def test_assert_no_leakage_rejects_forbidden_column():
    with pytest.raises(P.LeakageError):
        P.assert_no_leakage(["Credit Score", "Percent Paid"])  # second is post-funding outcome


def test_pipeline_fits_and_only_sees_permitted_columns():
    # Tiny synthetic frame using only permitted columns + the target.
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "Credit Score": rng.normal(670, 60, n),
        "Revenue": rng.gamma(2, 1000, n),
        "Time In Business": rng.integers(1, 200, n),
        "Industry": rng.choice(["construction", "retail", "other"], n),
        "Borrower State": rng.choice([f"S{i}" for i in range(30)], n),  # high-card -> target enc
        "y": rng.integers(0, 2, n),
    })
    pre, types = P.build_preprocessor(df)
    X = pre.fit_transform(df, df["y"])
    assert np.isfinite(X).all()
    assert X.shape[0] == n
    assert "Borrower State" in types["high_card"]
    assert "Industry" in types["low_card"]
