"""Tests for the Phase 5 decision-support demo (emerald_ai.serve).

These guard the scoring contract, not the cosmetics: every applicant gets a probability in [0, 1],
the riskiest-decile flag is consistent with the operating threshold, reasons map back to real
permitted features, and the leakage guard still holds through the serving path.
"""
from __future__ import annotations

import numpy as np

from emerald_ai import feature_audit as FA
from emerald_ai import serve as S


def test_scorer_builds_one_field_per_permitted_feature():
    sc = S.get_scorer()
    names = {f.name for f in sc.fields}
    assert names == set(FA.permitted_columns())
    # operating point is an honest decile cut, not 0.5
    assert 0.0 < sc.threshold < 1.0
    assert sc.threshold != 0.5
    assert 0.0 <= sc.catch_rate <= 1.0


def test_empty_payload_scores_via_defaults():
    """A blank form must still score (defaults fill in) and stay a valid probability."""
    sc = S.get_scorer()
    out = S.score_applicant(sc, {})
    assert 0.0 <= out["probability"] <= 1.0
    assert out["in_riskiest_decile"] == (out["probability"] >= sc.threshold)
    assert len(out["reasons"]) == 3


def test_reasons_are_named_permitted_features():
    sc = S.get_scorer()
    out = S.score_applicant(sc, {"Revenue": 9000, "Credit Score": 640})
    permitted = set(FA.permitted_columns())
    for r in out["reasons"]:
        assert r["feature"] in permitted
        assert r["direction"] in {"increases risk", "decreases risk"}
        # sign of the contribution must match the stated direction
        assert (r["contribution"] > 0) == (r["direction"] == "increases risk")


def test_higher_revenue_raises_risk_matching_the_data():
    """Defaulters skew to HIGHER revenue here; the served model must reflect that monotone direction."""
    sc = S.get_scorer()
    base = {"Time In Business": 60}
    lo = S.score_applicant(sc, {**base, "Revenue": 800})["probability"]
    hi = S.score_applicant(sc, {**base, "Revenue": 9000})["probability"]
    assert hi > lo


def test_decile_flag_respects_threshold():
    sc = S.get_scorer()
    out = S.score_applicant(sc, {"Revenue": 11000, "Time In Business": 12})
    assert out["in_riskiest_decile"] == (out["probability"] >= sc.threshold)
    assert np.isclose(out["threshold"], round(sc.threshold, 4))


def test_example_cases_span_a_risk_gradient():
    """The curated demo cases must run low->high so the demo actually shows a gradient."""
    sc = S.get_scorer()
    res = S.score_frame(sc, S.example_cases_frame())
    by_case = dict(zip(res["case"], res["percent"]))
    assert by_case["established_low_revenue"] < by_case["borderline"]
    assert by_case["borderline"] < by_case["high_revenue_short_history"]
    # at least one case lands in the review queue and at least one stays out
    assert res["in_riskiest_decile"].any() and (~res["in_riskiest_decile"]).any()


def test_score_frame_handles_partial_and_unknown_columns():
    """A batch with a passthrough id column, a missing feature, and a junk column still scores."""
    import pandas as pd

    sc = S.get_scorer()
    df = pd.DataFrame([
        {"id": "a", "Revenue": 800, "NOT_A_FEATURE": "x"},
        {"id": "b", "Revenue": 11000, "NOT_A_FEATURE": "y"},
    ])
    res = S.score_frame(sc, df)
    assert list(res["id"]) == ["a", "b"]
    assert {"probability", "in_riskiest_decile", "top_reasons"} <= set(res.columns)
    assert (res["probability"].between(0, 1)).all()
    assert res.loc[1, "probability"] > res.loc[0, "probability"]  # higher revenue -> higher risk


def test_random_applicants_are_in_distribution_and_unlabelled():
    sc = S.get_scorer()
    samp = S.random_applicants(n=20, seed=1)
    assert len(samp) == 20
    assert "y" not in samp.columns and "Deal Status" not in samp.columns
    # every emitted feature column is a permitted pre-funding feature (no leakage into test data)
    for c in samp.columns:
        if c != "id":
            assert c in FA.permitted_columns()
