"""Tests for the Phase 5 decision-support demo (emerald_ai.serve).

These guard the scoring contract, not the cosmetics: every applicant gets a probability in [0, 1],
the riskiest-decile flag is consistent with the operating threshold, reasons map back to real
permitted features, and the leakage guard still holds through the serving path.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from emerald_ai import config as C
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


def test_reasons_are_human_readable():
    """Each reason must carry a plain-English label + verdict consistent with its sign."""
    sc = S.get_scorer()
    out = S.score_applicant(sc, {"Revenue": 9000, "Credit Score": 610})
    for r in out["reasons"]:
        assert r["label"] == S._friendly(r["feature"])           # friendly name, not the raw column
        assert r["verdict"] in {"raises risk", "lowers risk"}
        assert (r["verdict"] == "raises risk") == (r["contribution"] > 0)
    # the batch column reads as arrowed friendly phrases, not "Revenue +"
    res = S.score_frame(sc, S.example_cases_frame())
    sample = res["top_reasons"].iloc[0]
    assert ("↑" in sample or "↓" in sample) and "+" not in sample


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


def test_batch_ranks_and_queues_within_the_uploaded_set():
    """The review queue is the riskiest decile *of the batch*, ranked by risk — the real operating unit."""
    import pandas as pd

    sc = S.get_scorer()
    res = S.score_frame(sc, S.random_applicants(n=30, seed=3))
    # rank is a 1..n permutation, rank 1 has the highest probability
    assert sorted(res["rank"]) == list(range(1, len(res) + 1))
    assert res.loc[res["rank"].idxmin(), "probability"] == res["probability"].max()
    # queue = ceil(10% of batch), and every queued row outranks every non-queued row
    assert res["review_queue"].sum() == max(1, int(np.ceil(0.10 * len(res))))
    assert res.loc[res["review_queue"], "rank"].max() < res.loc[~res["review_queue"], "rank"].min()


def test_upload_endpoint_scores_csv_and_ignores_extra_columns():
    """The /api/score-upload endpoint accepts a file (the raw dataset path), keeping only permitted cols."""
    client = TestClient(S.create_app())
    # a CSV carrying a permitted feature, an id, an unknown column, and the leakage label
    csv = "id,Revenue,NOT_A_FEATURE,Deal Status\na,800,x,paidOff\nb,11000,y,default\n"
    r = client.post("/api/score-upload", files={"file": ("book.csv", csv, "text/csv")})
    assert r.status_code == 200
    j = r.json()
    assert j["n"] == 2 and j["shown"] == 2
    assert {"rank", "id", "percent", "review_queue", "top_reasons"} <= set(j["rows"][0])
    # ranked riskiest-first; higher revenue is riskier here, so applicant b leads
    assert j["rows"][0]["id"] == "b"


def test_random_applicants_are_in_distribution_and_unlabelled():
    sc = S.get_scorer()
    samp = S.random_applicants(n=20, seed=1)
    assert len(samp) == 20
    assert "y" not in samp.columns and "Deal Status" not in samp.columns
    # every emitted feature column is a permitted pre-funding feature (no leakage into test data)
    for c in samp.columns:
        if c != "id":
            assert c in FA.permitted_columns()


# --------------------------------------------------------------------------- hardening (path-to-production)
def _valid_csv() -> str:
    return "id,Revenue\na,800\nb,11000\n"


def test_upload_rejects_disallowed_extension():
    client = TestClient(S.create_app())
    r = client.post("/api/score-upload", files={"file": ("book.pdf", _valid_csv(), "application/pdf")})
    assert r.status_code == 400
    assert "unsupported file type" in r.json()["detail"]


def test_upload_rejects_empty_file():
    client = TestClient(S.create_app())
    r = client.post("/api/score-upload", files={"file": ("book.csv", "", "text/csv")})
    assert r.status_code == 400


def test_upload_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(C, "MAX_UPLOAD_MB", 0.0000001)
    client = TestClient(S.create_app())
    r = client.post("/api/score-upload", files={"file": ("book.csv", _valid_csv(), "text/csv")})
    assert r.status_code == 400
    assert "too large" in r.json()["detail"]


def test_upload_rejects_unparseable_excel_bytes():
    client = TestClient(S.create_app())
    r = client.post("/api/score-upload",
                     files={"file": ("book.xlsx", b"this is not a real xlsx file", "application/octet-stream")})
    assert r.status_code == 400
    assert "could not parse" in r.json()["detail"]


def test_upload_rejects_no_recognised_columns():
    client = TestClient(S.create_app())
    csv = "id,NOT_A_FEATURE\na,1\nb,2\n"
    r = client.post("/api/score-upload", files={"file": ("book.csv", csv, "text/csv")})
    assert r.status_code == 400
    assert "no recognised applicant columns" in r.json()["detail"]


def test_batch_rejects_empty_pasted_csv():
    client = TestClient(S.create_app())
    r = client.post("/api/score-batch", json={"csv": "   "})
    assert r.status_code == 400


def test_batch_rejects_unparseable_pasted_csv():
    client = TestClient(S.create_app())
    ragged = "a,b\n1,2\n3,4,5,6,7\n"  # inconsistent field counts -> pandas ParserError
    r = client.post("/api/score-batch", json={"csv": ragged})
    assert r.status_code == 400
    assert "could not parse" in r.json()["detail"]


def test_api_key_open_when_unset(monkeypatch):
    """Locks in the safe default: no EMERALD_API_KEY set -> /api/* stays open."""
    monkeypatch.delenv("EMERALD_API_KEY", raising=False)
    client = TestClient(S.create_app())
    r = client.post("/api/score", json={"Revenue": 800})
    assert r.status_code == 200


def test_api_key_required_when_set(monkeypatch):
    monkeypatch.setenv("EMERALD_API_KEY", "s3cret")
    client = TestClient(S.create_app())
    # no header -> 401
    r = client.post("/api/score", json={"Revenue": 800})
    assert r.status_code == 401
    # wrong header -> 401
    r = client.post("/api/score", json={"Revenue": 800}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401
    # correct header -> 200
    r = client.post("/api/score", json={"Revenue": 800}, headers={"X-API-Key": "s3cret"})
    assert r.status_code == 200
    # the HTML form stays browsable regardless (not gated behind the key)
    assert client.get("/").status_code == 200


def test_unhandled_error_returns_clean_json_no_traceback(monkeypatch):
    """The global exception handler must not leak internals for an unexpected (non-HTTPException) error."""
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated internal failure: db connection string=secret")

    monkeypatch.setattr(S, "score_frame", _boom)
    client = TestClient(S.create_app(), raise_server_exceptions=False)
    r = client.post("/api/score-upload", files={"file": ("book.csv", _valid_csv(), "text/csv")})
    assert r.status_code == 500
    assert r.json() == {"detail": "internal error"}
    assert "secret" not in r.text and "RuntimeError" not in r.text and "Traceback" not in r.text


# --------------------------------------------------------------------------- deployment
def test_health_reports_ready_and_operating_point():
    """The probe must be unauthenticated, non-blocking, and describe the served model."""
    S.get_scorer()                                   # ensure the model is loaded
    client = TestClient(S.create_app())
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["model_loaded"] and j["ready"]
    sc = S.get_scorer()
    assert j["operating_threshold"] == round(sc.threshold, 4)
    assert j["training_rows"] == sc.n_rows and j["training_events"] == sc.n_events


def test_health_is_not_behind_the_api_key(monkeypatch):
    """A probe must keep working when API-key auth is switched on."""
    monkeypatch.setenv("EMERALD_API_KEY", "s3cret")
    client = TestClient(S.create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["auth_required"] is True
    # while /api/* is gated
    assert client.post("/api/score", json={"Revenue": 800}).status_code == 401


def test_model_cache_round_trip(tmp_path, monkeypatch):
    """A cached artefact is reused; one built by a different code path is ignored, not trusted."""
    monkeypatch.setattr(C, "ARTEFACT_DIR", tmp_path)
    monkeypatch.setattr(C, "MODEL_CACHE", tmp_path / "scorer.joblib")

    assert S._load_cached_scorer() is None            # nothing cached yet
    sc = S.get_scorer()
    S._save_cached_scorer(sc)
    assert (tmp_path / "scorer.joblib").exists()

    again = S._load_cached_scorer()
    assert again is not None
    assert again.threshold == sc.threshold and again.n_events == sc.n_events

    monkeypatch.setattr(C, "MODEL_CACHE_VERSION", "999")   # cache from another version
    assert S._load_cached_scorer() is None


def test_corrupt_model_cache_does_not_break_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODEL_CACHE", tmp_path / "scorer.joblib")
    (tmp_path / "scorer.joblib").write_bytes(b"not a joblib file")
    assert S._load_cached_scorer() is None             # falls back to fitting, never raises


def test_startup_warms_the_model_so_health_is_ready_immediately():
    """A public deployment must be ready before traffic arrives, not on the first request."""
    S.get_scorer.cache_clear()
    with TestClient(S.create_app()) as client:      # context manager fires startup handlers
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["ready"] is True


def test_rate_limit_returns_429_and_exempts_health(monkeypatch):
    monkeypatch.setenv("EMERALD_RATE_LIMIT", "3")
    S._RATE_STATE.clear()
    client = TestClient(S.create_app())
    codes = [client.post("/api/score", json={"Revenue": 800}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    assert client.get("/health").status_code == 200      # probes are never rate limited


def test_rate_limit_can_be_disabled(monkeypatch):
    monkeypatch.setenv("EMERALD_RATE_LIMIT", "0")
    S._RATE_STATE.clear()
    client = TestClient(S.create_app())
    assert all(client.post("/api/score", json={"Revenue": 800}).status_code == 200 for _ in range(4))


def test_build_artefact_writes_a_servable_file_without_row_level_data(tmp_path):
    """The deployable artefact must be small, self-sufficient, and free of record-level data."""
    import joblib, numpy as np

    out = tmp_path / "scorer.joblib"
    info = S.build_artefact(str(out))
    assert out.exists() and info["training_events"] == 50
    assert info["size_kb"] < 200                      # metadata and coefficients only

    scorer = joblib.load(out)["scorer"]
    assert scorer.model.coef_.shape[1] == len(scorer.feat_names)
    # nothing inside the artefact is big enough to hold the 3,898 training rows
    for obj in (scorer.train_mean, scorer.model.coef_):
        assert isinstance(obj, np.ndarray) and obj.size < 200


def test_service_starts_without_the_training_extras(monkeypatch):
    """A serving container installs requirements-serve.txt, which omits imbalanced-learn and
    xgboost. Importing them at module scope would break the deploy, so the serving path must not
    touch them when a fitted artefact is available."""
    import builtins, sys

    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name.split(".")[0] in {"imblearn", "xgboost"}:
            raise ModuleNotFoundError(f"No module named '{name}' (simulated serving container)")
        return real_import(name, *a, **k)

    for m in [m for m in sys.modules if m.split(".")[0] in {"imblearn", "xgboost"}]:
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.setattr(builtins, "__import__", guard)

    with TestClient(S.create_app()) as client:
        assert client.get("/health").json()["ready"] is True
        assert client.post("/api/score", json={"Revenue": 9000}).status_code == 200
        assert client.get("/").status_code == 200


def test_serving_requirements_cover_what_the_service_imports():
    """requirements-serve.txt must list every distribution the serving path needs."""
    from pathlib import Path

    text = Path("requirements-serve.txt").read_text(encoding="utf-8").lower()
    for pkg in ("pandas", "numpy", "scikit-learn", "joblib", "openpyxl",
                "fastapi", "uvicorn", "python-multipart"):
        assert pkg in text, f"{pkg} missing from requirements-serve.txt"
