"""Phase 5 — proof-of-concept decision-support demo (FastAPI + minimal UI).

A *demo of decision support*, not an MLOps stack (roadmap Phase 5). It serves the frozen headline
model — the regularised, class-weighted logistic regression on the 17 leakage-safe pre-funding
features — behind a single-page form. For each applicant it returns:

  * P(default) from ``predict_proba`` — never a hard 0.5 yes/no (see the threshold discussion: at
    1.28% prevalence a 0.5 cut predicts everyone "safe");
  * whether the applicant falls in the **riskiest decile**, the operating point a lending desk
    would actually review (threshold set from out-of-fold scores, not in-sample);
  * the **top-3 reasons** (SHAP local contributions aggregated back to the original named features),
    the "why was this flagged?" answer a regulator (FCA Consumer Duty) expects.

Honest framing, surfaced in the UI: the model *ranks* applications for review; it does not
approve or decline. Run: ``python -m emerald_ai serve``.
"""
from __future__ import annotations

import io
import logging
import os
import secrets
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# FastAPI types are imported at module level (not lazily) so that, under
# ``from __future__ import annotations``, FastAPI can resolve parameter annotations like
# ``UploadFile`` against the module globals — a lazy import inside create_app() leaves them as
# unresolvable forward references. FastAPI/uvicorn are declared in requirements.txt.
from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

# Module logger only — no ``basicConfig`` here, so importing this module (e.g. under pytest) has
# no side effects. ``run()`` below configures handlers for the real server process.
logger = logging.getLogger(__name__)

from . import config as C
from . import data as D
from . import feature_audit as FA
from . import metrics as M
from . import preprocess as P
# NOTE: emerald_ai.experiments is imported lazily inside _fit_scorer(), not here. It pulls in
# imbalanced-learn and xgboost, which the bake-off needs but a serving container does not: when
# a fitted artefact is present the service never fits, so a deployment image can omit those
# packages entirely. Importing them at module scope would make the service fail to start
# without them.

# Cap dropdown size for very high-cardinality categoricals (e.g. Loan Purpose has 154 levels).
# Rare levels are grouped by the preprocessor's ``min_frequency`` anyway, so the long tail is
# immaterial to the score; we just keep the form usable.
_MAX_CATEGORY_OPTIONS = 40
TOP_DECILE = 0.10

# Plain-English labels for the model's columns, so a reason reads "Monthly revenue" not "Revenue"
# and a non-technical reviewer can act on it without a data dictionary.
FRIENDLY_LABELS = {
    "Credit Score": "Credit score",
    "Amount Sought": "Loan amount requested",
    "Revenue": "Monthly revenue",
    "Average Monthly Sales": "Monthly sales",
    "Time In Business": "Time in business",
    "Days Since Last Opportunity": "Days since last enquiry",
    "Online App Completed": "Applied online",
    "Is Borrower Renewal": "Returning borrower",
    "Current Tier": "Risk tier",
    "Mktg Tier": "Marketing tier",
    "Industry": "Industry",
    "Loan Purpose": "Loan purpose",
    "Borrower State": "Borrower's state",
    "Deal Type": "Deal type",
    "Renewal Type": "Renewal type",
    "Channel": "Origination channel",
    "Medium": "Marketing medium",
}


def _friendly(feature: str) -> str:
    """Plain-English name for a model column (falls back to the raw name)."""
    return FRIENDLY_LABELS.get(feature, feature)


@dataclass
class FieldSpec:
    """One form field derived from the training data: how to render and what to default to."""
    name: str
    kind: str                       # "numeric" | "categorical"
    default: object
    options: list[str] = field(default_factory=list)  # categoricals only
    lo: float = None                # numeric only: 10th percentile (typical-range hint)
    hi: float = None                # numeric only: 90th percentile (typical-range hint)


@dataclass
class Scorer:
    """The fitted, frozen model plus everything needed to score and explain one new applicant."""
    pre: object
    model: object
    feat_names: list[str]           # transformed (post-encoding) feature names
    train_mean: np.ndarray          # mean of each transformed feature (SHAP baseline E[x])
    source_of: list[str]            # for each transformed feature, its originating permitted column
    fields: list[FieldSpec]
    threshold: float                # P(default) cut defining the riskiest decile (from OOF)
    catch_rate: float               # share of all defaults captured in that decile (OOF)
    prevalence: float
    n_rows: int
    n_events: int


def _map_to_source(transformed_names: list[str], permitted: list[str]) -> list[str]:
    """Map each transformed feature name back to the permitted column it came from.

    Numeric / target-encoded columns keep their name; one-hot columns are ``"{col}_{level}"``.
    We pick the longest permitted column that the name equals or prefixes — robust to levels that
    themselves contain underscores.
    """
    by_len = sorted(permitted, key=len, reverse=True)
    out = []
    for n in transformed_names:
        src = next((c for c in by_len if n == c or n.startswith(c + "_")), n)
        out.append(src)
    return out


def _build_fields(df: pd.DataFrame) -> list[FieldSpec]:
    """Derive a form field per permitted column, with data-driven defaults (median / mode)."""
    fields = []
    for c in FA.permitted_columns():
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            fields.append(FieldSpec(
                c, "numeric", float(np.round(s.median(), 2)),
                lo=float(np.round(s.quantile(0.10), 2)), hi=float(np.round(s.quantile(0.90), 2)),
            ))
        else:
            counts = s.value_counts()
            options = [str(v) for v in counts.index[:_MAX_CATEGORY_OPTIONS]]
            default = str(counts.index[0]) if len(counts) else ""
            fields.append(FieldSpec(c, "categorical", default, options))
    return fields


# Per-process rate-limit state: {client ip: (minute window, hits)}. See _rate_limit below.
_RATE_STATE: dict[str, tuple[int, int]] = {}


def build_artefact(path=None) -> dict:
    """Fit the model and write the deployable artefact, for a machine that HAS the dataset.

    The artefact is what a public deployment ships: model coefficients, the fitted preprocessor,
    the operating point and the form metadata - about 26 KB, with no row-level data. A server
    that carries only this file can serve every endpoint, so the lending book never has to leave
    the analyst's machine.
    """
    dest = Path(path) if path else C.MODEL_CACHE
    get_scorer.cache_clear()
    scorer = _fit_scorer()
    dest.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"key": _cache_key(), "scorer": scorer}, dest)
    return {
        "path": str(dest),
        "size_kb": round(dest.stat().st_size / 1024, 1),
        "threshold": round(scorer.threshold, 4),
        "catch_rate": round(scorer.catch_rate, 4),
        "training_rows": scorer.n_rows,
        "training_events": scorer.n_events,
    }


def _cache_key() -> str:
    """Identity of a cached artefact: cache format, global seed, and the served model."""
    return f"v{C.MODEL_CACHE_VERSION}|seed={C.SEED}|logreg+class_weight|paidoff_only"


def _load_cached_scorer() -> Scorer | None:
    """Return a previously fitted scorer if one exists and was built by this code path."""
    if not C.MODEL_CACHE.exists():
        return None
    try:
        import joblib
        blob = joblib.load(C.MODEL_CACHE)
        if blob.get("key") != _cache_key():
            logger.info("model cache ignored: built for %r, need %r", blob.get("key"), _cache_key())
            return None
        logger.info("model cache hit: %s", C.MODEL_CACHE)
        return blob["scorer"]
    except Exception as exc:  # noqa: BLE001 - a corrupt cache must never stop the service
        logger.warning("model cache unreadable (%s); retraining", exc)
        return None


def _save_cached_scorer(scorer: Scorer) -> None:
    try:
        import joblib
        C.ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"key": _cache_key(), "scorer": scorer}, C.MODEL_CACHE)
        logger.info("model cached -> %s", C.MODEL_CACHE)
    except Exception as exc:  # noqa: BLE001 - caching is an optimisation, not a requirement
        logger.warning("could not write model cache: %s", exc)


@lru_cache(maxsize=1)
def get_scorer() -> Scorer:
    """Return the frozen model, from the on-disk cache when possible, otherwise by fitting it.

    Fitting takes roughly 25 seconds (it also runs the out-of-fold pass that sets the operating
    point), which a restarted container should not repeat. The fitted object is cached to disk
    and reused whenever the cache was produced by the same code path and seed.
    """
    cached = _load_cached_scorer()
    if cached is not None:
        return cached

    logger.info("no usable model cache; fitting the model")
    scorer = _fit_scorer()
    _save_cached_scorer(scorer)
    return scorer


def _fit_scorer() -> Scorer:
    """Train the frozen model on all cleaned data and set its operating point.

    Requires the dataset and the modelling extras (imbalanced-learn, xgboost); a serving
    container that ships a fitted artefact never reaches this path.
    """
    from .experiments import _make_model, oof_predictions

    df = D.build_target(D.load_raw(), "paidoff_only").reset_index(drop=True)
    y = df["y"].to_numpy()

    pre, _ = P.build_preprocessor(df, scale=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.asarray(pre.fit_transform(df, y))
        model = _make_model("logreg", "class_weight", y)
        model.fit(X, y)

    names = list(pre.get_feature_names_out())
    source = _map_to_source(names, FA.permitted_columns())

    # Operating point: honest, out-of-fold riskiest-decile threshold (NOT in-sample, NOT 0.5).
    y_oof, p_oof = oof_predictions("logreg", "class_weight")
    threshold = float(np.quantile(p_oof, 1.0 - TOP_DECILE))
    catch_rate = float(M.recall_at_top_decile(y_oof, p_oof, TOP_DECILE))

    return Scorer(
        pre=pre, model=model, feat_names=names, train_mean=X.mean(axis=0),
        source_of=source, fields=_build_fields(df), threshold=threshold,
        catch_rate=catch_rate, prevalence=float(y.mean()),
        n_rows=int(len(y)), n_events=int(y.sum()),
    )


def _coerce_row(scorer: Scorer, payload: dict) -> pd.DataFrame:
    """Turn a form payload into a single-row frame over the permitted columns.

    Missing or blank fields fall back to the column's training default, so a partially-filled form
    still scores; numeric blanks become NaN and are median-imputed inside the pipeline.
    """
    row = {}
    for fs in scorer.fields:
        raw = payload.get(fs.name, None)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            row[fs.name] = fs.default
        elif fs.kind == "numeric":
            try:
                row[fs.name] = float(raw)
            except (TypeError, ValueError):
                row[fs.name] = np.nan
        else:
            row[fs.name] = str(raw)
    return pd.DataFrame([row], columns=FA.permitted_columns())


def score_applicant(scorer: Scorer, payload: dict, top_k: int = 3) -> dict:
    """Score one applicant: P(default), decile flag, and the top-k named SHAP reasons.

    For a linear model SHAP is exact: phi_j = coef_j * (x_j - E[x_j]). We compute it in the encoded
    space then aggregate signed contributions back to the original named features, so a reason reads
    "Revenue" rather than "Revenue (scaled column 3)".
    """
    row = _coerce_row(scorer, payload)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.asarray(scorer.pre.transform(row)).ravel()
    proba = float(scorer.model.predict_proba(X.reshape(1, -1))[0, 1])

    coef = scorer.model.coef_.ravel()
    contrib = coef * (X - scorer.train_mean)            # exact linear SHAP, encoded space
    agg: dict[str, float] = {}
    for src, phi in zip(scorer.source_of, contrib):
        agg[src] = agg.get(src, 0.0) + float(phi)       # back to the 17 named features

    ordered = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    reasons = [
        {"feature": k, "label": _friendly(k),
         "direction": "increases risk" if v > 0 else "decreases risk",
         "verdict": "raises risk" if v > 0 else "lowers risk",
         "contribution": round(v, 4), "value": _display_value(row, k)}
        for k, v in ordered
    ]

    in_decile = proba >= scorer.threshold
    return {
        "probability": round(proba, 4),
        "percent": round(100 * proba, 2),
        "in_riskiest_decile": bool(in_decile),
        "threshold": round(scorer.threshold, 4),
        "band": "RISKIEST DECILE — prioritise for review" if in_decile else "below review cut",
        "reasons": reasons,
    }


def score_frame(scorer: Scorer, df: pd.DataFrame, top_k: int = 3,
                review_frac: float = TOP_DECILE) -> pd.DataFrame:
    """Score a whole batch of applicants (one per row) — the real operational unit.

    Columns may be any subset of the permitted features (plus an optional free-text ``id``/``case``
    column, which is passed through). Unknown columns are ignored; missing ones fall back to the
    training defaults — the same contract as the single-applicant form.

    Beyond the per-row score, this computes the operating point the way a desk actually uses it:
    rank applicants by risk *within this batch* and flag the top ``review_frac`` as the review
    queue (``rank`` / ``review_queue``). The headline metric (recall@top-decile) is a population
    concept, so the queue is defined over the uploaded batch, not a frozen historical cut.
    ``in_riskiest_decile`` is also kept — that is the absolute, historical-threshold flag.

    Input row order is preserved (join-friendly); sort on ``rank`` for the review-queue view.
    """
    out = df.copy().reset_index(drop=True)

    # Build the model-input frame in one shot: every permitted column, blanks/NaN -> training default
    # (matching the single-applicant contract). This lets the whole batch be transformed at once —
    # the raw 14k-row dataset scores in ~1s instead of ~40s row-by-row.
    cols = {}
    for fs in scorer.fields:
        if fs.name in out.columns:
            s = out[fs.name]
            if fs.kind == "numeric":
                s = pd.to_numeric(s, errors="coerce")
            else:
                s = s.astype(object).replace("", np.nan)
            cols[fs.name] = s.fillna(fs.default)
        else:
            cols[fs.name] = fs.default
    model_df = pd.DataFrame(cols, index=out.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.asarray(scorer.pre.transform(model_df))
    proba = scorer.model.predict_proba(X)[:, 1]

    # Exact linear SHAP for every row at once, aggregated back to the named source features.
    contrib = (X - scorer.train_mean) * scorer.model.coef_.ravel()          # n x d
    agg = pd.DataFrame(contrib, columns=scorer.source_of).T.groupby(level=0).sum().T  # n x sources
    names = agg.columns.to_numpy()
    A = agg.to_numpy()
    order = np.argsort(-np.abs(A), axis=1)[:, :top_k]                        # top-k source idx / row
    reasons_txt = [
        ", ".join(f"{'↑' if A[i, j] > 0 else '↓'} {_friendly(names[j])}" for j in order[i])
        for i in range(len(A))
    ]

    out["probability"] = np.round(proba, 4)
    out["percent"] = np.round(100 * proba, 2)
    # within-batch operating point: rank by risk, queue the riskiest review_frac (at least one row)
    out["rank"] = pd.Series(proba).rank(method="first", ascending=False).astype(int)
    queue_size = max(1, int(np.ceil(len(out) * review_frac))) if len(out) else 0
    out["review_queue"] = out["rank"] <= queue_size
    out["in_riskiest_decile"] = proba >= scorer.threshold
    out["top_reasons"] = reasons_txt
    return out


def _parse_upload_bytes(raw: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded CSV/XLSX bytes into a frame, or raise ``HTTPException(400, ...)``.

    Boundary validation on a public upload endpoint — not defensive over-engineering: malformed
    files, wrong extensions, and oversized payloads are things a real caller WILL send.
    """
    name = (filename or "").lower()
    if not name.endswith(C.ALLOWED_UPLOAD_EXT):
        raise HTTPException(400, f"unsupported file type; expected one of {C.ALLOWED_UPLOAD_EXT}")
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > C.MAX_UPLOAD_MB:
        raise HTTPException(400, f"file too large ({size_mb:.1f} MB > {C.MAX_UPLOAD_MB} MB limit)")
    if not raw:
        raise HTTPException(400, "empty file")
    try:
        buf = io.BytesIO(raw)
        return pd.read_excel(buf) if name.endswith((".xlsx", ".xls")) else pd.read_csv(buf)
    except Exception as exc:  # noqa: BLE001 — genuinely "can happen" at a public upload boundary
        logger.warning("upload parse failed for %r: %s", filename, exc)
        raise HTTPException(400, f"could not parse {filename!r} as CSV/Excel: {exc}") from exc


def _require_recognised_columns(df: pd.DataFrame) -> None:
    """Reject a batch with none of the permitted pre-funding columns present."""
    if not any(c in df.columns for c in FA.permitted_columns()):
        raise HTTPException(400, "no recognised applicant columns found in the uploaded data")


def score_file(in_path: str, out_path: str | None = None) -> dict:
    """Batch-score a CSV/XLSX of applicants → write a results CSV. Returns a summary dict."""
    src = pd.read_excel(in_path) if str(in_path).lower().endswith((".xlsx", ".xls")) \
        else pd.read_csv(in_path)
    scored = score_frame(get_scorer(), src).sort_values("rank").reset_index(drop=True)
    if out_path is None:
        from pathlib import Path
        p = Path(in_path)
        out_path = str(p.with_name(p.stem + "_scored.csv"))
    scored.to_csv(out_path, index=False)
    return {
        "in_path": str(in_path), "out_path": out_path, "n": int(len(scored)),
        "n_review_queue": int(scored["review_queue"].sum()),
        "n_riskiest_decile": int(scored["in_riskiest_decile"].sum()),
    }


# --------------------------------------------------------------------------- example / test data
# Curated, in-distribution demo applicants. Numeric values sit inside each feature's typical
# p10–p90 band so the linear model never has to extrapolate. Only the load-bearing fields are set;
# the rest fall back to dataset defaults at score time.
EXAMPLE_CASES = [
    {"case": "established_low_revenue", "Credit Score": 730, "Revenue": 700,
     "Average Monthly Sales": 30000, "Time In Business": 150, "Amount Sought": 40000,
     "Is Borrower Renewal": 1},
    {"case": "typical_midbook", "Credit Score": 665, "Revenue": 1500,
     "Average Monthly Sales": 34000, "Time In Business": 51, "Amount Sought": 50000},
    {"case": "borderline", "Credit Score": 640, "Revenue": 3500,
     "Average Monthly Sales": 60000, "Time In Business": 30, "Amount Sought": 90000},
    {"case": "high_revenue_short_history", "Credit Score": 615, "Revenue": 9000,
     "Average Monthly Sales": 120000, "Time In Business": 12, "Amount Sought": 150000,
     "Is Borrower Renewal": 0},
    {"case": "thin_file_new_business", "Credit Score": 600, "Revenue": 4000,
     "Average Monthly Sales": 20000, "Time In Business": 6, "Amount Sought": 120000,
     "Online App Completed": 1, "Is Borrower Renewal": 0},
]


def example_cases_frame() -> pd.DataFrame:
    """The curated named demo cases as a frame (a ``case`` label column + partial features)."""
    return pd.DataFrame(EXAMPLE_CASES)


def random_applicants(n: int = 50, seed: int = C.SEED) -> pd.DataFrame:
    """N synthetic applicants for batch testing — privacy-safe, in-distribution.

    Each column is resampled *independently* (with replacement) from its own observed values, so
    every feature keeps its real marginal distribution but no output row reproduces any real
    applicant's record. Joint correlations are intentionally broken; this is test/demo data for
    exercising the batch path, not a statistical twin of the portfolio. Permitted columns only.
    """
    df = D.build_target(D.load_raw(), "paidoff_only")[FA.permitted_columns()]
    rng = np.random.default_rng(seed)
    cols = {}
    for c in df.columns:
        pool = df[c].dropna().to_numpy()
        cols[c] = rng.choice(pool, size=n, replace=True) if len(pool) else [np.nan] * n
    sample = pd.DataFrame(cols)
    sample.insert(0, "id", [f"app_{i:04d}" for i in range(n)])
    return sample


def write_sample_files(n: int = 50, seed: int = C.SEED) -> dict:
    """Write data/example_cases.csv (curated) + data/sample_applicants.csv (random) for batch tests."""
    out_dir = C.PROJECT_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    ex_path = out_dir / "example_cases.csv"
    rnd_path = out_dir / "sample_applicants.csv"
    example_cases_frame().to_csv(ex_path, index=False)
    random_applicants(n, seed).to_csv(rnd_path, index=False)
    return {"example_cases": str(ex_path), "sample_applicants": str(rnd_path), "n_random": n}


def _display_value(row: pd.DataFrame, col: str) -> str:
    v = row.iloc[0][col]
    if isinstance(v, float) and np.isnan(v):
        return "—"
    if isinstance(v, (int, float)) and float(v) == int(v):
        return f"{int(v):,}"                       # thousands separators, e.g. 90,000
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


# --------------------------------------------------------------------------- auth (D15/D17)
def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Gate ``/api/*`` behind a static key when ``EMERALD_API_KEY`` is set.

    **Default-open** (no-op) when the env var is unset, so the local/grading demo needs zero
    config. This is a single-developer academic-demo baseline, not real auth: no per-caller
    identity, no rotation, key travels in a plaintext header. See ``docs/path_to_production.md``
    §governance for the tradeoff and what real deployment would need instead.
    """
    expected = os.getenv("EMERALD_API_KEY")
    if expected is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        logger.warning("rejected request: missing or invalid X-API-Key")
        raise HTTPException(401, "missing or invalid X-API-Key")


# --------------------------------------------------------------------------- web layer
@asynccontextmanager
async def _lifespan(app):
    """Load the model when the process starts, not on the first request.

    A deployed service should be ready before it receives traffic; without this the first visitor
    would wait for the artefact load (or a full fit) and a probe would report 503. A failure is
    logged rather than raised, so /health can report the problem instead of the boot crashing.
    """
    try:
        get_scorer()
        logger.info("model warm: service ready")
    except Exception:  # noqa: BLE001 - surfaced through /health
        logger.error("model could not be loaded at startup", exc_info=True)
    yield


def create_app():
    """Build the FastAPI app (routes over the cached scorer)."""
    app = FastAPI(
        title="EMERALD-AI decision-support API",
        version="1.0",
        docs_url="/docs",
        lifespan=_lifespan,
        description=(
            "Ranks green-loan applications by delinquency risk so that a lending desk can review "
            "the riskiest first. **The service ranks for review; it does not approve or decline.**"
            "\n\n"
            "**Scores are for ordering, not probabilities.** The model was fitted on 50 delinquent "
            "events in 14,135 loans, and its probabilities are unreliable on the minority class, so "
            "`percent` should be treated as a ranking key rather than a likelihood of default.\n\n"
            "**Authentication.** Optional: if the server sets `EMERALD_API_KEY`, every `/api/*` "
            "call needs a matching `X-API-Key` header. `/` and `/health` are always reachable.\n\n"
            "**Rate limit.** 60 requests per client per minute by default "
            "(`EMERALD_RATE_LIMIT`); `/health` is exempt. Exceeding it returns 429.\n\n"
            "**Errors.** 400 unusable input, 401 bad key, 429 rate limited, 500 unexpected fault. "
            "All return `{\"detail\": \"...\"}` and never a stack trace."
        ),
        openapi_tags=[
            {"name": "service", "description": "Health and readiness."},
            {"name": "scoring", "description": "Score one application or a whole batch."},
        ],
    )
    MAX_TABLE_ROWS = 200  # cap rows returned to the browser; the summary still spans the whole file

    @app.exception_handler(Exception)
    async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        """Last-line safety net: never leak an internal traceback to the client."""
        logger.error("unhandled error on %s %s", request.method, request.url.path, exc_info=True)
        return JSONResponse({"detail": "internal error"}, status_code=500)

    def _batch_payload(df: pd.DataFrame) -> dict:
        _require_recognised_columns(df)
        scored = score_frame(get_scorer(), df).sort_values("rank").reset_index(drop=True)
        cols = ["rank"] + [c for c in ("id", "case") if c in scored.columns] + \
               ["percent", "review_queue", "top_reasons"]
        shown = scored.head(MAX_TABLE_ROWS)
        logger.info("scored batch: n=%d review_queue=%d riskiest_decile=%d",
                    len(scored), int(scored["review_queue"].sum()), int(scored["in_riskiest_decile"].sum()))
        return {
            "n": int(len(scored)),
            "shown": int(len(shown)),
            "n_review_queue": int(scored["review_queue"].sum()),
            "n_riskiest_decile": int(scored["in_riskiest_decile"].sum()),
            "rows": shown[cols].to_dict(orient="records"),
        }

    @app.middleware("http")
    async def _rate_limit(request: Request, call_next):
        """Small fixed-window limit per client IP, for public deployments.

        In-memory and therefore per-process: it protects a single instance from casual abuse and
        accidental loops, and is not a substitute for a gateway limiter (see the deployment
        notes). Disabled by setting EMERALD_RATE_LIMIT=0.
        """
        limit = int(os.getenv("EMERALD_RATE_LIMIT", "60"))
        if limit <= 0 or request.url.path == "/health":
            return await call_next(request)
        import time
        now = int(time.time() // 60)
        ip = (request.client.host if request.client else "?")
        window, hits = _RATE_STATE.get(ip, (now, 0))
        if window != now:
            window, hits = now, 0
        hits += 1
        _RATE_STATE[ip] = (window, hits)
        if hits > limit:
            logger.warning("rate limit hit for %s (%d requests this minute)", ip, hits)
            return JSONResponse({"detail": "too many requests, please retry shortly"},
                                status_code=429)
        return await call_next(request)

    @app.get("/health", tags=["service"], summary="Readiness probe")
    def health() -> JSONResponse:
        """Liveness/readiness probe for a container orchestrator or load balancer.

        Deliberately unauthenticated and non-blocking: it reports whether the model is already
        loaded rather than triggering a fit, so a probe never waits on training. ``ready`` is
        false only while the first fit is still in progress.
        """
        loaded = get_scorer.cache_info().currsize > 0
        body = {
            "status": "ok",
            "model_loaded": loaded,
            "ready": loaded,
            "cache_present": C.MODEL_CACHE.exists(),
            "auth_required": os.getenv("EMERALD_API_KEY") is not None,
        }
        if loaded:
            s = get_scorer()
            body |= {
                "operating_threshold": round(s.threshold, 4),
                "catch_rate": round(s.catch_rate, 4),
                "training_rows": s.n_rows,
                "training_events": s.n_events,
            }
        return JSONResponse(body, status_code=200 if loaded else 503)

    @app.get("/", response_class=HTMLResponse, tags=["service"], include_in_schema=False)
    def index() -> str:
        return _render_page(get_scorer())

    @app.post("/api/score", tags=["scoring"], dependencies=[Depends(_require_api_key)],
              summary="Score one application")
    def api_score(payload: dict = Body(...)) -> JSONResponse:
        logger.info("scored single applicant")
        return JSONResponse(score_applicant(get_scorer(), payload))

    @app.post("/api/score-batch", tags=["scoring"], dependencies=[Depends(_require_api_key)],
              summary="Score a batch supplied as CSV text")
    def api_score_batch(payload: dict = Body(...)) -> JSONResponse:
        """Score a pasted CSV (``{"csv": "...text..."}``) → ranked JSON records + summary."""
        text = payload.get("csv", "")
        if not text.strip():
            raise HTTPException(400, "empty csv")
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as exc:  # noqa: BLE001 — user-supplied text, genuinely can be malformed
            logger.warning("pasted-csv parse failed: %s", exc)
            raise HTTPException(400, f"could not parse pasted CSV: {exc}") from exc
        logger.info("scoring pasted batch: %d rows", len(df))
        return JSONResponse(_batch_payload(df))

    @app.post("/api/score-upload", tags=["scoring"], dependencies=[Depends(_require_api_key)],
              summary="Score an uploaded CSV or Excel file")
    async def api_score_upload(file: UploadFile = File(...)) -> JSONResponse:
        """Score an uploaded CSV **or Excel** file — including the raw ``All_Funded_*.xlsx`` dataset.

        Only the permitted pre-funding columns are used; the file's other 140+ columns (and the
        outcome label) are ignored, so the raw book can be dropped in as-is.
        """
        raw = await file.read()
        logger.info("upload received: filename=%r size=%dB", file.filename, len(raw))
        df = _parse_upload_bytes(raw, file.filename or "")
        return JSONResponse(_batch_payload(df))

    return app


def _render_page(scorer: Scorer) -> str:
    """Server-render the single-page form with data-driven defaults. Minimal CSS, one inline script."""
    rows = []
    for fs in scorer.fields:
        if fs.kind == "numeric":
            inp = (f'<input type="number" step="any" name="{fs.name}" '
                   f'value="{fs.default}" data-kind="numeric">'
                   f'<em class="hint">typical {fs.lo:g}–{fs.hi:g}</em>')
        else:
            opts = "".join(
                f'<option value="{o}"{" selected" if o == fs.default else ""}>{o}</option>'
                for o in fs.options
            )
            inp = f'<select name="{fs.name}" data-kind="categorical">{opts}</select>'
        rows.append(f'<label class="fld"><span>{fs.name}</span>{inp}</label>')

    fields_html = "\n".join(rows)
    catch_pct = round(100 * scorer.catch_rate)
    return _PAGE.format(
        fields=fields_html, n_rows=scorer.n_rows, n_events=scorer.n_events,
        prevalence=round(100 * scorer.prevalence, 2), threshold=round(scorer.threshold, 3),
        catch_pct=catch_pct,
    )


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EMERALD-AI — credit decision support</title>
<style>
 :root{{
   /* Restrained, work-tool palette: warm neutrals with a single slate-blue accent.
      Green is kept for the mark alone; red and amber carry meaning, not decoration. */
   --bg:#f4f4f1; --surface:#ffffff; --ink:#22261f; --muted:#5d635a; --faint:#8b9086;
   --line:#dcdcd4; --line-strong:#c6c7bd;
   --brand:#3c5a6e; --brand2:#4a6b80; --brand-deep:#2c4354; --brand-ink:#2c4354;
   --mark:#4a7c59;
   --risk:#9d4a42; --risk-soft:#f7edeb; --risk-ink:#7d3a33;
   --ok:#3c5a6e; --ok-soft:#eef1f3; --ok-ink:#33505f;
   --tint:#f0f0ea;
   --ring:rgba(60,90,110,.22);
   --sh-sm:none; --sh:none;
   --r:3px; --r-sm:3px;
 }}
 *{{box-sizing:border-box}}
 html{{-webkit-text-size-adjust:100%}}
 body{{font-family:"Helvetica Neue",Helvetica,Arial,"Segoe UI",sans-serif;
   margin:0;background:var(--bg);color:var(--ink);line-height:1.5;-webkit-font-smoothing:antialiased}}
 a{{color:var(--brand)}}
 .wrap{{max-width:1080px;margin:0 auto;padding:0 20px}}
 /* header */
 header{{background:var(--brand-deep);color:#fff;
   padding:26px 0 22px;border-bottom:1px solid var(--brand-deep)}}
 .brand{{display:flex;align-items:center;gap:12px}}
 .logo{{width:34px;height:34px;border-radius:5px;background:var(--mark);
   display:grid;place-items:center;flex:0 0 auto}}
 .logo svg{{width:22px;height:22px}}
 header h1{{margin:0;font-size:20px;font-weight:700;letter-spacing:-.02em}}
 header .tag{{font-size:12px;opacity:.8;margin-top:1px}}
 header p{{margin:16px 0 0;max-width:760px;font-size:14px;opacity:.92}}
 header p b{{font-weight:600}}
 /* layout */
 section,main{{margin-top:22px}}
 .card{{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);
   padding:16px}}

 /* heading bar spanning the full width of the panel */
 .sechead{{display:flex;align-items:center;gap:9px;margin:-16px -16px 13px;padding:9px 16px;
   background:var(--tint);border-bottom:1px solid var(--line)}}
 .num{{width:20px;height:20px;border-radius:2px;background:var(--brand-ink);color:#fff;font-size:11px;
   font-weight:700;display:grid;place-items:center;flex:0 0 auto}}
 .num.alt{{background:var(--line-strong);color:var(--surface)}}
 h2{{font-size:14px;margin:0;font-weight:700;letter-spacing:0}}
 .lead{{font-size:13px;color:var(--muted);margin:0 0 0;max-width:780px}}
 .lead b{{color:var(--ink);font-weight:600}}
 main{{display:grid;grid-template-columns:1.35fr 1fr;gap:22px;align-items:start}}
 @media(max-width:840px){{main{{grid-template-columns:1fr}}}}
 /* form fields */
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:13px;margin-top:18px}}
 .fld{{display:flex;flex-direction:column;gap:5px}}
 .fld span{{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}}
 input,select{{padding:8px 10px;border:1px solid var(--line-strong);border-radius:var(--r-sm);font-size:14px;
   background:var(--surface);color:var(--ink);font-family:inherit}}
 input:focus,select:focus{{outline:2px solid var(--brand);outline-offset:1px;border-color:var(--brand)}}
 .hint{{color:var(--faint);font-size:10.5px;font-style:normal;letter-spacing:.01em}}
 /* buttons */
 .btn{{appearance:none;border:0;cursor:pointer;font-family:inherit;font-weight:600;font-size:14px;
   border-radius:var(--r-sm);padding:10px 18px;color:#fff;background:var(--brand);
   transition:background .15s}}
 .btn:hover{{background:var(--brand-deep)}}
 .btn:focus-visible{{outline:2px solid var(--ink);outline-offset:2px}}
 .btn.block{{width:100%;margin-top:18px}}
 .meta{{font-size:12.5px;color:var(--muted);margin-top:12px}}
 .meta b{{color:var(--ink);font-weight:600}}
 code{{background:var(--tint);color:var(--brand-ink);padding:1.5px 6px;border-radius:6px;
   font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
 /* result panel */
 #out{{display:none}}
 .scorewrap{{display:flex;align-items:baseline;gap:12px;margin:6px 0 2px;flex-wrap:wrap}}
 .score{{font-size:48px;font-weight:800;letter-spacing:-.03em;font-variant-numeric:tabular-nums;line-height:1}}
 .caption{{font-size:11px;font-weight:600;color:var(--faint);text-transform:uppercase;letter-spacing:.06em}}
 .pill{{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:3px;
   font-size:12px;font-weight:700;letter-spacing:.01em}}
 .pill.risk{{background:var(--risk-soft);color:var(--risk-ink);border:1px solid var(--risk-soft)}}
 .pill.ok{{background:var(--ok-soft);color:var(--ok-ink);border:1px solid #bfdbfe}}
 .dot{{width:7px;height:7px;border-radius:50%}} .pill.risk .dot{{background:var(--risk)}} .pill.ok .dot{{background:var(--ok)}}
 .subhead{{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin:20px 0 9px}}
 .reason{{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:11px 13px;
   border-radius:var(--r-sm);margin-top:8px;font-size:13.5px;background:var(--tint);border:1px solid var(--line)}}
 .reason.up{{background:var(--risk-soft);border-color:#e6d3cf}} .reason.down{{background:var(--ok-soft);border-color:#d6dde1}}
 .reason b{{font-weight:600}} .reason small{{color:var(--faint)}}
 .tag-dir{{font-size:11.5px;font-weight:700;white-space:nowrap}}
 .up .tag-dir{{color:var(--risk-ink)}} .down .tag-dir{{color:var(--ok-ink)}}
 .disc{{font-size:11.5px;color:var(--faint);margin-top:18px;border-top:1px solid var(--line);padding-top:12px;line-height:1.55}}
 .legend{{margin-top:16px;background:var(--tint);border:1px solid var(--line);border-radius:var(--r-sm);padding:14px 16px}}
 .legend-h{{font-size:12px;font-weight:700;color:var(--brand-ink);text-transform:uppercase;letter-spacing:.04em;margin-bottom:8px}}
 .legend ul{{margin:0;padding-left:18px}} .legend li{{font-size:12.5px;color:var(--ink);margin:5px 0;line-height:1.5}}
 .legend-f{{margin-top:9px;font-size:12.5px;font-weight:600;color:var(--brand-ink)}}
 .action{{margin:12px 0 2px;font-size:13px;color:var(--ink);background:var(--tint);border:1px solid var(--line);border-radius:var(--r-sm);padding:10px 13px;line-height:1.5}}
 .empty{{display:grid;place-items:center;text-align:center;padding:30px 14px;color:var(--faint)}}
 .empty svg{{width:40px;height:40px;opacity:.5;margin-bottom:8px}}
 /* batch */
 .dropzone{{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:16px;padding:16px;
   border:1px dashed var(--line-strong);border-radius:var(--r-sm);background:var(--tint);transition:border-color .15s,background .15s}}
 .dropzone:hover{{border-color:var(--brand);background:var(--ok-soft)}}
 input[type=file]{{font-size:13px;color:var(--muted);background:transparent;border:0;padding:0;flex:1 1 200px}}
 input[type=file]::file-selector-button{{font-family:inherit;font-weight:600;font-size:13px;cursor:pointer;
   margin-right:12px;padding:7px 13px;border:1px solid var(--line-strong);border-radius:var(--r-sm);color:var(--brand-ink);
   background:var(--line);transition:background .15s}}
 input[type=file]::file-selector-button:hover{{background:#a7f3d0}}
 #batchsummary{{margin-top:14px}}
 .summary-card{{display:flex;gap:18px;flex-wrap:wrap;margin-top:14px}}
 .stat{{background:var(--tint);border:1px solid var(--line);border-radius:var(--r-sm);padding:12px 16px;min-width:120px}}
 .stat .v{{font-size:24px;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums}}
 .stat.flag .v{{color:var(--risk)}}
 .stat .l{{font-size:11px;color:var(--muted);margin-top:2px}}
 .tablewrap{{overflow-x:auto;margin-top:16px;border:1px solid var(--line);border-radius:var(--r-sm)}}
 table.bt{{border-collapse:collapse;font-size:12.5px;width:100%}}
 table.bt th,table.bt td{{padding:9px 13px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line)}}
 table.bt th{{background:var(--tint);color:var(--muted);font-weight:600;text-transform:uppercase;font-size:10.5px;letter-spacing:.04em}}
 table.bt tbody tr:last-child td{{border-bottom:0}}
 table.bt tbody tr:nth-child(even){{background:var(--tint)}}
 table.bt tbody tr:hover{{background:var(--ok-soft)}}
 table.bt tr.flag{{background:var(--risk-soft)}} table.bt tr.flag:hover{{background:#f1e2df}}
 table.bt tr.flag td:first-child{{font-weight:700;color:var(--risk-ink)}}
 footer{{text-align:center;color:var(--faint);font-size:11.5px;padding:30px 0 36px}}
 /* two-column shell: a fixed side rail, content to the right */
 .layout{{display:grid;grid-template-columns:210px minmax(0,1fr);gap:22px;align-items:start;
   padding-top:22px}}
 @media(max-width:820px){{ .layout{{grid-template-columns:1fr;gap:14px}} }}
 .content{{min-width:0}}
 .side{{position:sticky;top:22px;background:var(--surface);border:1px solid var(--line);
   border-radius:var(--r-sm)}}
 @media(max-width:820px){{ .side{{position:static}} }}
 .side-label{{font-size:10.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
   color:var(--faint);padding:11px 13px 6px;border-bottom:1px solid var(--line)}}
 .side [role="tablist"]{{display:flex;flex-direction:column}}
 .side [role="tablist"] button:last-child{{border-bottom:0}}
 @media(max-width:820px){{ .side [role="tablist"]{{flex-direction:row;flex-wrap:wrap}} }}
 .side button{{appearance:none;cursor:pointer;font-family:inherit;font-size:13.5px;text-align:left;
   padding:9px 13px;border:0;border-bottom:1px solid var(--line);background:var(--surface);
   color:var(--ink);border-left:3px solid transparent;transition:background .12s}}
 @media(max-width:820px){{ .side button{{flex:1 1 auto;border-left:0;border-bottom:2px solid transparent}} }}
 .side button:hover{{background:var(--tint)}}
 .side button[aria-selected="true"]{{background:var(--tint);border-left-color:var(--brand);
   font-weight:600;color:var(--brand-ink)}}
 @media(max-width:820px){{ .side button[aria-selected="true"]{{border-left-color:transparent;
   border-bottom-color:var(--brand)}} }}
 .side button:focus-visible{{outline:2px solid var(--brand);outline-offset:-2px}}
 .panel[hidden]{{display:none}}
 /* help + api content */
 .steps{{counter-reset:s;display:grid;gap:14px;margin:16px 0 0}}
 .step{{display:grid;grid-template-columns:28px 1fr;gap:13px;align-items:start}}
 .step .sn{{counter-increment:s;width:28px;height:28px;border-radius:50%;background:var(--brand-ink);
   color:#fff;display:grid;place-items:center;font-size:13px;font-weight:700}}
 .step .sn::before{{content:counter(s)}}
 .step h4{{margin:2px 0 3px;font-size:14.5px}}
 .step p{{margin:0;font-size:13.5px;color:var(--muted)}}
 .qa{{border-top:1px solid var(--line);padding:13px 0}}
 .qa:first-of-type{{border-top:0}}
 .qa h4{{margin:0 0 5px;font-size:14px}}
 .qa p{{margin:0;font-size:13.5px;color:var(--muted)}}
 .kv{{display:grid;grid-template-columns:auto 1fr;gap:9px 14px;font-size:13.5px;margin:12px 0 0}}
 .kv dt{{font-weight:600}} .kv dd{{margin:0;color:var(--muted)}}
 .warnbox{{background:var(--risk-soft);border:1px solid #e0c9c4;border-radius:var(--r-sm);
   padding:13px 15px;margin:16px 0 0;font-size:13.5px;color:var(--risk-ink)}}
 .warnbox b{{font-weight:700}}
 .ep{{border:1px solid var(--line);border-radius:var(--r-sm);padding:14px 15px;margin-top:12px;background:var(--tint)}}
 .ep .sig{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;font-weight:600}}
 .ep .verb{{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.06em;
   padding:2px 7px;border-radius:5px;margin-right:7px;background:var(--brand-ink);color:#fff}}
 .ep .verb.get{{background:var(--ok-ink)}}
 .ep p{{margin:7px 0 0;font-size:13.5px;color:var(--muted)}}
 pre.code{{background:#26302b;color:#dfe4dc;border-radius:var(--r-sm);padding:11px 13px;overflow-x:auto;
   font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.5;margin:10px 0 0}}
 pre.code .c{{color:#7dd3a0}}
</style></head><body>
<header><div class="wrap">
  <div class="brand">
    <div class="logo"><svg viewBox="0 0 24 24" fill="none"><path d="M12 21c5-2 8-6 8-12V4l-8 2-8-2v5c0 6 3 10 8 12z" fill="#fff" opacity=".95"/><path d="M12 17V8M9 11l3-3 3 3" stroke="#4a7c59" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
    <div><h1>EMERALD-AI</h1><div class="tag">green-loan credit decision support</div></div>
  </div>
  <p>The model <b>ranks a batch of applications</b> and routes the riskiest decile to human review —
    it does not approve or decline.</p>
</div></header>

<div class="wrap layout">
<nav class="side">
  <div class="side-label">Sections</div>
  <div role="tablist">
    <button role="tab" id="tab-score" aria-controls="panel-score" aria-selected="true">Score applications</button>
    <button role="tab" id="tab-help" aria-controls="panel-help" aria-selected="false">How to use this</button>
    <button role="tab" id="tab-api" aria-controls="panel-api" aria-selected="false">API reference</button>
  </div>
</nav>

<div class="content">
<div class="panel" id="panel-score" role="tabpanel" aria-labelledby="tab-score">
<section class="card">
  <div class="sechead"><span class="num">1</span><h2>Batch review queue</h2></div>
  <p class="lead">The operational use case. Upload the day's applications as a <b>CSV or Excel</b>
    file — the model ranks them by risk and flags the riskiest <b>10%</b> as the review queue
    (reviewing the top decile historically catches ~{catch_pct}% of all defaults). Only the
    pre-funding columns are used, so you can drop in the raw
    <code>All_Funded_2019_Green Loan.xlsx</code> as-is; extra columns are ignored. An optional
    <b>id</b>/<b>case</b> column is passed through. Sample files:
    <code>data/sample_applicants.csv</code>, <code>data/example_cases.csv</code>.</p>
  <div class="dropzone">
    <input type="file" id="file" accept=".csv,.xlsx,.xls">
    <button id="batchbtn" type="button" class="btn">Rank applications</button>
  </div>
  <div class="legend">
    <div class="legend-h">How to read the results</div>
    <ul>
      <li><b>Rank</b> &amp; <b>Percent</b> — the risk order and risk score (higher = riskier). Use it to
        <b>prioritise who to look at</b> — it is a ranking score, not a literal "% chance of default".</li>
      <li><b>Review queue</b> — the riskiest 10% of this file. <b>These are the applications to review first.</b></li>
      <li><b>Top reasons</b> — what is driving the score: <b>↑</b> a factor raising risk, <b>↓</b> one lowering it.</li>
    </ul>
    <div class="legend-f">→ The model tells you <b>who to check first</b>. A human still makes the approve/decline decision.</div>
  </div>
  <div id="batchsummary"></div>
  <div id="batchtable"></div>
</section>

<main>
  <form id="f" class="card">
    <div class="sechead"><span class="num">2</span><h2>Single application</h2></div>
    <p class="lead">Decompose one decision (the "why was this flagged?" answer for an adverse-action
      notice) or stress-test how the score moves as a feature changes.</p>
    <div class="grid">{fields}</div>
    <button type="submit" class="btn block">Score applicant</button>
    <p class="meta">Defaults are dataset medians/modes — change only the fields you care about.
      Trained on <b>{n_rows}</b> loans, <b>{n_events}</b> defaults ({prevalence}% prevalence).</p>
  </form>
  <div>
    <div class="card" id="placeholder">
      <div class="sechead"><span class="num alt">→</span><h2>Result</h2></div>
      <div class="empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 13h4l3 7 4-14 3 7h4" stroke-linecap="round" stroke-linejoin="round"/></svg>
        <div style="font-size:13px;max-width:230px">Fill the form and press <b>Score applicant</b> to
          see the default probability, review-queue status, and the top-3 reasons.</div>
      </div>
    </div>
    <div class="card" id="out">
      <div class="sechead"><span class="num alt">→</span><h2>Result</h2></div>
      <div class="caption">Risk score — higher means review sooner (a ranking score, not a literal % chance)</div>
      <div class="scorewrap"><div class="score" id="score">—</div><div id="band"></div></div>
      <div id="action" class="action"></div>
      <div class="subhead">Top reasons &mdash; what is driving this score</div>
      <div id="reasons"></div>
      <p class="disc">Reference threshold: historical riskiest decile = P(default) &ge; {threshold}
        (out-of-fold). For one applicant this is an absolute reference; the real review queue is the
        within-batch top decile above. A score is a prioritisation signal, not an adverse-action decision.</p>
    </div>
  </div>
</main>
</div><!-- /panel-score -->

<div class="panel" id="panel-help" role="tabpanel" aria-labelledby="tab-help" hidden>
<section class="card">
  <div class="sechead"><span class="num">?</span><h2>What this tool does</h2></div>
  <p class="lead">This tool <b>puts a pile of loan applications in order of risk</b> so that the
    riskiest ones get looked at first. That is all it does. It never approves anyone, never
    declines anyone, and never replaces a credit decision. Think of it as the colleague who reads
    the whole pile overnight and leaves the ones worth your attention on top.</p>
  <div class="warnbox">
    <b>Please do not use the percentage as a probability.</b> A score of 80% does not mean
    "80 out of 100 of these borrowers will default". The number is reliable for <b>ordering</b>
    applications, not for stating how likely any single default is. Use it to decide
    <i>who to look at first</i>, never to justify a decision on its own.
  </div>
</section>

<section class="card">
  <div class="sechead"><span class="num">1</span><h2>Reviewing a day's applications</h2></div>
  <div class="steps">
    <div class="step"><div class="sn"></div><div>
      <h4>Export your applications to a spreadsheet</h4>
      <p>A CSV or Excel file, one application per row. Extra columns are ignored, so an ordinary
        export works — no need to tidy it up first. If you include a column called <b>id</b>,
        it is carried through so you can match rows back to your system.</p></div></div>
    <div class="step"><div class="sn"></div><div>
      <h4>Upload it on the "Score applications" tab</h4>
      <p>Choose the file and press <b>Rank applications</b>. Scoring the whole file takes about a
        second.</p></div></div>
    <div class="step"><div class="sn"></div><div>
      <h4>Work down the review queue</h4>
      <p>The table comes back ordered from riskiest to safest, and the top 10% of that upload is
        highlighted as the <b>review queue</b>. Historically, reviewing that top 10% is where
        roughly {catch_pct}% of all eventual defaults were found.</p></div></div>
  </div>
</section>

<section class="card">
  <div class="sechead"><span class="num">2</span><h2>Reading the results</h2></div>
  <dl class="kv">
    <dt>Percentage</dt>
    <dd>A risk <i>ranking</i> score. Higher means "look at this one sooner". Compare applications
      with each other; do not read it as a chance of default.</dd>
    <dt>Review queue</dt>
    <dd>The riskiest 10% <b>of the file you just uploaded</b>. Upload 200 applications and 20 are
      flagged; upload 20 and 2 are flagged.</dd>
    <dt>Top reasons</dt>
    <dd>The three things that pushed this application's score up or down the most. An arrow up
      means that value raised the risk score, down means it lowered it. This is the answer to
      "why is this one near the top?"</dd>
    <dt>Riskiest decile flag</dt>
    <dd>On the single-application panel only: whether this applicant would fall in the riskiest
      10% of the <i>historical</i> book. It is a fixed reference point, unlike the queue above,
      which depends on the file you uploaded.</dd>
  </dl>
</section>

<section class="card">
  <div class="sechead"><span class="num">3</span><h2>What you should know before relying on it</h2></div>
  <div class="qa">
    <h4>It learned from very few actual defaults</h4>
    <p>The whole model is built on 50 loans that went wrong out of 14,135. That is enough to sort
      applications usefully, and not enough to say precisely how risky any one applicant is.</p>
  </div>
  <div class="qa">
    <h4>It has only seen loans that were approved</h4>
    <p>Applications your organisation turned down are not in the data, so the tool ranks
      <i>within</i> the kind of application you normally fund. It cannot tell you about applicants
      unlike those.</p>
  </div>
  <div class="qa">
    <h4>Fairness across groups has not been verified</h4>
    <p>Checking whether the model treats industries or regions even-handedly needs far more
      defaults per group than exist here. That check was attempted and could not be completed, so
      no claim of fairness is made in either direction.</p>
  </div>
  <div class="qa">
    <h4>The reasons explain the score, not the borrower</h4>
    <p>They tell you what drove this score in this model. They are a starting point for a
      conversation or a file review, not evidence about the business itself.</p>
  </div>
  <div class="qa">
    <h4>It should not be used to tell someone why they were declined</h4>
    <p>If a decision has to be explained to an applicant, that explanation must come from your
      own credit policy and a human reviewer, not from this screen.</p>
  </div>
</section>
</div><!-- /panel-help -->

<div class="panel" id="panel-api" role="tabpanel" aria-labelledby="tab-api" hidden>
<section class="card">
  <div class="sechead"><span class="num">&lt;/&gt;</span><h2>API reference</h2></div>
  <p class="lead">Every screen in this tool is a thin layer over these endpoints, so anything the
    interface can do can be scripted. An interactive, always-current schema is served at
    <a href="/docs">/docs</a>.</p>
  <dl class="kv">
    <dt>Base URL</dt><dd>The address you are reading this page from.</dd>
    <dt>Authentication</dt>
    <dd>Optional. If the server sets <code>EMERALD_API_KEY</code>, every <code>/api/*</code> call
      needs a matching <code>X-API-Key</code> header; <code>/</code> and <code>/health</code>
      stay open. Without that variable the API is unauthenticated.</dd>
    <dt>Rate limit</dt>
    <dd>60 requests per client per minute by default (<code>EMERALD_RATE_LIMIT</code>);
      <code>/health</code> is exempt. Exceeding it returns <b>429</b>.</dd>
    <dt>Errors</dt>
    <dd><b>400</b> unreadable or unusable input · <b>401</b> missing/incorrect key ·
      <b>429</b> rate limited · <b>500</b> unexpected fault. All return
      <code>{{"detail": "..."}}</code> and never a stack trace.</dd>
  </dl>
</section>

<section class="card">
  <div class="sechead"><span class="num alt">→</span><h2>Endpoints</h2></div>

  <div class="ep">
    <div class="sig"><span class="verb get">GET</span>/health</div>
    <p>Readiness probe. Unauthenticated and non-blocking: returns <b>503</b> while the model is
      still loading and <b>200</b> once it can serve.</p>
    <pre class="code">{{"status":"ok","model_loaded":true,"ready":true,
 "operating_threshold":{threshold},"catch_rate":0.62,
 "training_rows":{n_rows},"training_events":{n_events}}}</pre>
  </div>

  <div class="ep">
    <div class="sig"><span class="verb">POST</span>/api/score</div>
    <p>Score one application. Send any subset of the permitted fields; anything omitted falls back
      to that field's training median, so a partial payload still scores.</p>
    <pre class="code"><span class="c"># request</span>
{{"Revenue": 9000, "Credit Score": 620, "Time In Business": 12}}

<span class="c"># response</span>
{{"probability":0.9928,"percent":99.28,"band":"riskiest decile",
 "in_riskiest_decile":true,"threshold":{threshold},
 "reasons":[{{"feature":"Revenue","label":"Monthly revenue",
             "value":"9,000","contribution":1.83,
             "direction":"increases risk","verdict":"raises risk"}}]}}</pre>
  </div>

  <div class="ep">
    <div class="sig"><span class="verb">POST</span>/api/score-upload</div>
    <p>Score a whole file (<code>multipart/form-data</code>, field name <code>file</code>).
      Accepts <code>.csv</code>, <code>.xlsx</code>, <code>.xls</code> up to 10&nbsp;MB. Rows come
      back ranked, riskiest first, with the top decile of the upload flagged.</p>
    <pre class="code">curl -X POST https://HOST/api/score-upload \
  -H "X-API-Key: $KEY" \
  -F "file=@applications.csv"

<span class="c"># response (table truncated to the riskiest 200 rows)</span>
{{"n":12,"shown":12,"n_review_queue":2,"n_riskiest_decile":2,
 "rows":[{{"rank":1,"id":"app_0004","percent":93.54,
          "review_queue":true,"top_reasons":"↑ Monthly revenue, ..."}}]}}</pre>
  </div>

  <div class="ep">
    <div class="sig"><span class="verb">POST</span>/api/score-batch</div>
    <p>Same as the upload endpoint, but the file is sent as CSV text in JSON —
      <code>{{"csv": "id,Revenue\\na,800\\n"}}</code> — for callers that already hold the data in
      memory. The response is identical.</p>
  </div>
</section>

<section class="card">
  <div class="sechead"><span class="num alt">!</span><h2>Notes for integrators</h2></div>
  <div class="qa">
    <h4>Only pre-funding fields are read</h4>
    <p>Seventeen application-time fields are permitted; every other column in your file is ignored,
      including any outcome column. Sending extra data cannot influence the score.</p>
  </div>
  <div class="qa">
    <h4>The queue flag is relative to your upload</h4>
    <p><code>review_queue</code> marks the riskiest 10% of the rows you sent.
      <code>in_riskiest_decile</code> compares against the fixed historical threshold
      (P&nbsp;&ge;&nbsp;{threshold}). Use the first for routing work, the second for a stable
      reference.</p>
  </div>
  <div class="qa">
    <h4>Scores are for ranking</h4>
    <p>The probabilities are not calibrated on the rare default cases. Treat <code>percent</code>
      as an ordering key; do not surface it to applicants or use it as a probability in downstream
      pricing.</p>
  </div>
</section>
</div><!-- /panel-api -->

<footer>EMERALD-AI · proof-of-concept decision support · the model ranks for review, it does not decide</footer>
</div><!-- /content -->
</div>

<script>
const $=id=>document.getElementById(id);
// tabs
for(const t of document.querySelectorAll('[role="tab"]')){{
  t.addEventListener('click',()=>{{
    for(const o of document.querySelectorAll('[role="tab"]')){{
      const on=o===t;
      o.setAttribute('aria-selected',on?'true':'false');
      $(o.getAttribute('aria-controls')).hidden=!on;
    }}
    window.scrollTo({{top:0,behavior:'smooth'}});
  }});
}}
const f=$('f');
f.addEventListener('submit',async e=>{{
  e.preventDefault();
  const data={{}};
  for(const el of f.querySelectorAll('[name]')) data[el.name]=el.value;
  const r=await fetch('/api/score',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(data)}});
  const j=await r.json();
  $('placeholder').style.display='none';
  $('out').style.display='block';
  const s=$('score'); s.textContent=j.percent.toFixed(1)+'%';
  s.style.color=j.in_riskiest_decile?'var(--risk)':'var(--ink)';
  $('band').innerHTML='<span class="pill '+(j.in_riskiest_decile?'risk':'ok')+'"><span class="dot"></span>'+j.band+'</span>';
  $('action').innerHTML=j.in_riskiest_decile
    ? '→ <b>Send to manual review.</b> This applicant is among the riskiest — check the reasons below before funding.'
    : '→ <b>Lower priority.</b> Below the review cut-off; no extra review needed on risk grounds alone.';
  const box=$('reasons'); box.innerHTML='';
  for(const x of j.reasons){{
    const up=x.contribution>0;
    box.innerHTML+='<div class="reason '+(up?'up':'down')+'"><span><b>'+x.label+
      '</b> <small>= '+x.value+'</small></span><span class="tag-dir">'+(up?'▲ ':'▼ ')+x.verdict+'</span></div>';
  }}
  $('out').scrollIntoView({{behavior:'smooth',block:'nearest'}});
}});

$('batchbtn').addEventListener('click',async()=>{{
  const fi=$('file'); if(!fi.files.length){{alert('Choose a CSV or Excel file first');return;}}
  const btn=$('batchbtn'); btn.disabled=true; btn.textContent='Scoring…';
  try{{
    const fd=new FormData(); fd.append('file',fi.files[0]);
    const r=await fetch('/api/score-upload',{{method:'POST',body:fd}});
    if(!r.ok){{$('batchsummary').innerHTML='<p class="meta">Could not read that file — check it is a CSV or Excel file.</p>';return;}}
    const j=await r.json();
    const note=j.shown<j.n?' · showing the riskiest '+j.shown:'';
    $('batchsummary').innerHTML=
      '<div class="summary-card">'+
      '<div class="stat"><div class="v">'+j.n.toLocaleString()+'</div><div class="l">applications ranked'+note+'</div></div>'+
      '<div class="stat flag"><div class="v">'+j.n_review_queue.toLocaleString()+'</div><div class="l">review queue (top decile)</div></div>'+
      '<div class="stat"><div class="v">'+j.n_riskiest_decile.toLocaleString()+'</div><div class="l">clear historical threshold</div></div>'+
      '</div>';
    const keys=Object.keys(j.rows[0]||{{}});
    let h='<div class="tablewrap"><table class="bt"><thead><tr>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr></thead><tbody>';
    for(const row of j.rows){{
      h+='<tr class="'+(row.review_queue?'flag':'')+'">'+keys.map(k=>'<td>'+row[k]+'</td>').join('')+'</tr>';
    }}
    $('batchtable').innerHTML=h+'</tbody></table></div>';
  }} finally {{ btn.disabled=false; btn.textContent='Rank applications'; }}
}});
</script>
</body></html>"""


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Boot the demo server. Trains the model up front so the first request is instant."""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.info("training frozen model + setting operating point ...")
    s = get_scorer()
    logger.info("ready. riskiest-decile threshold P>=%.3f (OOF catch-rate %.0f%% of %d defaults).",
                s.threshold, 100 * s.catch_rate, s.n_events)
    if os.getenv("EMERALD_API_KEY"):
        logger.info("API key auth ENABLED for /api/* routes")
    else:
        logger.warning("API key auth DISABLED (EMERALD_API_KEY not set) — /api/* routes are open")
    logger.info("open http://%s:%s/  (Ctrl+C to stop)", host, port)
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
