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

import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import feature_audit as FA
from . import metrics as M
from . import preprocess as P
from .experiments import _make_model, oof_predictions

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


@lru_cache(maxsize=1)
def get_scorer() -> Scorer:
    """Train the frozen model once on all cleaned data and cache it (process-lifetime singleton)."""
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
    permitted = set(FA.permitted_columns())
    out = df.copy().reset_index(drop=True)
    probs, pcts, flags, reasons_txt = [], [], [], []
    for _, row in out.iterrows():
        payload = {k: row[k] for k in df.columns
                   if k in permitted and not (pd.isna(row[k]) if np.isscalar(row[k]) else False)}
        r = score_applicant(scorer, payload, top_k=top_k)
        probs.append(r["probability"])
        pcts.append(r["percent"])
        flags.append(r["in_riskiest_decile"])
        reasons_txt.append(", ".join(
            f"{'↑' if x['contribution'] > 0 else '↓'} {x['label']}" for x in r["reasons"]))
    out["probability"] = probs
    out["percent"] = pcts
    # within-batch operating point: rank by risk, queue the riskiest review_frac (at least one row)
    out["rank"] = (pd.Series(probs).rank(method="first", ascending=False)).astype(int)
    queue_size = max(1, int(np.ceil(len(out) * review_frac))) if len(out) else 0
    out["review_queue"] = out["rank"] <= queue_size
    out["in_riskiest_decile"] = flags
    out["top_reasons"] = reasons_txt
    return out


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


# --------------------------------------------------------------------------- web layer
def create_app():
    """Build the FastAPI app. Imported lazily so the package has no hard FastAPI dependency."""
    from fastapi import Body, FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="EMERALD-AI — decision-support demo", docs_url="/docs")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _render_page(get_scorer())

    @app.post("/api/score")
    def api_score(payload: dict = Body(...)) -> JSONResponse:
        return JSONResponse(score_applicant(get_scorer(), payload))

    @app.post("/api/score-batch")
    def api_score_batch(payload: dict = Body(...)) -> JSONResponse:
        """Score a pasted/uploaded CSV (``{"csv": "...text..."}``) → ranked JSON records + summary."""
        import io

        df = pd.read_csv(io.StringIO(payload.get("csv", "")))
        scored = score_frame(get_scorer(), df).sort_values("rank").reset_index(drop=True)
        cols = ["rank"] + [c for c in ("id", "case") if c in scored.columns] + \
               ["percent", "review_queue", "top_reasons"]
        return JSONResponse({
            "n": int(len(scored)),
            "n_review_queue": int(scored["review_queue"].sum()),
            "n_riskiest_decile": int(scored["in_riskiest_decile"].sum()),
            "rows": scored[cols].to_dict(orient="records"),
        })

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
   --bg:#eef2f6; --surface:#ffffff; --ink:#0f172a; --muted:#64748b; --faint:#94a3b8;
   --line:#e7ecf2; --brand:#059669; --brand2:#10b981; --brand-deep:#064e3b; --brand-ink:#065f46;
   --risk:#dc2626; --risk-soft:#fef2f2; --risk-ink:#991b1b;
   --ok:#2563eb; --ok-soft:#eff6ff; --ok-ink:#1e40af;
   --ring:rgba(16,185,129,.28);
   --sh-sm:0 1px 2px rgba(15,23,42,.05); --sh:0 6px 22px rgba(15,23,42,.07);
   --r:16px; --r-sm:11px;
 }}
 *{{box-sizing:border-box}}
 html{{-webkit-text-size-adjust:100%}}
 body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
   margin:0;background:var(--bg);color:var(--ink);line-height:1.5;-webkit-font-smoothing:antialiased}}
 a{{color:var(--brand)}}
 .wrap{{max-width:1080px;margin:0 auto;padding:0 20px}}
 /* header */
 header{{background:linear-gradient(135deg,var(--brand-deep),var(--brand-ink));color:#fff;
   padding:30px 0 26px;box-shadow:var(--sh)}}
 .brand{{display:flex;align-items:center;gap:12px}}
 .logo{{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,var(--brand2),var(--brand));
   display:grid;place-items:center;box-shadow:0 4px 14px rgba(5,150,105,.45);flex:0 0 auto}}
 .logo svg{{width:22px;height:22px}}
 header h1{{margin:0;font-size:20px;font-weight:700;letter-spacing:-.02em}}
 header .tag{{font-size:12px;opacity:.8;margin-top:1px}}
 header p{{margin:16px 0 0;max-width:760px;font-size:14px;opacity:.92}}
 header p b{{font-weight:600}}
 /* layout */
 section,main{{margin-top:22px}}
 .card{{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);
   padding:22px;box-shadow:var(--sh-sm);transition:box-shadow .2s ease}}
 .card:hover{{box-shadow:var(--sh)}}
 .sechead{{display:flex;align-items:center;gap:11px;margin:0 0 6px}}
 .num{{width:26px;height:26px;border-radius:8px;background:var(--brand-ink);color:#fff;font-size:13px;
   font-weight:700;display:grid;place-items:center;flex:0 0 auto}}
 .num.alt{{background:#e2e8f0;color:var(--muted)}}
 h2{{font-size:16px;margin:0;font-weight:700;letter-spacing:-.01em}}
 .lead{{font-size:13px;color:var(--muted);margin:6px 0 0;max-width:780px}}
 .lead b{{color:var(--ink);font-weight:600}}
 main{{display:grid;grid-template-columns:1.35fr 1fr;gap:22px;align-items:start}}
 @media(max-width:840px){{main{{grid-template-columns:1fr}}}}
 /* form fields */
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:13px;margin-top:18px}}
 .fld{{display:flex;flex-direction:column;gap:5px}}
 .fld span{{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}}
 input,select{{padding:9px 11px;border:1px solid var(--line);border-radius:10px;font-size:14px;
   background:#fbfdff;color:var(--ink);font-family:inherit;transition:border-color .15s,box-shadow .15s}}
 input:focus,select:focus{{outline:none;border-color:var(--brand2);box-shadow:0 0 0 3px var(--ring);background:#fff}}
 .hint{{color:var(--faint);font-size:10.5px;font-style:normal;letter-spacing:.01em}}
 /* buttons */
 .btn{{appearance:none;border:0;cursor:pointer;font-family:inherit;font-weight:600;font-size:14px;
   border-radius:11px;padding:12px 20px;color:#fff;background:linear-gradient(135deg,var(--brand2),var(--brand));
   box-shadow:0 4px 14px rgba(5,150,105,.32);transition:transform .12s,box-shadow .2s,filter .2s}}
 .btn:hover{{filter:brightness(1.04);box-shadow:0 6px 18px rgba(5,150,105,.4)}}
 .btn:active{{transform:translateY(1px)}}
 .btn.block{{width:100%;margin-top:18px}}
 .meta{{font-size:12.5px;color:var(--muted);margin-top:12px}}
 .meta b{{color:var(--ink);font-weight:600}}
 code{{background:#eef6f1;color:var(--brand-ink);padding:1.5px 6px;border-radius:6px;
   font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
 /* result panel */
 #out{{display:none}}
 .scorewrap{{display:flex;align-items:baseline;gap:12px;margin:6px 0 2px;flex-wrap:wrap}}
 .score{{font-size:48px;font-weight:800;letter-spacing:-.03em;font-variant-numeric:tabular-nums;line-height:1}}
 .caption{{font-size:11px;font-weight:600;color:var(--faint);text-transform:uppercase;letter-spacing:.06em}}
 .pill{{display:inline-flex;align-items:center;gap:6px;padding:6px 13px;border-radius:999px;
   font-size:12px;font-weight:700;letter-spacing:.01em}}
 .pill.risk{{background:var(--risk-soft);color:var(--risk-ink);border:1px solid #fecaca}}
 .pill.ok{{background:var(--ok-soft);color:var(--ok-ink);border:1px solid #bfdbfe}}
 .dot{{width:7px;height:7px;border-radius:50%}} .pill.risk .dot{{background:var(--risk)}} .pill.ok .dot{{background:var(--ok)}}
 .subhead{{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin:20px 0 9px}}
 .reason{{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:11px 13px;
   border-radius:11px;margin-top:8px;font-size:13.5px;background:#f8fafc;border:1px solid var(--line)}}
 .reason.up{{background:var(--risk-soft);border-color:#fde0e0}} .reason.down{{background:var(--ok-soft);border-color:#dbeafe}}
 .reason b{{font-weight:600}} .reason small{{color:var(--faint)}}
 .tag-dir{{font-size:11.5px;font-weight:700;white-space:nowrap}}
 .up .tag-dir{{color:var(--risk-ink)}} .down .tag-dir{{color:var(--ok-ink)}}
 .disc{{font-size:11.5px;color:var(--faint);margin-top:18px;border-top:1px solid var(--line);padding-top:12px;line-height:1.55}}
 .empty{{display:grid;place-items:center;text-align:center;padding:30px 14px;color:var(--faint)}}
 .empty svg{{width:40px;height:40px;opacity:.5;margin-bottom:8px}}
 /* batch */
 .dropzone{{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:16px;padding:16px;
   border:1.5px dashed #cbd5e1;border-radius:13px;background:#f8fafc;transition:border-color .15s,background .15s}}
 .dropzone:hover{{border-color:var(--brand2);background:#f0fdf8}}
 input[type=file]{{font-size:13px;color:var(--muted);background:transparent;border:0;padding:0;flex:1 1 200px}}
 input[type=file]::file-selector-button{{font-family:inherit;font-weight:600;font-size:13px;cursor:pointer;
   margin-right:12px;padding:8px 15px;border:0;border-radius:9px;color:var(--brand-ink);
   background:#d1fae5;transition:background .15s}}
 input[type=file]::file-selector-button:hover{{background:#a7f3d0}}
 #batchsummary{{margin-top:14px}}
 .summary-card{{display:flex;gap:18px;flex-wrap:wrap;margin-top:14px}}
 .stat{{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:12px 16px;min-width:120px}}
 .stat .v{{font-size:24px;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums}}
 .stat.flag .v{{color:var(--risk)}}
 .stat .l{{font-size:11px;color:var(--muted);margin-top:2px}}
 .tablewrap{{overflow-x:auto;margin-top:16px;border:1px solid var(--line);border-radius:12px}}
 table.bt{{border-collapse:collapse;font-size:12.5px;width:100%}}
 table.bt th,table.bt td{{padding:9px 13px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line)}}
 table.bt th{{background:#f8fafc;color:var(--muted);font-weight:600;text-transform:uppercase;font-size:10.5px;letter-spacing:.04em}}
 table.bt tbody tr:last-child td{{border-bottom:0}}
 table.bt tbody tr:hover{{background:#fafcff}}
 table.bt tr.flag{{background:var(--risk-soft)}} table.bt tr.flag:hover{{background:#fde8e8}}
 table.bt tr.flag td:first-child{{font-weight:700;color:var(--risk-ink)}}
 footer{{text-align:center;color:var(--faint);font-size:11.5px;padding:30px 0 36px}}
</style></head><body>
<header><div class="wrap">
  <div class="brand">
    <div class="logo"><svg viewBox="0 0 24 24" fill="none"><path d="M12 21c5-2 8-6 8-12V4l-8 2-8-2v5c0 6 3 10 8 12z" fill="#fff" opacity=".95"/><path d="M12 17V8M9 11l3-3 3 3" stroke="#059669" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
    <div><h1>EMERALD-AI</h1><div class="tag">green-loan credit decision support</div></div>
  </div>
  <p>The model <b>ranks a batch of applications</b> and routes the riskiest decile to human review —
    it does not approve or decline. Single-application scoring is available below for explanation
    and what-if analysis.</p>
</div></header>

<div class="wrap">
<section class="card">
  <div class="sechead"><span class="num">1</span><h2>Batch review queue</h2></div>
  <p class="lead">The operational use case. Upload the day's applications as a CSV — the model ranks
    them by risk and flags the riskiest <b>10%</b> as the review queue (reviewing the top decile
    historically catches ~{catch_pct}% of all defaults). Any subset of the form columns works; an
    optional <b>id</b>/<b>case</b> column is passed through. Try the bundled
    <code>data/sample_applicants.csv</code> or <code>data/example_cases.csv</code>.</p>
  <div class="dropzone">
    <input type="file" id="file" accept=".csv">
    <button id="batchbtn" type="button" class="btn">Rank applications</button>
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
      <div class="caption">Estimated probability of default</div>
      <div class="scorewrap"><div class="score" id="score">—</div><div id="band"></div></div>
      <div class="subhead">Top reasons</div>
      <div id="reasons"></div>
      <p class="disc">Reference threshold: historical riskiest decile = P(default) &ge; {threshold}
        (out-of-fold). For one applicant this is an absolute reference; the real review queue is the
        within-batch top decile above. A score is a prioritisation signal, not an adverse-action decision.</p>
    </div>
  </div>
</main>
<footer>EMERALD-AI · proof-of-concept decision support · the model ranks for review, it does not decide</footer>
</div>

<script>
const $=id=>document.getElementById(id);
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
  const box=$('reasons'); box.innerHTML='';
  for(const x of j.reasons){{
    const up=x.contribution>0;
    box.innerHTML+='<div class="reason '+(up?'up':'down')+'"><span><b>'+x.label+
      '</b> <small>= '+x.value+'</small></span><span class="tag-dir">'+(up?'▲ ':'▼ ')+x.verdict+'</span></div>';
  }}
  $('out').scrollIntoView({{behavior:'smooth',block:'nearest'}});
}});

$('batchbtn').addEventListener('click',async()=>{{
  const fi=$('file'); if(!fi.files.length){{alert('Choose a CSV first');return;}}
  const csv=await fi.files[0].text();
  const r=await fetch('/api/score-batch',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{csv}})}});
  const j=await r.json();
  $('batchsummary').innerHTML=
    '<div class="summary-card">'+
    '<div class="stat"><div class="v">'+j.n+'</div><div class="l">applications ranked</div></div>'+
    '<div class="stat flag"><div class="v">'+j.n_review_queue+'</div><div class="l">review queue (top decile)</div></div>'+
    '<div class="stat"><div class="v">'+j.n_riskiest_decile+'</div><div class="l">clear historical threshold</div></div>'+
    '</div>';
  const keys=Object.keys(j.rows[0]||{{}});
  let h='<div class="tablewrap"><table class="bt"><thead><tr>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr></thead><tbody>';
  for(const row of j.rows){{
    h+='<tr class="'+(row.review_queue?'flag':'')+'">'+keys.map(k=>'<td>'+row[k]+'</td>').join('')+'</tr>';
  }}
  $('batchtable').innerHTML=h+'</tbody></table></div>';
}});
</script>
</body></html>"""


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Boot the demo server. Trains the model up front so the first request is instant."""
    import uvicorn

    print("[emerald_ai] training frozen model + setting operating point ...")
    s = get_scorer()
    print(f"[emerald_ai] ready. riskiest-decile threshold P>={s.threshold:.3f} "
          f"(OOF catch-rate {100*s.catch_rate:.0f}% of {s.n_events} defaults).")
    print(f"[emerald_ai] open http://{host}:{port}/  (Ctrl+C to stop)")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
