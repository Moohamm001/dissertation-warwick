"""Can we do better than the headline LR? — the defensible small-N experiment (RQ1 follow-up).

Premise (established in `reports/learning_evidence.md`): the ceiling here is the *event count*
(50), not the model class. So rather than chase a bigger model — which at ~3-4 events per variable
would just overfit fold noise — this module runs the two experiments that genuinely bear on
"better", each ATTACKABLE in the Popperian sense (the result can come out null):

  1. **Respect the events-per-variable budget.** Compare the full-feature L2 logistic baseline to
     an L1-sparse model and an L1-sparse model *plus* three domain affordability ratios. Question:
     does enforcing sparsity / adding meaningful ratios *tighten the fold band*, or not?
  2. **Events-needed projection.** Subsample the positives, measure how the fold-band width shrinks
     with the event count, and project how many events would be needed to halve today's uncertainty.

All preprocessing is fit inside the CV fold. Affordability ratios are derived purely from permitted
pre-funding numerics, so they are leakage-safe by construction. Run: ``python -m emerald_ai improve``.

METHOD-CITATION STATUS (see the report's audit table; Rule 1 = no method without a paper) — all
closed 2026-06-29:
  * Events-per-variable / sample-size projection — Peduzzi 1996, Vittinghoff 2006, Riley 2020 (D7).
  * L1 / elastic-net penalised regression — Tibshirani 1996, Zou & Hastie 2005 (D10).
  * Affordability-ratio feature engineering — Altman 1968 (D11).
  * Feature selection under class imbalance — Wasikowski & Chen 2009 (D12).
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import data as D
from . import feature_audit as FA
from . import metrics as M

# Permitted pre-funding numerics — the leakage-safe inputs this experiment may use.
BASE_NUMERICS = [
    "Credit Score", "Amount Sought", "Revenue", "Average Monthly Sales", "Time In Business",
    "Days Since Last Opportunity", "Online App Completed", "Is Borrower Renewal",
    "Current Tier", "Mktg Tier",
]
RATIOS = ["loan_to_revenue", "loan_to_sales", "revenue_to_sales"]


def affordability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add three pre-funding affordability ratios — what a credit officer actually reads.

    Leakage-safe: every input is a permitted application-time numeric. Zero denominators are
    guarded to NaN (then median-imputed inside the fold), never silently turned into inf.
    """
    out = df.copy()
    rev = out["Revenue"].replace(0, np.nan)
    sales = out["Average Monthly Sales"].replace(0, np.nan)
    out["loan_to_revenue"] = out["Amount Sought"] / rev          # loan size vs monthly revenue
    out["loan_to_sales"] = out["Amount Sought"] / sales          # loan size vs monthly sales
    out["revenue_to_sales"] = out["Revenue"] / sales             # margin / model-of-business proxy
    return out.replace([np.inf, -np.inf], np.nan)


def _make_lr(penalty: str):
    """A class-weighted logistic model. C fixed at 1.0 — no inner-loop tuning on ~10 positives/fold."""
    kw = dict(max_iter=5000, class_weight="balanced", C=1.0)
    if penalty == "l2":
        return LogisticRegression(penalty="l2", solver="lbfgs", **kw)
    if penalty == "l1":
        return LogisticRegression(penalty="l1", solver="saga", **kw)
    if penalty == "elasticnet":
        return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, **kw)
    raise ValueError(penalty)


def _pipe(penalty: str):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("lr", _make_lr(penalty))])


# (label, penalty, feature list) — the configurations under test.
CONFIGS = [
    ("LR L2, base numerics (baseline)", "l2", BASE_NUMERICS),
    ("LR L1-sparse, base numerics", "l1", BASE_NUMERICS),
    ("LR L1-sparse + affordability ratios", "l1", BASE_NUMERICS + RATIOS),
    ("LR elastic-net + affordability ratios", "elasticnet", BASE_NUMERICS + RATIOS),
]


def _oof_and_band(X: pd.DataFrame, y: np.ndarray, penalty: str,
                  n_splits=5, n_repeats=5):
    """Repeated stratified CV. Return per-fold PR-AUC array and mean #non-zero coefficients."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=C.SEED)
    pr, nnz = [], []
    for tr, te in rskf.split(X, y):
        pipe = _pipe(penalty)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[te])[:, 1]
        pr.append(M.pr_auc(y[te], p))
        nnz.append(int(np.sum(np.abs(pipe.named_steps["lr"].coef_.ravel()) > 1e-8)))
    return np.array(pr, float), float(np.mean(nnz))


def sparse_bakeoff(df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Run every configuration; report median PR-AUC, the fold band, its width, and #features kept."""
    dfx = affordability_features(df)
    rows = []
    for label, penalty, feats in CONFIGS:
        pr, nnz = _oof_and_band(dfx[feats], y, penalty)
        b = M.fold_band(pr)
        rows.append({
            "configuration": label, "n_features_in": len(feats),
            "median_pr_auc": round(b["median"], 4),
            "fold_band": f"[{b['lo']:.3f}, {b['hi']:.3f}]",
            "band_width": round(b["hi"] - b["lo"], 4),
            "feats_kept": round(nnz, 1),
        })
    return pd.DataFrame(rows)


def events_projection(df: pd.DataFrame, y: np.ndarray, fracs=(0.5, 0.75, 1.0), n_draws=3):
    """Measure how the fold-band width falls with the event count, AT FIXED PREVALENCE, then project.

    BOTH classes are subsampled by the same fraction, so prevalence — and therefore the PR-AUC floor —
    is held constant and the band widths are comparable across sample sizes; only the event count (the
    binding quantity, Peduzzi/Riley) varies. To damp subsample noise at this N, each fraction is
    averaged over ``n_draws`` independent draws. A ``width ∝ 1/√(events)`` law (the standard-error
    scaling of a proportion-like estimate) is fitted and used to project the events needed to halve
    today's band.
    """
    dfx = affordability_features(df).reset_index(drop=True)
    feats = BASE_NUMERICS + RATIOS
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pts = []
    for fe in fracs:
        kp = max(10, int(round(len(pos_idx) * fe)))
        kn = int(round(len(neg_idx) * fe))
        widths, meds, nev = [], [], []
        for d in range(n_draws):
            rng = np.random.default_rng(C.SEED + d)
            sub = np.sort(np.concatenate([rng.choice(pos_idx, kp, replace=False),
                                          rng.choice(neg_idx, kn, replace=False)]))
            ys = y[sub]
            pr, _ = _oof_and_band(dfx.iloc[sub][feats], ys, "l1", n_repeats=3)
            b = M.fold_band(pr)
            widths.append(b["hi"] - b["lo"]); meds.append(b["median"]); nev.append(int(ys.sum()))
        pts.append({"n_events": int(np.mean(nev)), "median_pr_auc": round(float(np.mean(meds)), 4),
                    "band_width": round(float(np.mean(widths)), 4)})
    proj = pd.DataFrame(pts)
    # fit width = a / sqrt(n_events)  ->  a = width * sqrt(n)
    a = float(np.mean(proj["band_width"] * np.sqrt(proj["n_events"])))
    cur_n = int(proj["n_events"].max())
    cur_w = float(proj.loc[proj.n_events == cur_n, "band_width"].iloc[0])
    n_half = int(round((a / (cur_w / 2)) ** 2)) if cur_w > 0 else None
    # quality gate: does the band actually shrink as events grow? if not, the 1/sqrt projection
    # is not supported by the data at this N and must be reported as such, not asserted.
    corr = float(np.corrcoef(proj["n_events"], proj["band_width"])[0, 1]) if len(proj) > 1 else np.nan
    reliable = corr < 0
    return proj, {"a": a, "cur_n": cur_n, "cur_w": cur_w, "n_to_halve": n_half,
                  "corr": corr, "reliable": reliable}


def _fig_projection(proj: pd.DataFrame, fit: dict) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(proj["n_events"], proj["band_width"], "o-", color="#1565c0", label="observed band width")
    xs = np.linspace(proj["n_events"].min(), max(fit["n_to_halve"] or 0, proj["n_events"].max()), 100)
    ax.plot(xs, fit["a"] / np.sqrt(xs), "--", color="#9e9e9e", label="a / sqrt(events) fit")
    if fit["n_to_halve"]:
        ax.axhline(fit["cur_w"] / 2, color="#c62828", lw=1, ls=":")
        ax.axvline(fit["n_to_halve"], color="#c62828", lw=1, ls=":",
                   label=f"~{fit['n_to_halve']} events to halve band")
    ax.set_xlabel("number of delinquency events"); ax.set_ylabel("PR-AUC fold-band width")
    ax.set_title("Uncertainty is event-limited: projecting the data needed to do better")
    ax.legend(); fig.tight_layout()
    p = C.FIGURES_DIR / "improve_projection.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p.name


def build_report() -> str:
    from .eda import _md_table

    C.ensure_dirs()
    df = D.build_target(D.load_raw(), "paidoff_only").reset_index(drop=True)
    y = df["y"].to_numpy()

    bake = sparse_bakeoff(df, y)
    proj, fit = events_projection(df, y)
    f_proj = _fig_projection(proj, fit)

    base_w = bake.loc[bake.configuration.str.startswith("LR L2"), "band_width"].iloc[0]
    best = bake.loc[bake["band_width"].idxmin()]
    best_med = bake.loc[bake["median_pr_auc"].idxmax()]
    tightened = best["band_width"] < base_w - 1e-9
    moved_median = best_med["median_pr_auc"] > bake.loc[bake.configuration.str.startswith("LR L2"),
                                                         "median_pr_auc"].iloc[0] + 0.01

    md = f"""# Can we do better? — sparsity, affordability ratios, and the events ceiling

*Generated by `python -m emerald_ai improve`, seed {C.SEED}, label paidoff_only
({int(y.sum())} events / {len(y)} rows). Repeated stratified CV (5x5). All fitting inside the fold.
C fixed at 1.0 — no inner-loop tuning on ~10 positives/fold.*

## Experiment 1 — respect the events-per-variable budget
With {int(y.sum())} events and {len(BASE_NUMERICS) + len(RATIOS)} candidate numerics, events-per-variable
is ~{y.sum() / (len(BASE_NUMERICS) + len(RATIOS)):.1f} — far below the rule-of-ten (Peduzzi 1996),
so L1 sparsity is the principled lever, not a bigger model.

{_md_table(bake)}

- **band_width** = 97.5th − 2.5th fold percentile (smaller = less uncertain).
- **feats_kept** = mean non-zero coefficients (L1 selects; lower = sparser).

**Verdict:** sparsity/ratios **{'tighten' if tightened else 'do NOT tighten'}** the band
(best width {best['band_width']:.3f} vs L2 baseline {base_w:.3f}) and the median PR-AUC
**{'moves materially' if moved_median else 'does not move materially'}** (best median
{best_med['median_pr_auc']:.3f}). {'A genuine, if modest, gain — consistent with adding signal without spending EPV.' if (tightened or moved_median) else 'This is the expected null at 50 events: the fold band is dominated by sampling variance, which no model choice on the same data can remove. Reported as a null, not buried.'}

## Experiment 2 — how many events to actually do better?
Subsampling **both classes by the same fraction** (prevalence — and the PR-AUC floor — held fixed,
so the widths are comparable), averaged over 3 draws per point, the fold-band width falls with the
event count. Fitting the standard `width ∝ 1/√(events)` law:

{_md_table(proj)}

- Current: **{fit['cur_n']} events → band width {fit['cur_w']:.3f}** (prevalence held fixed across rows).
- Correlation(events, band width) = **{fit['corr']:.2f}**.
{('- **Projection: ~%d events (≈%.1fx today) would be needed to halve the uncertainty band.** '
  'The lever is *data*, not model complexity (Riley 2020).' % (fit['n_to_halve'], fit['n_to_halve'] / fit['cur_n']))
 if fit['reliable'] else
 '- **The band does NOT shrink monotonically with events at this N** (correlation is non-negative), '
 'so a 1/sqrt(events) projection is *not* supported by these three points — reported as such rather '
 'than fabricated. The honest reading is weaker but still directional: even halving the events barely '
 'moves the band, i.e. we are deep in the small-sample regime where estimates are unstable (Riley 2020). '
 'A trustworthy required-sample-size figure needs the analytic Riley/`pmsampsize` calculation, flagged '
 'as next work.'}

![projection]({C.FIGURES_DIR.name}/{f_proj})

## So what (Hamming)
The honest answer to "can we predict better": **not by changing the model on these 50 events** —
the band is event-limited, and Experiment 1 {'shows only a modest within-data gain' if (tightened or moved_median) else 'returns the expected null'}. {('The defensible route to a materially better model is **~%d events** — a longer observation window or a pooled portfolio — quantified above.' % fit['n_to_halve']) if fit['reliable'] else 'The route to a better model is **more events** (longer window / pooled portfolio); the exact figure needs the analytic sample-size calculation (next work), since the empirical scaling is itself too noisy at N=50 to pin down — which is the deeper point.'}

## Method → citation audit (Rule 1)
| Method (where) | Status | Paper(s) |
|---|---|---|
| Events-per-variable / sample-size projection (`events_projection`) | **COVERED** | Peduzzi 1996 `W2037668591`, Vittinghoff 2006 `W2130373985`, Riley 2020 `W3012413426` — all **[CURATED]** (D7) |
| L1 / elastic-net penalised logistic regression (`_make_lr`) | **COVERED** | Tibshirani 1996 `W2135046866`, Zou & Hastie 2005 `W2122825543` — **[CURATED]** (D10) |
| Affordability-ratio feature engineering (`affordability_features`) | **COVERED** | Altman 1968 `W2124532504` — **[CURATED]** (D11) |
| Feature selection under class imbalance | **COVERED** | Wasikowski & Chen 2009 `W2138776277` — **[CURATED]** (D12) |

All methods are now citation-backed (D10–D12 curated 2026-06-29); the experiment is no longer
provisional.

---
*Reproduce: `python -m emerald_ai improve`*
"""
    out = C.REPORTS_DIR / "improvement.md"
    out.write_text(md, encoding="utf-8")
    return str(out)
