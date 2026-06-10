"""Phase 3 — the empirical solution-finder (model x imbalance-strategy bake-off).

Answers RQ1: can a gradient-boosted model beat a regularised-LR baseline on early-delinquency
detection *beyond the fold band* at this prevalence — or is "no significant winner" the finding?

Discipline (all enforced here):
  * Repeated stratified CV (5 splits x 5 repeats = 25 folds) so each test fold holds ~10 events.
  * Preprocessing + ANY resampling fit strictly inside the training fold (no leakage, no SMOTE
    seeing held-out rows).
  * Metrics reported as median + 2.5-97.5 fold band, never a bare number.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

from . import config as C
from . import data as D
from . import metrics as M
from . import preprocess as P

# (model, strategy) combinations. logreg+class_weight is the RQ1 baseline.
COMBOS = [
    ("dummy", "none"),            # prevalence floor for PR-AUC context
    ("logreg", "class_weight"),   # <- baseline
    ("logreg", "smote"),
    ("xgboost", "class_weight"),  # scale_pos_weight
    ("xgboost", "smote"),
]


def _make_model(name: str, strategy: str, y_train: np.ndarray):
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        cw = "balanced" if strategy == "class_weight" else None
        return LogisticRegression(max_iter=2000, C=1.0, class_weight=cw)
    if name == "xgboost":
        pos = max(1, int(y_train.sum()))
        neg = int(len(y_train) - pos)
        spw = (neg / pos) if strategy == "class_weight" else 1.0
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
            scale_pos_weight=spw, tree_method="hist",
            random_state=C.SEED, n_jobs=-1,
        )
    raise ValueError(name)


def _fold_scores(name, strategy, X_tr, y_tr, X_te) -> np.ndarray:
    """Fit one combo on a training fold (resampling inside), return P(default) on the test fold."""
    if strategy == "smote":
        k = min(5, int(y_tr.sum()) - 1)
        if k >= 1:
            X_tr, y_tr = SMOTE(random_state=C.SEED, k_neighbors=k).fit_resample(X_tr, y_tr)
    model = _make_model(name, strategy, y_tr)
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def run_bakeoff(scheme: str = "paidoff_only", n_splits: int = 5, n_repeats: int = 5) -> pd.DataFrame:
    """Run every combo through repeated stratified CV; return per-fold metric rows."""
    df = D.build_target(D.load_raw(), scheme).reset_index(drop=True)
    y = df["y"].to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=C.SEED)

    rows = []
    for name, strategy in COMBOS:
        pre, _ = P.build_preprocessor(df, scale=(name == "logreg"))
        for fold, (tr, te) in enumerate(rskf.split(df, y)):
            df_tr, df_te = df.iloc[tr], df.iloc[te]
            y_tr, y_te = y[tr], y[te]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_tr = pre.fit_transform(df_tr, y_tr)
                X_te = pre.transform(df_te)
                p = _fold_scores(name, strategy, X_tr, y_tr, X_te)
            rows.append({
                "combo": f"{name}+{strategy}",
                "fold": fold,
                "pr_auc": M.pr_auc(y_te, p),
                "recall_top_decile": M.recall_at_top_decile(y_te, p),
                "within_min_ece": M.within_minority_ece(y_te, p),
            })
    return pd.DataFrame(rows)


def summarise(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-fold rows to median + fold band per combo per metric."""
    out = []
    for combo, g in per_fold.groupby("combo", sort=False):
        row = {"combo": combo}
        for metric in ("pr_auc", "recall_top_decile", "within_min_ece"):
            b = M.fold_band(g[metric])
            row[metric] = f"{b['median']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"
        out.append(row)
    return pd.DataFrame(out)


def build_report(scheme: str = "paidoff_only") -> str:
    """Run the bake-off and write reports/model_bakeoff.md with the honest RQ1 verdict."""
    from .eda import _md_table  # reuse the dependency-free md table

    C.ensure_dirs()
    per_fold = run_bakeoff(scheme)
    summary = summarise(per_fold)

    # RQ1 verdict: does XGBoost's PR-AUC band clear the LR baseline's median?
    piv = per_fold.pivot_table(index="fold", columns="combo", values="pr_auc")
    base = piv["logreg+class_weight"].median()
    best_xgb = max(
        ("xgboost+class_weight", "xgboost+smote"),
        key=lambda c: piv[c].median(),
    )
    xgb_lo = float(np.percentile(piv[best_xgb].dropna(), 2.5))
    separated = xgb_lo > base
    verdict = (
        f"Best GBM (`{best_xgb}`, median PR-AUC {piv[best_xgb].median():.3f}) "
        + ("**clears** " if separated else "**does NOT clear** ")
        + f"the LR baseline median ({base:.3f}) at its 2.5th fold percentile "
        f"({xgb_lo:.3f}). "
        + ("A defensible winner." if separated else
           "**The fold bands overlap — no significant winner. This is the honest RQ1 finding, "
           "not a failure: at ~10 events per test fold the data cannot separate the models.**")
    )
    prevalence = 100 * D.build_target(D.load_raw(), scheme)["y"].mean()

    md = f"""# Phase 3 — Model x Imbalance Bake-off (RQ1)

*Generated by `python -m emerald_ai bakeoff`, seed = {C.SEED}. Label scheme: **{scheme}**
(prevalence {prevalence:.2f}%). 5x5 repeated stratified CV = 25 folds. All preprocessing and
resampling fit inside the training fold. Metrics: median [2.5–97.5 fold percentile].*

## Results
{_md_table(summary)}

- **pr_auc** — area under precision-recall (primary). Prevalence floor ≈ {prevalence/100:.4f}.
- **recall_top_decile** — share of all defaults captured in the riskiest 10%.
- **within_min_ece** — calibration gap on actual defaults (lower = better; Phase 4 will address).

## RQ1 verdict
{verdict}

## Reading note (per the roadmap framing)
The *result* of this dissertation is not a leaderboard win; it is an honest, CI-aware answer
about what {int(round(prevalence/100*len(D.build_target(D.load_raw(), scheme))))} events can
support. Overlapping bands are reported as overlapping, not spun.

---
*Reproduce: `python -m emerald_ai bakeoff`*
"""
    out = C.REPORTS_DIR / "model_bakeoff.md"
    out.write_text(md, encoding="utf-8")
    return str(out)
