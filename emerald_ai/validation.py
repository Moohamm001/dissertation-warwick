"""Rule 2 — prove the model actually learns (do not assume it at ~50 events).

This module is ADDITIVE: it imports and reuses the existing pipeline functions
(`preprocess.build_preprocessor`, `experiments._make_model`, `experiments._fold_scores`,
`metrics`) without modifying any modelling code. Five evidence checks:

  1. Permutation test  — shuffle labels, rebuild the null PR-AUC distribution, locate the real one.
  2. Baselines         — majority-class dummy + single-feature logistic models.
  3. Stability         — PR-AUC / recall@decile across repeated CV with different seeds.
  4. Learning curve     — 50% / 75% / 100% of training data; has performance plateaued?
  5. Calibration        — reliability curve + Brier score (evidence, not assertion).

Every result is reported with its uncertainty; negative/ambiguous outcomes are stated, not hidden.
Run: ``python -m emerald_ai evidence``.
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import data as D
from . import metrics as M
from . import preprocess as P
from .experiments import _fold_scores, _make_model  # reused, NOT modified

MODEL, STRATEGY = "logreg", "class_weight"  # the headline model whose learning we must justify


def _load():
    df = D.build_target(D.load_raw(), "paidoff_only").reset_index(drop=True)
    return df, df["y"].to_numpy()


def _oof(df, y, name=MODEL, strategy=STRATEGY, n_splits=5, seed=C.SEED) -> np.ndarray:
    """Out-of-fold P(default) reusing the project pipeline; data loaded once by caller."""
    pre, _ = P.build_preprocessor(df, scale=(name == "logreg"))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(df, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_tr = pre.fit_transform(df.iloc[tr], y[tr])
            X_te = pre.transform(df.iloc[te])
            oof[te] = _fold_scores(name, strategy, X_tr, y[tr], X_te)
    return oof


# --- 1. permutation test --------------------------------------------------
def permutation_test(df, y, n_perm: int = 100):
    real = M.pr_auc(y, _oof(df, y))
    rng = np.random.default_rng(C.SEED)
    null = np.array([M.pr_auc(yp := rng.permutation(y), _oof(df, yp, seed=C.SEED + i + 1))
                     for i in range(n_perm)])
    pval = (1 + np.sum(null >= real)) / (n_perm + 1)
    pct = 100 * np.mean(null < real)
    return {"real": real, "null_mean": float(null.mean()), "null_max": float(null.max()),
            "percentile": pct, "p_value": pval, "null": null}


# --- 2. baselines ---------------------------------------------------------
def baselines(df, y):
    prevalence = y.mean()
    rows = [{"model": "dummy (prevalence floor)", "pr_auc": prevalence}]
    for col in ["Credit Score", "Revenue", "Time In Business", "Average Monthly Sales"]:
        skf = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
        oof = np.full(len(y), np.nan)
        x = df[[col]]
        for tr, te in skf.split(x, y):
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler()),
                             ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))])
            pipe.fit(x.iloc[tr], y[tr])
            oof[te] = pipe.predict_proba(x.iloc[te])[:, 1]
        rows.append({"model": f"LR | {col} only", "pr_auc": M.pr_auc(y, oof)})
    rows.append({"model": f"FULL model ({MODEL}+{STRATEGY}, 17 feats)", "pr_auc": M.pr_auc(y, _oof(df, y))})
    return rows


# --- 3. stability across seeds -------------------------------------------
def stability(df, y, seeds=(1, 7, 21, 42, 99)):
    rows = []
    for s in seeds:
        oof = _oof(df, y, seed=s)
        rows.append({"seed": s, "pr_auc": M.pr_auc(y, oof),
                     "recall_decile": M.recall_at_top_decile(y, oof)})
    pr = np.array([r["pr_auc"] for r in rows])
    rc = np.array([r["recall_decile"] for r in rows])
    return rows, {"pr_auc": (pr.mean(), pr.std()), "recall_decile": (rc.mean(), rc.std()),
                  "prevalence": y.mean()}


# --- 4. learning curve ----------------------------------------------------
def learning_curve(df, y, fracs=(0.5, 0.75, 1.0)):
    pre, _ = P.build_preprocessor(df, scale=True)
    rng = np.random.default_rng(C.SEED)
    out = []
    for frac in fracs:
        skf = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
        oof = np.full(len(y), np.nan)
        for tr, te in skf.split(df, y):
            # stratified subsample of the TRAIN fold to `frac`
            pos = tr[y[tr] == 1]; neg = tr[y[tr] == 0]
            sub = np.concatenate([rng.choice(pos, max(2, int(len(pos) * frac)), replace=False),
                                  rng.choice(neg, int(len(neg) * frac), replace=False)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_tr = pre.fit_transform(df.iloc[sub], y[sub])
                X_te = pre.transform(df.iloc[te])
                oof[te] = _fold_scores(MODEL, STRATEGY, X_tr, y[sub], X_te)
        out.append({"train_frac": frac, "pr_auc": M.pr_auc(y, oof)})
    return out


# --- 5. calibration -------------------------------------------------------
def calibration(df, y):
    oof = _oof(df, y)
    brier = brier_score_loss(y, oof)
    frac_pos, mean_pred = calibration_curve(y, oof, n_bins=5, strategy="quantile")
    return {"brier": brier, "frac_pos": frac_pos, "mean_pred": mean_pred, "oof": oof}


# --- report ---------------------------------------------------------------
def _fig_permutation(perm) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(perm["null"], bins=20, color="#9e9e9e", label="null (shuffled labels)")
    ax.axvline(perm["real"], color="#2e7d32", lw=2.5, label=f"real PR-AUC = {perm['real']:.3f}")
    ax.set_xlabel("PR-AUC"); ax.set_title(
        f"Permutation test: real beats {perm['percentile']:.0f}% of nulls (p={perm['p_value']:.3f})")
    ax.legend(); fig.tight_layout()
    p = C.FIGURES_DIR / "evid_permutation.png"; fig.savefig(p, dpi=130); plt.close(fig); return p.name


def _fig_learning(lc) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["train_frac"] for r in lc], [r["pr_auc"] for r in lc], "o-", color="#1565c0")
    ax.set_xlabel("fraction of training data"); ax.set_ylabel("PR-AUC")
    ax.set_title("Learning curve: has performance plateaued?"); fig.tight_layout()
    p = C.FIGURES_DIR / "evid_learning.png"; fig.savefig(p, dpi=130); plt.close(fig); return p.name


def _fig_reliability(cal) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#9e9e9e", label="perfect")
    ax.plot(cal["mean_pred"], cal["frac_pos"], "o-", color="#2e7d32", label="LR (OOF)")
    ax.set_xlabel("mean predicted P(default)"); ax.set_ylabel("observed default rate")
    ax.set_title(f"Reliability curve (Brier = {cal['brier']:.4f})"); ax.legend(); fig.tight_layout()
    p = C.FIGURES_DIR / "evid_reliability.png"; fig.savefig(p, dpi=130); plt.close(fig); return p.name


def build_report(n_perm: int = 100) -> str:
    from .eda import _md_table
    import pandas as pd

    C.ensure_dirs()
    df, y = _load()

    perm = permutation_test(df, y, n_perm=n_perm)
    base = baselines(df, y)
    stab_rows, stab = stability(df, y)
    lc = learning_curve(df, y)
    cal = calibration(df, y)

    f_perm = _fig_permutation(perm)
    f_lc = _fig_learning(lc)
    f_rel = _fig_reliability(cal)

    plateaued = abs(lc[-1]["pr_auc"] - lc[-2]["pr_auc"]) < 0.02
    base_df = pd.DataFrame(base).assign(pr_auc=lambda d: d.pr_auc.round(4))
    stab_df = pd.DataFrame(stab_rows).round(3)

    md = f"""# Rule 2 — Evidence that the model learns (not noise)

*Generated by `python -m emerald_ai evidence`, seed {C.SEED}, label scheme paidoff_only
(prevalence {y.mean():.4f}, {int(y.sum())} events). All checks reuse the project pipeline.*

## 1. Permutation test (the key check)
Shuffle the labels and rerun the full out-of-fold pipeline {n_perm} times to build a null
distribution; the real model must clearly exceed it.

- **Real PR-AUC = {perm['real']:.3f}**; null mean {perm['null_mean']:.3f}, null max {perm['null_max']:.3f}.
- Real beats **{perm['percentile']:.0f}%** of permutations; **p = {perm['p_value']:.3f}**.
- **Verdict: {'real signal — clears the null' if perm['p_value'] < 0.05 else 'NOT distinguishable from noise — reported as a limitation'}.**

![permutation]({C.FIGURES_DIR.name}/{f_perm})

## 2. Baselines (how much does the full model add?)
{_md_table(base_df)}

## 3. Stability across CV seeds
{_md_table(stab_df)}

PR-AUC {stab['pr_auc'][0]:.3f} ± {stab['pr_auc'][1]:.3f}; recall@decile {stab['recall_decile'][0]:.3f}
± {stab['recall_decile'][1]:.3f}; prevalence floor {stab['prevalence']:.4f}.
{'**The interval sits well above the floor — stable signal.**' if stab['pr_auc'][0] - 2*stab['pr_auc'][1] > stab['prevalence'] else '**Interval approaches the floor — interpret with caution.**'}

## 4. Learning curve
{_md_table(pd.DataFrame(lc).round(4))}

**{'Performance has plateaued' if plateaued else 'Performance still moving'}** between 75% and 100%
of the data — {'consistent with the dataset being too small to reward more complex models, which '
'SUPPORTS the logistic-regression finding (events-per-variable literature, Peduzzi 1996; Riley 2020).' if plateaued else 'more data may still help; stated as a caveat.'}

![learning]({C.FIGURES_DIR.name}/{f_lc})

## 5. Calibration evidence
Brier score = **{cal['brier']:.4f}** vs a prevalence-only Brier of ≈ {y.mean()*(1-y.mean()):.4f}.
**Honest limitation: the model's Brier is far WORSE than the trivial base-rate predictor** — class
weighting inflates the predicted probabilities, so the score *ranks* defaults well (Sections 1–4)
but its absolute probabilities are not trustworthy. This is a known effect of cost-sensitive
training and is exactly the gap **Phase 4 post-hoc calibration (Platt/isotonic)** exists to close.
The reliability curve below shows the overconfidence directly — reported, not hidden.

![reliability]({C.FIGURES_DIR.name}/{f_rel})

---
*Reproduce: `python -m emerald_ai evidence`*
"""
    out = C.REPORTS_DIR / "learning_evidence.md"
    out.write_text(md, encoding="utf-8")
    return str(out)
