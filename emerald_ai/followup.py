"""Review-round-1 follow-up checks (dissertation §6.3 item 4) — three analyses the pipeline
can run without new data, each answering a specific reviewer challenge:

  1. **Paired fold-level RQ1 comparison** (reviewer C1): the bake-off models share identical CV
     folds, so the correct comparison object is the per-fold *difference* distribution, not
     overlap of marginal bands. Reports median difference, its fold band, win counts, and a sign
     test (with the honest caveat that repeated-CV folds are correlated, so the p-value is
     approximate — Nadeau-Bengio territory).
  2. **Reason-code stability across folds** (reviewer C6): if LR coefficients are unstable at
     EPV ~ 2-4, the served SHAP reason codes could be seed artefacts. Measures pairwise Spearman
     correlation of per-fold global importance and top-3 membership frequency, at feature level
     and at the affordability-cluster level.
  3. **Score-distribution diagnostic** (reviewer C8): the evaluation population (terminal-outcome
     loans) is not the deployment population (full book). Scores both with the frozen scorer and
     compares the distributions.

Run: ``python -m emerald_ai followup-checks`` -> reports/followup_checks.md
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import metrics as M
from . import preprocess as P
from .experiments import run_bakeoff, _fold_scores

AFFORDABILITY_CLUSTER = {"Revenue", "Average Monthly Sales"}


# ------------------------------------------------------------------ 1. paired comparison
def paired_comparison(per_fold: pd.DataFrame, baseline: str = "logreg+class_weight") -> pd.DataFrame:
    """Per-fold paired PR-AUC differences (baseline − challenger) on shared folds."""
    from scipy.stats import binomtest

    piv = per_fold.pivot_table(index="fold", columns="combo", values="pr_auc")
    rows = []
    for challenger in [c for c in piv.columns if c not in (baseline, "dummy+none")]:
        d = (piv[baseline] - piv[challenger]).dropna()
        wins = int((d > 0).sum())
        ties = int((d == 0).sum())
        n = int(d.size - ties)
        p = binomtest(wins, n, 0.5).pvalue if n else float("nan")
        rows.append({
            "challenger": challenger,
            "median_diff": round(float(d.median()), 4),
            "diff_band": f"[{np.percentile(d, 2.5):.3f}, {np.percentile(d, 97.5):.3f}]",
            "baseline_wins": f"{wins}/{d.size}",
            "sign_test_p": round(float(p), 3),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2. reason-code stability
def _fold_importances(scheme: str = "paidoff_only", n_splits: int = 5) -> pd.DataFrame:
    """Global mean|SHAP| per source feature, computed independently per CV fold (LR+CW)."""
    from sklearn.model_selection import StratifiedKFold
    from .serve import _map_to_source
    from . import feature_audit as FA

    df = D.build_target(D.load_raw(), scheme, clean=True).reset_index(drop=True)
    y = df["y"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(df, y)):
        pre, _ = P.build_preprocessor(df, scale=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_tr = np.asarray(pre.fit_transform(df.iloc[tr], y[tr]))
            X_te = np.asarray(pre.transform(df.iloc[te]))
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        model.fit(X_tr, y[tr])
        names = list(pre.get_feature_names_out())
        source = _map_to_source(names, FA.permitted_columns())
        contrib = np.abs((X_te - X_tr.mean(axis=0)) * model.coef_.ravel())   # exact linear SHAP, |.|
        imp = pd.DataFrame(contrib, columns=source).T.groupby(level=0).sum().T.mean(axis=0)
        rows.append(imp.rename(fold))
    return pd.DataFrame(rows)  # folds x source features


def stability_summary(imps: pd.DataFrame, top_k: int = 3) -> dict:
    """Pairwise Spearman of fold importance vectors + top-k membership frequencies."""
    from scipy.stats import spearmanr

    n = len(imps)
    cors = [spearmanr(imps.iloc[i], imps.iloc[j]).statistic
            for i in range(n) for j in range(i + 1, n)]
    topk = imps.apply(lambda r: set(r.nlargest(top_k).index), axis=1)
    freq = pd.Series([f for s in topk for f in s]).value_counts() / n
    cluster_first = float(np.mean([imps.loc[i].idxmax() in AFFORDABILITY_CLUSTER for i in imps.index]))
    return {
        "mean_spearman": float(np.mean(cors)),
        "min_spearman": float(np.min(cors)),
        "top3_freq": freq,
        "cluster_rank1_share": cluster_first,
    }


# ------------------------------------------------------------------ 3. score-distribution diagnostic
def score_distribution_check(threshold: float | None = None) -> dict:
    """Frozen-scorer score distribution: full raw book vs the terminal-outcome evaluation subset."""
    from scipy.stats import ks_2samp
    from . import serve as S

    scorer = S.get_scorer()
    thr = float(threshold if threshold is not None else scorer.threshold)
    full = S.score_frame(scorer, D.load_raw())["probability"].to_numpy()
    eval_df = D.build_target(D.load_raw(), "paidoff_only", clean=True)
    evalp = S.score_frame(scorer, eval_df)["probability"].to_numpy()
    ks = ks_2samp(full, evalp)
    q = lambda a: {f"q{int(100*x)}": round(float(np.quantile(a, x)), 4) for x in (0.1, 0.5, 0.9)}
    return {
        "n_full": int(full.size), "n_eval": int(evalp.size),
        "full_quantiles": q(full), "eval_quantiles": q(evalp),
        "share_flagged_full": round(float((full >= thr).mean()), 4),
        "share_flagged_eval": round(float((evalp >= thr).mean()), 4),
        "ks_stat": round(float(ks.statistic), 4), "ks_p": float(ks.pvalue),
        "threshold": thr,
    }


# ------------------------------------------------------------------ 4. `behind` leave-one-out
def behind_sensitivity(scheme: str = "paidoff_only") -> dict:
    """Drop the single `behind` event and re-run the headline model: does anything change?

    The event definition is `default` ∪ `behind` with exactly one `behind` case (examiner
    challenge: its operational meaning is undocumented, so show the results do not hinge on it).
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    df_all = D.build_target(D.load_raw(), scheme, clean=True).reset_index(drop=True)
    df_wo = df_all.loc[df_all[C.LABEL_COL] != "behind"].reset_index(drop=True)
    out = {}
    for label, df in (("with_behind", df_all), ("without_behind", df_wo)):
        y = df["y"].to_numpy()
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=C.SEED)
        pre, _ = P.build_preprocessor(df, scale=True)
        praucs, recs = [], []
        for tr, te in rskf.split(df, y):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_tr = pre.fit_transform(df.iloc[tr], y[tr])
                X_te = pre.transform(df.iloc[te])
                p = _fold_scores("logreg", "class_weight", X_tr, y[tr], X_te)
            praucs.append(M.pr_auc(y[te], p))
            recs.append(M.recall_at_top_decile(y[te], p))
        out[label] = {"n_events": int(y.sum()),
                      "pr_auc": M.fold_band(praucs), "recall_decile": M.fold_band(recs)}
    return out


# ------------------------------------------------------------------ report
def build_report(scheme: str = "paidoff_only") -> str:
    from .eda import _md_table

    C.ensure_dirs()
    per_fold = run_bakeoff(scheme)
    paired = paired_comparison(per_fold)
    imps = _fold_importances(scheme)
    stab = stability_summary(imps)
    dist = score_distribution_check()
    loo = behind_sensitivity(scheme)

    top3_md = "\n".join(f"- **{f}**: in the top-3 in {100*v:.0f}% of folds"
                        for f, v in stab["top3_freq"].items())
    ratio = dist["share_flagged_full"] / dist["share_flagged_eval"] \
        if dist["share_flagged_eval"] else float("nan")

    md = f"""# Follow-up checks — paired RQ1 test, reason-code stability, population diagnostic

*Generated by `python -m emerald_ai followup-checks`, seed {C.SEED}, label {scheme}. Three
analyses answering review-round-1 challenges (see `docs/dissertation/REVIEW_ROUND1.md`).*

## 1. Paired fold-level model comparison (challenge C1)
All combos share identical 5x5 CV folds, so the comparison object is the per-fold PR-AUC
difference (LR+class-weight baseline minus challenger). Positive = baseline better.

{_md_table(paired)}

**Caveat stated up front:** repeated-CV folds share training data, so fold differences are
correlated and the sign test is approximate (anti-conservative). It is reported as descriptive
evidence, not confirmatory inference.

**Verdict:** {'the paired analysis agrees with the marginal-band verdict — no challenger separates from the baseline (all sign-test p > 0.05); the paired difference bands all straddle zero.' if (paired['sign_test_p'] > 0.05).all() else 'AT LEAST ONE challenger separates from the baseline under the paired test — this CHANGES the RQ1 reading; see table.'}

## 2. Reason-code stability across folds (challenge C6)
Global mean|SHAP| per source feature, recomputed independently in each of 5 folds (LR+CW):

- Pairwise Spearman rank correlation of fold importance vectors:
  **mean {stab['mean_spearman']:.3f}, min {stab['min_spearman']:.3f}**.
- Affordability cluster (Revenue / Average Monthly Sales) holds global rank 1 in
  **{100*stab['cluster_rank1_share']:.0f}%** of folds.
- Top-3 membership by fold:
{top3_md}

**Verdict:** {'rankings are highly stable across folds — the global reason-code ordering is not a seed artefact (though per-applicant codes on borderline cases still shift with the coefficient noise EPV predicts).' if stab['mean_spearman'] >= 0.8 else 'rankings shift materially across folds — global reason codes must be presented as cluster-level, not feature-level, exactly as the dissertation now words it.'}

## 3. Evaluation vs deployment population (challenge C8)
Frozen scorer applied to the full raw book (n={dist['n_full']:,}) vs the terminal-outcome
evaluation subset (n={dist['n_eval']:,}):

| | full book | evaluation subset |
|---|---|---|
| P(default) q10 / q50 / q90 | {dist['full_quantiles']['q10']} / {dist['full_quantiles']['q50']} / {dist['full_quantiles']['q90']} | {dist['eval_quantiles']['q10']} / {dist['eval_quantiles']['q50']} / {dist['eval_quantiles']['q90']} |
| share flagged at P >= {dist['threshold']:.3f} | {100*dist['share_flagged_full']:.1f}% | {100*dist['share_flagged_eval']:.1f}% |

- Two-sample KS statistic **{dist['ks_stat']}** (p = {dist['ks_p']:.2g}).
- The operating threshold set on the evaluation subset flags **{100*dist['share_flagged_full']:.1f}%**
  of the full book ({ratio:.1f}x the intended 10% decile) — quantifying the evaluation/deployment
  mismatch the label construction creates.

**Verdict:** the score distributions differ{' materially' if dist['ks_stat'] > 0.1 else ' only mildly'};
any operational use on the full application flow must re-derive the operating threshold on that
population (as `docs/path_to_production.md` already requires) rather than reusing the
terminal-outcome cut.

## 4. Leave-one-out on the single `behind` event (label-definition sensitivity)
The event set is `default` (49) ∪ `behind` (1); the lone `behind` case has no documented
operational definition, so the headline model (LR+CW, 5x5 CV) is re-run without it:

| | events | PR-AUC median [band] | recall@decile median [band] |
|---|---|---|---|
| with `behind` | {loo['with_behind']['n_events']} | {loo['with_behind']['pr_auc']['median']:.3f} [{loo['with_behind']['pr_auc']['lo']:.3f}, {loo['with_behind']['pr_auc']['hi']:.3f}] | {loo['with_behind']['recall_decile']['median']:.3f} [{loo['with_behind']['recall_decile']['lo']:.3f}, {loo['with_behind']['recall_decile']['hi']:.3f}] |
| without `behind` | {loo['without_behind']['n_events']} | {loo['without_behind']['pr_auc']['median']:.3f} [{loo['without_behind']['pr_auc']['lo']:.3f}, {loo['without_behind']['pr_auc']['hi']:.3f}] | {loo['without_behind']['recall_decile']['median']:.3f} [{loo['without_behind']['recall_decile']['lo']:.3f}, {loo['without_behind']['recall_decile']['hi']:.3f}] |

**Verdict:** {'the headline is insensitive to the `behind` case — medians move within fold noise.' if abs(loo['with_behind']['pr_auc']['median'] - loo['without_behind']['pr_auc']['median']) < 0.02 else 'dropping the `behind` case moves the median PR-AUC materially — the event definition matters and must be discussed.'}

---
*Reproduce: `python -m emerald_ai followup-checks`*
"""
    out = C.REPORTS_DIR / "followup_checks.md"
    out.write_text(md, encoding="utf-8")
    return str(out)
