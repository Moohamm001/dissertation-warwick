"""Phase 4a — probability calibration + split-conformal (RQ2).

Motivated by the Rule-2 finding that the class-weighted logistic model *ranks* defaults well but
its probabilities are inflated (Brier 0.124 ≫ base-rate 0.0127). Here we:
  * fit post-hoc calibrators (Platt / isotonic) on a held-out calibration split inside each CV fold;
  * report **Brier** and **within-minority ECE** for raw vs Platt vs isotonic, each with bootstrap CIs;
  * run a transparent **split-conformal** procedure (LAC/score method, Angelopoulos & Bates 2023)
    and report marginal coverage — framed as honest small-N transparency, not a precision claim.

Expected tension (and an honest RQ2 answer): calibrating for *marginal* Brier can leave *within-
minority* ECE unchanged or worse, because ~50 events barely constrain the calibrator. We report it
either way. Additive module; reuses the existing pipeline.

Run: ``python -m emerald_ai calibrate``.
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config as C
from . import data as D
from . import metrics as M
from . import preprocess as P
from .experiments import _make_model

MODEL, STRATEGY = "logreg", "class_weight"


def _calibrated_oof(df, y, n_splits: int = 5, seed: int = C.SEED) -> dict:
    """OOF probabilities for raw / Platt / isotonic. Calibrator fit on a held-out slice of train."""
    pre, _ = P.build_preprocessor(df, scale=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = {k: np.full(len(y), np.nan) for k in ("raw", "platt", "isotonic")}
    for tr, te in skf.split(df, y):
        ytr = y[tr]
        fit_i, cal_i = train_test_split(np.arange(len(tr)), test_size=0.3,
                                        stratify=ytr, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_fit = pre.fit_transform(df.iloc[tr].iloc[fit_i], ytr[fit_i])
            model = _make_model(MODEL, STRATEGY, ytr[fit_i]); model.fit(X_fit, ytr[fit_i])
            s_cal = model.predict_proba(pre.transform(df.iloc[tr].iloc[cal_i]))[:, 1]
            s_te = model.predict_proba(pre.transform(df.iloc[te]))[:, 1]
            y_cal = ytr[cal_i]
            platt = LogisticRegression(max_iter=1000).fit(s_cal.reshape(-1, 1), y_cal)
            iso = IsotonicRegression(out_of_bounds="clip").fit(s_cal, y_cal)
        oof["raw"][te] = s_te
        oof["platt"][te] = platt.predict_proba(s_te.reshape(-1, 1))[:, 1]
        oof["isotonic"][te] = iso.predict(s_te)
    return oof


def _boot_ci(y, s, fn, n_boot: int = 2000, seed: int = C.SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        v = fn(y[b], s[b])
        if not np.isnan(v):
            vals.append(v)
    v = np.array(vals)
    return float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def split_conformal(df, y, alphas=(0.10, 0.05), n_repeats: int = 30, seed: int = C.SEED) -> list[dict]:
    """Manual split-conformal (LAC). Marginal coverage + mean set size across repeated 60/20/20 splits."""
    pre, _ = P.build_preprocessor(df, scale=True)
    out = {a: {"cov": [], "size": []} for a in alphas}
    for r in range(n_repeats):
        tr, tmp = train_test_split(np.arange(len(y)), test_size=0.4, stratify=y, random_state=seed + r)
        cal, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed + r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_tr = pre.fit_transform(df.iloc[tr], y[tr])
            model = _make_model(MODEL, STRATEGY, y[tr]); model.fit(X_tr, y[tr])
            p_cal = model.predict_proba(pre.transform(df.iloc[cal]))[:, 1]
            p_te = model.predict_proba(pre.transform(df.iloc[te]))[:, 1]
        y_cal, y_te = y[cal], y[te]
        # nonconformity = 1 - p(true label); for label1 -> 1-p1, label0 -> p1
        s_cal = np.where(y_cal == 1, 1 - p_cal, p_cal)
        for a in alphas:
            n = len(s_cal)
            q = np.quantile(s_cal, min(1.0, np.ceil((n + 1) * (1 - a)) / n), method="higher")
            in1 = (1 - p_te) <= q
            in0 = p_te <= q
            covered = np.where(y_te == 1, in1, in0)
            out[a]["cov"].append(covered.mean())
            out[a]["size"].append((in0.astype(int) + in1.astype(int)).mean())
    return [{"nominal": 1 - a, "coverage": float(np.mean(out[a]["cov"])),
             "coverage_sd": float(np.std(out[a]["cov"])), "mean_set_size": float(np.mean(out[a]["size"]))}
            for a in alphas]


def _fig_reliability(y, oof) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color="#9e9e9e", label="perfect")
    colors = {"raw": "#ef6c00", "platt": "#1565c0", "isotonic": "#2e7d32"}
    for k, c in colors.items():
        fp, mp = calibration_curve(y, oof[k], n_bins=5, strategy="quantile")
        ax.plot(mp, fp, "o-", color=c, label=k)
    ax.set_xlabel("mean predicted P(default)"); ax.set_ylabel("observed default rate")
    ax.set_title("Reliability: raw vs Platt vs isotonic"); ax.legend()
    fig.tight_layout()
    p = C.FIGURES_DIR / "calib_reliability.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p.name


def build_report() -> str:
    from .eda import _md_table
    import pandas as pd

    C.ensure_dirs()
    df = D.build_target(D.load_raw(), "paidoff_only").reset_index(drop=True)
    y = df["y"].to_numpy()

    oof = _calibrated_oof(df, y)
    rows = []
    for k in ("raw", "platt", "isotonic"):
        bm, bl, bh = _boot_ci(y, oof[k], lambda a, b: brier_score_loss(a, b))
        em, el, eh = _boot_ci(y, oof[k], M.within_minority_ece)
        rows.append({"calibration": k,
                     "Brier [95% CI]": f"{bm:.4f} [{bl:.4f}, {bh:.4f}]",
                     "within-min ECE [95% CI]": f"{em:.3f} [{el:.3f}, {eh:.3f}]"})
    tbl = pd.DataFrame(rows)
    conf = pd.DataFrame(split_conformal(df, y)).round(3)
    f_rel = _fig_reliability(y, oof)

    base_brier = y.mean() * (1 - y.mean())
    raw_b = brier_score_loss(y, oof["raw"]); best = min(("platt", "isotonic"),
                                                        key=lambda k: brier_score_loss(y, oof[k]))
    best_b = brier_score_loss(y, oof[best])
    raw_ece = M.within_minority_ece(y, oof["raw"]); best_ece = M.within_minority_ece(y, oof[best])

    md = f"""# Phase 4a — Calibration & Conformal (RQ2)

*Generated by `python -m emerald_ai calibrate`, seed {C.SEED}, label paidoff_only
(prevalence {y.mean():.4f}, {int(y.sum())} events). Calibrators fit on a held-out slice inside each
CV fold; metrics are out-of-fold with 2,000-sample bootstrap CIs.*

## Calibration metrics (raw vs Platt vs isotonic)
{_md_table(tbl)}

Base-rate (prevalence-only) Brier ≈ {base_brier:.4f}.

## RQ2 verdict (honest)
- **Marginal Brier:** {raw_b:.4f} (raw) → {best_b:.4f} ({best}). Post-hoc calibration
  {'substantially improves' if best_b < raw_b * 0.7 else 'improves' if best_b < raw_b else 'does not improve'}
  the marginal Brier, pulling the inflated class-weighted probabilities back toward the base rate.
- **Within-minority ECE:** {raw_ece:.3f} (raw) → {best_ece:.3f} ({best}).
  {'Calibration also helps on the minority.' if best_ece < raw_ece else '**Calibration does NOT help (and may hurt) within-minority ECE** — pulling probabilities down for marginal Brier lowers confidence on the rare defaults. This is the proposal’s core point: marginal calibration and minority calibration are different objectives, and ~50 events cannot satisfy both.'}

## Split-conformal coverage (transparency, not precision)
Manual LAC split-conformal [Angelopoulos & Bates 2023], {30} repeated 60/20/20 splits:
{_md_table(conf)}

At {y.mean()*100:.2f}% prevalence, marginal coverage is near-vacuous — a set that always contains
the majority class trivially achieves it; mean set size shows how often the set collapses to a
single label. Reported as an honest small-N transparency artefact, as the roadmap frames it.

![reliability]({C.FIGURES_DIR.name}/{f_rel})

---
*Reproduce: `python -m emerald_ai calibrate`*
"""
    out = C.REPORTS_DIR / "calibration.md"
    out.write_text(md, encoding="utf-8")
    return str(out)
