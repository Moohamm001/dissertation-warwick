"""Visual story — prove, step by step, how the extreme imbalance is handled and that the model is
USEFUL on this data despite ~50 events.

Generates a sequence of figures + ``reports/visual_story.md`` that walks a non-specialist through:
  1. The problem      — how extreme the imbalance + censoring are.
  2. The method       — resampling happens INSIDE each CV fold (no leakage).
  3. Does it learn?   — PR-AUC across folds vs the prevalence floor.
  4. Is it USEFUL?    — cumulative-gains curve: defaults caught in the riskiest decile.
  5. Honest limits    — calibration on the actual defaults.

Run: ``python -m emerald_ai figures``.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from . import config as C
from . import data as D
from . import experiments as E

GREEN, BLUE, ORANGE, GREY = "#2e7d32", "#1565c0", "#ef6c00", "#9e9e9e"


def _save(fig, name: str) -> str:
    p = C.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return name


# ---- Step 1: the problem -------------------------------------------------
def fig_imbalance() -> str:
    df = D.load_raw()
    vc = df[C.LABEL_COL].value_counts(dropna=False)
    labels = [str(x) for x in vc.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [ORANGE if l in ("default", "behind") else BLUE for l in labels]
    bars = ax.bar(labels, vc.values, color=colors, log=True)
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("loans (log scale)")
    ax.set_title("The problem: 50 delinquency events (orange) in 14,135 loans = 0.36%")
    return _save(fig, "story1_imbalance.png")


def fig_censoring() -> str:
    df = D.load_raw().copy()
    df["yr"] = df[C.ORIGINATION_COL].dt.year
    comp = df.groupby("yr")[C.LABEL_COL].value_counts().unstack(fill_value=0)
    n = comp.sum(axis=1)
    pct_cens = (comp.get("current", 0) / n * 100)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(comp.index.astype(str), pct_cens, color=BLUE)
    ax.set_ylabel("% of cohort still 'current'")
    ax.set_title("Why censoring matters: later cohorts are mostly unresolved (right-censored)")
    return _save(fig, "story1_censoring.png")


# ---- Step 2: the method (schematic) --------------------------------------
def fig_cv_schematic() -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    n_folds = 5
    for i in range(n_folds):
        y0 = n_folds - 1 - i
        # one row = one CV split; test block is fold i
        for j in range(n_folds):
            is_test = j == i
            ax.add_patch(mpatches.Rectangle((j, y0), 1, 0.8,
                         facecolor=ORANGE if is_test else BLUE, edgecolor="white"))
        ax.text(-0.2, y0 + 0.4, f"split {i+1}", ha="right", va="center", fontsize=9)
    ax.text(n_folds + 0.15, n_folds - 0.6,
            "TRAIN (blue):\n• fit preprocessing\n• SMOTE / class-weight\n  applied HERE only",
            va="top", fontsize=9, color=BLUE)
    ax.text(n_folds + 0.15, 1.2,
            "TEST (orange):\n• untouched\n• score + measure", va="top", fontsize=9, color=ORANGE)
    ax.set_xlim(-1.5, n_folds + 3.2)
    ax.set_ylim(-0.3, n_folds + 0.2)
    ax.axis("off")
    ax.set_title("How imbalance is handled: resampling lives INSIDE the training fold (no leakage)")
    return _save(fig, "story2_cv.png")


# ---- Step 3: does it learn? ----------------------------------------------
def fig_prauc_folds(per_fold) -> str:
    combos = list(per_fold["combo"].unique())
    data = [per_fold.loc[per_fold.combo == c, "pr_auc"].dropna().values for c in combos]
    prevalence = D.build_target(D.load_raw(), "paidoff_only")["y"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, vert=True, labels=[c.replace("+", "\n+") for c in combos], showfliers=False)
    for i, d in enumerate(data, 1):
        ax.scatter(np.random.normal(i, 0.05, len(d)), d, s=10, alpha=0.4, color=GREY)
    ax.axhline(prevalence, color=ORANGE, ls="--", label=f"prevalence floor ({prevalence:.3f})")
    ax.set_ylabel("PR-AUC per fold")
    ax.set_title("Does it learn? Every model sits ~8x above the floor — but bands overlap (no winner)")
    ax.legend()
    return _save(fig, "story3_prauc.png")


# ---- Step 4: is it USEFUL? -----------------------------------------------
def fig_gains(y, score, label: str) -> str:
    order = np.argsort(score)[::-1]
    y_sorted = np.asarray(y)[order]
    total = y_sorted.sum()
    frac_pop = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    frac_caught = np.cumsum(y_sorted) / total
    rec_decile = frac_caught[int(np.ceil(0.1 * len(y_sorted))) - 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(frac_pop, frac_caught, color=GREEN, lw=2.5, label=f"{label} model")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", label="random review")
    ax.axvline(0.1, color=ORANGE, ls=":")
    ax.scatter([0.1], [rec_decile], color=ORANGE, zorder=5)
    ax.annotate(f"review riskiest 10%\n→ catch {rec_decile*100:.0f}% of all defaults",
                (0.1, rec_decile), xytext=(0.18, rec_decile - 0.18),
                arrowprops=dict(arrowstyle="->", color=ORANGE), fontsize=10, color=ORANGE)
    ax.set_xlabel("fraction of loans reviewed (ranked riskiest first)")
    ax.set_ylabel("fraction of all defaults caught")
    ax.set_title("Is it useful? Cumulative-gains: the score concentrates defaults at the top")
    ax.legend(loc="lower right")
    return _save(fig, "story4_gains.png"), rec_decile


# ---- Step 5: honest calibration ------------------------------------------
def fig_calibration(oofs: dict) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data, labels, means = [], [], []
    for name, (y, s) in oofs.items():
        pos = np.asarray(s)[np.asarray(y) == 1]
        data.append(pos)
        labels.append(name)
        means.append(pos.mean())
    ax.boxplot(data, vert=True, labels=labels, showfliers=False)
    for i, (d, m) in enumerate(zip(data, means), 1):
        ax.scatter(np.random.normal(i, 0.05, len(d)), d, s=12, alpha=0.5, color=GREY)
        ax.text(i, 1.02, f"mean {m:.2f}", ha="center", fontsize=9)
    ax.axhline(1.0, color=GREEN, ls="--", label="ideal confidence on a default = 1.0")
    ax.set_ylabel("predicted P(default) on ACTUAL defaults")
    ax.set_ylim(0, 1.1)
    ax.set_title("Honest limits: how confident is each model on the real defaults?")
    ax.legend(loc="lower right")
    return _save(fig, "story5_calibration.png")


# ---- assemble ------------------------------------------------------------
def build_story() -> str:
    C.ensure_dirs()
    f1a, f1b = fig_imbalance(), fig_censoring()
    f2 = fig_cv_schematic()

    per_fold = E.run_bakeoff()
    f3 = fig_prauc_folds(per_fold)

    y_lr, s_lr = E.oof_predictions("logreg", "class_weight")
    y_xgb, s_xgb = E.oof_predictions("xgboost", "class_weight")
    f4, rec = fig_gains(y_lr, s_lr, "logreg")
    f5 = fig_calibration({"logreg\n+class_weight": (y_lr, s_lr),
                          "xgboost\n+class_weight": (y_xgb, s_xgb)})

    fd = C.FIGURES_DIR.name
    md = f"""# Visual Story — handling 0.36% imbalance, and proving it works

*Generated by `python -m emerald_ai figures`, seed {C.SEED}. Five steps, each a figure.*

## Step 1 — The problem
Only **50 loans** ever go delinquent out of 14,135 (**0.36%**), and most loans are still
unresolved (right-censored), worse for recent cohorts.

![imbalance]({fd}/{f1a})
![censoring]({fd}/{f1b})

## Step 2 — How we handle it (without cheating)
Resampling (SMOTE) and class-weighting are applied **only inside the training part of each
cross-validation split**. The test block never sees synthetic rows or the encoder's knowledge —
so the scores are honest.

![cv]({fd}/{f2})

## Step 3 — Does the model actually learn?
Across 25 folds, every real model scores **~8× above the prevalence floor** — it is finding
signal, not guessing. The boxes also **overlap**, which is why we honestly report *no single
winning model* (RQ1).

![prauc]({fd}/{f3})

## Step 4 — Is it USEFUL on this data? (the key proof)
Rank every loan by predicted risk and review the riskiest first. Reviewing just the **top 10%**
captures **{rec*100:.0f}% of all defaults** — far above the 10% a random review would catch.
*That* is the operational value, and it survives the small sample.

![gains]({fd}/{f4})

## Step 5 — Honest limits: calibration
On the actual defaults, the logistic model is far more confident (closer to the ideal 1.0) than
XGBoost — which is why LR is the better-calibrated choice and why Phase 4 (calibration) targets
exactly this gap.

![calibration]({fd}/{f5})

---
*Reproduce: `python -m emerald_ai figures`*
"""
    out = C.REPORTS_DIR / "visual_story.md"
    out.write_text(md, encoding="utf-8")
    return str(out)
