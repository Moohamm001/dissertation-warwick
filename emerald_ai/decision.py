"""Cost-sensitive decision layer (option b) — turn the risk score into a defensible review policy.

The model only *ranks*; the desk still needs a cut-off. A 0.5 cut-off is wrong at 1.28% prevalence
(it floods the queue), and the top-decile cut-off is an arbitrary convention. The principled cut-off
minimises the lender's expected cost given the relative cost of the two errors:

    cost(t) = R · FN(t) + FP(t),   R = cost(false negative) / cost(false positive)

A false negative (a missed default) is far dearer than a false positive (a needless review), so R is
large. We do NOT use the analytic optimum t* = 1/(1+R): the class-weighted probabilities are
miscalibrated (Brier worse than base-rate, per `learning_evidence.md`), so we sweep the threshold
empirically on out-of-fold scores and pick the cost-minimiser. Because R is a business input we do
not truly know, we report a *range* of R and a sensitivity curve, never a single magic threshold —
and bootstrap the cost saving to show whether it survives the 50-event sampling noise.

CITATION NOTE (Rule 1): cost-sensitive framing is COVERED (D5, Xia et al. 2017). The canonical
expected-cost threshold result is now curated too — Elkan 2001 (`W167016754`, decision D14,
curated 2026-06-29). Run: ``python -m emerald_ai decide``.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from .experiments import oof_predictions

# Cost ratios to report: a missed default costs R times a needless review. Spanning a plausible
# lending range (a review is cheap; a default loss is large) keeps the policy honest to R-uncertainty.
COST_RATIOS = (5, 10, 20, 50, 100)


def _cumulative(y: np.ndarray, p: np.ndarray):
    """Sort by descending risk; return cumulative TP/FP when flagging the top-k, plus sorted scores."""
    order = np.argsort(p)[::-1]
    ys = y[order].astype(float)
    tp_cum = np.concatenate([[0.0], np.cumsum(ys)])          # k=0..N: defaults caught in top-k
    fp_cum = np.concatenate([[0.0], np.cumsum(1.0 - ys)])    # good loans reviewed in top-k
    return tp_cum, fp_cum, p[order]


def optimal_policy(y: np.ndarray, p: np.ndarray, R: float) -> dict:
    """Cost-minimising flag-the-top-k policy for cost ratio R, evaluated on OOF scores."""
    P = float(y.sum())
    tp_cum, fp_cum, p_sorted = _cumulative(y, p)
    cost = R * (P - tp_cum) + fp_cum                          # cost(flag top k), k = 0..N
    k = int(np.argmin(cost))
    thr = 1.0 if k == 0 else float(p_sorted[k - 1])          # k=0 => flag none
    return {
        "R": R, "threshold": round(thr, 4), "n_flagged": k,
        "defaults_caught": int(tp_cum[k]), "defaults_total": int(P),
        "recall": round(tp_cum[k] / P, 3) if P else float("nan"),
        "precision": round(tp_cum[k] / k, 4) if k else float("nan"),
        "cost": float(cost[k]),
    }


def _cost_at_k(y, p, R, k):
    tp_cum, fp_cum, _ = _cumulative(y, p)
    return float(R * (y.sum() - tp_cum[k]) + fp_cum[k])


def baselines_cost(y: np.ndarray, p: np.ndarray, R: float) -> dict:
    """Expected cost of the naive policies we must beat, for the same R."""
    N, P = len(y), int(y.sum())
    decile_k = max(1, int(np.ceil(0.10 * N)))
    return {
        "review_nothing": R * P,                              # miss every default
        "review_all": float(N - P),                           # review every good loan
        "threshold_0.5": _cost_at_k(y, p, R, int((p >= 0.5).sum())),
        "top_decile": _cost_at_k(y, p, R, decile_k),
    }


def bootstrap_saving(y: np.ndarray, p: np.ndarray, R: float, n_boot: int = 500) -> dict:
    """Bootstrap the % cost reduction of the cost-optimal policy vs BOTH baselines (0.5 cut and
    top-decile rule). The threshold is re-selected inside every resample, so the intervals
    propagate selection uncertainty; selection and evaluation share each resample, so the
    *location* of both intervals retains in-sample optimism (true savings may sit lower)."""
    rng = np.random.default_rng(C.SEED)
    n = len(y)
    sav05, savdec = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.sum() == 0:
            continue
        opt = optimal_policy(yb, pb, R)["cost"]
        base05 = _cost_at_k(yb, pb, R, int((pb >= 0.5).sum()))
        basedec = _cost_at_k(yb, pb, R, max(1, int(np.ceil(0.10 * n))))
        if base05 > 0:
            sav05.append(100 * (base05 - opt) / base05)
        if basedec > 0:
            savdec.append(100 * (basedec - opt) / basedec)
    s, d = np.array(sav05), np.array(savdec)
    return {"median": float(np.median(s)), "lo": float(np.percentile(s, 2.5)),
            "hi": float(np.percentile(s, 97.5)), "n": int(s.size),
            "dec_median": float(np.median(d)), "dec_lo": float(np.percentile(d, 2.5)),
            "dec_hi": float(np.percentile(d, 97.5))}


def policy_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    rows = []
    for R in COST_RATIOS:
        opt = optimal_policy(y, p, R)
        base = baselines_cost(y, p, R)
        saving = 100 * (base["threshold_0.5"] - opt["cost"]) / base["threshold_0.5"] \
            if base["threshold_0.5"] > 0 else float("nan")
        rows.append({
            "cost_ratio_R": R, "opt_threshold": opt["threshold"], "n_flagged": opt["n_flagged"],
            "defaults_caught": f"{opt['defaults_caught']}/{opt['defaults_total']}",
            "recall": opt["recall"], "precision": opt["precision"],
            "cost_vs_0.5_%saved": round(saving, 1),
        })
    return pd.DataFrame(rows)


def _fig_cost_curve(y, p, R) -> str:
    tp_cum, fp_cum, _ = _cumulative(y, p)
    ks = np.arange(len(tp_cum))
    cost = R * (y.sum() - tp_cum) + fp_cum
    opt = optimal_policy(y, p, R)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, cost, color="#1565c0", lw=1.5)
    ax.axvline(opt["n_flagged"], color="#c62828", ls="--",
               label=f"cost-optimal: flag {opt['n_flagged']} (recall {opt['recall']:.0%})")
    ax.axvline(int((p >= 0.5).sum()), color="#9e9e9e", ls=":", label="naive 0.5 cut")
    ax.set_xlabel("applications flagged for review (riskiest first)")
    ax.set_ylabel(f"expected cost (FP-equivalents), R={R}")
    ax.set_title("Cost-optimal review volume vs naive 0.5 cut-off")
    ax.legend(); fig.tight_layout()
    out = C.FIGURES_DIR / "decision_cost_curve.png"; fig.savefig(out, dpi=130); plt.close(fig)
    return out.name


def build_report(scheme: str = "paidoff_only") -> str:
    from .eda import _md_table

    C.ensure_dirs()
    y, p = oof_predictions("logreg", "class_weight", scheme=scheme)
    table = policy_table(y, p)
    R_demo = 20
    boot = bootstrap_saving(y, p, R_demo)
    f_cost = _fig_cost_curve(y, p, R_demo)

    opt20 = optimal_policy(y, p, R_demo)
    n05 = int((p >= 0.5).sum())
    robust = boot["lo"] > 0

    # Reference points the dissertation quotes: the demo's top-decile operating point (the same
    # OOF-decile cut serve.get_scorer() uses) and the cost of that decile rule at R_demo, so the
    # policy is compared against the operationally natural baseline, not only the 0.5 strawman.
    decile_thr = float(np.quantile(p, 0.90))
    in_decile = p >= decile_thr
    decile_catch = float(y[in_decile].sum() / y.sum()) if y.sum() else float("nan")
    base20 = baselines_cost(y, p, R_demo)
    saving_vs_decile = 100 * (base20["top_decile"] - opt20["cost"]) / base20["top_decile"] \
        if base20["top_decile"] > 0 else float("nan")

    md = f"""# Cost-sensitive decision policy (option b)

*Generated by `python -m emerald_ai decide`, seed {C.SEED}, label {scheme}
({int(y.sum())} events / {len(y)} rows). Threshold chosen on out-of-fold scores by minimising
`cost(t) = R·FN(t) + FP(t)`; R = cost(missed default) / cost(needless review).*

## Optimal review policy across cost ratios
{_md_table(table)}

- **opt_threshold** — risk cut-off that minimises expected cost (NOT 0.5, NOT a fixed decile).
- **n_flagged / recall** — how many applications enter the review queue and the share of defaults caught.
- **cost_vs_0.5_%saved** — expected-cost reduction against the naive 0.5 cut-off.

As R rises (a missed default hurts more), the optimal threshold falls, the queue grows, and recall
climbs — the policy trades cheap reviews for caught defaults exactly as a lender would want. The
`cost_vs_0.5_%saved` column is a ratio to a baseline whose own quality varies with R, so it is not
monotone in R — the non-monotonicity comes from the moving 0.5-cut denominator, not from
instability in the optimal policy (whose absolute cost is monotone in R).

## Reference operating point (as served by the demo)
The decision-support app flags the riskiest OOF decile: **threshold P >= {decile_thr:.3f}**,
catching **{100*decile_catch:.0f}%** of defaults ({int(y[in_decile].sum())}/{int(y.sum())}).
This is the catch-rate operating point used when R is unknown; the cost-optimal thresholds above
generalise it once a desk supplies its R.

## Does it actually work? (robustness at 50 events)
At **R = {R_demo}**, the cost-optimal policy flags **{opt20['n_flagged']}** applications
(vs **{n05}** under a 0.5 cut), catches **{opt20['defaults_caught']}/{opt20['defaults_total']}**
defaults, and cuts expected cost by **{table.loc[table.cost_ratio_R == R_demo, 'cost_vs_0.5_%saved'].iloc[0]:.1f}%**
vs the 0.5 cut, and by **{saving_vs_decile:.1f}%** vs the top-decile rule (the operationally
natural baseline).

Bootstrapping the OOF rows ({boot['n']} resamples), the cost saving vs the 0.5 cut is
**{boot['median']:.1f}% [{boot['lo']:.1f}, {boot['hi']:.1f}]**, and vs the top-decile rule
**{boot['dec_median']:.1f}% [{boot['dec_lo']:.1f}, {boot['dec_hi']:.1f}]** (95% intervals).
**Bootstrap procedure:** the threshold is **re-selected inside every resample** (selection and
evaluation both happen on the resampled rows), so the intervals propagate threshold-selection
uncertainty. Because selection and evaluation share each resample, both intervals retain
in-sample optimism in their *location* — the true savings may lie below the quoted bounds.
**Verdict vs 0.5 cut: {'the interval stays above zero — the policy beats the naive cut despite the 50-event noise (subject to the location caveat above).' if robust else 'the interval crosses zero — at 50 events the saving is NOT robust; reported honestly as inconclusive rather than spun.'}**
**Verdict vs top-decile rule: {'the interval stays above zero — a demonstrated (if modest) advantage.' if boot['dec_lo'] > 0 else 'the interval covers zero — NO demonstrated cost advantage over the simple decile rule at this R; the machinery earns its keep by deriving queue depth from stated costs, not by saving cost against a sensible default.'}**

![cost curve]({C.FIGURES_DIR.name}/{f_cost})

## Honest limitations
- **R is unknown.** We report a range and a sensitivity curve, not one threshold; the desk must
  supply its own FN:FP cost ratio. The *method* is the contribution, not a single number.
- **Probabilities are miscalibrated**, so the threshold is chosen empirically on OOF scores rather
  than from the analytic `1/(1+R)`; calibrating first (Phase 4) would change the absolute cut but
  not the policy logic.
- **50 events** still bound everything: the chosen threshold itself has sampling uncertainty (the
  bootstrap above quantifies it). This improves *decisions*, not discrimination (PR-AUC is unchanged).

## Method → citation audit (Rule 1)
| Method | Status | Paper(s) |
|---|---|---|
| Cost-sensitive treatment of imbalance | **COVERED** | Xia, Liu & Liu 2017 `W2700766797` **[CURATED]** (D5) |
| Expected-cost threshold selection | **COVERED** | Elkan 2001 `W167016754` — **[CURATED]** (D14) |

---
*Reproduce: `python -m emerald_ai decide`*
"""
    out = C.REPORTS_DIR / "decision_policy.md"
    out.write_text(md, encoding="utf-8")
    return str(out)
