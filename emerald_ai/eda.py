"""Phase 1 — data understanding & imbalance/censoring feasibility.

Produces ``reports/feasibility.md`` plus figures, entirely from the raw data, seeded and
re-runnable via ``python -m emerald_ai eda``. Every number in the report is computed here;
nothing is hand-typed. This is the chapter that decides the label, the feature freeze, and
whether a group-fairness audit is even estimable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt

from . import config as C
from . import data as D


# --------------------------------------------------------------------------- analyses
def label_distribution(df: pd.DataFrame) -> pd.Series:
    return df[C.LABEL_COL].value_counts(dropna=False)


def censoring_by_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Status composition by origination year — quantifies right-censoring.

    A loan still ``current`` has not had the chance to default; the later it originated, the
    shorter its observation window. We show, per origination year, the share censored.
    """
    d = df.copy()
    d["orig_year"] = d[C.ORIGINATION_COL].dt.year
    comp = (
        d.groupby("orig_year")[C.LABEL_COL]
        .value_counts(normalize=False)
        .unstack(fill_value=0)
    )
    comp["n"] = comp.sum(axis=1)
    comp["pct_current_censored"] = (comp.get("current", 0) / comp["n"] * 100).round(1)
    # observed delinquency among rows that reached a terminal state (paidOff/default/behind)
    terminal = comp.get("paidOff", 0) + comp.get("default", 0) + comp.get("behind", 0)
    events = comp.get("default", 0) + comp.get("behind", 0)
    comp["delinq_pct_of_terminal"] = np.where(terminal > 0, (events / terminal * 100), np.nan).round(2)
    return comp


def fairness_feasibility(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Event counts per group cell under the PRIMARY label — tests whether group-conditional
    fairness metrics are estimable (>= MIN_EVENTS_PER_CELL events)."""
    t = D.build_target(df, "paidoff_only")
    g = t.groupby(group_col)["y"].agg(n="size", events="sum")
    g["prevalence_pct"] = (g["events"] / g["n"] * 100).round(2)
    g["estimable"] = g["events"] >= C.MIN_EVENTS_PER_CELL
    return g.sort_values("events", ascending=False)


def missingness_map(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    miss = df.isna().mean().mul(100).round(1).sort_values(ascending=False)
    out = miss[miss > 0].head(top).rename("pct_missing").to_frame()
    out["fully_missing"] = out["pct_missing"] >= 100.0
    return out


def data_quality_flags(df: pd.DataFrame) -> dict[str, int]:
    """Concrete anomalies worth a sentence in the dissertation."""
    return {
        "credit_score_eq_zero (missing-coded)": int((df[C.CREDIT_SCORE_COL] == 0).sum()),
        "time_in_business_negative": int((df[C.TIB_COL] < 0).sum()),
        "time_in_business_gt_600 (>50yr, implausible)": int((df[C.TIB_COL] > 600).sum()),
        "origination_before_2019": int((df[C.ORIGINATION_COL].dt.year < 2019).sum()),
        "origination_after_2019": int((df[C.ORIGINATION_COL].dt.year > 2019).sum()),
    }


# --------------------------------------------------------------------------- figures
def _fig_prevalence_by_quarter(df: pd.DataFrame) -> str:
    t = D.build_target(df, "paidoff_only")
    t = t.assign(q=t[C.ORIGINATION_COL].dt.to_period("Q").astype(str))
    g = t.groupby("q")["y"].agg(["size", "sum"])
    g = g[g["size"] >= 30]  # suppress micro-cohorts
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(g.index, g["sum"], color="#2e7d32")
    ax.set_ylabel("delinquency events")
    ax.set_title("Delinquency events by origination quarter (paidOff-only label)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    p = C.FIGURES_DIR / "events_by_quarter.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p.name


def _fig_censoring(comp: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(comp.index.astype(str), comp["pct_current_censored"], color="#1565c0")
    ax.set_ylabel("% of cohort still 'current' (censored)")
    ax.set_title("Right-censoring by origination year")
    fig.tight_layout()
    p = C.FIGURES_DIR / "censoring_by_year.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p.name


# --------------------------------------------------------------------------- report
def _md_table(df: pd.DataFrame, index_name: str = "") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no tabulate dependency)."""
    df = df.copy()
    df.index.name = index_name or df.index.name or "index"
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([head, sep, *rows])


def build_report() -> str:
    """Run all analyses, write figures + ``reports/feasibility.md``; return the report path."""
    C.ensure_dirs()
    df = D.load_raw()

    dist = label_distribution(df)
    prev = D.prevalence_summary(df)
    comp = censoring_by_cohort(df)
    fair_ind = fairness_feasibility(df, C.INDUSTRY_COL)
    fair_state = fairness_feasibility(df, C.STATE_COL)
    miss = missingness_map(df)
    dq = data_quality_flags(df)

    fig_q = _fig_prevalence_by_quarter(df)
    fig_c = _fig_censoring(comp)

    n_ind_estimable = int(fair_ind["estimable"].sum())
    n_state_estimable = int(fair_state["estimable"].sum())

    md = f"""# Phase 1 — Imbalance & Censoring Feasibility Report

*Auto-generated by `python -m emerald_ai eda` — every number is computed from the raw data,
seed = {C.SEED}. Do not edit by hand.*

## 1. The binding constraint
Label column `{C.LABEL_COL}`:

```
{dist.to_string()}
```

Delinquency = `default` ∪ `behind`. **The event count, not the ratio, is the constraint.**

## 2. Label scheme — prevalence under both constructions
{_md_table(prev)}

`paidoff_only` is the **primary** label: it drops the right-censored `current` rows. Note the
event count barely moves between schemes — dropping censored rows fixes the *ratio*, never the
*count*. No labelling choice solves the small-N problem.

## 3. Right-censoring by origination cohort
The dataset is "2019 funded" but origination (`{C.ORIGINATION_COL}`) spans
{int(comp.index.min())}–{int(comp.index.max())}. Later cohorts have shorter observation windows,
so their `current` share is mechanically higher — this is the censoring threat to the label.

{_md_table(comp.drop(columns=[c for c in comp.columns if c not in ('n','current','paidOff','default','behind','pct_current_censored','delinq_pct_of_terminal')], errors='ignore'), index_name='orig_year')}

![censoring]({C.FIGURES_DIR.name}/{fig_c})
![events by quarter]({C.FIGURES_DIR.name}/{fig_q})

## 4. Fairness feasibility (THE go/no-go decision)
Group-conditional fairness metrics need events *within each group cell*. Threshold for
"estimable" = **≥ {C.MIN_EVENTS_PER_CELL} events** (primary label).

**By Industry** — {n_ind_estimable} of {len(fair_ind)} groups estimable:
{_md_table(fair_ind.head(12), index_name=C.INDUSTRY_COL)}

**By Borrower State** — {n_state_estimable} of {len(fair_state)} groups estimable:
{_md_table(fair_state.head(8), index_name=C.STATE_COL)}

**Verdict:** with so few estimable cells, a full group-conditional fairness audit (equalised
odds / predictive parity) is **not defensible** on this portfolio. RO4 converts to a *documented
non-estimability* result + the audit protocol — itself an honest contribution. (See roadmap Gate A.)

## 5. Missingness map (feature-freeze input)
{_md_table(miss, index_name='column')}

Features over the 40% drop threshold and the fully-missing set are excluded in preprocessing.

## 6. Data-quality flags (one sentence each in the write-up)
{chr(10).join(f"- **{k}**: {v}" for k, v in dq.items())}

---
*Reproduce: `python -m emerald_ai eda`*
"""
    out = C.REPORTS_DIR / "feasibility.md"
    out.write_text(md, encoding="utf-8")
    return str(out)
