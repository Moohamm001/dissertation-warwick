"""Data loading and label construction.

Two label schemes are produced side by side so the censoring assumption is never hidden:
  * ``paidoff_only``  (PRIMARY): non-event = paidOff; the 72% right-censored ``current`` rows
    are dropped. Prevalence rises (~1.3%) but the event count is unchanged (~50).
  * ``all_favourable`` (comparator): non-event = paidOff OR current; optimistic.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def load_raw() -> pd.DataFrame:
    """Load the raw Excel sheet. No cleaning here — characterisation comes first."""
    return pd.read_excel(C.RAW_DATA, sheet_name=C.RAW_SHEET)


def build_target(df: pd.DataFrame, scheme: str = "paidoff_only", clean: bool = True) -> pd.DataFrame:
    """Return a copy with a binary ``y`` column (1 = delinquency event) for the given scheme.

    Rows whose ``Deal Status`` is unlabelled (NaN) are always dropped. Under ``paidoff_only``
    the censored ``current`` rows are also dropped. The function never imputes a label.

    ``clean=True`` (default) applies the rule-based, leakage-safe impossible-value correction from
    :mod:`emerald_ai.clean` first. Pass ``clean=False`` for the raw-data sensitivity comparison.
    """
    if scheme not in {"paidoff_only", "all_favourable"}:
        raise ValueError(f"unknown scheme: {scheme!r}")

    if clean:
        from . import clean as _clean  # lazy import: clean.py imports this module
        df, _ = _clean.clean(df)

    status = df[C.LABEL_COL]
    non_event = C.FAVOURABLE_PAIDOFF if scheme == "paidoff_only" else C.FAVOURABLE_ALL
    keep = status.isin(C.DELINQUENT | non_event)

    out = df.loc[keep].copy()
    out["y"] = out[C.LABEL_COL].isin(C.DELINQUENT).astype(int)
    return out


def prevalence_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Event count / total / prevalence for both label schemes, as a tidy frame."""
    rows = []
    for scheme in ("all_favourable", "paidoff_only"):
        t = build_target(df, scheme)
        n, pos = len(t), int(t["y"].sum())
        rows.append(
            {
                "scheme": scheme,
                "n_total": n,
                "n_events": pos,
                "n_nonevents": n - pos,
                "prevalence_pct": round(100 * pos / n, 3),
            }
        )
    return pd.DataFrame(rows)
