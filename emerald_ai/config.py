"""Central configuration: paths, seed, label maps, column groups.

Every path constant lives here so a test can monkeypatch one place and so every figure is
traceable to a single source of truth (CLAUDE.md "Curie" rule).
"""
from __future__ import annotations

from pathlib import Path

# --- Reproducibility -------------------------------------------------------
SEED = 20260609  # global seed; set once, threaded everywhere.

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "All_Funded_2019_Green Loan.xlsx"
RAW_SHEET = "in"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
GOVERNANCE_DIR = PROJECT_ROOT / "data" / "governance"

# --- Label construction (proposal §5.1) ------------------------------------
# We model the EVENT (delinquency) as the positive class = 1. This is the minority we care
# about; PR-AUC and recall@top-decile are defined against it. (Note: the proposal text writes
# Y=1 for *favourable*; we invert to the conventional event=1 encoding and say so in the chapter.)
LABEL_COL = "Deal Status"
DELINQUENT = {"default", "behind"}          # event  -> y = 1
FAVOURABLE_ALL = {"paidOff", "current"}     # non-event under the all-favourable scheme
FAVOURABLE_PAIDOFF = {"paidOff"}            # non-event under the primary (censoring-safe) scheme
# `current` rows are right-censored; the primary label drops them entirely.

# --- Key columns for EDA / audit -------------------------------------------
ORIGINATION_COL = "Start"          # datetime; spans 2015-2020 despite the "2019" filename
INDUSTRY_COL = "Industry"
STATE_COL = "Borrower State"
REVENUE_COL = "Revenue"            # business-size proxy source
CREDIT_SCORE_COL = "Credit Score"
TIB_COL = "Time In Business"

# Minimum events per group cell for a group-conditional metric to be reported, not "undefined".
MIN_EVENTS_PER_CELL = 10

# Rule-based data-quality thresholds (domain-justified, leakage-safe — no distributional info).
CREDIT_SCORE_FLOOR = 1        # FICO 0 is impossible -> missing-coded; values < this -> NaN
TIB_MAX_MONTHS = 600         # > 50 years in business is implausible (likely a year typo) -> NaN
TIB_MIN_MONTHS = 0           # negative time-in-business is impossible -> NaN


def ensure_dirs() -> None:
    """Create output directories if absent. Safe to call repeatedly."""
    for d in (REPORTS_DIR, FIGURES_DIR, GOVERNANCE_DIR):
        d.mkdir(parents=True, exist_ok=True)
