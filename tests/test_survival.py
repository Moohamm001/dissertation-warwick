"""Tests for the survival-feasibility diagnostic (emerald_ai.survival).

The point of this module is a NEGATIVE result — that no trustworthy time-to-event clock exists — so
the test locks in the evidence that drives the non-estimability verdict, not a model's performance.
"""
from __future__ import annotations

from emerald_ai import config as C
from emerald_ai import data as D
from emerald_ai import survival as S


def test_candidate_clocks_are_built_for_every_status():
    df = D.load_raw()
    table, d = S.candidate_durations(df)
    assert {"cal_months", "term_months"} <= set(d.columns)
    assert set(table["status"]) == {"default", "behind", "current", "paidOff"}


def test_clocks_are_incoherent_so_survival_is_non_estimable():
    df = D.load_raw()
    _, d = S.candidate_durations(df)
    coh = S.coherence(d)
    # the two candidate durations do not agree -> there is no single reliable clock
    assert abs(coh["clock_corr"]) < 0.5
    # the red flags that make the censored rows unusable
    assert coh["paidoff_below_term_floor_pct"] > 25      # paidOff not near 100% term-complete
    assert coh["current_implausible_pct"] > 25           # current censored implausibly early
    assert coh["n_current_dropped"] > 1000               # this is the information we hoped to recover


def test_report_writes_and_states_the_verdict():
    path = S.build_report()
    assert path.endswith("survival_feasibility.md")
    text = (C.REPORTS_DIR / "survival_feasibility.md").read_text(encoding="utf-8")
    assert "NON-ESTIMABLE" in text
