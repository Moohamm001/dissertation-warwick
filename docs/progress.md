# EMERALD-AI — Progress Log

*Living ledger of every step done. Newest at top. Maps to `docs/roadmap.md` phases.
Updated as work lands; this is the answer to "what's going on in this project?".*

## Status at a glance
| Roadmap phase | State |
|---|---|
| Setup / scaffold | ✅ done |
| Phase 1 — EDA / imbalance feasibility | ✅ done |
| Phase 2 — leakage audit + preprocessing | ✅ done |
| Literature bot (lit-review aid) | ✅ built; ⏳ vetting not started (0 curated) |
| **Phase 3 — model × imbalance bake-off** | ⬜ **NEXT (the dissertation core)** |
| Phase 4 — calibration + conformal + SHAP | ⬜ |
| Phase 5 — audit (XAI / fairness / robustness) | ⬜ |
| Phase 6 — proof-of-concept demo | ⬜ |
| Phase 7 — write-up + release | ⬜ |

## Done (most recent first)
- **2026-06-10 — Git hygiene.** Untracked `CLAUDE.md` (local working file) via `.gitignore`.
- **2026-06-10 — Git initialised.** Repo on `main`, commit `5cc534f`. Authored as the user; no AI
  attribution in history (standing rule).
- **2026-06-10 — Phase 2: leakage-safe pipeline.** `feature_audit.py` (default-deny; **17 vetted
  pre-funding features** permitted, ~148 forbidden), `preprocess.py` (ColumnTransformer +
  `assert_no_leakage` guard), catalogue in `data/governance/`. 5 leakage tests pass.
  Run: `python -m emerald_ai audit` / `preprocess-check`.
- **2026-06-10 — Literature bot.** `research_bot` OpenAlex crawler; crawled **179 papers** into
  `literature/auto_index.yaml` across 6 method themes. 4 tests (path-isolation) pass.
- **2026-06-10 — Phase 1: EDA / feasibility.** `python -m emerald_ai eda` → `reports/feasibility.md`.
  Key results: event count = **50 under both label schemes**; **72.8%** of 2019 cohort censored;
  **group-fairness audit non-estimable** (0/27 industries, 0/51 states reach ≥10 events).
- **2026-06-10 — Package scaffold + README.** `emerald_ai` package, Windows-first CLI, pinned deps.
- **2026-06-09 — Roadmap.** v1 (maximalist) → v2 (MSc cut) → v2.1 (conformal restored Core-light;
  "rigour is the contribution" framing).
- **2026-06-09 — Proposal review.** Verified 50 events vs raw data; flagged N=50 as binding
  constraint, censoring under-weighted, fairness audit infeasible (later confirmed by EDA).

## Literature coverage (as of 2026-06-10)
179 auto-discovered; **117 on-topic (65%)**, 62 noise to discard. Per theme: imbalance 60,
calibration 29, green-finance 28, explainability/fairness 26, selection-bias 20, tabular 16.
117/179 are ≥2018. **Vetting (promote to curated `index.yaml`) not yet started.**

## Next action
Build **Phase 3**: LR vs XGBoost × class-weight vs SMOTE-in-fold, repeated stratified CV,
PR-AUC / recall@top-decile / within-minority ECE + bootstrap CIs → answers RQ1.
