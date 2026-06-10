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
| Phase 3 — model × imbalance bake-off | ✅ done (RQ1 answered) |
| **Phase 4 — calibration + conformal + SHAP** | ⬜ **NEXT** |
| Phase 5 — audit (XAI / fairness / robustness) | ⬜ |
| Phase 6 — proof-of-concept demo | ⬜ |
| Phase 7 — write-up + release | ⬜ |

## Done (most recent first)
- **2026-06-10 — Evidence-grounded methodology audit.** Rule 1: `docs/methods_citations.md`
  (every imbalance choice → paper); 3 gaps (PR-vs-ROC, repeated CV, EPV/small-n) closed by
  patching crawler seeds and re-crawling (179→292 papers; Saito&Rehmsmeier 2015, Peduzzi 1996,
  Riley 2020, Krstajić 2014, Kapoor&Narayanan 2023). Rule 2: `emerald_ai/validation.py` +
  `python -m emerald_ai evidence` → `reports/learning_evidence.md`. **Permutation test p=0.010
  (real beats 100% of nulls); stable across seeds (0.091±0.002); plateaus by 75% (supports LR).
  HONEST LIMITATION: Brier 0.124 ≫ base-rate 0.0127 — class-weighted probs miscalibrated → Phase 4.**
  Gap report: `docs/gap_report.md`. ⏳ curated promotion to index.yaml AWAITS user sign-off.
- **2026-06-10 — Visual story.** `python -m emerald_ai figures` → `reports/visual_story.md` + 6
  figures: imbalance, censoring, CV-resampling schematic, PR-AUC fold boxes vs floor, **cumulative
  gains (top 10% catches 64% of defaults — the usefulness proof)**, minority calibration.
- **2026-06-10 — Phase 3: model × imbalance bake-off (RQ1).** `experiments.py` + `metrics.py`,
  5×5 repeated stratified CV, resampling inside folds. `python -m emerald_ai bakeoff` →
  `reports/model_bakeoff.md`. **RQ1 finding: no significant winner — LR (PR-AUC 0.116) ≈ XGBoost
  (0.091), fold bands overlap. All models beat the 0.013 prevalence floor ~8× (features carry
  signal). LR is far better calibrated on defaults (ECE 0.31 vs 0.85) — flagged for Phase 4.**
  4 metric tests added (13 passing total).
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
Build **Phase 4**: post-hoc calibration (Platt/isotonic) on the best model + within-minority ECE
with bootstrap CIs + split-conformal (MAPIE, Core-light) + SHAP global/local → answers RQ2/RQ3.
The Phase 3 calibration gap (LR 0.31 vs XGBoost 0.85) makes calibration the obvious next lever.
