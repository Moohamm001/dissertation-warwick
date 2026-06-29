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
| Phase 4 — calibration + conformal + SHAP | ✅ done (RQ2/RQ3 answered) |
| Data-quality cleaning + sensitivity | ✅ done (integrated, robust) |
| **Phase 5 — proof-of-concept demo (FastAPI)** | ✅ done (`python -m emerald_ai serve`) |
| Fairness/robustness audit (light) | ⬜ **NEXT** (largely a documented non-estimability result) |
| Phase 6 — write-up + release | ⬜ |

## Done (most recent first)
- **2026-06-29 — Survival feasibility: can the censored loans be recovered? NON-ESTIMABLE.**
  `emerald_ai/survival.py` + `python -m emerald_ai survival-check` → `reports/survival_feasibility.md`.
  Tested option (a): use the ~10,124 censored `current` loans (dropped by `paidoff_only`) via a
  time-to-default model. **Verdict: non-estimable — the dataset has no trustworthy clock.** Two
  candidate durations (calendar `End−Start` vs term-based `Closed Max Term × Term Complete %`)
  **correlate −0.02**; 89.9% of `paidOff` sit below 90% term-complete (so the column ≠ elapsed loan
  life); 75.6% of `current` show <1 month calendar span (implausible for 2015–2019 originations →
  `End` is an admin booking date, not maturity/default). Did NOT fit a Cox model on a meaningless
  time axis. Second documented infeasibility, alongside the fairness audit. **Citation GAP:** brain
  has zero survival/Cox papers. Unlock = a default-date / last-payment-date field. 3 tests (28 total).
- **2026-06-29 — RQ1 follow-up: "can we do better?" experiment.** `emerald_ai/improve.py` +
  `python -m emerald_ai improve` → `reports/improvement.md`. **Exp 1:** L1-sparse / elastic-net /
  affordability ratios vs L2 baseline (10 numerics, EPV~3.8). **Expected NULL — nothing tightens the
  fold band** (~0.19, median PR-AUC ~0.122 all four); band is sampling-variance-limited, not
  model-limited. **Exp 2:** fixed-prevalence events projection (subsample both classes, 3 draws);
  width∝1/√events fit, corr(events,width)=−0.69 → **~245 events (≈4.9×) to halve the band** — lever
  is *data*, not model. Self-corrected a v1 bug: subsampling only positives confounded prevalence
  (band rose with events); fixed to subsample both. **Method→citation audit (Rule 1):** events/EPV
  COVERED (D7); **L1/elastic-net = GAP (D10)**, affordability features = GAP (D11), feature-selection
  under imbalance = PARTIAL (D12, 2 papers in auto_index uncurated). Experiment is PROVISIONAL until
  those papers are crawled + curated (awaiting approval). 3 tests (25 passing total).
- **2026-06-27 — Phase 5c: batch made the hero (use-case alignment).** User-driven reframe after
  the design question "batch vs single-field — which is the real use case?". Verdict: **batch is the
  operational unit** — the headline metric (recall@top-decile) is a *population* concept, so a single
  applicant has no decile; lending desks rank-and-route a pipeline, not type one form. Changes:
  `score_frame` now adds `rank` + `review_queue` = the riskiest decile **within the uploaded batch**
  (distinct from `in_riskiest_decile`, the absolute historical-threshold flag). UI reordered: ①
  batch review queue (hero, ranked table, highlighted queue) on top; ② single-application panel
  demoted to "explain / what-if" (adverse-action SHAP + sensitivity). `score-file` output now ranked
  + queue-flagged. Verified live (batch hero renders, ranked 99→19%, queue=top decile). 21 tests pass.
- **2026-06-27 — Phase 5b: batch scoring + demo/test data.** `serve.score_frame` / `score_file`
  + CLIs `python -m emerald_ai score-file <csv>` and `make-samples`, plus a CSV upload panel in the
  app (`/api/score-batch`). Two generated fixtures: `data/example_cases.csv` (5 curated
  in-distribution cases spanning the gradient — 5.8% → 45.5% → 77.7% → 91.3% → 99.3%) and
  `data/sample_applicants.csv` (50 **privacy-safe synthetic** rows: each column resampled
  independently from its real marginal, so no real record is reproduced; raw data is git-tracked, so
  this matters). 3 batch tests.
- **2026-06-27 — Phase 5: proof-of-concept decision-support demo.** `emerald_ai/serve.py` (FastAPI +
  minimal single-page UI), `python -m emerald_ai serve`. Serves the frozen class-weighted LR on the
  17 leakage-safe pre-funding features. Per applicant returns: **P(default) from `predict_proba`
  (never a 0.5 yes/no)**, a **riskiest-decile flag** (operating threshold P≥0.617 set from
  out-of-fold scores — OOF catch-rate 62% of 50 defaults), and **top-3 SHAP reasons** aggregated
  back to named features (exact linear SHAP). UI states honestly: model *ranks for review*, does not
  approve/decline. Numeric fields show typical p10–p90 range (extreme out-of-distribution inputs
  saturate the linear model — a documented limitation). `fastapi`/`uvicorn` added to requirements.
  5 scoring-contract tests (18 passing total).
- **2026-06-10 — Phase 4: calibration + conformal + SHAP (RQ2/RQ3).** `emerald_ai/calibrate.py`
  (`calibrate`) + `emerald_ai/explain.py` (`explain`). **RQ2:** Platt/isotonic fix *marginal* Brier
  (0.122→0.012) but **worsen within-minority ECE (0.35→0.97)** — the two calibration objectives
  conflict at 50 events (validates proposal §5.5). Split-conformal coverage exactly nominal
  (0.90→0.90, 0.95→0.951), near-vacuous as framed. **RQ3:** SHAP ranks Revenue #1 (corroborates
  single-feature finding), explanations faithful (linear-SHAP). D6 calibration citation found
  (Niculescu-Mizil & Caruana 2005) — PROPOSED, awaiting curation approval.
- **2026-06-10 — Cleaning integrated + sensitivity analysis.** User approved wiring `clean()` into
  the modelling path (`data.build_target(clean=True)` default). New `python -m emerald_ai sensitivity`
  → `reports/sensitivity_cleaning.md`. **RQ1 no-winner conclusion ROBUST to cleaning (LR 0.117 ≥
  XGBoost 0.093–0.105).** Honest correction: cleaning does NOT rescue Credit Score (0.0104→0.0104,
  only 2 rows) — the feature is genuinely weak here; Revenue (0.065) is the workhorse. Reports
  regenerated on cleaned basis.
- **2026-06-10 — Data-quality cleaning module (additive).** `emerald_ai/clean.py` +
  `python -m emerald_ai clean-report` → `reports/data_quality.md`. Rule-based, leakage-safe
  correction of 54 impossible values (Credit Score=0 ×2; Time In Business negative ×1 / >600mo ×51).
- **2026-06-10 — Evidence-grounded methodology audit.** Rule 1: `docs/methods_citations.md`
  (every imbalance choice → paper); 3 gaps (PR-vs-ROC, repeated CV, EPV/small-n) closed by
  patching crawler seeds and re-crawling (179→292 papers; Saito&Rehmsmeier 2015, Peduzzi 1996,
  Riley 2020, Krstajić 2014, Kapoor&Narayanan 2023). Rule 2: `emerald_ai/validation.py` +
  `python -m emerald_ai evidence` → `reports/learning_evidence.md`. **Permutation test p=0.010
  (real beats 100% of nulls); stable across seeds (0.091±0.002); plateaus by 75% (supports LR).
  HONEST LIMITATION: Brier 0.124 ≫ base-rate 0.0127 — class-weighted probs miscalibrated → Phase 4.**
  Gap report: `docs/gap_report.md`. ✅ user approved promoting all 11 papers → `index.yaml`
  (curated=11, auto=281); `methods_citations.md` marked CURATED.
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
The empirical core (Phases 1–5) is complete: model, calibration, explanations, and a working demo.
Remaining before write-up: (1) the **light fairness/robustness audit** — mostly a documented
*non-estimability* result (0 estimable cells from Phase 1), plus the robustness checks already in
`evidence`/`sensitivity`; (2) **Phase 6 write-up & release** — finalise dissertation chapters, add
demo screenshots, pin `requirements`, datasheet, tag the repo, and pass the clean-checkout
reproduction gate.
