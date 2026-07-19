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
| Fairness/robustness audit (light) | ✅ done (documented non-estimability, now §4.6 of the draft) |
| Phase 6 — write-up + release | 🟨 **IN PROGRESS** — full first draft built (`docs/dissertation/`); polish + release gate remain |

## Done (most recent first)
- **2026-07-17 — Polish loop: plain-language pass + contribution framing + Appendix E; round-3
  review launched.** Per user request to reduce hard vocabulary while keeping required technical
  terms: simplified ornate wording across all chapters (near-vacuous→"almost no information",
  trebles→triples, straddling→"span", "manufacture results"→"produce meaningless numbers",
  "governing fact"→"central fact", "flatters"→"makes look better than it is", "anti-conservative"
  glossed in plain words). Strengthened the contribution framing: §7.2 now invokes Hand (2006,
  *Classifier Technology and the Illusion of Progress*) to present methodological-rigour-as-
  contribution as a positive stance, not an apology for N=50. Fixed the last two places that led
  with the "19% vs 0.5-cut" strawman (§7.2 central claim, §1.4 contribution 5) — both now frame
  the decision layer as "cost-sized queue depth; clearly better than 0.5, modestly better than
  the decile rule", consistent with §4.6/§5.4. Ran the demo (`serve`) to verify it works end-to-
  end (all 4 endpoints, model trains, P>=0.617 operating point) — screenshot-capture tool was
  unavailable, so Appendix E is updated to the verified-working description with a precise
  capture instruction for the student. Every embedded figure path re-verified after the
  restructure. Round-3 adversarial review (both reviewers) launched on the WMG + plain-language
  version — **both returned "internally sealed on substance" / "Minor revision at the accept
  boundary"**. All findings fixed same session: §3.12 test count 44→49, §7.3 projection ref
  §4.3→§4.4, §3.7 stranded-"is" grammar, §2.6 pointer Chapter 5→§4.8, abstract 6.8% hedge, and
  the highest-leverage item — §7.2 now demarcates replication (imbalance-harms-calibration, EPV
  ceiling — confirms prior art) from the three original contributions (joint characterisation at
  N=50 on a real green-loan book; two entry-condition-tested non-estimability verdicts; pricing
  declined claims in events ~200–250). docx rebuilt (11,985 words); 49 tests pass. Full ledger:
  `docs/dissertation/REVIEW_ROUND1.md`.
  **Professional-formatting pass (same day):** `build.py` upgraded to emit A4, 2.54 cm margins,
  Times New Roman 12 / 1.5 line spacing / justified body, a bold heading hierarchy, italic centred
  captions, roman front-matter + arabic body page numbering (two Word sections), and
  `updateFields=true` (Word auto-populates TOC/lists/page numbers on open — no manual F9).
  Appendix B ethics-waiver rationale drafted (secondary/anonymised/no-participants; approval
  reference left blank). Demo re-verified working end-to-end; screenshot-capture tool unavailable
  in this environment (Appendix E left with a precise capture instruction). Core word count
  chapters 1–7 ≈ 10,324 + abstract 457 — under a typical 15k MSc limit. Remaining items all need
  the student (ethics certificate, approval ref, screenshots, name/ID) — none fabricable.
- **2026-07-18 — Strict WMG-template fidelity: build.py now populates the actual template file.**
  Rather than build a lookalike, `build.py` now opens `24-25_wmg_ft_msc_dissertation_template.docx`
  as the base document, clears its placeholder body, and pours the content in using the template's
  OWN named styles — so the output inherits WMG's real theme: blue headings (2E74B5), "Alt
  Heading 1" front matter, "Title" style title page, the template's fonts/margins/section setup.
  `build_from_template` is now the default (was build-from-scratch); `--scratch` keeps the
  house-style fallback. Fixed two template-specific issues found in verification: (a) the template
  lacks "List Bullet"/"List Number"/"Light Grid Accent 1" styles → added resilient fallbacks
  (`_list_para`, table-style try-list); (b) the template wraps its original Table of Contents in a
  `<w:sdt>` content control that the first `_clear_body` missed → it now strips every non-sectPr
  body child, so no duplicate TOC and no leftover template placeholder text survives. Verified:
  26 body fields (3 TOC + 23 SEQ captions), single main TOC, 2 sections (roman front / arabic
  body), updateFields on open, 7 chapters + References + Appendices, 8 tables. Both build modes
  confirmed working via temp builds. (Real dissertation.docx write pending — file was locked open
  in Word at build time; re-run `build.py` with Word closed.)
- **2026-07-14 — D19–D23 curated (9 papers, per supervisor recommendation) + WMG template
  restructure begun.** User approved the round-2 literature shortlist minus the King & Zeng
  software-paper variant (Firth 1993 is the rare-events anchor instead): curated 23 → 32.
  New citations written into the draft: §2.4 RQ2 prior art (van den Goorbergh 2022; Wallace &
  Dahabreh 2013; Van Calster 2016 calibration hierarchy), §2.3 Firth, §2.2 CV-comparison
  inference (Nadeau & Bengio 2003; Dietterich 1998), §2.8 Hand 2006, §2 opening green-domain
  note (Bonacorsi et al. 2024; Agosto, Cerchiello & Giudici 2023 — with the honest statement
  that no direct green-loan default literature exists); §5.2's placeholder attribution replaced
  with real citations; references 27 → 36 entries; D19–D23 rows added to
  `methods_citations.md` (**all decisions D1–D23 citation-closed**). User also supplied the
  official WMG dissertation template (`24-25_wmg_ft_msc_dissertation_template.docx`) —
  restructure to its required order — **DONE**. Chapters now: 1 Introduction, 2 Literature
  Review, 3 Methodology, 4 Results, 5 Analysis, 6 Discussion, 7 Conclusion (Results/Analysis/
  Discussion split into three; each of Ch2–6 opens with Introduction and closes with Summary).
  Files renamed 03_data_and_methodology→03_methodology, 05_discussion split into 05_analysis +
  06_discussion, 06_conclusion→07_conclusion, 07_references→08_references. Appendices
  restructured to WMG order: A ethics-training evidence, B ethical-approval/waiver (placeholders),
  C feature set, D reproduction, E demo, F supplementary figures. Full cross-reference sweep done
  (all §x.y / Chapter / Appendix refs remapped; verified no stale refs remain). `build.py`
  rewritten for the WMG template: submission pro-forma, title page, verbatim declaration,
  abstract, acknowledgements, auto TOC + List of Tables + List of Figures (Word fields — F9 to
  populate), auto SEQ captions above tables / below figures, footer page numbers. Structural
  build verified programmatically (9 H1 headings, 23 captions/SEQ, 3 TOC fields, footer PAGE).
  Note: pandoc/LibreOffice absent on this machine — python-docx fallback is the production path.
- **2026-07-12 — Review round 2 closed: examiner "internally sealed", adversarial reviewer
  Major→Minor.** Both reviewers re-audited the revised draft. Both independently caught the same
  process failure: the round-1 ledger claimed the "pre-registered" sweep complete when 4
  instances survived (abstract included) — fixed and grep-verified. Two new computations:
  **(1) `behind` leave-one-out** (`followup_checks.md` §4): PR-AUC 0.117→0.099 (inside fold
  band), recall unchanged — nothing turns on the undocumented n=1 case; **(2) vs-decile
  bootstrap** (`decision_policy.md`): **6.8% [1.2, 19.4]** — above zero (reviewer predicted it
  would cover zero) but modest + location-optimistic; abstract/§4.5/§5.3 restated at the right
  size ("queue-depth derivation is the durable contribution; the cost edge is the bonus").
  Abstract brought up to the body's standard (decile context, cluster-level driver claim,
  confidence-shortfall wording, "delinquency-detection"); "early" defined at §1.3;
  location-bias phrasing fixed ("true saving may lie below the quoted bounds"). Full round-2
  ledger appended to `docs/dissertation/REVIEW_ROUND1.md`. docx rebuilt; 49/49 tests.
  **Open:** D19–D23 curation approval, Appendix C screenshots, Warwick format check.
- **2026-07-12 — D6+D18 curated + follow-up checks module (all review-round-1 items closed).**
  User approved D6 (Niculescu-Mizil & Caruana 2005) + D18 (Lundberg & Lee 2017) → curated 21→23;
  **all decisions D1–D18 now citation-closed.** New `emerald_ai/followup.py` +
  `python -m emerald_ai followup-checks` → `reports/followup_checks.md`, answering the three
  reviewer challenges with data: **(1) paired fold-level RQ1 test** — baseline wins only 14/25
  shared folds, sign-test p≈0.69, difference bands straddle zero → no-winner CONFIRMED under the
  sharper instrument; **(2) reason-code stability** — mean pairwise Spearman 0.89 (min 0.80),
  Revenue top-3 in 100% of folds, affordability cluster rank-1 in 60% → global ordering not a
  seed artefact, cluster-level claim retained; **(3) population diagnostic** — terminal-outcome
  threshold flags 16.4% of the full 14,135-row book vs the intended 10% (1.6×, KS 0.065) → C8
  mismatch quantified. Draft updated (§4.7, §5.1, §5.4, §5.6, §6.3); 5 new tests (49 total).
  Literature round 2 crawled (52 seeds, 1,062 auto): 8 target papers found (van den Goorbergh,
  Wallace & Dahabreh, Van Calster, King & Zeng, Firth, Nadeau & Bengio, Dietterich, Hand) +
  2 green-domain candidates — **shortlist awaiting curation approval (D19–D23)**.
- **2026-07-12 — Review round 1 (two adversarial reviewers) + same-day revision pass.** Two
  independent reviews of the first draft: an external-examiner audit (3 FATAL, 10 MAJOR — numbers
  faithful to reports, but SHAP uncited, demo threshold untraceable, Brier inconsistency) and a
  top-tier adversarial review (verdict: Major revision — within-minority ECE near-tautological as
  "calibration", 0.5-cut a strawman baseline, 245-event fit from 3 points, funded-book range
  restriction and label/deployment-population mismatch unaddressed). All findings verified
  against code before action; one corrected (bootstrap DOES re-select threshold per resample).
  ~25 fixes applied across all chapters; `decision.py` now reports the demo operating point
  (0.617/62%) and the top-decile baseline (**cost-optimal beats decile rule by only 3.9% at
  R=20 — quoted honestly**); `clean.py` stale status fixed; Lundberg & Lee 2017 crawled and
  cited (D18 [PROPOSED]). Full ledger: `docs/dissertation/REVIEW_ROUND1.md`. Rebuilt docx;
  44 tests pass. **Open:** D6+D18 curation approvals, 3 queued analyses (paired RQ1 test,
  reason-code stability, score-distribution diagnostic), literature round 2, screenshots.
- **2026-07-12 — Phase 6 started: full dissertation first draft.** `docs/dissertation/` created —
  9 Markdown chapters (abstract, introduction, literature review, data & methodology, results,
  discussion, conclusion, references, appendices) + `build.py` (pandoc → python-docx fallback;
  pandoc absent on this machine, fallback verified). First complete draft ≈8,200 words, 8 tables,
  15 embedded figures, compiled to `dissertation.docx`. Structure per the agreed reading order:
  infeasibility results (fairness, survival) presented **in Results** as findings, not buried in
  limitations; no separate implementation chapter (demo = 1 section of Methodology + Appendix C);
  "rigour is the contribution" threaded from §1.3 through §6.2. Every number traceable to a
  generated report; citations restricted to the 21 curated papers + proposal BIB entries.
  Discussion includes the detection-bound argument (band ~0.19 vs plausible effect ~0.03) and the
  path-to-production scope split. **Remaining for Phase 6:** demo screenshots (Appendix C
  placeholder), D6 curation approval (Niculescu-Mizil & Caruana 2005 — cited in draft, flagged),
  Warwick word-limit/format check, pin requirements, datasheet, clean-checkout reproduction gate,
  repo tag. Prose polish passes to follow.
- **2026-07-10 — Path-to-production write-up + serve.py hardening.** User asked to "upgrade the app
  and prediction to production level" via more literature support. Pushed back on the framing first:
  no citation fixes N=50 events — statistical production-readiness (out-of-time validation, external
  validity, minority calibration) is bounded by sample size, not literature coverage. Split into two
  achievable tracks. **(A)** `docs/path_to_production.md` (new) + D15–D17 in `methods_citations.md`
  — ✅ **user approved same day; 4 papers promoted to `index.yaml` (curated 17 → 21)**:
  D15 model risk in ML credit scoring (Alonso-Robisco &
  Carbó Martínez 2022, `W4285089594`), D16 monitoring — concept drift (Gama et al. 2014,
  `W2099419573`) + Population Stability Index (Yurdakul & Naranjo 2020, `W3129493035`), D17
  regulatory explainability grounding the shipped SHAP `top_reasons` (Wachter, Mittelstadt & Floridi
  2017, `W3124443940`). Explicit scope note: this document does NOT claim statistical
  production-readiness. New `research_bot/seeds.yaml` theme `production_deployment` (877 unique
  papers fetched, 560 new -> auto_index, 863 total). **(B)** `emerald_ai/serve.py` hardened: stdlib
  `logging` (structured, never logs raw applicant values), clean `400`/`500` JSON errors instead of
  raw tracebacks on malformed uploads (bad extension/size/parse/no recognised columns), a global
  exception handler as a last-line safety net, and an optional static `EMERALD_API_KEY` header gate
  on `/api/*` (default-open, zero-config for local/grading use). 10 new tests in `test_serve.py`
  (34→44 total, all passing). Verified live via preview tooling: malformed upload → clean 400 JSON
  (not a traceback), normal scoring still 200 unauthenticated by default, structured log lines
  observed for every request.
- **2026-06-29 — Web batch upload accepts the raw Excel dataset + vectorised scoring.** New
  `/api/score-upload` endpoint (multipart, FastAPI `UploadFile`) takes a **CSV or .xlsx** file —
  the raw `All_Funded_2019_Green Loan.xlsx` can be dropped in as-is (permitted columns only; the
  other 140+ columns and the label are ignored). `score_frame` rewritten **vectorised**: the whole
  batch is transformed/predicted in one pass with exact linear SHAP aggregated per row — the full
  14,135-row book scores in **~0.07s** (was ~37s row-by-row). Summary spans the file; the table
  returns the riskiest 200. Fix: moved FastAPI type imports to module level so `UploadFile`/`Request`
  annotations resolve under `from __future__ import annotations` (same root cause as the earlier
  `Request`-as-query bug). `python-multipart` added to requirements. New TestClient test (34 total).
- **2026-06-29 — Friendly `top_reasons`.** Plain-English labels ("Monthly revenue"), ↑/↓ arrows +
  "raises/lowers risk", thousands separators — in the single panel, batch table, and exported CSV.
- **2026-06-29 — Cost-sensitive decision layer (option b) — WORKS.** `emerald_ai/decision.py` +
  `python -m emerald_ai decide` → `reports/decision_policy.md`. Picks the review threshold that
  minimises `R·FN + FP` (R = cost of missed default ÷ needless review) on OOF scores — not 0.5, not
  a fixed decile. Behaves correctly: as R rises (5→100) the threshold falls (0.998→0.426), the queue
  grows (27→816) and recall climbs (0.10→0.82). **At R=20: flags 319 vs 605 (0.5-cut), catches 29/50,
  expected cost −16% vs 0.5; bootstrap saving 19.0% [7.9, 32.8] — interval above zero, ROBUST at 50
  events.** Improves *decisions*, not PR-AUC. Honest caveats: R unknown (report a range, not one
  threshold); probs miscalibrated so threshold chosen empirically not analytic 1/(1+R). Citations:
  cost-sensitive COVERED (D5 Xia 2017); expected-cost threshold (Elkan 2001) a GAP, 2 uncurated
  candidates in auto_index. 4 tests (32 total).
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
The first full dissertation draft exists (`docs/dissertation/`, build verified). Remaining for
Phase 6: (1) polish passes on the prose (chapter by chapter, against the Warwick word limit and
format rules — check the handbook); (2) capture demo screenshots into Appendix C; (3) close D6
(the one remaining [PROPOSED] citation — Niculescu-Mizil & Caruana 2005, already cited in the
draft); (4) release gate — pin `requirements`, datasheet, tag the repo, clean-checkout
reproduction of every figure.
