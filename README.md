# EMERALD-AI

Explainable, calibrated credit-risk modelling for **funded green loans**, delivered as an MSc
Applied AI dissertation (University of Warwick) plus a proof-of-concept decision-support demo.

> **Binding constraint:** the dataset has **50 delinquency events in 14,135 loans (0.36%)**.
> The whole project is engineered around that small *event count* — not the ratio. No resampling
> creates information 50 events do not contain. See `docs/roadmap.md`.

## Explore this project step by step

New here? Walk it in this order — each step produces something you can open and read. Commands are
Windows-first (`python -m ...`); they work on any OS.

```powershell
pip install -r requirements.txt   # 0. set up
```

**Step 1 — Understand the plan and the constraint.**
Read [`docs/roadmap.md`](docs/roadmap.md). It explains the one fact everything hinges on: **50
delinquency events (0.36%)**, and how the project is scoped around it. Then skim
[`docs/progress.md`](docs/progress.md) to see exactly what is done and what is next.

**Step 2 — See the problem in the data (Phase 1).**
```powershell
python -m emerald_ai eda
```
Open [`reports/feasibility.md`](reports/feasibility.md): the imbalance, the 2015–2020 censoring,
and the proof that a group-fairness audit is *non-estimable* here (0 estimable cells).

**Step 3 — Check the data can't leak the answer (Phase 2).**
```powershell
python -m emerald_ai audit             # -> data/governance/feature_catalogue.yaml
python -m emerald_ai preprocess-check  # fits the leakage-safe pipeline, prints output shape
```
Default-deny: only 17 vetted *pre-funding* features may reach a model. `tests/test_preprocess.py`
proves post-funding fields (e.g. `Percent Paid`) can never get in.

**Step 4 — Find what works under the imbalance (Phase 3, answers RQ1).**
```powershell
python -m emerald_ai bakeoff
```
Open [`reports/model_bakeoff.md`](reports/model_bakeoff.md): LR vs XGBoost × class-weight vs SMOTE,
repeated stratified CV, metrics with fold bands. Honest verdict: **no significant winner**.

**Step 5 — See *why it is useful* (the visual story).**
```powershell
python -m emerald_ai figures
```
Open [`reports/visual_story.md`](reports/visual_story.md) — five figures from problem to proof,
ending in the cumulative-gains curve: **reviewing the riskiest 10% catches ~64% of all defaults**.

**Step 5b — Prove the model learns, not memorises (Rule 2 evidence).**
```powershell
python -m emerald_ai evidence       # permutation test, baselines, stability, learning curve
python -m emerald_ai sensitivity    # raw-vs-cleaned robustness check
```

**Step 5c — "Can we do better?" (RQ1 follow-up).**
```powershell
python -m emerald_ai improve        # L1-sparse + affordability ratios vs the events ceiling
```
Open [`reports/improvement.md`](reports/improvement.md): respecting events-per-variable (L1) and
adding domain affordability ratios is the **expected null** — the fold band is event-limited, not
model-limited. A fixed-prevalence projection estimates the events needed to halve the uncertainty.
Includes a **method→citation audit** that flags methods not yet backed by the literature brain
(penalised regression, financial-ratio features).

**Step 5d — Can we use the censored loans? (survival feasibility).**
```powershell
python -m emerald_ai survival-check  # is there a reliable time-to-event clock?
```
Open [`reports/survival_feasibility.md`](reports/survival_feasibility.md): a time-to-default model
could in principle recover the ~10k censored `current` loans the primary label drops — but the
dataset has **no trustworthy duration** (the two candidate clocks correlate −0.02; `paidOff` loans
sit at 31% median term-complete). Verdict: **survival analysis is NON-ESTIMABLE here** — a second
documented infeasibility, alongside the fairness audit.

**Step 6 — Calibration & explanations (Phase 4, answers RQ2/RQ3).**
```powershell
python -m emerald_ai calibrate      # Platt/isotonic + within-min ECE (CIs) + split-conformal
python -m emerald_ai explain        # SHAP global + local explanations
```
`reports/calibration.md` — calibration fixes *marginal* Brier (0.12→0.01) but **worsens
within-minority ECE**: the two objectives conflict at 50 events. `reports/explainability.md` —
SHAP confirms `Revenue` is the workhorse feature.

**Step 7 — See it with your own eyes (Phase 5, the decision-support demo).**
```powershell
python -m emerald_ai serve      # -> http://127.0.0.1:8000
```
A FastAPI app whose layout mirrors the real use case. **Primary path — batch review queue:** upload a
CSV of applications; the model ranks them by risk and flags the riskiest **decile *of that batch*** as
the review queue (the operating point behind "review the top 10%, catch ~62% of defaults"). The
headline metric is a population concept, so the queue is defined over the uploaded batch — *never* a
0.5 yes/no. **Secondary path — single-application panel:** decompose one decision into its **top-3
SHAP reasons** (the "why was this flagged?" answer for an adverse-action notice) or stress-test how
the score moves as a feature changes. Framed honestly: the model *ranks for review*, it does not
approve or decline.

Batch scoring from the command line (the natural path for volume):
```powershell
python -m emerald_ai make-samples            # -> data/example_cases.csv + data/sample_applicants.csv
python -m emerald_ai score-file data/sample_applicants.csv   # -> *_scored.csv, ranked, queue-flagged
```
`data/example_cases.csv` holds five curated, in-distribution demo applicants spanning the risk
gradient (≈6% → 99%); `data/sample_applicants.csv` is privacy-safe synthetic test data (each column
resampled independently from its real marginal — no row reproduces a real record). Output is ranked
by risk with a `review_queue` flag (within-batch top decile) and an `in_riskiest_decile` flag
(absolute historical threshold).

**Step 8 — (Optional) grow the literature brain.**
```powershell
python -m research_bot crawl    # OpenAlex -> literature/auto_index.yaml
python -m research_bot status
```

**Step 9 — Verify everything.**
```powershell
python -m pytest -q              # 28 tests: leakage guard, metrics, bot isolation, demo/batch, improve + survival feasibility
```

## What's here
| Path | What |
|---|---|
| `docs/roadmap.md` | MSc-scoped roadmap (v2.1): RQs, imbalance playbook, phase plan, decision gates. |
| `docs/progress.md` | **Living log** of every step done; newest first. Start here for status. |
| `emerald_ai/` | Analysis package + CLI (`__main__.py`). `config.py` (paths/seed/labels), `data.py` (label construction), `eda.py` (Phase 1), `feature_audit.py` + `preprocess.py` (Phase 2 leakage-safe), `experiments.py` + `metrics.py` (Phase 3 bake-off), `figures.py` (visual story). |
| `reports/feasibility.md` | **Generated.** Phase 1 imbalance/censoring numbers + figures. |
| `reports/model_bakeoff.md` | **Generated.** Phase 3 RQ1 results (metrics with fold bands). |
| `reports/visual_story.md` | **Generated.** Five-step figure narrative + the gains-curve proof. |
| `reports/learning_evidence.md` | **Generated.** Permutation test, baselines, stability, learning curve. |
| `reports/improvement.md` | **Generated.** RQ1 follow-up: L1-sparse + affordability ratios vs the events ceiling, with a method→citation audit (flags lit-brain gaps). |
| `reports/survival_feasibility.md` | **Generated.** Whether the censored loans support a time-to-event model — verdict: non-estimable (no reliable duration). |
| `reports/calibration.md` | **Generated.** Phase 4 RQ2: Platt/isotonic + conformal. |
| `reports/explainability.md` | **Generated.** Phase 4 RQ3: SHAP global + local. |
| `docs/methods_citations.md` | Every imbalance/calibration choice → a paper (evidence audit). |
| `data/governance/` | **Generated.** Leakage audit: `feature_catalogue.yaml` + `feature_audit_summary.md`. |
| `emerald_ai/serve.py` | **Phase 5 demo.** FastAPI decision-support app (`python -m emerald_ai serve`): batch review-queue (primary) + single-application explain/what-if panel (secondary) → ranked P(default), within-batch decile queue, top-3 SHAP reasons. Also `score-file` / `make-samples` CLIs. |
| `data/example_cases.csv` | **Generated.** Five curated in-distribution demo applicants (risk gradient ≈6%→99%) for the batch path. |
| `data/sample_applicants.csv` | **Generated.** 50 privacy-safe synthetic applicants (column-wise resample) for batch testing. |
| `research_bot/` | Small OpenAlex crawler (lit-review aid). `discovery.py` (queries), `state.py` (brain), `seeds.yaml`. |
| `literature/` | The literature brain: `index.yaml` (curated) + `auto_index.yaml` (**generated**, auto-discovered). |
| `tests/` | 28 tests: leakage guard, metric panel, bot isolation, demo/batch, improve + survival feasibility. |
| `All_Funded_2019_Green Loan.xlsx` | Raw dataset (14,135 × 166). |

## Literature bot
A deliberately small lit-review aid — **not** a dissertation contribution. It crawls OpenAlex for
papers on methods for imbalanced green-credit scoring and files them in `literature/auto_index.yaml`,
kept separate from the hand-vetted `literature/index.yaml`. Raw keyword discovery is noisy by
design; you promote the relevant hits into the curated index. Path constants live in **both**
`state.py` and `discovery.py` — tests must patch both to avoid leaking into the real brain
(`tests/test_research_bot.py`).

## Key Phase 1 findings (reproduce with `python -m emerald_ai eda`)
- Two label schemes: `paidoff_only` (**primary**, drops censored `current` rows → 1.28% prevalence)
  vs `all_favourable` (0.36%). **Event count = 50 under both.**
- **Censoring is real:** origination spans 2015–2020; 72.8% of the 2019 cohort is right-censored.
- **Group fairness audit is non-estimable:** 0 / 27 industries and 0 / 51 states reach ≥10 events.
  RO4 becomes a *documented non-estimability* result, not a full audit.

## Reproducibility
Single seed (`emerald_ai.config.SEED`), headless figures, every report number computed in code —
nothing hand-typed. Clean-checkout reproduction is a release gate (roadmap Phase 6).
