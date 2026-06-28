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
A single-page FastAPI app: fill the pre-funding application form and get **P(default)**, whether the
applicant falls in the **riskiest decile** (the desk's review queue — threshold set from out-of-fold
scores, *never* a 0.5 yes/no), and the **top-3 SHAP reasons** mapped back to named features. Framed
honestly in the UI: the model *ranks for review*, it does not approve or decline.

For **many applicants at once**, upload a CSV in the app's batch panel, or use the CLI:
```powershell
python -m emerald_ai make-samples            # -> data/example_cases.csv + data/sample_applicants.csv
python -m emerald_ai score-file data/sample_applicants.csv   # -> data/sample_applicants_scored.csv
```
`data/example_cases.csv` holds five curated, in-distribution demo applicants spanning the risk
gradient (≈6% → 99%); `data/sample_applicants.csv` is privacy-safe synthetic test data (each column
resampled independently from its real marginal — no row reproduces a real record).

**Step 8 — (Optional) grow the literature brain.**
```powershell
python -m research_bot crawl    # OpenAlex -> literature/auto_index.yaml
python -m research_bot status
```

**Step 9 — Verify everything.**
```powershell
python -m pytest -q              # 21 tests: leakage guard, metrics, bot path-isolation, demo + batch scoring
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
| `reports/calibration.md` | **Generated.** Phase 4 RQ2: Platt/isotonic + conformal. |
| `reports/explainability.md` | **Generated.** Phase 4 RQ3: SHAP global + local. |
| `docs/methods_citations.md` | Every imbalance/calibration choice → a paper (evidence audit). |
| `data/governance/` | **Generated.** Leakage audit: `feature_catalogue.yaml` + `feature_audit_summary.md`. |
| `emerald_ai/serve.py` | **Phase 5 demo.** FastAPI decision-support app (`python -m emerald_ai serve`): single-applicant form + CSV batch panel → P(default) + riskiest-decile flag + top-3 SHAP reasons. Also `score-file` / `make-samples` CLIs. |
| `data/example_cases.csv` | **Generated.** Five curated in-distribution demo applicants (risk gradient ≈6%→99%) for the batch path. |
| `data/sample_applicants.csv` | **Generated.** 50 privacy-safe synthetic applicants (column-wise resample) for batch testing. |
| `research_bot/` | Small OpenAlex crawler (lit-review aid). `discovery.py` (queries), `state.py` (brain), `seeds.yaml`. |
| `literature/` | The literature brain: `index.yaml` (curated) + `auto_index.yaml` (**generated**, auto-discovered). |
| `tests/` | 21 tests: leakage guard, metric panel, bot path-isolation, demo + batch scoring contract. |
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
