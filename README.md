# EMERALD-AI

Explainable, calibrated credit-risk modelling for **funded green loans**, delivered as an MSc
Applied AI dissertation (University of Warwick) plus a proof-of-concept decision-support demo.

> **Binding constraint:** the dataset has **50 delinquency events in 14,135 loans (0.36%)**.
> The whole project is engineered around that small *event count* — not the ratio. No resampling
> creates information 50 events do not contain. See `docs/roadmap.md`.

## Quickstart (Windows-first)
```powershell
pip install -r requirements.txt
python -m emerald_ai eda          # Phase 1 imbalance/censoring feasibility -> reports/feasibility.md
python -m research_bot crawl      # grow the literature brain from OpenAlex
python -m research_bot status     # curated vs auto-discovered counts
```

## What's here
| Path | What |
|---|---|
| `docs/roadmap.md` | MSc-scoped roadmap (v2.1): RQs, imbalance playbook, phase plan, decision gates. |
| `emerald_ai/` | Analysis package. `config.py` (paths/seed/labels), `data.py` (label construction), `eda.py` (Phase 1), `__main__.py` (CLI). |
| `reports/feasibility.md` | **Generated.** Real imbalance/censoring numbers + figures. |
| `research_bot/` | Small OpenAlex crawler (lit-review aid). `discovery.py` (queries), `state.py` (brain), `seeds.yaml` (imbalance + green-credit method queries). |
| `literature/` | The literature brain: `index.yaml` (curated) + `auto_index.yaml` (**generated**, auto-discovered). |
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
