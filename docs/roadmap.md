# EMERALD-AI — Project Roadmap (MSc scope)

**Status:** draft v2.0 · 2026-06-09 · supervisor-authored · *re-scoped to MSc dissertation size*
**Binding constraint:** 50 delinquent events (49 `default` + 1 `behind`) in 14,135 rows — **0.36%**.

**Scoping rule for this document:** an MSc dissertation is marked on *one question answered with
depth and honesty*, not on technique count. Anything not essential to that is moved to **Stretch**
and only attempted if the Core is finished. At N=50, breadth lowers the mark; depth raises it.

**Framing note (carry into the write-up):** the contribution of this dissertation is the
*methodological rigour*, not the technique inventory. The imbalance-aware CV protocol, the
within-minority ECE with CIs, the cohort-exposure censoring analysis, the pre-registration of
metrics, and the *documented non-estimability* of group fairness are the intellectual result —
they must be presented as findings, not buried as plumbing. "2 models + SHAP" is only thin if the
rigour is hidden.

---

## The dissertation in one sentence
Build and *honestly* evaluate a calibrated, explainable early-delinquency model for funded green
loans under extreme imbalance (0.36%), shipped as a proof-of-concept decision-support demo — with
explicit treatment of what 50 events permit and forbid.

## Research questions (cut to three)
- **RQ1.** Can a gradient-boosted model beat a regularised-logistic-regression baseline on
  early-delinquency detection (PR-AUC, recall@top-decile) *beyond bootstrap-CI overlap* at 0.36%
  prevalence — and if not, is that itself the finding?
- **RQ2.** Does post-hoc calibration meaningfully improve **within-minority** reliability, given
  only ~10 minority cases in the calibration split?
- **RQ3.** Do SHAP explanations yield a coherent, lender-legible account of individual decisions,
  and where does the small event count make group-level fairness claims non-estimable?

---

## Imbalance playbook (the one idea that runs through every chapter)
| Stage | Decision |
|---|---|
| **Label** | `paidOff`-only is **primary** (drops 72% censored `current` rows; prevalence ~1.3%). All-favourable reported once as an optimistic comparator. `behind` (N=1) ignored analytically. |
| **Split** | **Repeated stratified 5-fold** (×5 repeats) so each test fold holds ~10 positives. Report the *distribution* across repeats, not a single number. Nested only where HPO needs it. |
| **Resampling** | Class weighting (`scale_pos_weight` / `class_weight`) as default; **SMOTE-within-fold** as the one comparator. Strictly inside folds [Santos 2018]. No other resamplers. |
| **Metrics** | **PR-AUC, recall@top-decile, within-minority ECE** — each with bootstrap 95% CIs. Accuracy banned (constant predictor = 99.64%). |
| **Claims** | Null results ("no significant winner") are publishable findings, stated plainly. |

---

## Core plan (~14 weeks)

### Phase 1 — Data understanding & feasibility · Weeks 1–3
*The chapter that makes or breaks the viva.*
- Reproducible EDA: prevalence & defaults **by origination month** (censoring/exposure), missingness
  map (Term 86.4% / APR 59.6% / Factor 42%), leakage audit → locked feature set.
- **Fairness feasibility check:** event counts per Industry / State / size group. This *decides*
  whether RQ3's fairness half is a real audit or a documented "non-estimable at this N".
- **Deliverable:** `reports/feasibility.md` + figures. **Gate:** primary label confirmed,
  feature set frozen, metrics pre-registered, fairness go/no-go decided.

### Phase 2 — Preprocessing pipeline · Weeks 3–5
- One leakage-safe `scikit-learn` Pipeline: drop >40%-missing, impute + missing-indicators,
  winsorise, encode (fit inside folds), scale for LR only. Light, sensible feature engineering.
- **Deliverable:** `pipeline.py` + a CI-free automated leakage assertion (no post-funding field
  reachable). **Gate:** pipeline passes the leakage check.

### Phase 3 — Modelling · Weeks 5–8
- **Two core models:** regularised **logistic regression** (baseline) + **XGBoost** (primary,
  monotonic constraints on Credit Score / Annual Revenue / Time in Business). *LightGBM optional.*
- Imbalance: class-weighting vs SMOTE-within-fold, HPO via a modest search (RandomizedSearchCV is
  fine — full Bayesian nested HPO is overkill here).
- **Deliverable:** results table with bootstrap CIs across repeats. **Gate (RQ1):** does XGBoost
  beat LR beyond CI overlap? Record the honest answer either way.

### Phase 4 — Calibration & explainability · Weeks 8–11
- Calibration: Platt **or** isotonic, chosen on a held-out split; reliability diagram; within-minority
  ECE with bootstrap CIs (caveated — high variance at N≈10). Answers RQ2.
- **Split-conformal (Core-light):** one MAPIE call producing marginal-coverage prediction sets,
  reported as an *honest small-N transparency artefact* — "marginal coverage is near-vacuous at
  0.36%, and here is why." This is the technique that most directly embodies the thesis, so it
  stays in Core despite the trim. Mondrian/class-conditional coverage only as a footnote diagnostic.
- Explainability: **SHAP only** — global beeswarm + local waterfall for sample decisions.
- Fairness: light descriptive audit (calibration / error rates across the *one or two* groups that
  cleared Phase 1's count gate), framed honestly; non-estimable groups reported as such. Answers RQ3.
- **Deliverable:** calibration + interpretability chapter, model card.

### Phase 5 — Proof-of-concept demo · Weeks 11–13
- **FastAPI** endpoint serving the model + a SHAP explanation, with a **minimal** UI (single-page
  form → score + "top reasons" panel). This is a *demo of decision support*, not an MLOps stack.
- **Deliverable:** `python -m emerald_ai.serve` runs locally + screenshots in the dissertation.

### Phase 6 — Write-up & release · Weeks 13–14 (+ buffer)
- Dissertation finalised; every figure regenerated from a seeded script; pinned `requirements`,
  README, datasheet. Tag the repo. **Gate:** clean checkout reproduces the headline figures.

---

## Stretch (only if Core is genuinely finished — never at its expense)
- A third model (CatBoost) or a one-shot deep-learning **negative control**.
- LIME / DiCE counterfactuals; Quantus fidelity validation.
- MLflow run tracking; DVC data versioning; React frontend instead of the minimal UI.
- Temporal H1/H2-2019 generalisation test.

## What I cut from v1 and why
| Cut | Reason |
|---|---|
| 8 models → 2 (+1 optional) | At 50 events, comparing 8 models is noise theatre; 2 done well shows more rigour. |
| Nested Bayesian HPO → Randomized search | Inner-loop Bayesian HPO is unjustifiable tuning on ~10 positives/fold. |
| Conformal, Quantus, LIME, DiCE → Stretch | Each is a mini-project; none is *expected* of an MSc; SHAP + calibration suffice. |
| Full AIF360 4-metric audit → light descriptive | The counts don't support a full group-conditional audit (Phase 1 confirms). |
| MLflow + DVC + Prometheus + Evidently → none/stretch | Production MLOps is PhD/industry surface area, not MSc marking criteria. |
| FastAPI + React → FastAPI + minimal UI | A working demo earns the practical marks; a polished SPA does not earn proportionally more. |

## Top risks
1. **Underpowered comparison** → accept that "no significant winner" is a valid RQ1 answer.
2. **Censoring** → `paidOff`-only primary + cohort analysis (Phase 1).
3. **Scope creep back up** → the Stretch list is a fence, not a to-do list.
