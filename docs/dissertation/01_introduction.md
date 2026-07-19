# 1. Introduction

## 1.1 Context and motivation

Green lending — credit extended to small businesses for environmentally beneficial purposes — has
expanded faster than the evidence base needed to underwrite it. A lender entering this market
inherits a familiar problem in an unfamiliar setting: it must estimate the probability that a
funded loan becomes delinquent, using a book that is young, small, and overwhelmingly composed of
loans that have not (yet) gone wrong. The portfolio studied here is typical of the situation. It
contains 14,135 funded green loans originated principally in 2019, of which only 50 ever became
delinquent — 49 recorded as `default` and one as `behind`. Every substantive question this
dissertation asks is downstream of that single number.

Class imbalance of this severity changes what counts as a competent analysis. A classifier that
predicts "no default" for every applicant achieves 99.64% accuracy on the raw book while being
operationally useless; ordinary model comparisons are lost in sampling noise; probability
estimates inflated by imbalance corrections cannot be taken at face value; and subgroup analyses
— fairness audits included — silently run out of data. The machine-learning literature offers a
large toolbox for imbalanced classification, but most of it was validated on datasets with
hundreds or thousands of minority instances. Whether any of it *does anything measurable* at 50
events is an empirical question, and it is the question this dissertation answers.

## 1.2 The problem, stated honestly

The practical task is delinquency detection for decision support: rank incoming
applications by risk so that a lending desk can review the most dangerous cases first. The
scientific task is to establish, with defensible uncertainty quantification, which claims about
such a model the data can support. These tasks are deliberately separated throughout. A model can
be operationally useful (its top decile catches a large share of defaults) while remaining
scientifically indistinguishable from a simpler competitor (its PR-AUC confidence band overlaps
the baseline's). Conflating the two produces the overclaiming this dissertation is designed to
avoid.

## 1.3 Research questions

Three research questions were fixed before modelling began, together with the metrics and
uncertainty conventions used to answer them:

- **RQ1.** Can a gradient-boosted model beat a regularised-logistic-regression baseline on
  early-delinquency detection (PR-AUC, recall at the top decile) *beyond bootstrap-CI overlap* at
  0.36% prevalence — and if not, is that itself the finding?
- **RQ2.** Does post-hoc calibration meaningfully improve **within-minority** reliability, given
  only around ten minority cases in any calibration split?
- **RQ3.** Do SHAP explanations yield a coherent, lender-legible account of individual decisions,
  and where does the small event count make group-level fairness claims non-estimable?

A terminological note on RQ1's "early-delinquency detection", which is quoted verbatim from the
project plan: "early" refers to the *decision point* — risk assessed at application time, before
funding, from pre-funding features only — not to a time-to-event claim; the label records
whether a loan ever became delinquent and carries no timing information (a limitation §6.3
develops). The phrasing of RQ1 and RQ3 is deliberate: both admit a negative or non-estimable
answer as a legitimate result. This reflects the dissertation's central methodological stance — at N = 50,
the honest reporting of what cannot be estimated is as much a contribution as any estimate.

## 1.4 Contributions

The dissertation makes five contributions:

1. **A leakage-audited, imbalance-aware evaluation protocol** for extreme-imbalance credit
   scoring: default-deny feature vetting (17 of 166 columns admitted), all data-dependent
   transformations confined to training folds, repeated stratified cross-validation, and a
   pre-specified metric panel with uncertainty intervals throughout (Chapter 3).
2. **An honest answer to the model-choice question** (RQ1): no significant winner between
   logistic regression and XGBoost, established with fold-percentile bands, a label-permutation
   test, seed-stability checks and a learning curve — plus a sample-size projection showing that
   roughly 200–250 events (4–5× the current count) would be needed to halve the uncertainty band
   (Chapters 4–5).
3. **A quantification of the structural conflict between marginal calibration and confidence on
   the minority** (RQ2): Platt and isotonic recalibration repair the marginal Brier score while
   collapsing the model's confidence on the defaults themselves — an inherent trade-off at low
   prevalence that this dissertation measures, with disjoint confidence intervals, at its most
   extreme operating point (Chapter 4).
4. **Two documented non-estimability results** — group-conditional fairness (no industry or state
   cell reaches ten events) and survival modelling of the censored book (no trustworthy
   time-to-event clock exists in the data) — reported with the evidence, as findings (Chapter 4).
5. **A decision layer and deployable artefact that the data does support**: an expected-cost
   review threshold that sets the review-queue depth from a stated cost ratio rather than a fixed
   cutoff — clearly better than a naive 0.5 cut and modestly better than a simple top-decile
   rule — served through a hardened FastAPI decision-support application with per-applicant SHAP
   reason codes, together with a literature-grounded account of the governance, monitoring and
   regulatory work that separates this proof of concept from a production system (Chapters 4
   and 6).

## 1.5 Scope and limitations, up front

Three boundaries are declared at the outset. First, the analysis concerns a single lender's
funded book, originated in a narrow window; nothing here estimates out-of-time or out-of-lender
generalisation, and no such claim is made. Second, only funded loans are observed, so the model
ranks *funded* applications; reject inference is out of scope by construction, since rejected
applicants do not appear in the data. Third, the deployed artefact is a proof of concept for
decision support — it ranks applications for human review and explicitly does not approve or
decline.

## 1.6 Structure of the dissertation

Chapter 2 reviews the literature by theme: learning under class imbalance, evaluation and
validation under imbalance, small-sample constraints, probability calibration, explainability and
its regulatory context, and model risk in deployed credit models. Chapter 3 describes the data,
the label construction and its censoring rationale, the leakage audit, and the experimental
protocol. Chapter 4 reports results in the order of the research questions, followed by the
decision layer and the two non-estimability findings. Chapter 5 analyses those findings —
reading the null correctly, the calibration conflict, decisions over discrimination, and
non-estimability as a finding. Chapter 6 discusses what the system is and is not, grounding the
path-to-production requirements in the literature and stating the study's limitations. Chapter 7
concludes and sets out the data acquisitions that would unlock the analyses this dataset forbids.
