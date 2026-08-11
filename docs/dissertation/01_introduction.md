# 1. Introduction

## 1.1 Motivation

Green lending — credit extended to small businesses for environmentally beneficial purposes — has
grown faster than the evidence base needed to underwrite it. Three forces explain the growth.
First, climate risk has become a recognised concern for financial stability: Battiston et al.
(2021) show that the exposure of bank portfolios to climate-sensitive assets is material enough
to matter for systemic risk, which pushes lenders to build books that are transparently aligned
with transition goals. Second, small firms attempting green investment face a documented
financing gap; Yu et al. (2021) find that demand for green finance is driven precisely by
financing constraints on green innovation, so a lender entering the market is meeting real,
unmet demand. Third, policy has actively steered credit towards green purposes — Yao et al.
(2021) trace how a national green-credit policy changed the financing and performance of the
firms it targeted. The result is a rapidly expanding category of loans that lenders are
underwriting with limited historical experience of how they perform.

That inexperience is the practical problem this dissertation addresses. A lender in this market
must estimate the probability that a funded loan becomes delinquent, using a book that is young,
small, and overwhelmingly composed of loans that have not (yet) gone wrong. The portfolio studied
here is typical. It contains 14,135 funded green loans, originated principally in 2019, of which
only 50 ever became delinquent — 49 recorded as `default` and one as `behind`. Every substantive
question this dissertation asks is downstream of that single number.

Class imbalance of this severity changes what counts as a competent analysis. A classifier that
predicts "no default" for every applicant achieves 99.64% accuracy on the raw book while being
operationally useless; ordinary model comparisons are lost in sampling noise; probability
estimates inflated by imbalance corrections cannot be taken at face value; and subgroup analyses
— fairness audits included — silently run out of data. The machine-learning literature offers a
large toolbox for imbalanced classification, but most of it was validated on datasets with
hundreds or thousands of minority instances. Whether any of it *does anything measurable* at 50
events is an empirical question, and it is the question this dissertation answers.

The difficulty is compounded by the borrower type. Small and medium-sized enterprises are harder
to score than consumers or large corporates: their financial reporting is thinner, their failure
processes are heterogeneous, and the predictor sets that work for them are still contested.
Ciampi et al. (2021), reviewing three decades of SME default-prediction research, report a field
that has not converged on a stable feature set or method; Altman et al. (2022) revisit SME
default predictors and find that models continue to need re-specification for this segment; and
Kou et al. (2020) turn to transactional data precisely because conventional SME financial ratios
are often unavailable. A green-loan book therefore combines three difficulties at once — a new
product, a hard-to-score borrower segment, and an extremely rare outcome.

There is also a professional motivation. A model that ranks credit applications is a model that
affects who receives finance. If it is deployed on the strength of a benchmark number whose
uncertainty was never quantified, the cost of that overconfidence is borne by borrowers who are
declined or delayed for no defensible reason, and by a lender who believes it is managing risk
that it is not. The discipline this dissertation applies — reporting what 50 events can and
cannot support — is therefore not academic fastidiousness but a condition of using such a model
responsibly at all.

## 1.2 Problem statement

The practical task is delinquency detection for decision support: rank incoming applications by
risk so that a lending desk can review the most dangerous cases first. The scientific task is to
establish, with defensible uncertainty quantification, which claims about such a model the data
can support. These tasks are deliberately separated throughout. A model can be operationally
useful (its top decile catches a large share of defaults) while remaining scientifically
indistinguishable from a simpler competitor (its PR-AUC confidence band overlaps the baseline's).
Conflating the two produces the overclaiming this dissertation is designed to avoid.

A second separation matters as much. The deployable unit here is not a single application but a
*batch*: the headline operating measure — the share of defaults captured in the riskiest decile —
is a population statistic, and a lending desk ranks and routes a pipeline of applications rather
than scoring one form at a time. The system built in this dissertation is therefore designed
around batch review, with single-application scoring retained for explanation and what-if
analysis rather than as the primary interface.

## 1.3 Research aim and objectives

**Aim.** To determine what a small, extremely imbalanced green-loan portfolio permits a lender to
claim about machine-learned delinquency risk, and to deliver the strongest decision-support
artefact those data genuinely support.

Six research objectives (ROs) operationalise that aim. Each is stated so that its success
criterion is independent of whether the result is positive or negative.

- **RO1 — Establish the analytical envelope.** Quantify the event count, prevalence and censoring
  structure of the portfolio, map missingness, and decide *before modelling* which analyses are
  feasible — in particular, whether group-conditional fairness metrics are estimable at this event
  count (Chapter 3, §4.2).
- **RO2 — Build a leakage-safe modelling pipeline.** Vet every candidate feature for
  post-funding information under a default-deny rule, and confine all data-dependent
  transformations to training folds, with an automated guard that fails the pipeline if a
  forbidden field becomes reachable (§3.4, §3.6).
- **RO3 — Test whether model choice matters at this sample size.** Compare a regularised
  logistic regression against gradient boosting under two imbalance treatments, using
  precision-recall-based metrics with fold-level uncertainty, and treat a null result as a
  legitimate answer (§4.3–§4.4; addresses RQ1).
- **RO4 — Assess the trustworthiness of the probabilities and explanations.** Evaluate post-hoc
  calibration and distribution-free coverage on the minority class, produce per-applicant
  explanations, and report where subgroup claims become non-estimable (§4.5, §4.7–§4.8;
  addresses RQ2 and RQ3).
- **RO5 — Convert the model into a defensible operating policy.** Derive a review threshold from
  an explicit cost ratio rather than an arbitrary cutoff, quantify its advantage against both a
  naive and an operationally realistic baseline, and ship it as a working batch decision-support
  application (§3.11, §4.6).
- **RO6 — Specify what production would require.** Set out, with reference to the model-risk,
  monitoring and regulatory literatures, the governance and monitoring work that separates this
  proof of concept from a deployable system, and price the additional data needed to answer the
  questions this dataset cannot (§6.3, §7.4).

## 1.4 Research questions

Three research questions were fixed before modelling began, together with the metrics and
uncertainty conventions used to answer them. RO3 addresses RQ1; RO4 addresses RQ2 and RQ3; RO1,
RO2, RO5 and RO6 supply the conditions under which the answers can be trusted and used.

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
whether a loan ever became delinquent and carries no timing information (a limitation §6.4
develops). The phrasing of RQ1 and RQ3 is deliberate: both admit a negative or non-estimable
answer as a legitimate result. This reflects the dissertation's central methodological stance — at
N = 50, the honest reporting of what cannot be estimated is as much a contribution as any
estimate.

## 1.5 Contributions

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
   rule — served through a hardened FastAPI batch decision-support application with per-applicant
   SHAP reason codes, together with a literature-grounded account of the governance, monitoring
   and regulatory work that separates this proof of concept from a production system (Chapters 4
   and 6).

## 1.6 Scope and limitations, up front

Three boundaries are declared at the outset. First, the analysis concerns a single lender's
funded book, originated in a narrow window; nothing here estimates out-of-time or out-of-lender
generalisation, and no such claim is made. Second, only funded loans are observed, so the model
ranks *funded* applications; reject inference is out of scope by construction, since rejected
applicants do not appear in the data. Third, the deployed artefact is a proof of concept for
decision support — it ranks applications for human review and explicitly does not approve or
decline.

## 1.7 Structure of the dissertation

Chapter 2 reviews the literature by theme: the green-lending and SME-credit context, learning
under class imbalance, evaluation and validation under imbalance, small-sample constraints,
probability calibration, explainability and its regulatory context, and model risk in deployed
credit models. Chapter 3 describes the data, the label construction and its censoring rationale,
the leakage audit, and the experimental protocol. Chapter 4 reports results in the order of the
research questions, followed by the decision layer and the two non-estimability findings.
Chapter 5 analyses those findings — reading the null correctly, the calibration conflict,
decisions over discrimination, and non-estimability as a finding. Chapter 6 discusses what the
system is and is not, grounding the path-to-production requirements in the literature and stating
the study's limitations. Chapter 7 concludes and sets out the data acquisitions that would unlock
the analyses this dataset forbids.
