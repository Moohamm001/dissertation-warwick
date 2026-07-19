# Abstract

Green-loan portfolios are growing rapidly, yet the credit-risk evidence base for them remains
thin: funded books are young, defaults are rare, and the resulting datasets are extremely
imbalanced. This dissertation builds and honestly evaluates a delinquency-detection model for a
portfolio of 14,135 funded green loans containing only 50 delinquent events (0.36% of the raw
book; 1.28% under the censoring-safe primary label), and asks what such a dataset permits a
lender to claim — and what it forbids.

Three research questions are addressed under a pre-specified, leakage-audited protocol
(default-deny feature vetting retained 17 of 166 columns; all resampling and encoding confined to
training folds; 5×5 repeated stratified cross-validation; accuracy banned in favour of PR-AUC,
recall at the top decile, and within-minority expected calibration error, each with uncertainty
intervals). First, a gradient-boosted model does not beat a regularised logistic regression
beyond fold-band overlap (median PR-AUC 0.117 vs 0.093–0.105); at roughly ten events per test
fold the data cannot separate the models, and a permutation test (p = 0.010) confirms the signal
itself is real. Second, post-hoc calibration repairs marginal Brier score (0.122 → 0.012) while collapsing the
model's confidence on the rare defaults themselves (the mean confidence shortfall on actual
defaults rises from 0.35 to 0.97): at this prevalence, marginal calibration and confidence on
the minority are structurally conflicting objectives, and recalibration resolves the conflict
in the majority's favour. Third, exact linear
SHAP explanations are faithful and decision-legible, identifying the affordability signals
(monthly revenue and sales) — not credit score — as the dominant risk cluster, an ordering shown
to be stable across folds; however, group-conditional fairness auditing is empirically
non-estimable at the audit's granularity (no industry or state cell reaches ten events), as is
survival modelling of the 10,124 right-censored loans (the two candidate duration measures
correlate −0.02).

The work contributes a decision layer that the small sample *does* support: an expected-cost
review threshold that derives the review-queue depth from stated costs, beating the naive 0.5
cut-off by 19.0% (95% bootstrap interval [7.9, 32.8]) and the simpler top-decile rule by a
modest, and less certain, 6.8% [1.2, 19.4] at a plausible cost ratio, shipped in a hardened
proof-of-concept decision-support application with per-applicant reason codes. A sample-size projection quantifies the route to a materially
better model: on the order of four to five times the current event count (roughly 200–250
events) would be required to halve the present uncertainty band. The dissertation's central claim is methodological: under extreme
imbalance, rigour about what is estimable is itself the contribution, and null and non-estimable
results are reported as findings rather than failures.
