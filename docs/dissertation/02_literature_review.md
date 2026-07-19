# 2. Literature review

## 2.1 Introduction

This review is organised by the methodological decisions the dissertation had to make, rather
than by chronology. Each theme closes with the position adopted.

A note on the domain literature first, because its thinness shapes the whole review: there is,
to the best of this search (a curated crawl of ~1,000 candidate papers), **no direct
green-loan default-prediction literature** to build on. The nearest anchors are ESG-credit-risk
studies — ESG factors as drivers of firm credit risk (Bonacorsi et al., 2024) and Bayesian
estimation of the relative impact of ESG factors on credit ratings (Agosto, Cerchiello and
Giudici, 2023) —
which establish that sustainability attributes carry credit-relevant signal but say nothing
about default modelling *within* a funded green-loan book. This dissertation therefore leans on
the general credit-scoring and imbalanced-learning literatures for method, and contributes to
the green-lending literature chiefly by documenting what a real green-loan book permits. Every design decision in
Chapters 3–4 is traceable to a specific source here; the mapping is maintained as an auditable
artefact in the project repository (`docs/methods_citations.md`), and only verified,
hand-curated papers are cited.

## 2.2 Learning from class-imbalanced data

The survey literature (Haixiang et al., 2016) organises imbalance remedies into three families:
data-level resampling, algorithm-level cost sensitivity, and ensemble hybrids. Among resamplers,
SMOTE (Chawla et al., 2002) remains the canonical minority-oversampling method; the
fifteen-year retrospective by Fernández et al. (2018) documents both its successes
and the regime where it degrades — sparse minorities, where synthetic interpolation manufactures
off-manifold points. With 50 events, this project sits squarely in that warning's territory,
which is why SMOTE is included as a *comparator*, not a default. Cost-sensitive alternatives
reweight the loss instead of the data: class weighting avoids synthetic points entirely, and Xia,
Liu and Liu (2017) demonstrate cost-sensitive boosting on a peer-to-peer lending task directly
analogous to this one. Focal loss (Lin et al., 2017) extends the idea to extreme ratios in dense
detection. The position adopted: class weighting as the default treatment, SMOTE-within-fold as
the single resampling comparator, and no further resamplers — at 50 events, additional imbalance
machinery adds variance, not information.

## 2.3 Evaluation and validation under imbalance

Two well-established results discipline the evaluation design. First, accuracy and ROC-AUC are
misleading under severe imbalance: Saito and Rehmsmeier (2015) show that the precision-recall
curve shows minority-detection performance that ROC analysis makes look better than it is, and Hossin and Sulaiman
(2015) catalogue the metric families and their failure modes. Benchmarking practice in credit
scoring specifically (Lessmann et al., 2015) reinforces the need for multiple
decision-relevant metrics. Accordingly, accuracy is banned from the metric panel; PR-AUC and
recall at the top decile are primary.

A third result disciplines model *comparison* specifically: estimates from cross-validation
folds are correlated because folds share training data, so naive significance tests over folds
make a difference look more significant than it really is — Dietterich (1998) set out which comparison tests are trustworthy, and
Nadeau and Bengio (2003) showed no unbiased variance estimator for the CV generalisation error
exists. This is why the paired fold-level comparison in this dissertation is reported as
descriptive evidence rather than confirmatory inference.

Second, validation must respect the resampling boundary. Data-dependent steps performed before
the train/test split leak information and inflate scores — the general mechanism is documented by
Krstajić et al. (2014) and its prevalence in applied ML by Kapoor and Narayanan (2023);
Santos et al. (2018) demonstrate the specific case of resampling outside cross-validation folds
on imbalanced data. Arlot and Celisse (2010) ground the choice of repeated stratified k-fold:
stratification preserves the rare-event ratio in every fold, and repetition averages the
split-to-split variance that dominates at small n. The position adopted: 5×5 repeated stratified
CV, every data-dependent transformation (imputation, encoding, scaling, SMOTE) fitted strictly
inside the training fold, and fold-percentile bands reported instead of point estimates.

## 2.4 Small samples: events per variable and sample-size planning

The events-per-variable (EPV) literature sets expectations for what 50 events can support.
Peduzzi et al. (1996) established the rule-of-ten for logistic regression; Vittinghoff and
McCulloch (2006) showed the rule is a guideline rather than a cliff, but that instability grows
as EPV falls; Riley et al. (2020) provide the modern sample-size machinery for clinical
prediction models, framing minimum events as a design input rather than an afterthought. With 13
candidate numeric features this project operates at EPV ≈ 3.8, and at ≈ 2.0 on the 25 encoded
model inputs — far below ten on either denominator — which generates
two testable predictions: complex models should fail to beat a regularised linear baseline
(Chapter 4 confirms this), and shrinkage should be the principled response to feature-set growth.
The shrinkage tools are classical: the lasso (Tibshirani, 1996) and the elastic net (Zou and
Hastie, 2005); Firth's (1993) penalised likelihood is the canonical correction for the
small-sample bias of maximum-likelihood logistic estimates in exactly this rare-events regime;
and Wasikowski and Chen (2010) show feature selection is the effective lever
specifically for small-sample imbalanced problems. Altman (1968) grounds the domain-specific
feature engineering — financial ratios as bankruptcy predictors — used in the improvement
experiments.

## 2.5 Probability calibration and uncertainty quantification

Class-weighted training deliberately distorts predicted probabilities, so post-hoc calibration is
standard practice: Platt scaling and isotonic regression are the canonical approaches, compared
empirically by Niculescu-Mizil and Caruana (2005). That imbalance corrections damage probability
quality is established prior art, not this dissertation's discovery: Wallace and Dahabreh (2013)
showed class-probability estimates from imbalance-corrected classifiers are systematically
unreliable, and van den Goorbergh et al. (2022) demonstrated on clinical risk models that
SMOTE-style and reweighting corrections harm calibration while barely moving discrimination. Van
Calster et al. (2016) supply the vocabulary this dissertation adopts for what "calibrated" means
at different strengths — mean calibration, weak, moderate, strong — which matters here because
the event-level metric used in RQ2 sits below that hierarchy altogether (it measures confidence
on positives, not calibration in the bin-frequency sense; §3.6). What RQ2 contributes against
this literature is a measurement of the trade-off at an extreme operating point — 50 events,
roughly ten positives per calibration split — where marginal calibration and confidence on the
minority pull in exactly opposite directions. For distribution-free uncertainty, split conformal prediction
(Angelopoulos and Bates, 2023) provides finite-sample marginal coverage guarantees; the
dissertation implements it as an honesty check while showing why marginal coverage carries
almost no information at 1.28% prevalence.

## 2.6 Explainability and its regulatory context

SHAP values (Lundberg and Lee, 2017) provide additively decomposed, per-prediction attributions
grounded in Shapley values; for linear models the decomposition is exact rather than
approximated, which removes explanation-fidelity risk — the explanation is faithful to the model
by construction, though §4.8 notes this says nothing about the stability of the model's own
coefficients at low EPV. The regulatory motivation is real but frequently overstated: Wachter, Mittelstadt and
Floridi (2017) — the canonical legal analysis — show that a general "right to explanation" of
automated decisions does not, in fact, exist in the GDPR, while articulating what decision
subjects are plausibly owed: a legible account of the principal reasons. Per-applicant signed
reason codes of the kind this project ships are the right *shape* of artefact for that
obligation, a point developed further in the path-to-production analysis (Chapter 6), without
claiming jurisdiction-specific compliance.

## 2.7 Model risk, monitoring, and deployment

A model that performs in a notebook is not a deployed system. Alonso-Robisco and Carbó Martínez
(2022) quantify model risk for machine-learning credit-default models specifically, showing that
apparent performance shrinks once model uncertainty is priced — directly relevant to a project
whose fold bands are wide. Post-deployment, two monitoring literatures apply: concept drift
(Gama et al., 2014, the canonical survey) for changes in the underlying relationship, and the
population stability index (Yurdakul and Naranjo, 2020) — the credit industry's standard
statistic — for distributional shift in scores and features. These sources ground the
recommended (not implemented) monitoring design in Chapter 6.

## 2.8 Survival analysis and censoring

Right-censoring is the correct lens for a young loan book: loans still current at snapshot have
simply not been observed long enough. The proportional-hazards framework (Cox, 1972) is the
natural recovery route for censored observations — *if* a trustworthy time-to-event clock exists.
Chapter 4 shows that in this dataset it does not, converting the survival route into the second
documented non-estimability result.

## 2.9 Summary

The literature supplies well-understood components: imbalance treatments, leakage-safe
validation, EPV-based expectations, calibration methods, exact linear SHAP, and deployment risk
frameworks. It also supplies a standing warning that this dissertation takes as method: Hand
(2006) showed that the marginal gains of sophisticated classifiers over simple ones routinely
evaporate under real-world conditions — population drift, label noise, changing costs — and
that progress claims built on small benchmark differences are largely illusory. What the
literature does not supply is an account of how these components behave *jointly* at 50
events on a real lending book — where most published effect sizes are undetectable, where the two
calibration objectives collide, and where standard audits (fairness, survival) cease to be
estimable at all. Occupying that gap honestly — with pre-specified metrics, uncertainty
intervals on every claim, and non-estimability reported as evidence — is the contribution.
