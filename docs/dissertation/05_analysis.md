# 5. Analysis

## 5.1 Introduction

This chapter interprets the findings of Chapter 4. It examines what the null model-comparison
result does and does not license, why the calibration conflict is structural, why decision-level
improvements remain estimable when discriminative ones do not, and what the two non-estimability
verdicts mean as evidence. Interpretation is separated from the deployment implications, which
follow in Chapter 6.

## 5.2 Reading the null correctly: why "no winner" is the finding

The headline RQ1 result, logistic regression and XGBoost statistically indistinguishable, will
disappoint a reader expecting a leaderboard, and that disappointment is worth confronting
directly, because the null is exactly what the design predicted and exactly what a competent
practitioner should expect at this sample size. The events-per-variable literature (Peduzzi et
al., 1996; Riley et al., 2020) implies instability below roughly ten events per predictor; this
study operates at EPV ≈ 3.8. The learning curve plateaus by 75% of the data. The improvement
experiments (§4.4) show that principled shrinkage and domain feature engineering move neither the
median nor the band. Every arrow points the same way: the constraint is informational, not
algorithmic.

The null also sits comfortably within the credit-scoring literature rather than against it.
Dumitrescu et al. (2022), working with far larger samples than are available here, found that
most of the advantage machine learning shows over logistic regression in credit scoring can be
recovered by giving the logistic model non-linear decision-tree effects; that is, the gap is
narrower than the benchmark culture suggests even when the data are ample. Lessmann et al. (2015)
reach a compatible conclusion from their benchmark study: differences between competent
classifiers are often small relative to the choice of metric and evaluation design. If the
advantage is modest at scale, its disappearance at 50 events is not an anomaly to be explained
away but the expected extension of a known pattern. The finding this dissertation adds is the
*measurement* of where that pattern terminates: not "gradient boosting is no better", but "at
this event count the question is beneath the resolution of the instrument, and here is how much
data would raise the resolution".

There is a further, domain-specific reason not to over-read the null. The SME-scoring literature
reports that predictor sets for this segment are unstable across samples and periods (Ciampi et
al., 2021), and that models frequently require re-specification rather than transfer (Altman et
al., 2022). A comparison run on one lender's book, in one product category, in one origination
window, is therefore poorly placed to settle a general question about model families even in
principle: a limitation of scope that the sample size makes moot here, but that would remain
after the sample size problem was solved.

Two honest qualifications sharpen rather than weaken the null. First, the challenger ran
untuned: the project plan had contemplated a modest randomised search with monotonic
constraints, and the final protocol dropped both on the principled ground that model selection
over ~10 positives per inner fold is itself noise (§3.6), but the consequence must be owned:
the demonstrated result is that the comparison is *unpowered to crown* a winner, not that
XGBoost optimally configured could never win. Second, marginal-band overlap is not itself a
test of no difference: the folds are shared between models, so the sharper instrument is the
per-fold *paired* difference, and that analysis was run (`reports/followup_checks.md`). It
agrees with the marginal verdict: the baseline's median paired advantage over XGBoost is +0.02
to +0.03 PR-AUC with difference bands that span zero ([−0.126, 0.149] against
XGBoost+class-weight), the baseline wins only 14 of 25 shared folds, and the sign test finds
nothing (p ≈ 0.69; reported descriptively, since repeated-CV folds are correlated and the test
is approximate). Neither qualification rescues the challenger, because of a second, less
obvious layer: a **detection bound**. Even if some method genuinely
improved PR-AUC by, say, 0.03 (a large effect by the standards of the tabular credit
literature), that improvement could not be *demonstrated* here, because the fold band spans roughly 0.19.
Published improvements from imbalance-aware methods were validated on datasets with hundreds or
thousands of events; at 50 events such effects are not merely absent but undetectable, and any
claim to have detected one would be irreproducible. This reframes the "why not try method X?"
question that any examiner of such work should ask: the answer is not "we tried and it failed"
but "at this sample size, the claim that X helps is structurally unverifiable, and the events
projection quantifies what data would make it verifiable" (roughly 200–250 events; §4.4).

## 5.3 The calibration conflict and its operational consequence

The RQ2 result, Platt scaling repairing marginal Brier while nearly tripling the event-level
calibration error with disjoint confidence intervals, is best read as the quantification of a
structural trade-off, not the discovery of a paradox. As §3.6 and §4.5 make explicit, the
within-minority metric measures confidence on actual defaults, and shrinking every probability
toward a 1.28% base rate *must* collapse that confidence for any model with imperfect
discrimination; the general phenomenon is established prior art: Wallace and Dahabreh (2013)
on unreliable class-probability estimates under imbalance, van den Goorbergh et al. (2022) on
imbalance corrections harming calibration, rather than novel to this study, and Van Calster et
al.'s (2016) calibration hierarchy supplies the precise sense in which the event-level metric
here is not a calibration measure at all. What the measurement adds is its severity at this extreme
operating point and the plainness of the consequence: **for rare-event credit models, a marginal
calibration metric can certify as "well calibrated" a model whose probabilities are worthless on
precisely the cases that matter.** The operational response adopted here is
consistent across the system: the score is treated as a ranking quantity; the user interface
never presents it as a literal default probability; and the decision layer selects its threshold
empirically on out-of-fold scores rather than analytically from Elkan's 1/(1+R), which assumes
calibrated inputs. The conformal result reinforces the same lesson from another direction: a
formally exact marginal-coverage guarantee is substantively empty at this prevalence, and
reporting its emptiness (mean set size 1.15) is more informative than reporting its coverage.

## 5.4 What the data does support: decisions over discrimination

The decision layer is the constructive counterpart to the nulls. Its saving over the naive
cut-off is bootstrap-robust (19.0% [7.9, 32.8], threshold re-selected per resample) *because it
asks less of the data*: choosing a threshold on an existing ranking requires estimating one
operating point, not separating two models' full score distributions. The general principle (**at extreme sample sizes, decision-theoretic improvements, which reuse the ranking, remain
estimable long after discriminative improvements have become undetectable**) holds, but its
demonstration here must be stated at the right size. Against the 0.5 cut the saving is large
and robust; against the operationally sensible top-decile rule it is small (3.9% point
estimate; bootstrap 6.8% [1.2, 19.4], an interval above zero but optimistic in location, so
the true advantage may be marginal). The honest operational message is therefore twofold: the
simple decile queue is already near-optimal at moderate cost ratios, and the expected-cost
machinery's durable contribution is *deriving* the queue size from stated costs, with a small,
probably-positive cost edge as a bonus rather than the headline. The same logic
explains why recall-at-top-decile is the operational headline the study permits (a
population-level statement about queue composition, not a per-applicant probability claim)
though its own fold band [0.36, 0.90] means "catches a majority of defaults" is the strongest
phrasing the evidence supports; the ~60% centre should never travel without that band. The
practical reading for a lending desk is deliberately modest: review the model-ranked top decile
first, and choose the queue depth by an explicit cost ratio rather than by a probability
threshold whose scale cannot be trusted.

## 5.5 Non-estimability as a finding

The fairness and survival results (§4.7) share a structure worth making explicit: each method has
an entry condition (events per cell; a trustworthy clock), the condition was tested before the
method was applied, and the test failed with documented evidence. The alternative in each case
was available and tempting (a fairness table over cells of 2–9 events, a Cox model on the
administrative `End` date) and would have produced publishable-looking numbers that were pure
artefact. Declining to produce them is a result, not an omission: the fairness gate specifies the
audit that *should* run when the data can support it, and the survival check identifies the
single missing field (a default or last-payment date) that would reverse the verdict.

The `Credit Score` finding requires more care than a bare "surprising null". The feature a
credit dataset is nominally organised around performs below the prevalence floor in isolation,
survives data cleaning unchanged, and ranks eighth by SHAP: on this book, affordability
signals (`Revenue`, `Average Monthly Sales`) do the work. But an alternative explanation is
unexcluded and probably operative: **range restriction on a funded book**. Every observed loan
passed the incumbent underwriting screen, which plausibly conditions hard on credit score; a
variable used to select the sample is mechanically drained of marginal predictive power within
it. The correct statement is therefore conditional: *given funding under the incumbent policy*,
credit score adds little rank information beyond affordability, a finding about this decision
context, not about credit scores in general. The same selection lens applies to the label
itself: `paidoff_only` conditions non-events on resolution by the snapshot date, so the
evaluation population (terminal-outcome loans) is systematically faster-resolving than the full
application flow the demo scores. That diagnostic was run (`reports/followup_checks.md`):
scoring the full 14,135-row book against the 3,898-row evaluation subset, the distributions
differ modestly overall (KS statistic 0.065) but materially at the operating point (the
threshold set on the evaluation subset flags **16.4% of the full book against the intended
10%**) a 1.6× inflation of the review queue (the figure's coincidence with the R = 20 cost
saving of §4.6 is exactly that, a coincidence between unrelated quantities). The mismatch is thus quantified rather than
hypothetical, and the operational rule follows directly: any deployment must re-derive its
threshold on the population it actually scores.

## 5.6 Summary

The null model comparison reflects an informational constraint, not an algorithmic one: at this
event count a winner is not merely absent but undetectable. The calibration conflict is
structural, recalibration towards the base rate must collapse confidence on the rare positives,
and the operational response is to treat the score as a ranking quantity. Decision-layer
improvements remain estimable because they ask less of the data than discriminative comparisons.
Finally, the fairness and survival verdicts show that testing a method's entry condition, and
reporting its failure with evidence, is itself a result.
