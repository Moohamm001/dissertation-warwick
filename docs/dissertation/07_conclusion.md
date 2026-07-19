# 7. Conclusion and future work

## 7.1 Answers to the research questions

**RQ1 — Can a gradient-boosted model beat a regularised logistic regression beyond CI overlap at
this prevalence?** No — and the design anticipated that the null, if it occurred, would be the
finding. Median PR-AUC favours logistic regression (0.117 vs 0.093–0.105) with fully overlapping
fold bands; the signal itself is real (permutation p = 0.010, stable across seeds, eight times
the prevalence floor); and the follow-up experiments show the uncertainty band is event-limited,
with on the order of 4–5× the current events (roughly 200–250) required to halve it. At this
sample size the model-choice question is not merely unanswered but unanswerable — the
comparison is unpowered, and the detection bound makes tuning gains equally unverifiable — and
demonstrating that, with the machinery to say when it *would* become answerable, is the
contribution.

**RQ2 — Does post-hoc calibration improve within-minority reliability?** No; it actively harms
it. Platt scaling repairs marginal Brier (0.122 → 0.012) while the event-level calibration error
deteriorates from 0.347 to 0.969 with disjoint CIs. The trade-off is structural — recalibration
toward a 1.28% base rate must collapse confidence on the rare positives — and this dissertation
measures its severity at an extreme operating point: a marginally "well-calibrated" rare-event
model can be worthless on exactly the cases that matter. Conformal prediction's exact but almost empty marginal coverage (mean
set size 1.15 at 90% nominal) completes the same picture from the coverage side.

**RQ3 — Are SHAP explanations coherent and lender-legible, and where do group fairness claims
become non-estimable?** The explanations are exact (linear SHAP), coherent with the model's
coefficients (r = 0.61, the mathematically expected relationship), corroborated by independent
single-feature evidence, and rendered as signed plain-English reason codes per applicant.
Group-conditional fairness is non-estimable at the audit's granularity: no industry or state
cell reaches ten events. The audit protocol is specified; passing its gate requires either more
events or coarser groupings, and — since the data holds no protected characteristics — any audit
on these fields remains a proxy audit.

## 7.2 The central claim

Under extreme class imbalance, careful method is not preparation for the contribution — it *is*
the contribution. This is a positive stance, not an apology for a small dataset. Hand (2006)
argued that the apparent progress of ever more sophisticated classifiers is largely an illusion:
the gains measured on clean benchmarks rarely survive real conditions, and the honest, simple
model that reports its own uncertainty is usually worth more than the complex one that hides it.
A dataset of 50 events makes that argument concrete and unavoidable. Every headline in this
dissertation is either a carefully bounded positive result (a ranking that concentrates ~60% of
defaults in the top decile; a decision rule that turns that ranking into a cost-sized review
queue) or a clearly documented negative one (no detectable model winner; a structural calibration
trade-off; two non-estimability verdicts). None of the negative results could have been stated
safely without the protocol — pre-specified metrics, leakage-audited features, in-fold
resampling, and an uncertainty interval on every claim — and several results that would have
*looked* publishable (a fairness table built on cells of two or three events, a survival model
fitted to an administrative date, a leaderboard "win" sitting inside a fold band 0.19 wide) were
declined because the protocol forbade them.

It is worth being exact about what is new here and what is not. Two of the findings *confirm*
results already established in the literature rather than discovering them: that correcting for
imbalance damages probability quality (Wallace and Dahabreh, 2013; van den Goorbergh et al.,
2022) and that a small event count caps what a complex model can add over a simple one (the
events-per-variable literature; Hand, 2006). The dissertation's *own* contributions are three,
and they are narrower and more honest than a leaderboard win. First, it characterises these
phenomena *together*, on a real green-loan book at 50 events, showing how the imbalance
correction, the calibration conflict, the ranking usefulness and the decision rule interact at
one operating point. Second, it produces two non-estimability verdicts — group fairness and
survival modelling — each reached by testing a method's entry condition and reporting the failure
with evidence, rather than by quietly running the method anyway. Third, it prices the claims it
declines to make: it says, in events, how much more data (roughly 200–250) would be needed before
the model-choice question becomes answerable. Knowing which claims a dataset cannot support,
proving it, and quantifying what would change the answer — that is the durable result, and it is
exactly the discipline a lender, a regulator, or an examiner should want.

## 7.3 Future work: data first, then models

The projection of §4.4 orders the future-work list by what actually moves the needle.

1. **More events (of order 200–250) via a longer observation window or pooled portfolios.** This single
   acquisition would halve the uncertainty band, make the model-choice question answerable, give
   calibration splits enough minority mass to test the RQ2 conflict at scale, and begin
   populating fairness cells.
2. **One field — a default date or last-payment date per loan** — would reverse the survival
   non-estimability verdict and recover the 10,124 censored loans through discrete-time survival
   modelling, plausibly the cheapest large gain available to the lender.
3. **Declined-applicant data** would permit reject inference and extend the model's validity
   beyond the funded book.
4. **On the modelling side, only after the data moves**: class-conditional (Mondrian) conformal
   prediction, minority-aware calibration objectives, and the temporal out-of-time validation
   that a multi-cohort book would finally permit.
5. **Toward deployment**: the governance, monitoring and jurisdiction-specific regulatory work
   scoped in §6.2, none of which is a modelling task.

The dissertation closes where it began: 50 events is not a nuisance parameter to be engineered
around but the central fact of the problem. A discipline that reports what those events can
and cannot support — and prices, in events, the claims it declines to make — serves lenders,
regulators and borrowers better than a leaderboard ever could.
