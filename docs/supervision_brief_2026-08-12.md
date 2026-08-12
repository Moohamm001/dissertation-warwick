# Supervision meeting brief — 12 August 2026, Coventry

Candidate: Tatphong Kruerattanakul (5700836) · MSc Applied Artificial Intelligence
Repository: https://github.com/Moohamm001/dissertation-warwick

---

## 1. Actions from the last meeting

| Action | Status |
|---|---|
| **Include RO and motivation in Chapter 1** | **Done.** §1.1 Motivation grounds green lending in the climate-risk, green-finance-demand and green-credit-policy literatures and explains why SME borrowers are hard to score. §1.3 states the aim and **RO1–RO6**, each mapped to the section that discharges it. §7.1 gives an RO-by-RO verdict. |
| **Improve the writing across the chapters** | **Done.** New §2.2 (green lending, SME credit, credit-scoring practice); §3.1 now states the research design; §5.2 places the null result inside the credit-scoring literature; new §6.2 gives three practical rules for a lending desk; §6.3 explains why monitoring matters more for a green book. Plain-language pass throughout. |
| **Complete the deployment part** (batch, not per-application scoring) | **Done.** The batch review queue is the primary interface, with single-application scoring kept for explanation. The service is containerised, restarts in 0.02 s from a cached model, exposes a health probe, and can be deployed publicly **without the lending book leaving this machine**. |
| **Put the code on GitHub and send the link** | **Done** — link above. Please confirm whether the repository should be private for the assessment. |

## 2. Where the dissertation stands

- **7 chapters** in the WMG template: Introduction, Literature Review, Methodology, Results,
  Analysis, Discussion, Conclusion, plus References and Appendices A–F.
- **≈13,600 words** in Chapters 1–7 and the abstract (≈15,600 including front matter,
  references and appendices).
- **46 references**: 42 peer-reviewed journal articles and 4 papers at major refereed
  conferences (IJCAI, ICCV, NeurIPS, ICML). No preprints or web sources. One citation was
  withdrawn on quality grounds this week (see §5).
- **59 automated tests** pass; every figure and number regenerates from a seeded command.

## 3. Findings to discuss

| Research question | Answer |
|---|---|
| **RQ1** Does a gradient-boosted model beat regularised logistic regression? | **No, and the comparison is unpowered to say otherwise.** Median PR-AUC 0.117 vs 0.093–0.105, all fold bands overlapping; a paired fold-level test agrees (baseline ahead in 14 of 25 shared folds, sign test p ≈ 0.69). The signal itself is real (permutation test p = 0.010). A projection puts the data needed to halve the uncertainty at roughly 200–250 events, four to five times what exists. |
| **RQ2** Does post-hoc calibration help on the minority class? | **No; it harms it.** Platt scaling repairs marginal Brier (0.122 → 0.012) while confidence on actual defaults collapses (0.347 → 0.969, disjoint intervals). The trade-off is structural at 1.28% prevalence. |
| **RQ3** Are the explanations coherent, and where do fairness claims fail? | Explanations are exact (linear SHAP), coherent with the coefficients (r = 0.61) and stable across folds (Spearman 0.89). **Group-conditional fairness is non-estimable**: no industry (0 of 27) or state (0 of 51) cell reaches ten events. |

**What the data does support:** reviewing the model-ranked top decile captures on the order of
60% of defaults, and an expected-cost rule sets the queue depth from a stated cost ratio. Against
the naive 0.5 cut-off the saving is 19.0% [7.9, 32.8]; against the operationally sensible
top-decile rule it is a modest 6.8% [1.2, 19.4]. Both figures are reported.

**A second non-estimability result:** survival modelling of the 10,124 censored loans was
abandoned before fitting, because the two candidate duration measures correlate −0.02 — the data
has no trustworthy clock.

## 4. Questions for you

1. **Ethics.** Appendix A needs the training certificate and Appendix B the approval or waiver
   reference. Which route applies to a secondary-data project of this kind, and is a WMG REC
   reference required? A full self-assessment is drafted in Appendix B for review.
2. **Public deployment.** The service can be published (Docker on Render; the dataset stays
   here, only a 26 KB model artefact is deployed). Does the existing ethics position cover
   making a model of the lender's book publicly available, and should the lender's permission be
   sought first? My proposal is to publish behind an API key shared with the examiners.
3. **Repository visibility.** The dataset is currently tracked in the repository. Should it be
   private for assessment?
4. **Word limit.** Confirming the limit for the course: 13,600 words in the assessed chapters
   leaves room if more depth is wanted anywhere.
5. **Framing.** Chapter 7 argues the contribution is methodological — what 50 events permit,
   two documented non-estimability results, and pricing the missing data in events. Is that the
   right emphasis for the viva, or would you weight the deployed artefact more heavily?

## 5. Decisions taken since the last meeting that you may want to review

- **A citation was withdrawn.** Hossin & Sulaiman (2015) appeared in a journal published by
  AIRCC, indexed in neither Scopus nor Web of Science. It was a supporting citation for metric
  selection; Saito & Rehmsmeier (PLoS ONE) and Lessmann et al. (EJOR) carry that decision, so the
  argument is unchanged.
- **A reproducibility defect was fixed.** `imbalanced-learn` was used by the bake-off but never
  listed in `requirements.txt`, so a clean checkout could not have reproduced it. Found when the
  first deployment failed to boot.
- **Em dashes were removed** from the report at your stylistic preference, with the punctuation
  each context requires.

## 6. Outstanding before submission

| Item | Owner |
|---|---|
| Ethics training certificate (Appendix A) | Candidate |
| Approval / waiver reference and date (Appendix B, pro-forma) | Candidate, pending your guidance |
| Interface screenshots (Appendix E; a verified session transcript is already included) | Candidate |
| Signature and date on the pro-forma and declaration | Candidate |
| Confirm word limit and repository visibility | This meeting |
