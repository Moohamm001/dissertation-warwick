# Evidence-Grounded Methodology Audit — Gap Report & Plain Summary

*Companion to `docs/methods_citations.md` (Rule 1) and `reports/learning_evidence.md` (Rule 2).*

## A. Literature gaps: found → seeds added → resolved
Initial search of the crawled brain (179 papers) + proposal §5.4 bib left three decisions
**unsupported** and one **brain-blind but bib-covered**:

| Decision | Initial state | Action | Outcome |
|---|---|---|---|
| D2 within-fold resampling | brain 0 hits; **bib had Santos 2018** | added seeds "resampling within CV folds leakage", "CV imbalanced overoptimism" | ✅ brain now surfaces Krstajić 2014, Kapoor & Narayanan 2023 |
| D3 PR-AUC/recall over ROC/acc | **no support** | added seeds "precision recall curve imbalanced", "ROC vs PR" | ✅ Saito & Rehmsmeier 2015 (the canonical paper), Hossin & Sulaiman 2015 |
| D4 repeated stratified CV | **no support** | added seed "repeated stratified k-fold variance" | ✅ Arlot & Celisse 2010, Krstajić 2014 |
| D7 LR competitive at small-n | **no support** | added seeds "events per variable logistic", "sample size clinical prediction model" | ✅ Peduzzi 1996, Vittinghoff & McCulloch 2006, Riley 2020 |

Crawler grew 179 → **292 papers** (+113). The new `methodology_justification` seed group is committed
in `research_bot/seeds.yaml`. This is direct proof the bot can now *inform* the methodology.

## B. Unresolved (flagged, not faked)
- **D6 — calibration method (Platt/isotonic).** Not yet implemented (Phase 4); brain still lacks a
  clean Platt/isotonic reference. Will add seeds + close when Phase 4 is built. **Not cited until then.**

## C. Plain summary — what is now evidence-backed vs a limitation

### Evidence-backed (defensible in a viva)
1. **The model learns real signal.** Permutation test: real PR-AUC 0.096 **beats 100% of 100
   shuffled-label runs, p = 0.010**. Not noise.
2. **The result is stable**, not seed luck: PR-AUC **0.091 ± 0.002**, recall@decile **0.608 ± 0.027**
   across 5 seeds — well above the 0.013 prevalence floor.
3. **The full model adds value** over the best single feature (0.096 vs Revenue-only 0.065 vs
   floor 0.013).
4. **Performance plateaus** by 75% of the data → the dataset is too small to reward complex models,
   which **supports the logistic-regression finding** (EPV literature: Peduzzi 1996, Riley 2020).
5. **Every imbalance design choice (D1, D2, D4, D5) is now citation-backed** (see methods_citations).

### Limitations (stated honestly, not softened)
1. **Absolute probability calibration is poor.** Brier 0.124 ≫ base-rate 0.0127 — class weighting
   inflates probabilities. The score *ranks* well but its probabilities are untrustworthy until
   **Phase 4 calibration**. (Reported in `learning_evidence.md` §5.)
2. **`Credit Score` alone scores BELOW the dummy floor** (0.0104 vs 0.0128); `Revenue` is the real
   workhorse. Surprising for a credit dataset — worth a sentence and possibly a data-quality look
   (recall `Credit Score = 0` missing-coding).
3. **recall@top-decile interval is wide** at N=50 (per-fold 95% band 0.36–0.90); the point estimate
   ~0.60–0.64 must always carry that band.
4. **D6 calibration citation open** (above).

## D. Reproduce
`python -m emerald_ai evidence` (Rule 2 proofs) · `python -m research_bot crawl` (literature) ·
all seeded.
