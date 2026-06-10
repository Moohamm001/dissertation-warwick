# Methods → Citations map (evidence-grounded methodology audit)

*Every imbalance-related design decision traced to a specific paper. Rule: no decision ships
without published support; a flagged gap is acceptable, a fabricated link is not. Sources are
either the **proposal §5.4 bibliography** (already vetted) or the **crawled brain**
(`literature/auto_index.yaml`, OpenAlex id given).*

> **Curated-promotion status:** papers below marked **[PROPOSED]** are awaiting the user's
> relevance sign-off before being written to `literature/index.yaml` (per the audit's gating rule).
> Papers marked **[BIB]** are already in the proposal bibliography.

| # | Decision (where implemented) | Supporting paper(s) | One-sentence justification |
|---|---|---|---|
| **D1** | **SMOTE oversampling** — `experiments.py:_fold_scores` (SMOTE applied to the training matrix only) | Chawla et al. 2002 **[BIB]**; Fernández, García & Herrera 2018, *SMOTE: 15-year anniversary*, JAIR — `W2800788706` **[PROPOSED]** | SMOTE is the canonical minority-oversampling method; the 2018 review documents when it helps and its off-manifold risks on sparse minorities — exactly our 50-event caveat. |
| **D2** | **Resampling strictly INSIDE CV folds** — `experiments.py:run_bakeoff` (preprocessor + SMOTE fit on train fold only) | Santos et al. 2018 **[BIB]**; Krstajić et al. 2014, *Cross-validation pitfalls*, J. Cheminformatics — `W2154290668` **[PROPOSED]**; Kapoor & Narayanan 2023, *Leakage & the reproducibility crisis*, Patterns — `W4385576721` **[PROPOSED]** | Resampling or encoding before the split leaks test information and inflates scores; these establish that all data-dependent steps must be confined to the training fold. |
| **D3** | **PR-AUC + recall@top-decile over accuracy/ROC** — `metrics.py` (accuracy explicitly banned) | Saito & Rehmsmeier 2015, *PR plot more informative than ROC on imbalanced data*, PLoS ONE — `W1966716734` **[PROPOSED]**; Hossin & Sulaiman 2015, *Review of evaluation metrics*, IJDKP — `W2330219538` **[PROPOSED]**; Lessmann et al. 2015 **[BIB]** | On extreme imbalance, accuracy is dominated by the majority and ROC-AUC is optimistic; precision-recall reflects minority detection, which is the decision-relevant quantity. |
| **D4** | **Repeated stratified k-fold CV** — `experiments.py` (`RepeatedStratifiedKFold`, 5×5) | Arlot & Celisse 2010, *Survey of cross-validation procedures*, Statistics Surveys — `W1981552604` **[PROPOSED]**; Krstajić et al. 2014 — `W2154290668` **[PROPOSED]** | Stratification preserves the rare-event ratio in every fold; repetition averages out the high split-to-split variance that dominates at small n. |
| **D5** | **Cost-sensitive / class weighting** — `experiments.py:_make_model` (`scale_pos_weight`, `class_weight='balanced'`) | Lin et al. 2017 (focal loss) **[BIB]**; Xia, Liu & Liu 2017, *Cost-sensitive boosted tree for P2P loan evaluation*, ECRA — `W2700766797` **[PROPOSED]** | Re-weighting the loss is an alternative to resampling that avoids synthetic points; the P2P-lending study shows it on a directly analogous credit task. |
| **D7** | **Regularised LR competitive at small n** — interpretation of `reports/model_bakeoff.md` | Peduzzi et al. 1996, *Events per variable in logistic regression*, J. Clin. Epidemiol. — `W2037668591` **[PROPOSED]**; Vittinghoff & McCulloch 2006 — `W2130373985` **[PROPOSED]**; Riley et al. 2020, *Sample size for a clinical prediction model*, BMJ — `W3012413426` **[PROPOSED]** | The EPV literature shows complex models are unstable below ~10 events per predictor; with ~50 events our finding that LR matches gradient boosting is the expected, well-supported outcome, not an anomaly. |
| **D8** | **General imbalance framing** | Haixiang et al. 2016, *Learning from class-imbalanced data: review*, ESWA — `W2562319768` **[PROPOSED]** | Survey establishing the method landscape (resampling vs cost-sensitive vs ensemble) the project draws from. |

## Decisions NOT yet citation-closed
- **D6 — calibration method (Platt/isotonic).** *Not yet implemented* (Phase 4). The brain currently
  lacks a clean Platt/isotonic reference (only a weakly-related loss paper). **Action:** when Phase 4
  is built, add seeds for "Platt scaling probability calibration" / "isotonic regression calibration"
  / Niculescu-Mizil & Caruana 2005, then close. Flagged, not faked.

## How to read the code links
- `experiments.py:_fold_scores` — SMOTE/encoder fit happen here, after the train/test split → D1, D2.
- `experiments.py:run_bakeoff` — `RepeatedStratifiedKFold`; preprocessor `fit_transform` on train only → D2, D4.
- `experiments.py:_make_model` — `scale_pos_weight` / `class_weight` → D5.
- `metrics.py` — `pr_auc`, `recall_at_top_decile`; accuracy deliberately absent → D3.
