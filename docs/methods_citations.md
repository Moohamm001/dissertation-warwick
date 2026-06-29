# Methods → Citations map (evidence-grounded methodology audit)

*Every imbalance-related design decision traced to a specific paper. Rule: no decision ships
without published support; a flagged gap is acceptable, a fabricated link is not. Sources are
either the **proposal §5.4 bibliography** (already vetted) or the **crawled brain**
(`literature/auto_index.yaml`, OpenAlex id given).*

> **Curated-promotion status:** all 11 candidate papers below were **approved by the user on
> 2026-06-10** and promoted to `literature/index.yaml` (marked **[CURATED]**). Papers marked
> **[BIB]** are already in the proposal bibliography.

| # | Decision (where implemented) | Supporting paper(s) | One-sentence justification |
|---|---|---|---|
| **D1** | **SMOTE oversampling** — `experiments.py:_fold_scores` (SMOTE applied to the training matrix only) | Chawla et al. 2002 **[BIB]**; Fernández, García & Herrera 2018, *SMOTE: 15-year anniversary*, JAIR — `W2800788706` **[CURATED]** | SMOTE is the canonical minority-oversampling method; the 2018 review documents when it helps and its off-manifold risks on sparse minorities — exactly our 50-event caveat. |
| **D2** | **Resampling strictly INSIDE CV folds** — `experiments.py:run_bakeoff` (preprocessor + SMOTE fit on train fold only) | Santos et al. 2018 **[BIB]**; Krstajić et al. 2014, *Cross-validation pitfalls*, J. Cheminformatics — `W2154290668` **[CURATED]**; Kapoor & Narayanan 2023, *Leakage & the reproducibility crisis*, Patterns — `W4385576721` **[CURATED]** | Resampling or encoding before the split leaks test information and inflates scores; these establish that all data-dependent steps must be confined to the training fold. |
| **D3** | **PR-AUC + recall@top-decile over accuracy/ROC** — `metrics.py` (accuracy explicitly banned) | Saito & Rehmsmeier 2015, *PR plot more informative than ROC on imbalanced data*, PLoS ONE — `W1966716734` **[CURATED]**; Hossin & Sulaiman 2015, *Review of evaluation metrics*, IJDKP — `W2330219538` **[CURATED]**; Lessmann et al. 2015 **[BIB]** | On extreme imbalance, accuracy is dominated by the majority and ROC-AUC is optimistic; precision-recall reflects minority detection, which is the decision-relevant quantity. |
| **D4** | **Repeated stratified k-fold CV** — `experiments.py` (`RepeatedStratifiedKFold`, 5×5) | Arlot & Celisse 2010, *Survey of cross-validation procedures*, Statistics Surveys — `W1981552604` **[CURATED]**; Krstajić et al. 2014 — `W2154290668` **[CURATED]** | Stratification preserves the rare-event ratio in every fold; repetition averages out the high split-to-split variance that dominates at small n. |
| **D5** | **Cost-sensitive / class weighting** — `experiments.py:_make_model` (`scale_pos_weight`, `class_weight='balanced'`) | Lin et al. 2017 (focal loss) **[BIB]**; Xia, Liu & Liu 2017, *Cost-sensitive boosted tree for P2P loan evaluation*, ECRA — `W2700766797` **[CURATED]** | Re-weighting the loss is an alternative to resampling that avoids synthetic points; the P2P-lending study shows it on a directly analogous credit task. |
| **D7** | **Regularised LR competitive at small n** — interpretation of `reports/model_bakeoff.md` | Peduzzi et al. 1996, *Events per variable in logistic regression*, J. Clin. Epidemiol. — `W2037668591` **[CURATED]**; Vittinghoff & McCulloch 2006 — `W2130373985` **[CURATED]**; Riley et al. 2020, *Sample size for a clinical prediction model*, BMJ — `W3012413426` **[CURATED]** | The EPV literature shows complex models are unstable below ~10 events per predictor; with ~50 events our finding that LR matches gradient boosting is the expected, well-supported outcome, not an anomaly. |
| **D8** | **General imbalance framing** | Haixiang et al. 2016, *Learning from class-imbalanced data: review*, ESWA — `W2562319768` **[CURATED]** | Survey establishing the method landscape (resampling vs cost-sensitive vs ensemble) the project draws from. |

| **D6** | **Post-hoc calibration (Platt + isotonic)** — `calibrate.py` (`_calibrated_oof`) | Niculescu-Mizil & Caruana 2005, *Predicting good probabilities with supervised learning* — `W2098824882` **[PROPOSED]**; conformal: Angelopoulos & Bates 2023 **[BIB]** | The canonical empirical comparison of Platt scaling vs isotonic regression for post-hoc calibration; grounds both calibrators used in Phase 4a. |

| **D9** | **Events-needed / sample-size projection** — `improve.py:events_projection` | Peduzzi 1996 `W2037668591`, Vittinghoff 2006 `W2130373985`, Riley 2020 `W3012413426` — all **[CURATED]** (reuses D7) | The EPV/sample-size literature grounds the claim that the uncertainty band is event-limited and that more events (not more model) is the lever. |

## RQ1-follow-up & decision-layer decisions (closed 2026-06-29)
| # | Decision (where) | Supporting paper(s) | Justification |
|---|---|---|---|
| **D10** | **L1 / elastic-net penalised logistic regression** — `improve.py:_make_lr` | Tibshirani 1996, *Regression shrinkage and selection via the lasso*, JRSS-B — `W2135046866` **[CURATED]**; Zou & Hastie 2005, *Regularization and variable selection via the elastic net*, JRSS-B — `W2122825543` **[CURATED]** | The canonical L1 and elastic-net penalisation papers; ground the sparse models used to respect the events-per-variable budget at ~3.8 EPV. |
| **D11** | **Affordability / financial-ratio features** — `improve.py:affordability_features` | Altman 1968, *Financial ratios, discriminant analysis and the prediction of corporate bankruptcy*, J. Finance — `W2124532504` **[CURATED]** | The seminal use of financial ratios to predict default; grounds the loan-to-revenue / loan-to-sales / revenue-to-sales affordability features. |
| **D12** | **Feature selection under class imbalance** — `improve.py` | Wasikowski & Chen 2009, *Combating the small-sample class-imbalance problem using feature selection*, IEEE TKDE — `W2138776277` **[CURATED]** | Directly on-point: feature selection (not more capacity) is the lever for small, imbalanced samples. |
| **D13** | **Survival / time-to-event modelling** — `survival.py` | Cox 1972, *Regression models and life-tables*, JRSS-B — `W3147894994` **[CURATED]** | The foundational proportional-hazards reference; cited in `survival_feasibility.md` to frame *why* a clock is needed and *what* would be fitted IF the data supported it (it does not — non-estimable). |
| **D14** | **Expected-cost threshold selection** — `decision.py` | Elkan 2001, *The foundations of cost-sensitive learning*, IJCAI — `W167016754` **[CURATED]**; cost-sensitive framing reuses D5 (Xia et al. 2017) | The canonical expected-cost decision-theory result; grounds choosing the review threshold that minimises `R·FN + FP`. |

## Decisions NOT yet citation-closed
- D6 closes once the **[PROPOSED]** D6 paper above is approved for curation.
- *(D10–D14 closed 2026-06-29: 6 papers crawled from OpenAlex and promoted to `literature/index.yaml`,
  curated total 11 → 17. The `improve` / `survival` / `decide` experiments are no longer provisional.)*

## How to read the code links
- `experiments.py:_fold_scores` — SMOTE/encoder fit happen here, after the train/test split → D1, D2.
- `experiments.py:run_bakeoff` — `RepeatedStratifiedKFold`; preprocessor `fit_transform` on train only → D2, D4.
- `experiments.py:_make_model` — `scale_pos_weight` / `class_weight` → D5.
- `metrics.py` — `pr_auc`, `recall_at_top_decile`; accuracy deliberately absent → D3.
