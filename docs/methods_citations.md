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

## Decisions NOT yet citation-closed
- D6 closes once the **[PROPOSED]** D6 paper above is approved for curation.
- **D10 — L1 / elastic-net penalised logistic regression** (`improve.py:_make_lr`). **GAP — no
  supporting paper in the brain.** Needs Tibshirani 1996 (LASSO), Zou & Hastie 2005 (elastic-net),
  and ideally a small-sample shrinkage/penalisation reference (Riley/Van Calster). Until curated,
  the `improve` experiment is **provisional** and must not ship in the dissertation.
- **D11 — affordability / financial-ratio feature engineering** (`improve.py:affordability_features`).
  **GAP (domain).** Candidates: Altman 1968 (financial ratios), Lessmann et al. 2015 **[BIB]**.
- **D12 — feature selection under class imbalance.** **PARTIAL** — two on-topic papers already sit
  in `auto_index.yaml` (not yet curated): "Combating Small Sample Class Imbalance Using Feature
  Selection" (2009); "Cost-based feature selection for SVM: credit scoring" (2017).
- **D13 — survival / time-to-event modelling** (`survival.py`). **GAP — zero survival/censoring/Cox
  papers in the brain.** Would need Cox 1972 + a discrete-time survival reference (e.g. Tutz &
  Schmid 2016) IF a survival result ever became estimable. NB: `reports/survival_feasibility.md`
  finds survival **non-estimable** on this data (no reliable duration), so D13 is currently moot —
  the documented-non-estimability framing itself is COVERED by the Phase-1 feasibility precedent.
- **Next action:** patch `research_bot/seeds.yaml` with penalised-regression and financial-ratio
  queries, re-crawl, and promote D10–D12 candidates — then mark them CURATED here (ask first, per
  standing rule).

## How to read the code links
- `experiments.py:_fold_scores` — SMOTE/encoder fit happen here, after the train/test split → D1, D2.
- `experiments.py:run_bakeoff` — `RepeatedStratifiedKFold`; preprocessor `fit_transform` on train only → D2, D4.
- `experiments.py:_make_model` — `scale_pos_weight` / `class_weight` → D5.
- `metrics.py` — `pr_auc`, `recall_at_top_decile`; accuracy deliberately absent → D3.
