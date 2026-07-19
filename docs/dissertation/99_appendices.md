# Appendices

## Appendix A: Ethics training evidence

[Insert evidence of completing all required ethics training — certificate screenshot(s).]

## Appendix B: Ethical approval / waiver confirmation

**Ethical basis of the study.** This project analyses a pre-existing, anonymised commercial
lending dataset (14,135 funded green-loan records). It involves no human participants, no
recruitment, no intervention, and no collection of new data. The dataset contains no direct
personal identifiers: the modelling features are firmographic and financial (credit score,
revenue, sales, time in business, industry, state), and no name, address, contact detail, or
account identifier is used at any stage. The leakage audit (Chapter 3, Appendix C) further
restricts the model to 17 pre-funding attributes. On this basis the study falls within the scope
normally granted a low-risk secondary-data-analysis waiver under the University's research-ethics
framework.

**Confirmation.** [Insert the University/WMG ethical-approval reference number, or the confirming
email/waiver decision for a secondary-data project, by replacing this highlighted text. If a
formal waiver was issued, attach the confirmation here.]

## Appendix C: The permitted feature set

The 17 pre-funding features admitted by the default-deny leakage audit (10 numeric,
7 categorical). The full audit trail, including the exclusion reason for each of the ~148
forbidden columns, is versioned at `data/governance/feature_catalogue.yaml`.

| Feature | Type | Plain-English label (as served) |
|---|---|---|
| Credit Score | numeric | Credit score |
| Amount Sought | numeric | Loan amount requested |
| Revenue | numeric | Monthly revenue |
| Average Monthly Sales | numeric | Monthly sales |
| Time In Business | numeric | Time in business |
| Days Since Last Opportunity | numeric | Days since last enquiry |
| Online App Completed | numeric | Applied online |
| Is Borrower Renewal | numeric | Returning borrower |
| Current Tier | numeric | Risk tier |
| Mktg Tier | numeric | Marketing tier |
| Industry | categorical | Industry |
| Loan Purpose | categorical | Loan purpose |
| Borrower State | categorical | Borrower's state |
| Deal Type | categorical | Deal type |
| Renewal Type | categorical | Renewal type |
| Channel | categorical | Origination channel |
| Medium | categorical | Marketing medium |

## Appendix D: Reproduction instructions

Every figure and number in this dissertation regenerates from a clean checkout with the commands
below (Windows-first; global seed 20260609 is set in `emerald_ai/config.py`).

```powershell
pip install -r requirements.txt

python -m emerald_ai eda              # Ch.4 §4.2  -> reports/feasibility.md
python -m emerald_ai audit            # Ch.3 §3.4  -> data/governance/
python -m emerald_ai clean-report     # Ch.3 §3.5  -> reports/data_quality.md
python -m emerald_ai bakeoff          # Ch.4 §4.3  -> reports/model_bakeoff.md
python -m emerald_ai evidence         # Ch.4 §4.3  -> reports/learning_evidence.md
python -m emerald_ai sensitivity      # Ch.4 §4.3  -> reports/sensitivity_cleaning.md
python -m emerald_ai improve          # Ch.4 §4.4  -> reports/improvement.md
python -m emerald_ai calibrate        # Ch.4 §4.5  -> reports/calibration.md
python -m emerald_ai decide           # Ch.4 §4.6  -> reports/decision_policy.md
python -m emerald_ai survival-check   # Ch.4 §4.7  -> reports/survival_feasibility.md
python -m emerald_ai explain          # Ch.4 §4.8  -> reports/explainability.md
python -m emerald_ai followup-checks  # Ch.5 §5.2/§5.5, Ch.4 §4.8 -> reports/followup_checks.md
python -m emerald_ai figures          # visual story -> reports/visual_story.md

python -m pytest -q                   # 49 tests

python -m emerald_ai serve            # Ch.3 §3.11 demo -> http://127.0.0.1:8000
```

This document itself is compiled from the Markdown chapters in `docs/dissertation/` by
`python docs/dissertation/build.py`.

## Appendix E: Decision-support application

The served application (`python -m emerald_ai serve`) was run and verified for this submission:
the model trains on start-up, the operating point is set from out-of-fold scores
(P ≥ 0.617, 62% default catch-rate), and all four endpoints respond as designed (single-applicant
scoring, pasted-CSV batch, file upload, and the rejection path for malformed uploads). The batch
review queue is the primary interface: an uploaded CSV/Excel file is ranked by risk and the
riskiest within-batch decile is flagged for review; the single-application panel provides
per-decision SHAP reason codes and what-if analysis on the plain-English feature form.

*[Insert three screenshots captured from the running application on the submission machine
(`python -m emerald_ai serve`, then open http://127.0.0.1:8000): (i) the batch review queue with
its ranked table and highlighted top-decile; (ii) the single-application panel showing the risk
score and top-3 reason codes; (iii) a rejected malformed upload showing the clean 400 error
response rather than a raw traceback.]*

API surface: `GET /` (UI), `POST /api/score` (single applicant), `POST /api/score-batch`
(pasted CSV), `POST /api/score-upload` (file upload). Hardening: structured request logging (no
applicant field values logged), upload validation (extension allowlist, 10 MB cap, parse and
column checks returning HTTP 400), a global exception handler (generic HTTP 500, no traceback
leakage), and optional static API-key auth on `/api/*` via the `EMERALD_API_KEY` environment
variable (default-open for examination use).

## Appendix F: Supplementary figures

![Events by origination quarter](../../reports/figures/events_by_quarter.png)

![Permutation test null distribution](../../reports/figures/evid_permutation.png)

![Learning curve](../../reports/figures/evid_learning.png)

![Raw-model reliability (pre-calibration)](../../reports/figures/evid_reliability.png)

![Local explanation, low-risk case](../../reports/figures/shap_local_low-risk.png)

![Class imbalance overview](../../reports/figures/story1_imbalance.png)

![Cross-validation and in-fold resampling schematic](../../reports/figures/story2_cv.png)
