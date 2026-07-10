# Path to Production — Governance, Monitoring, Regulatory Grounding

*Companion to the D15–D17 rows in `docs/methods_citations.md`. Read this before quoting anything
here as "production-ready" — it is not that.*

## Scope note (read this first)

"Production-ready" conflates two independent questions, and this document answers only one of
them.

1. **Statistical production-readiness of the prediction itself** — out-of-time validation,
   external validity beyond one 2019 cohort at one lender, calibration at minority scale, coverage
   under distribution shift. **Not addressed here.** It is bounded by **N = 50 events** and cannot
   be improved by citing more papers — `docs/progress.md` and `reports/improvement.md` already show
   the uncertainty band is sampling-variance-limited (an events projection puts ~245 events, ≈4.9×
   the current count, as the ask to halve it). Nothing below changes that arithmetic.
2. **Engineering / governance readiness of the demo** — logging, error handling, access control,
   what a monitoring plan would look like, what a regulator asking "why was this applicant flagged"
   is legally owed. **This is what this document covers.** Some of it is already shipped (Track B,
   §regulatory below); the rest is a literature-grounded statement of what would be needed before
   go-live, explicitly flagged as *recommended*, not *done*.

If a future chapter draft contains a sentence like "the model is production-ready," it should cite
this scope note and be rewritten — the honest claim is "the demo has been hardened to a defensible
PoC standard; a validated production model would additionally require §2 (governance) and §3
(monitoring), and remains bounded by the sample-size limitation in §1 regardless."

## §1. Governance — what model risk management would require (D15)

Alonso-Robisco & Carbó Martínez (2022) quantify **model risk** specifically for ML credit-default
models — the gap between a model's apparent performance and its risk-adjusted performance once
model uncertainty is priced in. Applied to this project, a model risk management (MRM) process
before any real go-live would require, at minimum:

- **Validation sign-off independent of the builder** — a second party re-deriving the bake-off
  (`reports/model_bakeoff.md`), calibration (`reports/calibration.md`), and decision-layer
  (`reports/decision_policy.md`) results before the model is trusted for a real queue. This
  dissertation's own "Popper" discipline (ablations, permutation tests in `reports/learning_evidence.md`)
  is a partial substitute but is not independent review.
- **Explicit model risk documentation** — the frozen logistic regression's known failure modes
  (poor within-minority calibration, ECE 0.35→0.97 after post-hoc fixing; non-estimable fairness
  and survival extensions) must travel with the model artefact, not live only in `docs/progress.md`.
- **Versioning and change control** — `emerald_ai/serve.py` currently loads one hardcoded frozen
  model via `get_scorer()`'s `lru_cache`; there is no model registry, no retraining trigger, no
  rollback path. Recommended, not implemented.

**Status: recommended framework, not built.** The engineering changes below (logging, auth, error
handling) are the concrete first steps toward the "documented, access-controlled, auditable"
half of MRM — they do not constitute MRM on their own.

## §2. Monitoring — what a deployed model would need watched (D16)

Two literatures ground the recommendation, at different grain:

- **Concept drift** (Gama et al. 2014, the canonical survey): the joint distribution `P(X, y)` a
  model was trained on can shift after deployment — new industries, a change in underwriting mix,
  a macro shock. Green-loan lending is plausibly non-stationary (policy-driven lending, e.g. green
  incentive schemes changing year to year), so drift is a real risk here, not a textbook caveat.
- **Population Stability Index** (Yurdakul & Naranjo 2020): the industry-standard statistic credit
  risk teams use to flag when a scored population has drifted from the training population, feature
  by feature and on the score itself. It is cheap to compute and interpretable to a non-technical
  reviewer — the right first instrument for this project's scale.

**Recommended minimum monitoring plan** (not implemented — this project ships a single frozen
model with no live feedback loop):
1. Compute PSI on the score distribution and on each of the 17 permitted features, batch-over-batch,
   flagging PSI > 0.25 (the conventional "significant shift" threshold in the credit-risk literature)
   for manual review.
2. Track realised outcomes for the review queue against the model's implied ranking, once labels
   mature — the current project has no feedback loop at all; `Deal Status` is only ever read once,
   at training time.
3. Re-run `reports/model_bakeoff.md`'s bake-off periodically as new labelled data accrues, since
   N=50 means every ~10 new events is a meaningful addition to statistical power (§1 of this doc).

## §3. Regulatory — grounding the reason codes already shipped (D17)

Unlike §1–§2, this section describes something **already built**, not aspirational: `emerald_ai/serve.py`
returns a `top_reasons` field — the top-3 SHAP local contributions aggregated back to named,
plain-English features — on every scored applicant (`_friendly()`, `FRIENDLY_LABELS`).

Wachter, Mittelstadt & Floridi (2017) is the canonical legal analysis of what "explanation of an
automated decision" obligations actually require under GDPR-style regimes: not necessarily a full
causal account, but — where a right exists at all — something closer to the counterfactual "what
would need to change" framing SHAP contributions approximate. Two honest caveats this project must
carry into any regulatory claim:

- Wachter et al.'s analysis is **EU/GDPR-specific**; this project's own docstring references FCA
  Consumer Duty (UK) and the dataset's `Borrower State` field suggests a US-originated book (where
  the operative regime would be ECOA/Regulation B adverse-action notices, not GDPR). **No
  jurisdiction-specific compliance review has been done** — D17 grounds the *general shape* of the
  obligation (a decision subject is owed some account of the reasons), not a specific regulator's
  sign-off.
- SHAP reason codes are a **local linear decomposition of a linear model**, exact by construction
  here (`contrib = (X - train_mean) * model.coef_`) — this is a genuine strength (no post-hoc
  approximation error, unlike SHAP on tree ensembles) but the explanation is only as trustworthy as
  the underlying model's calibration, which §1 already flags as weak within the minority class.

## Cross-reference: engineering changes shipped alongside this document

`emerald_ai/serve.py` was hardened in the same change as this write-up:
- **Logging** — structured `logging` output for every request and scoring event (never raw
  applicant field values), replacing the previous silent/print-only behaviour.
- **Error handling** — malformed uploads (wrong extension, oversized, unparseable, no recognised
  columns) now return a clean `400` JSON body instead of a raw Python traceback; a global handler
  catches anything unexpected and returns a generic `500` rather than leaking internals.
- **Access control** — an optional static API key (`EMERALD_API_KEY` env var) gates the `/api/*`
  routes, default-open so the local/grading demo needs no configuration. This is a single-developer
  academic-demo baseline, explicitly **not** real auth (no per-caller identity, no rotation, no
  audit trail) — a real deployment would need proper credential management, which is out of scope
  here.

These are the "documented, access-controlled, auditable" first steps referenced in §1 — necessary
for a governance process to exist, but not sufficient on their own, and orthogonal to the §1 scope
note's statistical-readiness question.

## Reproduce
`python -m research_bot crawl` (adds the `production_deployment` seed group in
`research_bot/seeds.yaml`) · `python -m pytest tests/test_serve.py` (hardening tests) ·
D15–D17 citation status: `docs/methods_citations.md`.
