# 6. Discussion

## 6.1 Introduction

This chapter turns from interpretation to implications. It sets out what would separate the
proof-of-concept system from a production deployment, and states the study's limitations without
softening.

## 6.2 Path to production: what this system is, and what it is not

"Production-ready" conflates two questions that this dissertation has kept separate throughout.

**Statistical readiness is bounded and not claimed.** No out-of-time validation is possible
within a book this concentrated in one cohort; no external-lender validation exists; calibration
on the minority is demonstrably unreliable (§5.3). These are sample-size facts, and no
engineering or citation effort changes them.

**Engineering and governance readiness is partially delivered and honestly scoped.** The served
application was hardened to a defensible proof-of-concept standard: structured request logging
that records scoring events without applicant field values, boundary validation returning clean
4xx errors for malformed uploads, a global handler preventing internal traceback leakage, and
optional API-key access control (default-open for examination use, explicitly not
production-grade authentication). Beyond the code, the path-to-production analysis
(`docs/path_to_production.md`) grounds three requirements in the literature. *Governance*: model
risk management for ML credit models (Alonso-Robisco and Carbó Martínez, 2022) would require
independent validation sign-off, model documentation travelling with the artefact, and change
control — none of which a single-author project can self-provide. *Monitoring*: a deployed
version would watch the population stability index on the score and each feature (Yurdakul and
Naranjo, 2020) against concept drift (Gama et al., 2014), with a feedback loop from realised
outcomes that the current one-shot dataset cannot supply. *Regulatory*: the per-applicant signed
reason codes already served are the right shape of artefact for adverse-action explanation
obligations, but Wachter et al. (2017) is an EU/GDPR analysis, the book appears US-originated,
and no jurisdiction-specific compliance review has been performed — the claim is about artefact
form, not legal sufficiency.

## 6.3 Limitations

Stated without softening. (1) **Single lender, effectively single cohort**: 92.8% of loans
originate in 2019, so temporal generalisation is untested and untestable within this data. (2)
**Funded-book selection**: the model sees only approved loans; its ranking is conditional on the
incumbent underwriting policy, and no reject-inference correction is possible without data on
declined applicants. (3) **Absolute probabilities are unreliable** on the minority class even
after calibration; every downstream use here treats the score ordinally, and any deployment must
do the same. (4) **The delinquency label is thin**: one `behind` case and a `default` flag whose
operational definition (days past due, charge-off policy) is not documented in the export. A
leave-one-out check (`reports/followup_checks.md`) shows nothing turns on the single `behind`
case: dropping it moves the headline median PR-AUC from 0.117 to 0.099 — well inside the fold
band — and leaves recall@top-decile unchanged at 0.600; the undocumented `default` semantics
remain the substantive gap. (5)
**The improvement and projection experiments** (§4.4) share the same 50 events as everything
else; the 200–250-event projection extrapolates a fitted 1/√events law and should be read as an
order-of-magnitude planning figure, not a precise requirement. (6) **The demo's operating
threshold** (P ≥ 0.617) was set on out-of-fold scores from the same book — a real deployment
would re-derive it on its own population, per the monitoring plan above. (7) **The evaluation
population is not the deployment population**: the primary label conditions non-events on
resolution (§3.3, §5.5), so all reported operating characteristics — the top-decile catch-rate
included — are estimated on terminal-outcome loans while the batch scorer ingests the full
application flow; the diagnostic in §5.5 quantifies the consequence (the terminal-outcome
threshold over-flags the full book 1.6×) without removing it — the *catch-rate* on the full
flow remains unknowable until labels mature.

## 6.4 Summary

The system is a hardened proof of concept for decision support: it ranks funded applications for
human review and does not approve or decline. Statistical readiness is bounded by sample-size
facts that no engineering effort changes, while the governance, monitoring and regulatory work
required for production is scoped but not delivered. The limitation register — single cohort,
funded-book selection, unreliable absolute probabilities, a thin label, and an evaluation
population that is not the deployment population — bounds every claim in this dissertation.
