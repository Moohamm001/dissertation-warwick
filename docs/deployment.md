# Deploying the decision-support service publicly

This note covers what to deploy, what must **not** be deployed, and the decisions to make before
a public URL exists. It complements `docs/path_to_production.md`, which sets out the governance
and monitoring work that a genuinely production deployment would additionally require.

## 1. What actually gets deployed

**The dataset never leaves the analyst's machine.** The service does not need it at run time.
Fit the model where the data lives, and ship only the resulting artefact:

```powershell
python -m emerald_ai build-artefact     # -> artefacts/scorer.joblib
```

The artefact is **~26 KB** and contains:

| Included | Not included |
|---|---|
| 25 model coefficients and the intercept | Any borrower record |
| The fitted preprocessor (encoder categories, imputation values, scaler statistics) | Any row-level value |
| Form metadata: per-feature median default and the 10th/90th percentile hint | The outcome label |
| The operating point (threshold, catch-rate) and aggregate counts (3,898 rows, 50 events) | The `.xlsx` export |

This was verified rather than assumed: no array inside the artefact is large enough to hold the
3,898 training rows, and a sandbox containing only `emerald_ai/`, `data/governance/` and the
artefact serves the UI and every scoring endpoint with no dataset present. A test enforces both
properties so the guarantee cannot silently regress.

What the artefact *does* expose is **aggregate** information about the portfolio: the direction
and size of each coefficient, the typical range of each feature, and the category levels present
in the book. That is the normal disclosure surface of any deployed scoring model, but it is a
disclosure, and §3 below is where to decide whether it is acceptable to publish.

## 2. Deploying

**Container (any host that runs images):**

```powershell
docker compose up --build          # local
```

The compose file mounts the dataset read-only for local use; a remote deployment should omit
that mount entirely and rely on the committed artefact.

**Render (blueprint provided):** `render.yaml` builds from `requirements.txt`, starts
`python -m emerald_ai serve --host 0.0.0.0 --port $PORT`, and health-checks `/health`. Any
comparable host (Railway, Fly.io, Azure App Service, a university VM) works the same way; the
only requirements are Python 3.12, the dependencies, and the artefact.

**Environment variables:**

| Variable | Effect |
|---|---|
| `EMERALD_API_KEY` | When set, `/api/*` requires a matching `X-API-Key` header. The UI and `/health` stay reachable. Unset means open. |
| `EMERALD_RATE_LIMIT` | Requests per client IP per minute (default 60; `0` disables). In-process and per-instance. |

## 3. Decisions to take before publishing a URL

These are judgements for the researcher and supervisor, not defaults this repository can set.

1. **Ethical scope.** The ethics position recorded in Appendix B covers *secondary analysis* of
   an anonymised extract. Publishing a service derived from that data is a different activity:
   it makes a model of a real lender's book continuously available to anyone. Confirm with the
   supervisor that the existing approval or waiver covers it, or seek an amendment, before the
   URL is shared.
2. **Data-owner permission.** The book belongs to the lender that supplied it. Even though no
   records are served, the coefficients and feature ranges describe their portfolio. Publication
   should be agreed with them.
3. **Access model.** Three sensible options, in increasing openness: keep it private and demo it
   live in the viva; publish it with `EMERALD_API_KEY` set and share the key with the examiners;
   publish it fully open. The middle option gives a working public URL without leaving an
   unauthenticated model endpoint exposed indefinitely.
4. **What the page must say.** Whatever is deployed should repeat, on the page itself, what the
   dissertation says: the model ranks applications for human review, it does not approve or
   decline; its probabilities are unreliable on the minority class; and it was fitted on 50
   events from one lender in one origination window.

## 4. What this deployment still is not

The service is a hardened proof of concept, not a production system. Beyond the model's own
statistical limits (Chapter 6), the deployment lacks secrets management, TLS termination and a
gateway rate limiter, image scanning and signed build provenance, an orchestration manifest with
resource limits and rollback, a persistent request log or audit trail, and a
continuous-integration gate. `docs/path_to_production.md` sets out the governance and monitoring
work that would accompany them.
