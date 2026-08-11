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
disclosure, and §4 below is where to decide whether it is acceptable to publish.

## 2. Deploying to Render (step by step)

Deployment uses the **Docker runtime deliberately**. The repository still tracks the lending
book, so a plain Python runtime would clone it onto the server; building an image applies
`.dockerignore`, which excludes it. The build context was measured: **211 KB** in total
(package 171 KB, feature catalogue 13 KB, model artefact 26.5 KB, requirements 0.7 KB), with the
`.xlsx` excluded from both locations it exists in.

1. **Refresh the artefact** on the machine that has the data, and push it:

   ```powershell
   python -m emerald_ai build-artefact
   git add artefacts/scorer.joblib
   git commit -m "Refresh deployed model artefact"
   git push
   ```

2. **Create the service.** On Render: *New → Blueprint*, point it at the GitHub repository, and
   it will read `render.yaml` (Docker runtime, free plan, health check on `/health`).

3. **Decide the access model** in the Render dashboard before the first deploy finishes:
   - leave `EMERALD_API_KEY` unset for a fully open demo, or
   - set it to a value and share that key with the examiners, which keeps `/api/*` closed to
     anyone else while the UI and `/health` stay reachable.

4. **Verify** once the deploy is live:

   ```powershell
   curl https://<your-service>.onrender.com/health
   ```

   A healthy service returns HTTP 200 with `"ready": true` and the operating point. The free
   plan sleeps when idle, so the first request after a pause takes a few seconds to wake; the
   model itself loads from the artefact in about 0.02 s.

## 3. Deploying elsewhere

**Container (any host that runs images):**

```powershell
docker compose up --build          # local
```

The compose file mounts the dataset read-only for local use; a remote deployment should omit
that mount entirely and rely on the committed artefact.

**Any other host** (Railway, Fly.io, Azure App Service, Google Cloud Run, a university VM) works
the same way: build the image from the same `Dockerfile`, or run `python -m emerald_ai serve`
with the package, the feature catalogue and the artefact present. The CLI reads `PORT` and
`HOST` from the environment, so platforms that inject a port need no extra configuration.

**Environment variables:**

| Variable | Effect |
|---|---|
| `EMERALD_API_KEY` | When set, `/api/*` requires a matching `X-API-Key` header. The UI and `/health` stay reachable. Unset means open. |
| `EMERALD_RATE_LIMIT` | Requests per client IP per minute (default 60; `0` disables). In-process and per-instance. |

## 4. Decisions to take before publishing a URL

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

## 5. What this deployment still is not

The service is a hardened proof of concept, not a production system. Beyond the model's own
statistical limits (Chapter 6), the deployment lacks secrets management, TLS termination and a
gateway rate limiter, image scanning and signed build provenance, an orchestration manifest with
resource limits and rollback, a persistent request log or audit trail, and a
continuous-integration gate. `docs/path_to_production.md` sets out the governance and monitoring
work that would accompany them.
