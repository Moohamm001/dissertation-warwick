# EMERALD-AI decision-support service.
#
# Build:  docker build -t emerald-ai .
# Run:    docker run --rm -p 8000:8000 -v "%cd%/All_Funded_2019_Green Loan.xlsx:/app/All_Funded_2019_Green Loan.xlsx:ro" emerald-ai
#
# The dataset is NOT copied into the image: it is a real lending book and must stay out of any
# artefact that could be pushed to a registry. Mount it read-only at run time (see README).
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Serving dependencies only: the container scores with a fitted artefact and never trains, so
# xgboost, imbalanced-learn, matplotlib, shap and mapie are deliberately absent. Dependencies are
# installed before the source is copied so a code change does not invalidate the wheel layer.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Application source, the feature catalogue, and the fitted model. Everything else - above all
# the lending book - is excluded by .dockerignore, so the image serves without any dataset.
COPY emerald_ai/ ./emerald_ai/
COPY data/governance/ ./data/governance/
COPY artefacts/scorer.joblib ./artefacts/scorer.joblib

# Run as an unprivileged user, and give it a writable place for the fitted-model cache.
RUN useradd --create-home --uid 10001 emerald \
 && mkdir -p /app/artefacts \
 && chown -R emerald:emerald /app
USER emerald

EXPOSE 8000

# The probe reports readiness without triggering a fit, so the first ~25 s (model training) is
# reported as "starting" rather than failing. start-period covers that window.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "emerald_ai", "serve", "--host", "0.0.0.0", "--port", "8000"]
