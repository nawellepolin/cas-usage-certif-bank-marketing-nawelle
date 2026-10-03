# Dockerfile — Bank Marketing Scoring API (Scénario 3)
# Patron validé en M1-B2 : image slim, user non-root, layers ordonnées, healthcheck.

FROM python:3.11-slim

# User non-root
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

WORKDIR /home/appuser/app

# Dépendances d'abord (cache de layer) — fichier dédié API, plus léger que requirements.txt
COPY --chown=appuser:appuser requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Code applicatif : app/ + src/ (préprocessing réutilisé par l'API) + models/
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

RUN mkdir -p /home/appuser/app/logs && chown -R appuser:appuser /home/appuser/app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
