"""API de scoring — Bank Marketing (Scénario 3). Point d'entrée."""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from app.middleware import LoggingMiddleware
from app.schemas import ContactAttempt, HealthResponse, Prediction
from src.preprocess import add_engineered_features

# --- Métrique métier : distribution des classes prédites (cf. M5-B1/M6-B1) --
# Counter, pas Gauge : on compte des événements cumulés, Prometheus en déduit
# un débit via rate(). Label à cardinalité FINIE uniquement ("yes"/"no") —
# jamais un request_id ou une valeur continue en label (cf. piège M5-B1).
PREDICTIONS = Counter(
    "bank_marketing_predictions_total",
    "Nombre de prédictions émises, par classe prédite",
    ["predicted_class"],
)

# --- Loguru : logs d'accès uniquement, jamais de PII (cf. middleware) -------

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)
logger.add(
    LOGS_DIR / "api.log",
    rotation="10 MB",
    retention="7 days",
    compression="gz",
    serialize=True,
    enqueue=True,
    level="INFO",
)


# --- Lifespan : charge le modèle une seule fois au démarrage ---------------

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_NAME = "bank_marketing_scenario3"
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.joblib"
META_PATH = MODELS_DIR / f"{MODEL_NAME}.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modèle introuvable : {MODEL_PATH}")
    if not META_PATH.exists():
        raise RuntimeError(f"Métadonnées introuvables : {META_PATH}")

    app.state.model = joblib.load(MODEL_PATH)
    app.state.metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    logger.info(
        "Modèle chargé : {name} {version}",
        name=app.state.metadata.get("model_name", MODEL_PATH.stem),
        version=app.state.metadata["model_version"],
    )
    yield
    app.state.model = None
    logger.info("Modèle libéré")


app = FastAPI(
    title="Bank Marketing Scoring API",
    version="0.1.0",
    description="Probabilité de souscription à un dépôt à terme (Scénario 3 — sans variables sensibles).",
    lifespan=lifespan,
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Métriques HTTP automatiques (latence, RPS, codes retour) + endpoint /metrics.
# Exclu du schéma OpenAPI pour ne pas polluer la doc Swagger (cf. M5-B1).
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# Pré-câblage pour une authentification (M5+) — pas encore exigée en M1-B2/basique.
bearer_scheme = HTTPBearer(auto_error=False)


# --- Routes ------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Vérifie que le service est vivant ET que le modèle est chargé."""
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return HealthResponse(status="ok")


@app.get("/info")
async def info() -> dict:
    """Métadonnées du modèle actuellement servi — pour vérifier en prod quelle version tourne."""
    meta = app.state.metadata
    return {
        "api_version": app.version,
        "model_name": meta.get("model_name", MODEL_PATH.stem),
        "model_version": meta["model_version"],
        "scenario": meta.get("scenario"),
        "created_at": meta["created_at"],
        "sklearn_version": meta["sklearn_version"],
        "dataset_sha256": meta["dataset_sha256"],
        "metrics_holdout": meta["metrics_holdout"],
    }


@app.post("/predict", response_model=Prediction, status_code=status.HTTP_200_OK)
async def predict(
    attempt: ContactAttempt,
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> Prediction:
    """Probabilité de souscription pour un client, à partir des informations
    disponibles AVANT l'appel (cf. §1.2 du notebook — aucune variable sensible,
    aucune donnée connue seulement après l'appel)."""
    request_id = request.state.request_id
    try:
        raw = {
            "contact": attempt.contact,
            "month": attempt.month,
            "day_of_week": attempt.day_of_week,
            "campaign": attempt.campaign,
            "pdays": attempt.pdays,
            "previous": attempt.previous,
            "poutcome": attempt.poutcome,
            "emp.var.rate": attempt.emp_var_rate,
            "cons.price.idx": attempt.cons_price_idx,
            "cons.conf.idx": attempt.cons_conf_idx,
            "euribor3m": attempt.euribor3m,
            "nr.employed": attempt.nr_employed,
        }
        X = add_engineered_features(pd.DataFrame([raw]))
        proba = float(app.state.model.predict_proba(X)[0, 1])
    except Exception as exc:
        # Détail complet dans les logs serveur uniquement — jamais dans la réponse HTTP
        # (les erreurs pandas/sklearn exposent des noms de colonnes internes).
        logger.bind(request_id=request_id).exception("Échec de la prédiction")
        raise HTTPException(
            status_code=500,
            detail=f"Échec de la prédiction (request_id={request_id})",
        ) from exc

    predicted_class = "yes" if proba >= 0.5 else "no"
    PREDICTIONS.labels(predicted_class=predicted_class).inc()

    return Prediction(
        probability=round(proba, 4),
        prediction=predicted_class,
        model_version=app.state.metadata["model_version"],
        request_id=request_id,
    )
