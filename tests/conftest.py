"""Fixtures partagées pour les tests de l'API bank-marketing."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

MODEL_PATH = Path(__file__).parent.parent / "models" / "bank_marketing_scenario3.joblib"


@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient avec lifespan déclenché (modèle chargé).

    Skip propre tant que le modèle n'est pas présent — lancer d'abord
    la cellule §8.1 du notebook pour le produire dans models/.
    """
    if not MODEL_PATH.exists():
        pytest.skip(f"Modèle absent : {MODEL_PATH}. Exécute §8.1 du notebook d'abord.")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> dict:
    """Payload valide — reprend l'exemple documenté dans le schéma Pydantic."""
    return {
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp_var_rate": 1.1,
        "cons_price_idx": 93.994,
        "cons_conf_idx": -36.4,
        "euribor3m": 4.857,
        "nr_employed": 5191.0,
    }
