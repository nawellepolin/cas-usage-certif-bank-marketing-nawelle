"""Contract test du modèle servi par l'API — premier filet avant les routes.

Reprend le patron `contract_test_model` (M1-B1 mini-cours 05) : si le `.joblib`
n'a pas la bonne signature, aucun test d'API ne peut être fiable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocess import add_engineered_features  # noqa: E402

MODEL_PATH = Path(__file__).parent.parent / "models" / "bank_marketing_scenario3.joblib"


@pytest.fixture(scope="module")
def loaded_model():
    if not MODEL_PATH.exists():
        pytest.skip(f"Modèle absent : {MODEL_PATH}. Exécute §8.1 du notebook d'abord.")
    return joblib.load(MODEL_PATH)


@pytest.fixture
def raw_row() -> pd.DataFrame:
    """Ligne brute, colonnes exactement telles qu'attendues par le pipeline
    (noms avec points, `has_prior_contact` dérivée de `pdays`)."""
    row = pd.DataFrame([{
        "contact": "cellular", "month": "may", "day_of_week": "mon",
        "campaign": 1, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
        "emp.var.rate": 1.1, "cons.price.idx": 93.994, "cons.conf.idx": -36.4,
        "euribor3m": 4.857, "nr.employed": 5191.0,
    }])
    return add_engineered_features(row)


def test_model_contract(loaded_model, raw_row: pd.DataFrame) -> None:
    prediction = loaded_model.predict(raw_row)
    proba = loaded_model.predict_proba(raw_row)

    assert prediction.shape == (1,), f"shape predict={prediction.shape}, attendu (1,)"
    assert proba.shape == (1, 2), f"shape predict_proba={proba.shape}, attendu (1, 2)"
    assert int(prediction[0]) in (0, 1), f"classe inattendue : {prediction[0]}"
    assert 0.0 <= float(proba[0, 1]) <= 1.0, "probabilité hors [0, 1]"


def test_model_handles_unknown_category(loaded_model, raw_row: pd.DataFrame) -> None:
    """OneHotEncoder(handle_unknown='ignore') ne doit pas planter sur une
    modalité jamais vue à l'entraînement (ex. day_of_week='sat')."""
    row = raw_row.copy()
    row["day_of_week"] = "sat"
    proba = loaded_model.predict_proba(row)
    assert 0.0 <= float(proba[0, 1]) <= 1.0
