"""Schémas Pydantic de l'API bank-marketing — Scénario 3 (sans variables sensibles).

Convention M1-B2 : le schéma d'entrée doit refléter exactement les
`feature_columns` du modèle persisté (cf. `models/bank_marketing_scenario3.json`),
pas approximer. Les bornes viennent de l'EDA (§3 du notebook).

⚠️ `has_prior_contact` n'est PAS un champ d'entrée : c'est une variable dérivée
de `pdays` (cf. `src/preprocess.py::add_engineered_features`), calculée côté
API avant l'appel au pipeline — le client n'a pas à la fournir lui-même.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Contact = Literal["cellular", "telephone"]
Month = Literal["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DayOfWeek = Literal["mon", "tue", "wed", "thu", "fri"]
Poutcome = Literal["failure", "nonexistent", "success"]


class ContactAttempt(BaseModel):
    """Entrée de /predict — informations disponibles AVANT l'appel (cf. §1.2)."""

    contact: Contact = Field(..., description="Canal de contact")
    month: Month = Field(..., description="Mois envisagé pour le contact")
    day_of_week: DayOfWeek = Field(..., description="Jour de la semaine envisagé")
    campaign: int = Field(..., ge=1, le=60, description="Nombre de contacts déjà effectués dans cette campagne, dernier inclus")
    pdays: int = Field(..., ge=0, le=999, description="Jours depuis le dernier contact d'une campagne précédente (999 = jamais recontacté)")
    previous: int = Field(..., ge=0, le=10, description="Nombre de contacts avant cette campagne")
    poutcome: Poutcome = Field(..., description="Résultat de la campagne précédente")
    emp_var_rate: float = Field(..., ge=-4.0, le=2.0, description="Taux de variation de l'emploi (indicateur trimestriel)")
    cons_price_idx: float = Field(..., ge=92.0, le=95.0, description="Indice des prix à la consommation")
    cons_conf_idx: float = Field(..., ge=-51.0, le=-26.0, description="Indice de confiance des consommateurs")
    euribor3m: float = Field(..., ge=0.0, le=6.0, description="Taux Euribor à 3 mois")
    nr_employed: float = Field(..., ge=4900.0, le=5300.0, description="Nombre de salariés (indicateur trimestriel)")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class Prediction(BaseModel):
    """Sortie de /predict."""

    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilité de souscription")
    prediction: Literal["yes", "no"] = Field(..., description="Classe prédite au seuil 0.5")
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
