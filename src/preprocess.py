"""Pipeline de préparation des données — sans fuite (split d'abord).

Une seule fonction `make_preprocessor(scenario)` construit le ColumnTransformer
adapté à chaque scénario : seul le PÉRIMÈTRE des colonnes change d'un scénario
à l'autre, pas les traitements (encodage, normalisation) eux-mêmes.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCENARIOS = ["scenario_1", "scenario_2", "scenario_3", "scenario_4"]

SENSITIVE_COLS = ["age", "job", "marital", "education", "default", "housing", "loan"]
CAMPAIGN_HISTORY_COLS = ["campaign", "pdays", "previous", "poutcome", "has_prior_contact"]

ALL_CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
                    "contact", "month", "day_of_week", "poutcome"]
ALL_NUMERIC = ["age", "duration", "campaign", "pdays", "previous", "has_prior_contact",
               "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dérive `has_prior_contact` de `pdays`, puis neutralise le sentinelle.

    999 ne veut pas dire "999 jours", ça veut dire "jamais recontacté" :
    ce n'est pas une vraie valeur numérique. On le remplace par NaN (plutôt
    que de le garder brut ou de supprimer la colonne) pour ne pas perdre la
    récence réelle des ~3.7 % de clients effectivement recontactés — NaN
    sera imputé par la médiane des vraies valeurs dans make_preprocessor,
    en plus de l'indicateur binaire explicite."""
    df = df.copy()
    df["has_prior_contact"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, np.nan)
    return df


def get_scenario_columns(scenario: str) -> tuple[list[str], list[str]]:
    """Retourne (colonnes catégorielles, colonnes numériques) pour un scénario."""
    cat_cols = list(ALL_CATEGORICAL)
    num_cols = list(ALL_NUMERIC)

    if scenario in ("scenario_2", "scenario_3", "scenario_4"):
        num_cols.remove("duration")

    if scenario in ("scenario_3", "scenario_4"):
        cat_cols = [c for c in cat_cols if c not in SENSITIVE_COLS]
        num_cols = [c for c in num_cols if c not in SENSITIVE_COLS]

    if scenario == "scenario_4":
        cat_cols = [c for c in cat_cols if c not in CAMPAIGN_HISTORY_COLS]
        num_cols = [c for c in num_cols if c not in CAMPAIGN_HISTORY_COLS]

    return cat_cols, num_cols


def make_preprocessor(scenario: str) -> ColumnTransformer:
    """Construit le ColumnTransformer pour le scénario demandé
    (scenario_1 | scenario_2 | scenario_3 | scenario_4).

    Le pipeline numérique impute (médiane, calculée sur le train par le CV/
    split) avant de standardiser : seule `pdays` a des NaN (cf.
    add_engineered_features), les autres colonnes traversent l'imputer
    sans effet."""
    cat_cols, num_cols = get_scenario_columns(scenario)
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", numeric_pipeline, num_cols),
        ],
        remainder="drop",
    )
