"""Calibration d'un modèle en exploitation — reliability diagram & ECE
(convention M6-B1). Nécessite les vraies cibles : mesure en différé, jamais
en temps réel (cf. notebook §9)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_table(proba: pd.Series, true_label: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Un bin = une plage de probabilité prédite. Colonnes : confiance moyenne
    prédite vs fréquence réellement observée dans ce bin."""
    df = pd.DataFrame({"p": proba.to_numpy(), "y": true_label.to_numpy()})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, n_bins + 1), include_lowest=True)
    g = df.groupby("bin", observed=True).agg(n=("y", "size"), confiance=("p", "mean"), observe=("y", "mean"))
    return g.reset_index()


def expected_calibration_error(proba: pd.Series, true_label: pd.Series, n_bins: int = 10) -> float:
    """ECE : moyenne pondérée des écarts |confiance - observé|. 0 = parfaitement calibré."""
    g = reliability_table(proba, true_label, n_bins)
    w = g["n"] / g["n"].sum()
    return float((w * (g["confiance"] - g["observe"]).abs()).sum())
