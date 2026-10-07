"""Détection de dérive — PSI / KS / Chi² (convention M6-B1).

On croise 3 méthodes complémentaires plutôt qu'une seule : PSI (indice
d'ampleur, repères conventionnels <0.10 / 0.10-0.25 / >0.25), KS (test
statistique sur numériques), Chi² (sur catégorielles). Un signal statistique
n'est pas encore un diagnostic — c'est `drift_report` qui agrège, l'usage qui
tranche (cf. notebook §9).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


def population_stability_index(reference: pd.Series, current: pd.Series, n_bins: int = 10, eps: float = 1e-6) -> float:
    """PSI entre deux distributions numériques. Bins issus de la RÉFÉRENCE
    uniquement (sinon le PSI n'est plus comparable)."""
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p_ref = np.histogram(reference.dropna(), edges)[0] / len(reference.dropna())
    p_cur = np.histogram(current.dropna(), edges)[0] / len(current.dropna())
    # Lissage anti-zéro puis renormalisation (sinon ln(0) et proportions qui ne somment plus à 1)
    p_ref, p_cur = p_ref + eps, p_cur + eps
    p_ref, p_cur = p_ref / p_ref.sum(), p_cur / p_cur.sum()
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi_verdict(psi: float) -> str:
    """Repères conventionnels — pas une frontière absolue (cf. mini-cours M6-B1)."""
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "à investiguer"
    return "dérive forte"


def ks_pvalue(reference: pd.Series, current: pd.Series) -> float:
    """Test de Kolmogorov-Smirnov — p < 0.05 : assez d'évidence contre l'égalité des distributions."""
    return float(ks_2samp(reference.dropna(), current.dropna()).pvalue)


def chi2_pvalue(reference: pd.Series, current: pd.Series) -> float:
    """Test du Chi² sur une catégorielle — table de contingence référence vs actuel."""
    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()
    categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    table = pd.DataFrame({
        "reference": [ref_counts.get(c, 0) for c in categories],
        "current": [cur_counts.get(c, 0) for c in categories],
    })
    _, p, _, _ = chi2_contingency(table.T)
    return float(p)


def drift_report(df_reference: pd.DataFrame, df_current: pd.DataFrame,
                  numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    """Tableau de synthèse : une ligne par feature, calcul ET interprétation séparés."""
    rows = []
    for col in numeric_cols:
        psi = population_stability_index(df_reference[col], df_current[col])
        p = ks_pvalue(df_reference[col], df_current[col])
        rows.append({"feature": col, "type": "numérique", "psi": round(psi, 3),
                      "p_value": p, "verdict": psi_verdict(psi)})
    for col in categorical_cols:
        p = chi2_pvalue(df_reference[col], df_current[col])
        rows.append({"feature": col, "type": "catégorielle", "psi": None,
                      "p_value": p, "verdict": "dérive" if p < 0.05 else "stable"})
    return pd.DataFrame(rows)
