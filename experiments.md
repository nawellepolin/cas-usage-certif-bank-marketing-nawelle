# Expériences — Cas d'usage Bank Marketing

> Convention M1-B1/M4-B1 : chaque run = date, modèle, dataset, split, hyperparamètres, métriques (test interne), verdict. Détail complet dans `notebooks/M9_nawelle_bank_marketing.ipynb` (§5).

---

## Benchmark principal — 4 scénarios × 2 modèles (validation croisée)

- **Date** : 2026-09-05
- **Dataset** : `data/bank-additional-full.csv` (sha256 `74adfc57...` — cf. notebook §0.5), n=41 188 (41 176 après dédoublonnage)
- **Split** : `test_size=0.2`, `stratify=y`, `random_state=42` → train 32 940 / test 8 236
- **Validation croisée** : `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Scoring CV** : f1, roc_auc, recall, precision, accuracy
- **Préprocessing** : `src/preprocess.py::make_preprocessor(scenario)` — imputation médiane + standardisation (numériques), one-hot (catégorielles), fit uniquement sur train (via Pipeline)

| Scénario | Modèle | Hyperparamètres | F1 (CV) | ROC-AUC (CV) |
|---|---|---|---|---|
| scenario_1 | LogisticRegression | `class_weight="balanced"`, `max_iter=1000` | 0.586 | 0.937 |
| scenario_1 | RandomForest | `class_weight="balanced"`, `n_estimators=200` | 0.630 | 0.943 |
| scenario_2 | LogisticRegression | idem | 0.450 | 0.791 |
| scenario_2 | RandomForest | idem | 0.450 | 0.776 |
| scenario_3 | LogisticRegression | idem | 0.450 | 0.791 |
| scenario_3 | RandomForest | idem | 0.403 | 0.738 |
| scenario_4 | LogisticRegression | idem | 0.453 | 0.783 |
| scenario_4 | RandomForest | idem | 0.474 | 0.782 |

**Verdict par scénario** (meilleur modèle CV retenu pour l'éval finale) :
- scenario_1 → ✅ Random Forest — mais ⛔ écarté globalement : fuite de `duration`, non déployable.
- scenario_2 → ✅ LogReg (meilleur ROC-AUC/recall à F1 quasi égal avec RF).
- scenario_3 → ✅ LogReg — **retenu pour la production** (cf. §6.2 du notebook).
- scenario_4 → ✅ Random Forest — alternative documentée, non retenue par défaut.

## Hyperparamètres — Random Forest (scenario_1, exploration légère)

- **Date** : 2026-09-05
- Comparaison 3 configurations, CV identique au benchmark principal :

| n_estimators | max_depth | F1 (CV) | ROC-AUC (CV) |
|---|---|---|---|
| 100 | None | 0.626 | 0.941 |
| 300 | None | 0.632 | 0.943 |
| 300 | 15 | 0.624 | 0.943 |

**Verdict** : gain marginal (+0.006 F1 pour 100→300 arbres), pas de sur-apprentissage visible. Pas assez significatif pour justifier le coût de calcul supplémentaire en routine — gardé `n_estimators=200` comme compromis.

## Évaluation finale — test set scellé (une seule mesure par scénario)

- **Date** : 2026-09-05
- **Test set** : 8 236 lignes, jamais utilisées pour le choix de modèle/scénario (utilisées une seule fois ici)

| Scénario | Modèle | F1 (test) | Rappel | Précision | ROC-AUC |
|---|---|---|---|---|---|
| scenario_1 | Random Forest | 0.642 | 0.732 | 0.572 | 0.946 |
| scenario_2 | LogReg | 0.462 | 0.645 | 0.359 | 0.800 |
| scenario_3 | LogReg | 0.464 | 0.644 | 0.363 | 0.801 |
| scenario_4 | Random Forest | 0.500 | 0.593 | 0.433 | 0.792 |

**Baseline (`DummyClassifier`, prédit toujours "no", ajoutée le 2026-09-19)** : accuracy 0.887, F1 (classe yes) 0.000 — tous les modèles ci-dessus battent largement ce plancher.

**Verdict final** : **Scénario 3 + régression logistique** retenu pour la recommandation au client (cf. notebook §6.2) — quasi même performance que le Scénario 2 (coût de la fuite déjà payé), sans variables sensibles, le plus rapide (1.1 ms/prédiction) et le plus explicable.

## Bug corrigé (2026-09-05)

Les modèles du dict `MODELS` étaient réutilisés tels quels (même instance Python) entre plusieurs scénarios partageant le même type de modèle (ex. Random Forest pour scenario_1 et scenario_4), ce qui écrasait l'état entraîné du premier par le second lors du stockage dans `fitted_pipelines`. Corrigé avec `sklearn.base.clone()` — chaque scénario obtient désormais sa propre instance de modèle. Les métriques déjà publiées restaient correctes (la prédiction avait lieu juste après l'entraînement, avant l'écrasement) ; seule la réutilisation ultérieure du pipeline stocké était affectée.
