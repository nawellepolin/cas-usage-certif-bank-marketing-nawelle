"""Entraînement du modèle servi (Scénario 3) — script autonome.

Réutilise exactement la même logique que le notebook (§5/§8.1), via
`src/preprocess.py`, pour qu'il n'y ait jamais de divergence entre ce qui est
raconté dans le notebook et ce qui est réellement persisté. Permet aussi un
ré-entraînement non interactif, appelable en CI (`--check-only`, cf.
`.github/workflows/ci.yml`) sans écraser le modèle versionné à chaque run.

Usage (depuis la racine du repo, pour que les imports `src.*` se résolvent) :
    python -m src.train                        # ré-entraîne et écrase models/*.joblib+json
    python -m src.train --check-only            # ré-entraîne, vérifie, n'écrit rien sur disque
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import add_engineered_features, get_scenario_columns, make_preprocessor

RANDOM_STATE = 42
SCENARIO = "scenario_3"
MODEL_NAME = "bank_marketing_scenario3"

# Repère : F1 obtenu en §5.7 du notebook sur ce même scénario = 0.464. Un
# ré-entraînement qui tombe nettement sous ce seuil signale une vraie
# régression (code, données, dépendances), pas juste du bruit de CV à
# accepter silencieusement.
MIN_F1 = 0.40


def train(data_path: Path, model_version: str) -> tuple[Pipeline, dict]:
    df = pd.read_csv(data_path, sep=";")
    df_clean = df.drop_duplicates().reset_index(drop=True)
    df_fe = add_engineered_features(df_clean)

    y = (df_fe["y"] == "yes").astype(int)
    X = df_fe.drop(columns=["y"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    pipe = Pipeline([("prep", make_preprocessor(SCENARIO)), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    cat_cols, num_cols = get_scenario_columns(SCENARIO)

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "scenario": SCENARIO,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "dataset_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "hyperparameters": {"class_weight": "balanced", "max_iter": 1000, "random_state": RANDOM_STATE},
        "feature_columns": {"numeric": num_cols, "categorical": cat_cols},
        "target": {"column": "y", "mapping": {"no": 0, "yes": 1}},
        "metrics_holdout": {
            "f1": round(float(f1_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        },
    }
    return pipe, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/bank-additional-full.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--model-version", default="v1.0.0")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Ré-entraîne et vérifie le F1 contre MIN_F1, sans écrire sur disque "
             "(utilisé en CI pour valider la reproductibilité sans écraser le modèle versionné).",
    )
    args = parser.parse_args()

    pipe, metadata = train(args.data, args.model_version)
    metrics = metadata["metrics_holdout"]
    print(f"Entraînement terminé — F1 (classe 'yes') = {metrics['f1']:.4f}, ROC-AUC = {metrics['roc_auc']:.4f}")

    if metrics["f1"] < MIN_F1:
        raise SystemExit(f"F1={metrics['f1']:.4f} < seuil minimal {MIN_F1} — ré-entraînement rejeté.")

    if args.check_only:
        print("--check-only : pipeline de ré-entraînement validé, rien n'est écrit sur disque.")
        return

    args.output_dir.mkdir(exist_ok=True)
    model_path = args.output_dir / f"{MODEL_NAME}.joblib"
    meta_path = args.output_dir / f"{MODEL_NAME}.json"
    joblib.dump(pipe, model_path, compress=3)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modèle persisté : {model_path} ({model_path.stat().st_size / 1024:.1f} Ko)")
    print(f"Métadonnées : {meta_path}")


if __name__ == "__main__":
    main()
