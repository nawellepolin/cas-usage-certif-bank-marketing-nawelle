# Cas d'usage certif — Cibler une campagne de marketing bancaire de manière responsable

> Certification Dev-id — Promo Pros IT. Cas d'usage basé sur le dataset public
> **UCI Bank Marketing** (banque portugaise). Objectif : prédire si un client
> souscrira à un dépôt à terme, à partir d'informations disponibles **avant**
> l'appel, en comparant plusieurs scénarios de variables (performance vs
> fuite d'information vs équité), puis exposer le modèle retenu via une API.

---

## 📁 Structure du repo

```
.
├── notebooks/
│   ├── M9_nawelle_bank_marketing.ipynb   # notebook certif — le livrable principal
│   └── journal-de-bord.ipynb             # démarche, choix, difficultés — séance par séance
├── src/
│   └── preprocess.py                     # pipeline de préparation, 4 scénarios — réutilisé par le notebook ET l'API
├── app/                                  # API FastAPI (Scénario 3 retenu)
│   ├── main.py                           # /health, /info, /predict
│   ├── schemas.py                        # validation Pydantic stricte
│   └── middleware.py                     # logs structurés (Loguru), sans PII
├── tests/
│   ├── test_api.py                       # tests des routes
│   └── test_model_contract.py            # contrat du modèle persisté
├── models/
│   ├── bank_marketing_scenario3.joblib   # pipeline complet persisté (versionné, < 3 Ko)
│   └── bank_marketing_scenario3.json     # métadonnées (version, métriques, hash dataset)
├── data/
│   └── bank-additional-full.csv          # dataset fourni (41 188 lignes, séparateur `;`)
├── .github/workflows/ci.yml              # tests -> build Docker (bloquant si tests rouges)
├── Dockerfile / .dockerignore / requirements-api.txt
├── cas_usage_certif/                     # canevas vierge + feuille de route — référence, non modifié
├── datasheet.md                          # documentation du dataset (format Gebru et al.)
├── experiments.md                        # traçabilité des runs (convention M1-B1/M4-B1)
├── requirements.txt
└── README.md
```

> `consigne/` (sujet d'examen, mail, dataset original fourni par la formation)
> reste en local mais n'est pas versionné — ce sont des documents d'examen,
> pas des artefacts de travail.

---

## 🚀 Démarrage — notebook

```bash
git clone git@github.com:nawellepolin/cas-usage-certif-bank-marketing-nawelle.git
cd cas-usage-certif-bank-marketing-nawelle

python3 -m venv .venv && source .venv/bin/activate   # ou: uv venv && source .venv/bin/activate
pip install -r requirements.txt                       # ou: uv pip install -r requirements.txt

jupyter notebook notebooks/M9_nawelle_bank_marketing.ipynb
```

## 🚀 Démarrage — API

```bash
# En local
pip install -r requirements-api.txt
uvicorn app.main:app --reload
# -> http://localhost:8000/docs (Swagger)

# Tests
pytest -v tests/

# En conteneur
docker build -t bank-marketing-api:v1.0.0 .
docker run -d -p 8000:8000 bank-marketing-api:v1.0.0
curl http://localhost:8000/health
```

---

## 🗺️ Avancement (phases, cf. feuille de route dans `cas_usage_certif/`)

| Phase | Statut |
|---|---|
| 1. Cadrer (§1) | ✅ |
| 2. Explorer — identification données (§2) | ✅ |
| 3. Explorer — EDA (§3) | ✅ |
| 4. Préparer (§4) | ✅ |
| 5. Modéliser & comparer (§5) | ✅ |
| 6. Arbitrer (§6-7) | ✅ |
| Analyse éthique et réglementaire (RGPD, AI Act, biais) | ✅ |
| 6bis. Exposer & fiabiliser — API (§8) | ✅ |
| 7. Surveiller (§9) | 🔒 pas avant M6 |
| 8. Architecturer (§8.4 + dossier) | 🔒 pas avant M7-M8 |

---

## ✅ Conventions de code

- Python 3.12, `random_state=42` partout, `pathlib.Path` pour les chemins.
- Le notebook (`notebooks/`) raconte la démarche ; le préprocessing vit dans
  `src/preprocess.py`, **importé à la fois par le notebook et par l'API**
  (`app/main.py`) — aucune divergence possible entre entraînement et service.
- Le modèle servi est le **pipeline scikit-learn complet** (préprocessing +
  classifieur), jamais le classifieur seul.
