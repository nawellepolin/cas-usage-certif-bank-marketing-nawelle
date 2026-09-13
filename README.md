# Cas d'usage certif — Cibler une campagne de marketing bancaire de manière responsable

> Certification Dev-id — Promo Pros IT. Cas d'usage basé sur le dataset public
> **UCI Bank Marketing** (banque portugaise). Objectif : prédire si un client
> souscrira à un dépôt à terme, à partir d'informations disponibles **avant**
> l'appel, en comparant plusieurs scénarios de variables (performance vs
> fuite d'information vs équité).

---

## 📁 Structure du repo

```
.
├── notebooks/
│   ├── M9_nawelle_bank_marketing.ipynb   # notebook certif — le livrable principal
│   └── journal-de-bord.ipynb             # démarche, choix, difficultés — séance par séance
├── src/
│   ├── preprocess.py                     # pipeline de préparation (4 scénarios), Phase 4
│   ├── train.py                          # benchmark multi-modèles, Phase 5
│   └── evaluate.py                       # métriques (classification déséquilibrée)
├── data/
│   └── bank-additional-full.csv          # dataset fourni (41 188 lignes, séparateur `;`)
├── models/                               # modèles entraînés (gitignorés, Phase 5+)
├── cas_usage_certif/                     # canevas vierge + feuille de route — référence, non modifié
├── datasheet.md                          # documentation du dataset (format Gebru et al.)
├── requirements.txt
└── README.md
```

> `consigne/` (sujet d'examen, mail, dataset original fourni par la formation)
> reste en local mais n'est pas versionné — ce sont des documents d'examen,
> pas des artefacts de travail.

---

## 🚀 Démarrage

```bash
git clone git@github.com:nawellepolin/cas-usage-certif-bank-marketing-nawelle.git
cd cas-usage-certif-bank-marketing-nawelle

python3 -m venv .venv && source .venv/bin/activate   # ou: uv venv && source .venv/bin/activate
pip install -r requirements.txt                       # ou: uv pip install -r requirements.txt

jupyter notebook notebooks/M9_nawelle_bank_marketing.ipynb
```

---

## 🗺️ Avancement (phases, cf. feuille de route dans `cas_usage_certif/`)

| Phase | Statut |
|---|---|
| 1. Cadrer (§1) | ✅ |
| 2. Explorer — identification données (§2) | ✅ |
| 3. Explorer — EDA (§3) | ✅ |
| 4. Préparer (§4) | 🔜 |
| 5. Modéliser & comparer (§5) | 🔜 |
| 6. Arbitrer (§6-7) | 🔜 |
| 6bis. Exposer & fiabiliser — API (§8) | 🔜 |
| 7. Surveiller (§9) | 🔒 pas avant M6 |
| 8. Architecturer (§8.4 + dossier) | 🔒 pas avant M7-M8 |

---

## ✅ Conventions de code

- Python 3.12, `random_state=42` partout, `pathlib.Path` pour les chemins.
- Le notebook (`notebooks/`) raconte la démarche ; le code réutilisable
  (pipeline, boucle de benchmark) vit dans `src/` pour éviter la duplication
  entre les 4 scénarios.
