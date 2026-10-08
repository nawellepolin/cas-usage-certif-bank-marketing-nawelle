# Datasheet — bank-additional-full.csv

> Format Gebru et al., *"Datasheets for Datasets"* (2018) — 7 sections.
> Auteur : Nawelle Polin · Version 1.0.0 · 2026-09-19

---

## 1. Motivation

**Pourquoi ce dataset existe-t-il ?** Il a été constitué à l'origine pour une recherche académique (Moro, Cortez & Rita, *"A data-driven approach to predict the success of bank telemarketing"*, 2014) étudiant la capacité de modèles prédictifs à anticiper le succès de campagnes de démarchage téléphonique d'une banque de détail portugaise, menées entre mai 2008 et novembre 2010 — pour proposer des dépôts à terme à ses clients.

**Qui l'a créé et financé ?** Les auteurs académiques cités ci-dessus, à partir des données opérationnelles d'une banque portugaise anonymisée dans la publication. Le fichier est aujourd'hui distribué publiquement par le UCI Machine Learning Repository.

**Pourquoi l'utilise-t-on ici ?** Pour ce cas d'usage de certification (promo Dev-id), il sert de simulation réaliste d'un problème de ciblage commercial bancaire, avec un déséquilibre de classes et des variables socio-démographiques sensibles — exactement le type de terrain qui permet de travailler la compétence C2 (risques éthiques) autant que C4/C5 (choix et entraînement de modèle).

---

## 2. Composition

**Volume** : 41 188 lignes (dont 12 doublons stricts, retirés en préparation) × 20 variables explicatives + 1 variable cible `y`.

**Distribution de la cible** : 11.3 % `yes` (souscription) / 88.7 % `no` — fort déséquilibre (cf. §3.4 du notebook).

**Schéma complet** : voir §2.3 du notebook (`notebooks/M9_nawelle_bank_marketing.ipynb`) pour le dictionnaire variable par variable (type, plage, sensibilité).

**Variables sensibles signalées** (ciblage commercial, pas art. 9 RGPD) : `age`, `job`, `marital`, `education`, et dans une moindre mesure `default`/`housing`/`loan` (situation financière). Aucune variable ne relève des catégories particulières de l'article 9 RGPD (santé, origine, opinions, orientation sexuelle, données biométriques/génétiques).

**Identifiant direct** : aucun — pas de nom, téléphone ou numéro client dans le fichier.

**Quasi-identifiant résiduel** : la combinaison `age` + `job` + `marital` + `education` (+ `month`) n'a **pas été testée en k-anonymat** — à traiter par prudence comme donnée personnelle, pas comme anonymisée au sens légal (cf. notebook, section Analyse éthique §B.1).

**Résumé du verdict éthique chiffré** (cf. notebook, section Analyse éthique) :
- Disparate Impact sur les étiquettes historiques : `age` 0.181, `job` 0.219, `marital` 0.677, `education` 0.352 — tous sous le seuil d'alerte 0.8.
- Disparate Impact sur les prédictions du modèle retenu (Scénario 3) : `age` 0.154, `job` 0.195, `marital` 0.326, `education` 0.510 — amplification sur 3 des 4 variables malgré leur retrait des entrées du modèle.
- Intersectionnalité `age × job` : DI = 0.050, écart ×20 entre le groupe le mieux et le moins bien servi.

**Ce que le dataset ne contient pas** : pas d'identifiant client (donc pas de suivi longitudinal fiable d'un même client à travers plusieurs campagnes, en dehors des champs agrégés `pdays`/`previous`/`poutcome`), pas de variable de revenu direct, pas de localisation géographique.

---

## 3. Processus de collecte

Les données n'ont **pas été collectées par nous** : elles proviennent d'un export déjà constitué, fourni par la formation (dossier `consigne/` de l'examen), lui-même une republication par le UCI ML Repository du dataset original Moro/Cortez/Rita.

**Ce qu'on ne sait pas et ne peut pas vérifier** : le protocole de consentement des clients au moment de la collecte originale (2008-2010), la méthode exacte d'échantillonnage (exhaustif sur la clientèle contactée, ou un sous-ensemble), et si une procédure d'anonymisation formelle a été appliquée avant publication par les auteurs académiques.

**Point de vigilance temporel** : les variables de contexte socio-économique (`emp.var.rate`, `euribor3m`, `cons.price.idx`...) reflètent la crise financière de 2008-2010 — un contexte macroéconomique daté, qui ne transposerait pas directement à une situation actuelle si ce dataset servait de base à un vrai déploiement (cf. notebook, risques de dérive à anticiper en Phase 7/§9, hors périmètre de ce cas d'usage jusqu'à M5).

---

## 4. Preprocessing appliqué

Détaillé et versionné dans `src/preprocess.py` et le notebook §4 :
- 12 doublons stricts retirés avant tout traitement.
- Split stratifié 80/20 (train 32 940 / test 8 236), effectué **avant** toute transformation (anti-fuite).
- `has_prior_contact` dérivée de `pdays` (999 → NaN → imputation médiane sur le train, + indicateur binaire).
- Encodage one-hot des catégorielles, standardisation des numériques — appliqués via `ColumnTransformer`, un même traitement pour les 4 scénarios, seul le périmètre de colonnes changeant (`src/preprocess.py::make_preprocessor`).

Aucune transformation n'altère les lignes individuelles au-delà de ces étapes (pas de ré-échantillonnage, pas de génération de données synthétiques).

---

## 5. Usages prévus / à éviter

**Usages prévus** :
- Entraînement et comparaison de modèles de classification binaire (probabilité de souscription) pour prioriser un effort de démarchage commercial.
- Support pédagogique pour travailler la détection et la mitigation de biais algorithmiques (exercice de certification Dev-id).
- Référence pour tester une architecture de service de scoring (API, cf. §8 du notebook).

**Usages à éviter** :
- **Déployer le Scénario 1 (avec `duration`) en production** — fuite d'information, performance non reproductible en conditions réelles (cf. §1.2, §5.7).
- **Utiliser les variables sensibles (`age`, `job`, `marital`, `education`) pour exclure ou refuser un service**, plutôt que pour prioriser un contact commercial positif — l'usage prévu est additif (qui contacter en priorité), pas soustractif (qui écarter).
- **Automatiser entièrement la décision de contact sans supervision humaine réelle** — risque de basculer dans le champ de l'article 22 RGPD (cf. notebook, section B.3) si le conseiller suit le score sans examen.
- **Réutiliser ce modèle pour une décision de crédit ou d'éligibilité à un service essentiel** — changerait la qualification AI Act (actuellement "risque minimal", cf. section C) vers "haut risque" (Annexe III).
- **Partager ce jeu de données hors du cadre pédagogique sans revalider le risque de ré-identification** (quasi-identifiants non testés, cf. section 2).
- **Considérer le contexte socio-économique de 2008-2010 comme représentatif d'une situation actuelle** sans réentraînement sur des données récentes.

---

## 6. Distribution

**Format** : CSV, séparateur `;`, encodage standard. Fichier `data/bank-additional-full.csv` (gitignoré par défaut dans ce repo sauf exception explicite, cf. `.gitignore`).

**Source publique** : [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).

**Conditions d'usage** : dataset public à vocation académique (UCI ML Repository) — vérifier les conditions de licence exactes avant tout usage commercial réel allant au-delà du cadre pédagogique de cette certification.

---

## 7. Maintenance

**Responsable de cette datasheet** : Nawelle Polin (`npolin@dev-id.fr`).

**Statut** : dataset statique (snapshot historique 2008-2010), aucune mise à jour prévue de notre part — un déploiement réel nécessiterait un dataset vivant et réentraîné périodiquement (hors périmètre de ce cas d'usage jusqu'à M5, traité en Phase 7/§9 du canevas).

**Signalement d'un problème** : via le dépôt Git de ce projet (issues) ou directement auprès de l'auteure.

**Versioning** : v1.0.0 — première version, couvre les Phases 1 à 6 du cas d'usage (cadrage à arbitrage des scénarios).
