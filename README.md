# Smart School Project

**Smart School** est un projet d'intelligence artificielle visant à automatiser des processus clés pour les établissements scolaires : l'identification précoce des élèves en difficulté et la correction automatisée de copies manuscrites.

---

## Partie 1 — Prédiction d'Échec Scolaire (Régression)

Cette partie traite d'un problème de **régression supervisée**. L'objectif est de prédire le `score_examen` pour déclencher des interventions pédagogiques ciblées.

### Analyse Exploratoire & Features
* **Dataset :** 630 000 observations, 14 variables prédictives.
* **Variables clés :** L'analyse de corrélation de Spearman et l'**Information Mutuelle (MI)** montrent que `heures_etude` (MI: 0.489) et `assiduite_classe` (MI: 0.128) sont les facteurs les plus déterminants.
* **Taux d'échec :** 25.57 % des élèves ont un score < 50.

### Résultats des Modèles
| Modèle | Val MAE | Test MAE | Statut |
| :--- | :---: | :---: | :--- |
| Baseline (Moyenne) | 15.62 | — | Référence |
| Régression Linéaire | 9.12 | — | / |
| Random Forest | 7.51 | — | / |
| **XGBoost** | **7.24** | **7.22** | **Meilleur Modèle** |
| MLP (Deep Learning) | 8.10 | — | / |

**Performance finale :** XGBoost réduit l'erreur de **53.8 %** par rapport à la baseline, avec une stabilité remarquable (écart-type CV ±0.04).

---

## Partie 2 — Correction Automatique par OCR (Classification)

L'objectif est la reconnaissance de caractères manuscrits (62 classes : 0-9, A-Z, a-z) à partir d'images 28x28 pixels issues du jeu de données **EMNIST**.

### Pipeline Technique
* **Preprocessing :** Normalisation [0, 1], augmentation de données (rotations ±10°, translations) et réduction dimensionnelle via **PCA (64 composantes)** pour les modèles classiques.
* **Architecture CNN :** 3 blocs Conv2D (32/64/128 filtres).
    * BatchNormalization & Dropout (0.4) pour la régularisation.
    * Optimiseur Adam.

### Comparaison des Performances
| Modèle | Accuracy | F1-macro | Statut |
| :--- | :---: | :---: | :--- |
| Baseline | ~5 % | 0.01 | Référence |
| MLP (sklearn) | ~78 % | 0.77 | / |
| SVM (RBF) | ~82 % | 0.81 | / |
| **CNN TensorFlow** | **~88 %** | **0.87** | Meilleur Modèle |