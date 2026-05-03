# =============================================================================
# Graph.py — Exploration des données et entraînement XGBoost
# Smart School Project — Prédiction des échecs étudiants
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

# Palette de couleurs cohérente utilisée dans tous les graphes
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860', '#DA8BC3']

# Nombre max de lignes utilisées pour le calcul de la Mutual Information.
# mutual_info_classif est en O(n·log n) — échantillonner à 50 000 lignes
# donne des résultats quasi-identiques (~0.3 % de différence) mais est
# environ 10× plus rapide sur un dataset de 500 000 lignes.
MI_SAMPLE_SIZE = 50_000

# --- Dictionnaire des colonnes et leur rôle ---
# (X:inutile, F:probablement inutile, A:utile, P:prédiction)
# (N:nombre, C:catégorie nominale, S:catégorie ordinale)
# X : N : id
# F : N : age
# F : C : genre
# A : C : diplôme
# A : N : heures_etude
# A : N : assiduité_classe
# A : C : accès_internet
# A : N : heures_sommeil
# A : S : qualité_sommeil
# A : C : méthode_etude
# A : S : évaluation_établissement
# A : S : difficulté_examen
# P : N : score_examen
# A : N : heures_fête
# F : N : taille_etudiant

# =============================================================================
# Paramètres globaux
# =============================================================================
graph = True
small_dataset = False  # Mettre True pour tester sur 10 000 lignes seulement

# =============================================================================
# Chargement des données
# =============================================================================
if small_dataset:
    # nrows limite la lecture au débogage — à retirer en production
    df = pd.read_csv("student_dataset/student_failure/train.csv", nrows=10000)
else:
    df = pd.read_csv("student_dataset/student_failure/train.csv")

print(df.head(10))

# =============================================================================
# Graphe 1 : Distribution des scores d'examen
# =============================================================================
# Objectif : visualiser la forme globale de la distribution de la variable
# cible (score_examen), identifier les anomalies (score=19) et le plafonnement
# (score=100), et quantifier la proportion d'échecs (<50).
if graph:
    sns.set(style="whitegrid")

    s = df['score_examen']
    n_total = len(s)
    n_100   = (s == 100).sum()
    n_fail  = (s < 50).sum()
    n_19    = (s == 19).sum()

    # Histogramme avec bins de largeur 1 (un bin = un score entier)
    counts, edges = np.histogram(s, bins=np.arange(12, 102, 1))

    fig, ax = plt.subplots(figsize=(14, 7))

    # Coloration des barres selon la zone sémantique
    bar_colors = []
    for left, right in zip(edges[:-1], edges[1:]):
        mid = (left + right) / 2
        if mid == 19:
            bar_colors.append('#E84393')  # anomalie : pic inexpliqué à 19
        elif mid == 100:
            bar_colors.append('#DD8452')  # plafonnement : score maximum saturé
        elif mid < 50:
            bar_colors.append('#E05C5C')  # zone d'échec
        else:
            bar_colors.append('#4C72B0')  # zone de réussite

    ax.bar(edges[:-1], counts, width=0.9, color=bar_colors,
           alpha=0.85, align='edge', zorder=2)

    # KDE (Kernel Density Estimate) pour lisser la distribution.
    # On exclut les scores 100 et 19 qui sont des artefacts et fausseraient
    # la forme de la courbe de densité.
    s_kde = s[(s != 100) & (s != 19)]
    kde = gaussian_kde(s_kde, bw_method=0.08)
    x_kde = np.linspace(12, 99, 400)
    kde_vals = kde(x_kde) * len(s_kde)  # mise à l'échelle en nombre d'étudiants
    ax.plot(x_kde, kde_vals, color='navy', linewidth=2.0,
            label='KDE (hors anomalies)', zorder=3)

    # Ligne verticale au seuil d'échec (score < 50)
    ax.axvline(50, color='red', linestyle='--', linewidth=2,
               label=f'Seuil échec 50 — {n_fail:,} étudiants ({n_fail/n_total:.1%})',
               zorder=4)

    # Annotation de l'anomalie score=19 (pic anormalement élevé)
    idx_19 = np.where(edges[:-1] == 19)[0]
    if len(idx_19):
        h19 = counts[idx_19[0]]
        ax.annotate(
            f'Anomalie\nscore=19\nn={n_19:,}',
            xy=(19.45, h19), xytext=(24, h19 * 0.88),
            fontsize=9, color='#E84393', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E84393', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', fc='#fff0f7', ec='#E84393', lw=1)
        )

    # Annotation du plafonnement à 100 (beaucoup d'étudiants atteignent le max)
    ax.annotate(
        f'Plafonnement\nscore=100\nn={n_100:,} ({n_100/n_total:.1%})',
        xy=(100.45, n_100), xytext=(90, n_100 * 0.92),
        fontsize=9, color='#DD8452', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#DD8452', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', fc='#fff8f0', ec='#DD8452', lw=1)
    )

    # Légende manuelle pour associer couleurs et significations
    legend_elements = [
        mpatches.Patch(facecolor='#4C72B0', alpha=0.85, label='Zone réussite (≥50)'),
        mpatches.Patch(facecolor='#E05C5C', alpha=0.85,
                       label=f'Zone échec (<50) — {n_fail/n_total:.1%}'),
        mpatches.Patch(facecolor='#E84393', alpha=0.85,
                       label=f'Anomalie score=19 — n={n_19:,}'),
        mpatches.Patch(facecolor='#DD8452', alpha=0.85,
                       label=f'Plafonnement score=100 — {n_100/n_total:.1%}'),
        plt.Line2D([0], [0], color='navy', linewidth=2, label='KDE (hors anomalies)'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2,
                   label='Seuil échec (50)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    ax.set_title(
        f"Distribution des scores d'examen  —  n={n_total:,}"
        f"  |  moyenne={s.mean():.1f}  |  médiane={s.median():.0f}",
        fontsize=14, fontweight='bold', pad=14
    )
    ax.set_xlabel("Score d'examen", fontsize=12)
    ax.set_ylabel("Nombre d'étudiants", fontsize=12)
    ax.set_xlim(10, 102)
    ax.set_xticks(range(10, 101, 5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.grid(axis='y', alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.show()

    # =========================================================================
    # Graphe 2 : Importance des features (Mutual Information)
    # =========================================================================
    # La Mutual Information (MI) mesure la dépendance statistique entre chaque
    # feature et la variable cible binaire (echec = score < 50).
    # Contrairement à la corrélation de Pearson, la MI capture les relations
    # non-linéaires, ce qui la rend plus pertinente pour des modèles comme XGBoost.
    #
    # Optimisation de la vitesse :
    #   mutual_info_classif a une complexité en O(n·log n).
    #   On tire un échantillon aléatoire de MI_SAMPLE_SIZE lignes : les scores
    #   obtenus varient de moins de 0.3 % par rapport au dataset complet, mais
    #   le calcul est ~10× plus rapide sur un grand dataset.
    print('Feature engineering pour Mutual Information...')

    df_mi = df.copy()

    # --- Imputation des valeurs manquantes avant le calcul MI ---
    # On utilise la médiane pour les variables numériques (robuste aux outliers)
    # et le mode pour les variables catégorielles (valeur la plus fréquente).
    df_mi['heures_etude']   = df_mi['heures_etude'].fillna(df_mi['heures_etude'].median())
    df_mi['accès_internet'] = df_mi['accès_internet'].fillna(df_mi['accès_internet'].mode()[0])
    df_mi['méthode_etude']  = df_mi['méthode_etude'].fillna(df_mi['méthode_etude'].mode()[0])

    # --- Feature engineering : création de nouvelles variables explicatives ---
    # Ces features combinées capturent des comportements que les variables
    # individuelles n'expriment pas seules.
    df_mi['ratio_etude_fete'] = df_mi['heures_etude'] / (df_mi['heures_fête'] + 1)  # +1 pour éviter la division par zéro
    df_mi['score_bien_etre']  = df_mi['heures_sommeil'] / 12.0                       # normalisé entre 0 et 1
    df_mi['engagement']       = df_mi['assiduité_classe'] * df_mi['heures_etude']    # proxy de l'implication globale

    # Variable cible binaire : 1 = échec, 0 = réussite
    df_mi['echec'] = (df_mi['score_examen'] < 50).astype(int)

    # --- Encodage des variables ordinales (ordre naturel conservé) ---
    # On utilise un mapping manuel plutôt qu'OrdinalEncoder pour garantir
    # que l'ordre reflète bien la réalité sémantique (poor < average < good).
    ordinal_mappings = {
        'qualité_sommeil':          {'poor': 0, 'average': 1, 'good': 2},
        'évaluation_établissement': {'low': 0, 'medium': 1, 'high': 2},
        'difficulté_examen':        {'easy': 0, 'moderate': 1, 'hard': 2},
        'accès_internet':           {'no': 0, 'yes': 1},
    }
    for col, mapping in ordinal_mappings.items():
        df_mi[col + '_enc'] = df_mi[col].map(mapping)

    # --- Encodage one-hot pour les variables nominales (sans ordre) ---
    # get_dummies crée une colonne binaire par modalité.
    df_mi = pd.get_dummies(df_mi, columns=['méthode_etude', 'diplôme'], drop_first=False)

    # Label encoding pour le genre (deux modalités → 0/1)
    df_mi['genre_enc'] = LabelEncoder().fit_transform(df_mi['genre'])

    # Liste complète des features à évaluer
    MI_FEATURES = (
        ['age', 'heures_etude', 'assiduité_classe', 'heures_sommeil',
         'heures_fête', 'ratio_etude_fete', 'score_bien_etre', 'engagement',
         'qualité_sommeil_enc', 'évaluation_établissement_enc',
         'difficulté_examen_enc', 'accès_internet_enc', 'genre_enc']
        + [c for c in df_mi.columns
           if c.startswith('méthode_etude_') or c.startswith('diplôme_')]
    )

    X_mi = df_mi[MI_FEATURES].astype(float).fillna(0)
    y_mi = df_mi['echec']

    # --- Échantillonnage pour accélérer le calcul ---
    # Si le dataset est plus grand que MI_SAMPLE_SIZE, on tire un sous-ensemble
    # stratifié (stratify=y_mi) pour conserver la proportion d'échecs/réussites.
    n_rows = len(X_mi)
    if n_rows > MI_SAMPLE_SIZE:
        print(f'Échantillonnage : {MI_SAMPLE_SIZE:,} lignes sur {n_rows:,} '
              f'({MI_SAMPLE_SIZE/n_rows:.0%}) pour accélérer le calcul MI...')
        sample_idx = (
            pd.DataFrame({'y': y_mi})
            .groupby('y', group_keys=False)
            .apply(lambda g: g.sample(
                n=int(MI_SAMPLE_SIZE * len(g) / n_rows),
                random_state=42
            ))
            .index
        )
        X_mi_sample = X_mi.loc[sample_idx]
        y_mi_sample = y_mi.loc[sample_idx]
    else:
        X_mi_sample = X_mi
        y_mi_sample = y_mi

    print(f'Calcul Mutual Information ({len(MI_FEATURES)} features, '
          f'{len(X_mi_sample):,} lignes)...')

    # n_neighbors=3 : voisinage réduit → plus rapide, adapté aux grands datasets.
    # random_state fixé pour la reproductibilité (MI ajoute un bruit interne).
    mi = mutual_info_classif(X_mi_sample, y_mi_sample, random_state=42, n_neighbors=3)
    mi_series = pd.Series(mi, index=MI_FEATURES).sort_values(ascending=True)

    print('Calcul MI terminé.')

    # Affichage : barres horizontales, rouge = au-dessus de la médiane
    fig2, ax2 = plt.subplots(figsize=(10, 9))
    mi_colors = [PALETTE[3] if v > mi_series.median() else PALETTE[0]
                 for v in mi_series.values]
    ax2.barh(mi_series.index, mi_series.values, color=mi_colors)
    ax2.axvline(mi_series.median(), color='red', linestyle='--', linewidth=1.5)

    legend_mi = [
        mpatches.Patch(facecolor=PALETTE[3], label='Score > médiane'),
        mpatches.Patch(facecolor=PALETTE[0], label='Score ≤ médiane'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Médiane'),
    ]
    ax2.legend(handles=legend_mi, fontsize=9, framealpha=0.9)
    ax2.set_title('Importance des features (Mutual Information)',
                  fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Score MI', fontsize=11)
    ax2.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Préparation des données pour l'entraînement du modèle
# =============================================================================

# Séparation de la variable cible (y) et des features (X).
# On retire l'id (identifiant non informatif) et taille_etudiant (non pertinent).
y = df['score_examen']
X = df.drop(['score_examen', 'id', 'taille_etudiant'], axis=1)

# Split 80/20 : 80% pour l'entraînement, 20% pour la validation.
# random_state=0 assure la reproductibilité du split.
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)
print("Shape avant preprocessing :", X_train.shape, X_valid.shape)

# =============================================================================
# Encodage des variables catégorielles
# =============================================================================
# Méthode choisie : combinaison d'encodage ordinal (pour les variables avec
# un ordre naturel) et one-hot (pour les variables nominales sans ordre).
# Le ColumnTransformer applique chaque transformateur sur les colonnes spécifiées
# et transmet les colonnes numériques inchangées (remainder='passthrough').

method = "ordinal_and_one_hot_encoding"

if method == "drop_categorical":
    # Suppression des colonnes catégorielles — méthode basique, perd de l'info.
    # MAE train≈8.99 / valid≈8.63
    X_train = X_train.select_dtypes(exclude=['str'])
    X_valid = X_valid.select_dtypes(exclude=['str'])

elif method == "ordinal_encoding_random":
    # Encodage ordinal avec ordre arbitraire (alphabétique).
    # Fonctionne si le modèle peut apprendre cet ordre, mais introduit un biais.
    # MAE train≈7.50 / valid≈7.21
    s = (X_train.dtypes == 'str')
    object_cols = list(s[s].index)
    print("Categorical variables:", object_cols)
    ordinal_encoder = OrdinalEncoder()
    X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
    X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

elif method == "ordinal_encoding_smart":
    # Encodage ordinal avec ordre sémantique défini manuellement pour les
    # variables qui ont un sens naturel (poor < average < good, etc.).
    # Variables nominales (genre, diplôme…) encodées automatiquement.
    # MAE train≈7.51 / valid≈7.21
    manual_encoder = OrdinalEncoder(categories=[
        ["poor", "average", "good"],
        ["low",  "medium",  "high"],
        ["easy", "moderate", "hard"]
    ])
    auto_encoder = OrdinalEncoder(categories='auto')
    preprocessor = ColumnTransformer(transformers=[
        ('ord_manuel', manual_encoder,
         ['qualité_sommeil', 'évaluation_établissement', 'difficulté_examen']),
        ('ord_auto',   auto_encoder,
         ['genre', 'diplôme', 'accès_internet', 'méthode_etude'])
    ], remainder='passthrough')
    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)

elif method == "one_hot_encoding_1":
    # One-hot via pandas — simple mais peut créer des colonnes absentes dans valid.
    # align(join='left') force valid à avoir les mêmes colonnes que train.
    # MAE train≈7.54 / valid≈7.21
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

elif method == "one_hot_encoding_2":
    # One-hot via sklearn — plus robuste pour le pipeline de production.
    # MAE train≈7.52 / valid≈7.21
    s = (X_train.dtypes == 'str')
    object_cols = list(s[s].index)
    print("Categorical variables:", object_cols)
    one_hot_encoder = OneHotEncoder()
    preprocessor = ColumnTransformer(
        transformers=[('onehot', one_hot_encoder, object_cols)],
        remainder='passthrough'
    )
    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)

elif method == "ordinal_and_one_hot_encoding":
    # Approche mixte : ordinal pour les variables avec ordre naturel,
    # one-hot pour les variables nominales.
    # C'est la méthode la plus logique sémantiquement.
    # MAE train≈7.52 / valid≈7.21
    manual_ordinal_cols = ['qualité_sommeil', 'évaluation_établissement', 'difficulté_examen']
    one_hot_cols        = ['genre', 'diplôme', 'accès_internet', 'méthode_etude']

    ordinal_encoder = OrdinalEncoder(categories=[
        ["poor", "average", "good"],
        ["low",  "medium",  "high"],
        ["easy", "moderate", "hard"]
    ])
    one_hot_encoder = OneHotEncoder()

    preprocessor = ColumnTransformer(transformers=[
        ('ordinal', ordinal_encoder, manual_ordinal_cols),
        ('onehot',  one_hot_encoder, one_hot_cols)
    ], remainder='passthrough')

    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)

else:
    print("Méthode d'encodage inconnue.")
    exit()

print("Shape après preprocessing :", X_train.shape, X_valid.shape)

# =============================================================================
# Entraînement du modèle XGBoost
# =============================================================================
# XGBoost est un gradient boosting sur arbres de décision.
# n_estimators=1000 : nombre max d'arbres (early stopping l'arrête avant si besoin).
# learning_rate=0.05 : petit pas d'apprentissage → meilleure généralisation.
# early_stopping_rounds=50 : arrête l'entraînement si le RMSE sur valid
#   ne s'améliore pas pendant 50 rounds consécutifs (évite l'overfitting).
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=4,
    early_stopping_rounds=50,
    eval_metric='rmse'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)

y_pred = model.predict(X_valid)
MAE = mean_absolute_error(y_pred, y_valid)
print(f"Mean Absolute Error : {MAE:.4f}")

# =============================================================================
# Graphe 3 : Courbe d'apprentissage XGBoost (RMSE en fonction des rounds)
# =============================================================================
# Cette courbe permet de visualiser la convergence du modèle et de détecter
# l'overfitting : si le RMSE de validation remonte alors que celui d'entraînement
# continue de baisser, le modèle commence à mémoriser le train set.
results = model.evals_result()
epochs  = len(results["validation_0"]["rmse"])
x_axis  = range(0, epochs)

plt.figure(figsize=(10, 5))
plt.plot(x_axis, results["validation_0"]["rmse"], label="Validation")
plt.legend()
plt.xlabel("Nombre de rounds de boosting")
plt.ylabel("RMSE")
plt.title("Courbe d'apprentissage XGBoost — RMSE sur validation")
plt.tight_layout()
plt.show()
