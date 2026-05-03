# =============================================================================
# Graph.py — Exploration des données et entraînement des modèles
# Smart School Project — Prédiction des échecs étudiants
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import issparse
from scipy.stats import gaussian_kde
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler
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
# Section EDA — Exploration des données (Graphes 1 à 5)
# =============================================================================
if graph:
    sns.set(style="whitegrid")

    # =========================================================================
    # Graphe 1 : Valeurs manquantes par colonne
    # =========================================================================
    # Objectif : identifier quelles colonnes nécessitent une imputation avant
    # l'entraînement. Un traitement rigoureux des valeurs manquantes évite
    # des biais silencieux dans le modèle.
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]  # ne garder que les colonnes avec des NaN

    fig0, ax0 = plt.subplots(figsize=(10, 5))
    if missing.empty:
        # Cas sans valeurs manquantes : on l'affiche explicitement
        ax0.text(0.5, 0.5, 'Aucune valeur manquante dans le dataset',
                 ha='center', va='center', fontsize=14, color='green',
                 transform=ax0.transAxes)
    else:
        bars0 = ax0.bar(missing.index, missing.values, color=PALETTE[3], alpha=0.85)
        # Annotation du pourcentage au-dessus de chaque barre
        for bar, val in zip(bars0, missing.values):
            ax0.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + missing.max() * 0.01,
                     f'{val / len(df):.1%}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax0.set_ylabel("Nombre de valeurs manquantes", fontsize=11)
        plt.xticks(rotation=30, ha='right')

    ax0.set_title("Valeurs manquantes par colonne", fontsize=13, fontweight='bold', pad=12)
    ax0.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    # =========================================================================
    # Graphe 2 : Distribution des scores d'examen (variable cible)
    # =========================================================================
    # Objectif : visualiser la forme globale de la distribution de la variable
    # cible (score_examen), identifier les anomalies (score=19) et le
    # plafonnement (score=100), et quantifier la proportion d'échecs (<50).
    s = df['score_examen']
    n_total = len(s)
    n_100   = (s == 100).sum()
    n_fail  = (s < 50).sum()
    n_19    = (s == 19).sum()

    # Histogramme avec bins de largeur 1 (un bin = un score entier)
    counts, edges = np.histogram(s, bins=np.arange(12, 102, 1))

    fig1, ax1 = plt.subplots(figsize=(14, 7))

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

    ax1.bar(edges[:-1], counts, width=0.9, color=bar_colors,
            alpha=0.85, align='edge', zorder=2)

    # KDE (Kernel Density Estimate) pour lisser la distribution.
    # On exclut les scores 100 et 19 qui sont des artefacts et fausseraient
    # la forme de la courbe de densité.
    s_kde = s[(s != 100) & (s != 19)]
    kde = gaussian_kde(s_kde, bw_method=0.08)
    x_kde = np.linspace(12, 99, 400)
    kde_vals = kde(x_kde) * len(s_kde)  # mise à l'échelle en nombre d'étudiants
    ax1.plot(x_kde, kde_vals, color='navy', linewidth=2.0,
             label='KDE (hors anomalies)', zorder=3)

    # Ligne verticale au seuil d'échec (score < 50)
    ax1.axvline(50, color='red', linestyle='--', linewidth=2,
                label=f'Seuil échec 50 — {n_fail:,} étudiants ({n_fail/n_total:.1%})',
                zorder=4)

    # Annotation de l'anomalie score=19 (pic anormalement élevé)
    idx_19 = np.where(edges[:-1] == 19)[0]
    if len(idx_19):
        h19 = counts[idx_19[0]]
        ax1.annotate(
            f'Anomalie\nscore=19\nn={n_19:,}',
            xy=(19.45, h19), xytext=(24, h19 * 0.88),
            fontsize=9, color='#E84393', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E84393', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', fc='#fff0f7', ec='#E84393', lw=1)
        )

    # Annotation du plafonnement à 100
    ax1.annotate(
        f'Plafonnement\nscore=100\nn={n_100:,} ({n_100/n_total:.1%})',
        xy=(100.45, n_100), xytext=(90, n_100 * 0.92),
        fontsize=9, color='#DD8452', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#DD8452', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', fc='#fff8f0', ec='#DD8452', lw=1)
    )

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
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
    ax1.set_title(
        f"Distribution des scores d'examen  —  n={n_total:,}"
        f"  |  moyenne={s.mean():.1f}  |  médiane={s.median():.0f}",
        fontsize=14, fontweight='bold', pad=14
    )
    ax1.set_xlabel("Score d'examen", fontsize=12)
    ax1.set_ylabel("Nombre d'étudiants", fontsize=12)
    ax1.set_xlim(10, 102)
    ax1.set_xticks(range(10, 101, 5))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax1.grid(axis='y', alpha=0.4, zorder=0)
    plt.tight_layout()
    # =========================================================================
    # Graphe 3 : Score d'examen selon les variables catégorielles (box plots)
    # =========================================================================
    # Objectif : comparer la distribution du score selon chaque modalité
    # des variables catégorielles. Si les médianes diffèrent significativement
    # entre modalités, la variable est discriminante et utile pour le modèle.
    # La ligne rouge à 50 rappelle le seuil d'échec.
    # Les boîtes sont triées par médiane croissante pour faciliter la lecture.
    cat_cols = [
        'diplôme', 'méthode_etude', 'accès_internet',
        'qualité_sommeil', 'difficulté_examen', 'évaluation_établissement'
    ]
    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))

    for ax, col in zip(axes3.flat, cat_cols):
        # Tri des modalités par médiane croissante du score
        order = df.groupby(col)['score_examen'].median().sort_values().index
        sns.boxplot(data=df, x=col, y='score_examen',
                    order=order, ax=ax, hue=col, palette='Set2', legend=False)
        # Ligne rouge = seuil d'échec
        ax.axhline(50, color='red', linestyle='--', linewidth=1.2, label='Seuil échec (50)')
        ax.set_title(f'Score selon {col}', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel("Score d'examen", fontsize=9)
        ax.tick_params(axis='x', rotation=15)
        ax.legend(fontsize=8)

    fig3.suptitle(
        "Distribution du score d'examen selon les variables catégorielles",
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    # =========================================================================
    # Graphe 4 : Heatmap de corrélation (variables numériques)
    # =========================================================================
    # Objectif : mesurer la dépendance linéaire (corrélation de Pearson) entre
    # toutes les paires de variables numériques. Une forte corrélation avec
    # score_examen confirme l'utilité d'une variable. Une forte corrélation
    # entre deux features peut indiquer une redondance (multicolinéarité).
    # On exclut 'id' (identifiant sans valeur prédictive) et 'taille_etudiant'
    # (jugé non pertinent).
    num_cols = [c for c in df.select_dtypes(include='number').columns
                if c not in ('id', 'taille_etudiant')]
    corr_matrix = df[num_cols].corr()

    # Masque du triangle supérieur pour éviter la redondance (la matrice est symétrique)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, mask=mask,
        annot=True, fmt='.2f',
        cmap='coolwarm', center=0,
        linewidths=0.5, ax=ax4,
        annot_kws={'size': 9}
    )
    ax4.set_title(
        'Matrice de corrélation (Pearson) — variables numériques',
        fontsize=13, fontweight='bold', pad=12
    )
    plt.tight_layout()
    # =========================================================================
    # Graphe 5 : Importance des features (Mutual Information)
    # =========================================================================
    # La Mutual Information (MI) mesure la dépendance statistique entre chaque
    # feature et la variable cible binaire (echec = score < 50).
    # Contrairement à la corrélation de Pearson, la MI capture les relations
    # non-linéaires, ce qui la rend plus pertinente pour des modèles comme XGBoost.
    #
    # Optimisation vitesse :
    #   mutual_info_classif est en O(n·log n). On tire un échantillon stratifié
    #   de MI_SAMPLE_SIZE lignes : résultats quasi-identiques (~0.3 % de diff),
    #   mais ~10× plus rapide sur un grand dataset.
    print('Feature engineering pour Mutual Information...')

    df_mi = df.copy()

    # Imputation des valeurs manquantes avant le calcul MI :
    # médiane pour les variables numériques (robuste aux outliers),
    # mode pour les variables catégorielles (valeur la plus fréquente).
    df_mi['heures_etude']   = df_mi['heures_etude'].fillna(df_mi['heures_etude'].median())
    df_mi['accès_internet'] = df_mi['accès_internet'].fillna(df_mi['accès_internet'].mode()[0])
    df_mi['méthode_etude']  = df_mi['méthode_etude'].fillna(df_mi['méthode_etude'].mode()[0])

    # Feature engineering : nouvelles variables combinant plusieurs signaux.
    # Ces features capturent des comportements que les variables individuelles
    # n'expriment pas seules.
    df_mi['ratio_etude_fete'] = df_mi['heures_etude'] / (df_mi['heures_fête'] + 1)  # +1 évite div/0
    df_mi['score_bien_etre']  = df_mi['heures_sommeil'] / 12.0   # normalisé entre 0 et 1
    df_mi['engagement']       = df_mi['assiduité_classe'] * df_mi['heures_etude']  # proxy d'implication

    # Variable cible binaire : 1 = échec (<50), 0 = réussite
    df_mi['echec'] = (df_mi['score_examen'] < 50).astype(int)

    # Encodage ordinal manuel : l'ordre sémantique est conservé (poor < average < good)
    ordinal_mappings = {
        'qualité_sommeil':          {'poor': 0, 'average': 1, 'good': 2},
        'évaluation_établissement': {'low': 0, 'medium': 1, 'high': 2},
        'difficulté_examen':        {'easy': 0, 'moderate': 1, 'hard': 2},
        'accès_internet':           {'no': 0, 'yes': 1},
    }
    for col, mapping in ordinal_mappings.items():
        df_mi[col + '_enc'] = df_mi[col].map(mapping)

    # One-hot pour les variables nominales (sans ordre naturel)
    df_mi = pd.get_dummies(df_mi, columns=['méthode_etude', 'diplôme'], drop_first=False)

    # Label encoding pour le genre (deux modalités → 0/1)
    df_mi['genre_enc'] = LabelEncoder().fit_transform(df_mi['genre'])

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

    # Échantillonnage stratifié : conserve la proportion échec/réussite du dataset
    n_rows = len(X_mi)
    if n_rows > MI_SAMPLE_SIZE:
        print(f'Échantillonnage : {MI_SAMPLE_SIZE:,} / {n_rows:,} lignes '
              f'({MI_SAMPLE_SIZE/n_rows:.0%}) pour accélérer le calcul MI...')
        sample_idx = (
            pd.DataFrame({'y': y_mi})
            .groupby('y', group_keys=False)
            .apply(lambda g: g.sample(
                n=int(MI_SAMPLE_SIZE * len(g) / n_rows),
                random_state=42
            ), include_groups=False)
            .index
        )
        X_mi_sample = X_mi.loc[sample_idx]
        y_mi_sample = y_mi.loc[sample_idx]
    else:
        X_mi_sample = X_mi
        y_mi_sample = y_mi

    print(f'Calcul Mutual Information ({len(MI_FEATURES)} features, '
          f'{len(X_mi_sample):,} lignes)...')

    # n_neighbors=3 : voisinage réduit → plus rapide, adapté aux grands datasets
    mi = mutual_info_classif(X_mi_sample, y_mi_sample, random_state=42, n_neighbors=3)
    mi_series = pd.Series(mi, index=MI_FEATURES).sort_values(ascending=True)
    print('Calcul MI terminé.')

    fig5, ax5 = plt.subplots(figsize=(10, 9))
    mi_colors = [PALETTE[3] if v > mi_series.median() else PALETTE[0]
                 for v in mi_series.values]
    ax5.barh(mi_series.index, mi_series.values, color=mi_colors)
    ax5.axvline(mi_series.median(), color='red', linestyle='--', linewidth=1.5)
    legend_mi = [
        mpatches.Patch(facecolor=PALETTE[3], label='Score > médiane'),
        mpatches.Patch(facecolor=PALETTE[0], label='Score ≤ médiane'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Médiane'),
    ]
    ax5.legend(handles=legend_mi, fontsize=9, framealpha=0.9)
    ax5.set_title('Importance des features (Mutual Information)',
                  fontsize=13, fontweight='bold', pad=12)
    ax5.set_xlabel('Score MI', fontsize=11)
    ax5.grid(axis='x', alpha=0.4)
    plt.tight_layout()

# =============================================================================
# Préparation des données pour l'entraînement
# =============================================================================

# Séparation variable cible / features.
# id (identifiant) et taille_etudiant (non pertinent) sont retirés.
y = df['score_examen']
X = df.drop(['score_examen', 'id', 'taille_etudiant'], axis=1)

# Split 80/20 reproductible (random_state=0).
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)
print("Shape avant preprocessing :", X_train.shape, X_valid.shape)

# =============================================================================
# Encodage des variables catégorielles
# =============================================================================
# Méthode choisie : ordinal pour les variables avec ordre naturel,
# one-hot pour les variables nominales.
# Le ColumnTransformer applique chaque transformateur sur les colonnes
# spécifiées et transmet les colonnes numériques inchangées (remainder='passthrough').

method = "ordinal_and_one_hot_encoding"

if method == "drop_categorical":
    # Suppression des colonnes catégorielles — méthode basique, perd de l'info.
    # MAE train≈8.99 / valid≈8.63
    X_train = X_train.select_dtypes(exclude=['str'])
    X_valid = X_valid.select_dtypes(exclude=['str'])

elif method == "ordinal_encoding_random":
    # Encodage ordinal avec ordre arbitraire (alphabétique).
    # MAE train≈7.50 / valid≈7.21
    obj_mask = X_train.dtypes == 'str'
    object_cols = list(obj_mask[obj_mask].index)
    print("Categorical variables:", object_cols)
    ordinal_encoder = OrdinalEncoder()
    X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
    X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

elif method == "ordinal_encoding_smart":
    # Encodage ordinal avec ordre sémantique défini manuellement.
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
    # One-hot via pandas — align force valid à avoir les mêmes colonnes que train.
    # MAE train≈7.54 / valid≈7.21
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

elif method == "one_hot_encoding_2":
    # One-hot via sklearn — plus robuste pour un pipeline de production.
    # MAE train≈7.52 / valid≈7.21
    obj_mask = X_train.dtypes == 'str'
    object_cols = list(obj_mask[obj_mask].index)
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
    # one-hot pour les variables nominales — méthode la plus logique sémantiquement.
    # MAE train≈7.52 / valid≈7.21
    manual_ordinal_cols = ['qualité_sommeil', 'évaluation_établissement', 'difficulté_examen']
    one_hot_cols        = ['genre', 'diplôme', 'accès_internet', 'méthode_etude']

    ordinal_encoder = OrdinalEncoder(categories=[
        ["poor", "average", "good"],
        ["low",  "medium",  "high"],
        ["easy", "moderate", "hard"]
    ])
    one_hot_encoder = OneHotEncoder(sparse_output=False)

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

# Conversion dense pour les modèles incompatibles avec les matrices creuses
X_train_dense = X_train.toarray() if issparse(X_train) else np.array(X_train)
X_valid_dense = X_valid.toarray() if issparse(X_valid) else np.array(X_valid)

# Imputation des NaN résiduels dans les colonnes numériques passthrough.
# Le ColumnTransformer encode les colonnes catégorielles mais laisse les
# colonnes numériques intactes (remainder='passthrough'), donc leurs NaN
# subsistent. SimpleImputer remplace chaque NaN par la médiane de la colonne
# (médiane = robuste aux outliers, contrairement à la moyenne).
# fit uniquement sur X_train pour éviter toute fuite vers X_valid.
imputer = SimpleImputer(strategy='median')
X_train_dense = imputer.fit_transform(X_train_dense)
X_valid_dense = imputer.transform(X_valid_dense)

# =============================================================================
# Entraînement de plusieurs modèles + baseline
# =============================================================================
# La grille d'évaluation demande de comparer au moins 3 modèles dont un
# modèle de deep learning, et d'inclure une baseline (modèle naïf).

# --- Baseline : prédire la moyenne du jeu d'entraînement ---
# Ce modèle ne regarde pas les features : il prédit toujours la même valeur.
# C'est le "pire cas raisonnable" — tout modèle correct doit faire mieux.
baseline_pred = np.full(len(y_valid), y_train.mean())
mae_baseline  = mean_absolute_error(baseline_pred, y_valid)
print(f"Baseline (moyenne)         MAE : {mae_baseline:.4f}")

# --- Régression linéaire ---
# Modèle simple qui suppose une relation linéaire entre les features et le score.
# Sert de référence pour évaluer le gain apporté par les modèles non-linéaires.
lr = LinearRegression()
lr.fit(X_train_dense, y_train)
mae_lr = mean_absolute_error(lr.predict(X_valid_dense), y_valid)
print(f"Régression linéaire        MAE : {mae_lr:.4f}")

# --- Random Forest ---
# Ensemble d'arbres de décision : capture les non-linéarités sans nécessiter
# de normalisation. n_estimators=200 offre un bon compromis vitesse/performance.
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=4)
rf.fit(X_train_dense, y_train)
mae_rf = mean_absolute_error(rf.predict(X_valid_dense), y_valid)
print(f"Random Forest              MAE : {mae_rf:.4f}")

# --- XGBoost (Gradient Boosting) ---
# Boosting d'arbres : entraîne des arbres successifs pour corriger les erreurs
# des précédents. Early stopping arrête si le RMSE sur valid stagne pendant
# 50 rounds consécutifs.
model_xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=4,
    early_stopping_rounds=50,
    eval_metric='rmse'
)
model_xgb.fit(
    X_train_dense, y_train,
    eval_set=[(X_valid_dense, y_valid)],
    verbose=False
)
mae_xgb = mean_absolute_error(model_xgb.predict(X_valid_dense), y_valid)
print(f"XGBoost                    MAE : {mae_xgb:.4f}")

# --- MLP (Réseau de neurones — Deep Learning) ---
# Perceptron multi-couches : modèle de deep learning généraliste.
# Très sensible à l'échelle des features → on applique un StandardScaler
# (moyenne=0, écart-type=1) UNIQUEMENT sur les données du MLP, entraîné
# sur X_train pour éviter toute fuite d'information vers X_valid.
scaler_mlp = StandardScaler()
X_train_scaled = scaler_mlp.fit_transform(X_train_dense)
X_valid_scaled = scaler_mlp.transform(X_valid_dense)

mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64),  # 2 couches cachées : 128 puis 64 neurones
    activation='relu',             # ReLU : standard pour la régression
    max_iter=300,
    random_state=42,
    early_stopping=True,           # arrête si la val loss ne s'améliore plus
    validation_fraction=0.1,
    n_iter_no_change=20
)
mlp.fit(X_train_scaled, y_train)
mae_mlp = mean_absolute_error(mlp.predict(X_valid_scaled), y_valid)
print(f"MLP (réseau de neurones)   MAE : {mae_mlp:.4f}")

# =============================================================================
# Section Résultats — Graphes 6 à 8
# =============================================================================
if graph:

    # =========================================================================
    # Graphe 6 : Comparaison des modèles (MAE)
    # =========================================================================
    # Objectif : visualiser d'un coup d'œil quel modèle est le plus performant
    # et de combien il surpasse la baseline. La barre mise en valeur (orange)
    # est celle avec le MAE le plus bas (meilleur modèle).
    # La ligne rouge représente le MAE de la baseline — tout modèle en dessous
    # de cette ligne apporte une valeur ajoutée réelle.
    model_names = [
        'Baseline\n(moyenne)',
        'Régression\nLinéaire',
        'Random\nForest',
        'XGBoost',
        'MLP\n(neurones)'
    ]
    model_maes = [mae_baseline, mae_lr, mae_rf, mae_xgb, mae_mlp]

    # Le meilleur modèle est mis en valeur avec une couleur différente
    best_idx  = int(np.argmin(model_maes))
    bar_cols6 = [PALETTE[1] if i == best_idx else PALETTE[0]
                 for i in range(len(model_maes))]

    fig6, ax6 = plt.subplots(figsize=(11, 6))
    bars6 = ax6.bar(model_names, model_maes, color=bar_cols6, alpha=0.85, width=0.5)

    # Affichage du MAE au sommet de chaque barre
    for bar, mae in zip(bars6, model_maes):
        ax6.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(model_maes) * 0.01,
            f'{mae:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    # Ligne de référence : MAE de la baseline
    ax6.axhline(mae_baseline, color='red', linestyle='--', linewidth=1.5,
                label=f'Baseline MAE = {mae_baseline:.3f}')

    ax6.set_title('Comparaison des modèles — Mean Absolute Error (MAE)\n'
                  '(plus bas = meilleur)',
                  fontsize=13, fontweight='bold', pad=12)
    ax6.set_ylabel('MAE', fontsize=11)
    ax6.legend(fontsize=10)
    ax6.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    # =========================================================================
    # Graphe 7 : Courbe d'apprentissage XGBoost (RMSE vs rounds)
    # =========================================================================
    # Objectif : visualiser la convergence du modèle XGBoost au fil des rounds
    # de boosting. Une courbe qui descend puis se stabilise indique une bonne
    # convergence. Si la courbe remontait, ce serait un signe d'overfitting
    # (ici évité par l'early stopping).
    results = model_xgb.evals_result()
    epochs  = len(results["validation_0"]["rmse"])

    fig7, ax7 = plt.subplots(figsize=(10, 5))
    ax7.plot(range(epochs), results["validation_0"]["rmse"],
             color=PALETTE[0], linewidth=2, label="Validation RMSE")
    # Marquer le meilleur round (RMSE minimal)
    best_round = int(np.argmin(results["validation_0"]["rmse"]))
    best_rmse  = results["validation_0"]["rmse"][best_round]
    ax7.axvline(best_round, color='red', linestyle='--', linewidth=1.5,
                label=f'Meilleur round : {best_round}  (RMSE={best_rmse:.4f})')
    ax7.scatter([best_round], [best_rmse], color='red', zorder=5, s=60)
    ax7.set_xlabel("Nombre de rounds de boosting", fontsize=11)
    ax7.set_ylabel("RMSE", fontsize=11)
    ax7.set_title("Courbe d'apprentissage XGBoost — RMSE sur validation",
                  fontsize=13, fontweight='bold', pad=12)
    ax7.legend(fontsize=10)
    ax7.grid(alpha=0.4)
    plt.tight_layout()
    # =========================================================================
    # Graphe 8 : Importance des features XGBoost (gain)
    # =========================================================================
    # Objectif : comprendre quelles variables XGBoost utilise le plus pour
    # prendre ses décisions. Le "gain" mesure l'amélioration moyenne apportée
    # par chaque feature à chaque fois qu'elle est utilisée dans un arbre.
    # Ce graphe complète la Mutual Information (Graphe 5) : la MI est calculée
    # avant l'entraînement du modèle, l'importance XGBoost est calculée après.
    fi_vals = model_xgb.feature_importances_

    # Récupération des noms de features depuis le preprocessor sklearn.
    # get_feature_names_out() retourne des noms préfixés ('ordinal__qualité_sommeil')
    # que l'on nettoie en retirant le préfixe.
    try:
        raw_names = preprocessor.get_feature_names_out()
        feat_names = [n.split('__', 1)[-1] for n in raw_names]
    except Exception:
        # Fallback si get_feature_names_out() n'est pas disponible (sklearn < 1.0)
        feat_names = [f'feature_{i}' for i in range(len(fi_vals))]

    fi_series = pd.Series(fi_vals, index=feat_names).sort_values(ascending=True)

    # On n'affiche que les 20 features les plus importantes pour la lisibilité
    fi_top = fi_series.tail(20)

    fig8, ax8 = plt.subplots(figsize=(10, 8))
    fi_colors = [PALETTE[3] if v > fi_top.median() else PALETTE[0]
                 for v in fi_top.values]
    ax8.barh(fi_top.index, fi_top.values, color=fi_colors)
    ax8.axvline(fi_top.median(), color='red', linestyle='--', linewidth=1.5,
                label='Médiane')
    legend_fi = [
        mpatches.Patch(facecolor=PALETTE[3], label='Importance > médiane'),
        mpatches.Patch(facecolor=PALETTE[0], label='Importance ≤ médiane'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                   label='Médiane'),
    ]
    ax8.legend(handles=legend_fi, fontsize=9, framealpha=0.9)
    ax8.set_title('Importance des features XGBoost (gain) — Top 20',
                  fontsize=13, fontweight='bold', pad=12)
    ax8.set_xlabel('Importance (gain moyen par split)', fontsize=11)
    ax8.grid(axis='x', alpha=0.4)
    plt.tight_layout()

# Affichage simultané de tous les graphes.
# Tous les appels plt.show() intermédiaires ont été retirés — matplotlib
# conserve les figures en mémoire jusqu'ici. Un seul plt.show() à la fin
# ouvre toutes les fenêtres en même temps sans bloquer l'exécution du code.
plt.show()
