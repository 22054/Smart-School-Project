#plot information
#import featuretools as ft # featuretools for automated feature engineering

#from learntools.core import binder
#binder.bind(globals())
#from learntools.feature_engineering_new.ex5 import *

import matplotlib.pyplot as plt
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

PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860', '#DA8BC3']

# --- Données ---
# (X:inutile, F:probablement inutile, A:utile, P:prédiction)
# (N:nombre, C:catégorie nominale, S:catégorie ordinale)
# X : N : id : 
# F : N : age : 
# F : C : genre : 
# A : C : diplôme : 
# A : N : heures_etude : 
# A : N : assiduité_classe : 
# A : C : accès_internet : 
# A : N : heures_sommeil : 
# A : S : qualité_sommeil : 
# A : C : méthode_etude : 
# A : S : évaluation_établissement : 
# A : S : difficulté_examen : 
# P : N : score_examen : 
# A : N : heures_fête : 
# F : N : taille_etudiant : 

#pd.set_option('display.max_columns', None)
#pd.set_option('display.expand_frame_repr', True)
#pd.set_option('display.width', 1000)
#pd.set_option('display.max_rows', None)
graph = True
small_dataset = False
if small_dataset:
	df = pd.read_csv("student_dataset/student_failure/train.csv", nrows=10000) # WARNING : REMOVE NROWS FOR PRODUCTION
else:
	df = pd.read_csv("student_dataset/student_failure/train.csv")
print(df.head(10))
#print(df.qualité_sommeil)
if graph:
	sns.set(style="whitegrid")
	s = df['score_examen']
	n_total = len(s)
	n_100 = (s == 100).sum()
	n_fail = (s < 50).sum()
	n_19   = (s == 19).sum()

	counts, edges = np.histogram(s, bins=np.arange(12, 102, 1))

	fig, ax = plt.subplots(figsize=(14, 7))

	# --- Barres colorées par zone ---
	bar_colors = []
	for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
		mid = (left + right) / 2
		if mid == 19:
			bar_colors.append('#E84393')   # anomalie score=19
		elif mid == 100:
			bar_colors.append('#DD8452')   # plafonnement
		elif mid < 50:
			bar_colors.append('#E05C5C')   # zone echec
		else:
			bar_colors.append('#4C72B0')   # zone normale

	bars = ax.bar(edges[:-1], counts, width=0.9, color=bar_colors, alpha=0.85, align='edge', zorder=2)

	# --- KDE (hors score=100 et hors anomalie 19) ---
	s_kde = s[(s != 100) & (s != 19)]
	from scipy.stats import gaussian_kde
	kde = gaussian_kde(s_kde, bw_method=0.08)
	x_kde = np.linspace(12, 99, 400)
	kde_vals = kde(x_kde) * len(s_kde)   # mise à l'échelle en nombre d'étudiants
	ax.plot(x_kde, kde_vals, color='navy', linewidth=2.0, label='KDE (hors anomalies)', zorder=3)

	# --- Seuil d'échec ---
	ax.axvline(50, color='red', linestyle='--', linewidth=2,
	           label=f'Seuil échec 50 — {n_fail:,} étudiants ({n_fail/n_total:.1%})', zorder=4)

	# --- Annotation : anomalie score=19 ---
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

	# --- Annotation : plafonnement score=100 ---
	ax.annotate(
		f'Plafonnement\nscore=100\nn={n_100:,} ({n_100/n_total:.1%})',
		xy=(100.45, n_100), xytext=(90, n_100 * 0.92),
		fontsize=9, color='#DD8452', fontweight='bold',
		arrowprops=dict(arrowstyle='->', color='#DD8452', lw=1.5),
		bbox=dict(boxstyle='round,pad=0.3', fc='#fff8f0', ec='#DD8452', lw=1)
	)

	# --- Légende manuelle ---
	from matplotlib.patches import Patch
	legend_elements = [
		Patch(facecolor='#4C72B0', alpha=0.85, label='Zone réussite (≥50)'),
		Patch(facecolor='#E05C5C', alpha=0.85, label=f'Zone échec (<50) — {n_fail/n_total:.1%}'),
		Patch(facecolor='#E84393', alpha=0.85, label=f'Anomalie score=19 — n={n_19:,}'),
		Patch(facecolor='#DD8452', alpha=0.85, label=f'Plafonnement score=100 — {n_100/n_total:.1%}'),
		plt.Line2D([0], [0], color='navy', linewidth=2, label='KDE (hors anomalies)'),
		plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2,
		           label=f'Seuil échec (50)'),
	]
	ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

	# --- Mise en forme ---
	ax.set_title(
		f"Distribution des scores d'examen  —  n={n_total:,}  |  moyenne={s.mean():.1f}  |  médiane={s.median():.0f}",
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

	# ── Graphe 2 : Mutual Information ────────────────────────────────────────
	from matplotlib.patches import Patch
	print('Feature engineering pour Mutual Information...')
	df_mi = pd.read_csv("student_dataset/student_failure/train.csv") if small_dataset else df.copy()
	# Échantillon de 50k : résultats quasi-identiques, ~10x plus rapide

	df_mi['heures_etude']     = df_mi['heures_etude'].fillna(df_mi['heures_etude'].median())
	df_mi['accès_internet']   = df_mi['accès_internet'].fillna(df_mi['accès_internet'].mode()[0])
	df_mi['méthode_etude']    = df_mi['méthode_etude'].fillna(df_mi['méthode_etude'].mode()[0])
	df_mi['ratio_etude_fete'] = df_mi['heures_etude'] / (df_mi['heures_fête'] + 1)
	df_mi['score_bien_etre']  = df_mi['heures_sommeil'] / 12.0
	df_mi['engagement']       = df_mi['assiduité_classe'] * df_mi['heures_etude']
	df_mi['echec']            = (df_mi['score_examen'] < 50).astype(int)
	for col, mapping in {
		'qualité_sommeil':          {'poor': 0, 'average': 1, 'good': 2},
		'évaluation_établissement': {'low': 0, 'medium': 1, 'high': 2},
		'difficulté_examen':        {'easy': 0, 'moderate': 1, 'hard': 2},
		'accès_internet':           {'no': 0, 'yes': 1},
	}.items():
		df_mi[col + '_enc'] = df_mi[col].map(mapping)
	df_mi = pd.get_dummies(df_mi, columns=['méthode_etude', 'diplôme'], drop_first=False)
	df_mi['genre_enc'] = LabelEncoder().fit_transform(df_mi['genre'])

	MI_FEATURES = (
		['age', 'heures_etude', 'assiduité_classe', 'heures_sommeil',
		 'heures_fête', 'ratio_etude_fete', 'score_bien_etre', 'engagement',
		 'qualité_sommeil_enc', 'évaluation_établissement_enc',
		 'difficulté_examen_enc', 'accès_internet_enc', 'genre_enc']
		+ [c for c in df_mi.columns if c.startswith('méthode_etude_') or c.startswith('diplôme_')]
	)
	X_mi = df_mi[MI_FEATURES].astype(float).fillna(0)
	y_mi = df_mi['echec']

	print(f'Calcul Mutual Information ({len(MI_FEATURES)} features, {len(X_mi):,} lignes)...')
	mi = mutual_info_classif(X_mi, y_mi, random_state=42, n_neighbors=3)
	mi_series = pd.Series(mi, index=MI_FEATURES).sort_values(ascending=True)

	fig2, ax2 = plt.subplots(figsize=(10, 9))
	mi_colors = [PALETTE[3] if v > mi_series.median() else PALETTE[0] for v in mi_series.values]
	ax2.barh(mi_series.index, mi_series.values, color=mi_colors)
	ax2.axvline(mi_series.median(), color='red', linestyle='--', linewidth=1.5)
	legend_mi = [
		Patch(facecolor=PALETTE[3], label='Score > médiane'),
		Patch(facecolor=PALETTE[0], label='Score ≤ médiane'),
		plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Médiane'),
	]
	ax2.legend(handles=legend_mi, fontsize=9, framealpha=0.9)
	ax2.set_title('Importance des features (Mutual Information)', fontsize=13, fontweight='bold', pad=12)
	ax2.set_xlabel('Score MI', fontsize=11)
	ax2.grid(axis='x', alpha=0.4)
	plt.tight_layout()
	plt.show()




#separate target from predictors
y = df.score_examen
X = df.drop(['score_examen', 'id', 'taille_etudiant'], axis=1)

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
print("shape before preprocessing :", X_train.shape, X_valid.shape)

if False:
	# Drop columns with missing values (simplest approach)
	cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()] 
	X_train.drop(cols_with_missing, axis=1, inplace=True)
	X_valid.drop(cols_with_missing, axis=1, inplace=True)

# Selecting approache to deal with categorical variables
method = "ordinal_and_one_hot_encoding"
if method == "drop_categorical": # 8.995743871459961 and 8.630448577480498
	# Drop Categorical Variables
	X_train = X_train.select_dtypes(exclude=['str'])
	X_valid = X_valid.select_dtypes(exclude=['str'])
elif method == "ordinal_encoding_random": # 7.4998913270568845 and 7.21429521
	# Get list of categorical variables
	s = (X_train.dtypes == 'str')
	object_cols = list(s[s].index)
	print("Categorical variables:", object_cols)
	# Apply ordinal encoder to each column with categorical data
	ordinal_encoder = OrdinalEncoder()
	X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
	X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])
elif method == "ordinal_encoding_smart": # 7.508872240333558 and 7.211080674552554
	# Apply ordinal encoder to each column with categorical data
	manual_encoder = OrdinalEncoder(categories=[["poor","average","good"],
												["low","medium","high"],
												["easy","moderate","hard"]])
	auto_encoder = OrdinalEncoder(categories='auto')
	preprocessor = ColumnTransformer(transformers=[
		# (Nom, Transformateur, Liste des colonnes)
		('ord_manuel', manual_encoder, ['qualité_sommeil', 'évaluation_établissement', 'difficulté_examen']),
		('ord_auto',   auto_encoder,   ['genre', 'diplôme', 'accès_internet', 'méthode_etude'])
	], remainder='passthrough') # 'passthrough' veut dire : "garde les autres colonnes telles quelles"

	X_train = preprocessor.fit_transform(X_train)
	X_valid = preprocessor.transform(X_valid)
elif method == "one_hot_encoding_1": # 7.537241737747192 and 7.210295896459731
	# One-hot encode the data (to shorten the code, we use pandas)
	X_train = pd.get_dummies(X_train)
	X_valid = pd.get_dummies(X_valid)
	X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
elif method == "one_hot_encoding_2": # 7.521292720184326 and 7.210252136884901
	# One-hot encode the data
	# Get list of categorical variables
	s = (X_train.dtypes == 'str')
	object_cols = list(s[s].index)
	print("Categorical variables:", object_cols)
	# Apply ordinal encoder to each column with categorical data
	one_hot_encoder = OneHotEncoder()
	preprocessor = ColumnTransformer(transformers=[('onehot', one_hot_encoder, object_cols)], remainder='passthrough')
	X_train = preprocessor.fit_transform(X_train)
	X_valid = preprocessor.transform(X_valid)
elif method == "ordinal_and_one_hot_encoding": # 7.520595026741028 and 7.211823333794004
	# ordinal or one-hot encoding depending of the category
	manual_ordinal_cols = ['qualité_sommeil', 'évaluation_établissement', 'difficulté_examen']
	one_hot_cols = ['genre', 'diplôme', 'accès_internet', 'méthode_etude']
	
	# Apply ordinal encoder to each column with categorical data
	ordinal_encoder = OrdinalEncoder(categories=[["poor","average","good"],
												["low","medium","high"],
												["easy","moderate","hard"]])
	# Apply ordinal encoder to each column with categorical data
	one_hot_encoder = OneHotEncoder()
	preprocessor = ColumnTransformer(transformers=[
		# (Nom, Transformateur, Liste des colonnes)
		('ordinal', ordinal_encoder, manual_ordinal_cols),
		('onehot', one_hot_encoder, one_hot_cols)
	], remainder='passthrough') # 'passthrough' veut dire : "garde les autres colonnes telles quelles"

	X_train = preprocessor.fit_transform(X_train)
	X_valid = preprocessor.transform(X_valid)
else:
	print("unknown method")
	exit()
print("shape after preprocessing :", X_train.shape, X_valid.shape)

#model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4, early_stopping_rounds=50)
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4, early_stopping_rounds=50, eval_metric='rmse')

#For regression problems, common choices include “rmse” (root mean squared error) and “mae” (mean absolute error).
#For binary classification, “error” (binary classification error), “logloss” (binary log loss), and “auc” (area under the receiver operating characteristic curve) are frequently used.
#For multi-class classification settings, “merror” (multi-class classification error) and “mlogloss” (multi-class log loss) are popular options.

model.fit(X_train, y_train,
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

y_pred = model.predict(X_valid)
MAE = mean_absolute_error(y_pred, y_valid)
print("Mean Absolute Error: " + str(MAE))

# Retrieve the RMSE values from the training process
results = model.evals_result()
epochs = len(results["validation_0"]["rmse"])
x_axis = range(0, epochs)

# Plot the RMSE values
plt.figure()
plt.plot(x_axis, results["validation_0"]["rmse"], label="Test")
plt.legend()
plt.xlabel("Number of Boosting Rounds")
plt.ylabel("RMSE") # Root Mean Squared Error
plt.title("XGBoost Regressor RMSE Performance")
plt.show()