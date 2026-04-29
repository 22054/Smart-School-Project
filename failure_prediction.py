#import featuretools as ft # featuretools for automated feature engineering

#from learntools.core import binder
#binder.bind(globals())
#from learntools.feature_engineering_new.ex5 import *

import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
#import seaborn as sns
#from sklearn.decomposition import PCA
#from sklearn.feature_selection import mutual_info_regression
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

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
small_dataset = True
if small_dataset:
	df = pd.read_csv("student_dataset/student_failure/train.csv", nrows=10000) # WARNING : REMOVE NROWS FOR PRODUCTION
else:
	df = pd.read_csv("student_dataset/student_failure/train.csv")
print(df.head(10))
#print(df.qualité_sommeil)

#separate target from predictors
y = df.score_examen
X = df.drop(['score_examen', 'id', 'taille_etudiant'], axis=1)

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
print("shape after preprocessing :", X_train.shape, X_valid.shape)

if False:
	# Drop columns with missing values (simplest approach)
	cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()] 
	X_train.drop(cols_with_missing, axis=1, inplace=True)
	X_valid.drop(cols_with_missing, axis=1, inplace=True)

# Selecting approache to deal with categorical variables
method = "one_hot_encoding_2"
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
	# Get list of non-categorical variables
	s = (X_train.dtypes == 'str')
	object_cols = list(s[s].index)
	print("Categorical variables:", object_cols)
	# Apply ordinal encoder to each column with categorical data
	one_hot_encoder = OneHotEncoder()
	preprocessor = ColumnTransformer(transformers=[('onehot', one_hot_encoder, object_cols)], remainder='passthrough')
	X_train = preprocessor.fit_transform(X_train)
	X_valid = preprocessor.transform(X_valid)
elif method == "smart_encoding":
	# ordinal and one-hot encoding depending of the category
	pass
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