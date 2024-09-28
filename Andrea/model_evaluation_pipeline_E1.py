# Modelo predictivo para Housing Pricing

# Mariluz Daniela Sánchez Morales - A01422953
# Eric Manuel Navarro Martínez    - A01746219
# Pablo Spínola López             - A01753922
# Gustavo Téllez Mireles          - A01747114
# Gustavo Alejandro Téllez Valdés - A01747869

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# Parameters
csv_dataset_train = "train.csv"
csv_test = "test.csv"
target_label = "SalePrice"

# Load the dataset
df = pd.read_csv(csv_dataset_train)
df_test = pd.read_csv(csv_dataset_train)

# Train-test split
has_id = "Id" in df
if has_id:
    df = df.drop(columns=["Id"])

has_id = "Id" in df_test
if has_id:
    df_test = df_test.drop(columns=["Id"])

x_train, x_val_test, y_train, y_val_test = train_test_split(
    df.drop(columns=[target_label]),
    df[target_label],
    test_size=0.2,
    random_state=35
)

x_test = df_test.drop(columns=[target_label])
y_test = df_test[target_label]

# 1. Handle Year Columns
year_columns = ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold"]  # Add other year-related columns if any

# Convert years to age
current_year = 2024
for col in year_columns:
    if col in x_train.columns:
        x_train[col + "_Age"] = current_year - x_train[col]
        x_val_test[col + "_Age"] = current_year - x_val_test[col]
        x_test[col + "_Age"] = current_year - x_val_test[col]
        x_train.drop(columns=[col], inplace=True)
        x_val_test.drop(columns=[col], inplace=True)
        x_test.drop(columns=[col], inplace=True)

# 2. Preprocessing Pipelines

# Numerical data preprocessing
xTr_num = x_train.select_dtypes(include=["int64", "float64"])
xTr_num_test = x_val_test.select_dtypes(include=["int64", "float64"])
xTr_test_num = x_test.select_dtypes(include=["int64", "float64"])

# Handle skewness and imputation
for col in xTr_num.columns:
    #

    skewness1 = xTr_num[col].skew()
    if skewness1 > 1 or skewness1 < -1:
        xTr_num[col] = np.log1p(xTr_num[col])
    
    skewnessv = xTr_num_test[col].skew()
    if skewnessv > 1 or skewnessv < -1:
        xTr_num_test[col] = np.log1p(xTr_num_test[col])
    
    skewnesst = xTr_test_num[col].skew()
    if skewnesst > 1 or skewnesst < -1:
        xTr_test_num[col] = np.log1p(xTr_test_num[col])
    
    #

    if xTr_num[col].isnull().sum() > 0:
        if skewness1 > 1 or skewness1 < -1:
            xTr_num[col] = xTr_num[col].fillna(xTr_num[col].median())
        else:
            xTr_num[col] = xTr_num[col].fillna(xTr_num[col].mean())
    
    if xTr_num_test[col].isnull().sum() > 0:
        if skewnessv > 1 or skewnessv < -1:
            xTr_num_test[col] = xTr_num_test[col].fillna(xTr_num[col].median())
        else:
            xTr_num_test[col] = xTr_num_test[col].fillna(xTr_num[col].mean())
    
    if xTr_test_num[col].isnull().sum() > 0:
        if skewnesst > 1 or skewnesst < -1:
            xTr_test_num[col] = xTr_test_num[col].fillna(xTr_num[col].median())
        else:
            xTr_test_num[col] = xTr_test_num[col].fillna(xTr_num[col].mean())

# Categorical data preprocessing
xTr_cat = x_train.select_dtypes(include=["object"])
xTr_cat_test = x_val_test.select_dtypes(include=["object"])
xTr_test_cat = x_test.select_dtypes(include=["object"])

for col in xTr_cat.columns:
    most_common_class = xTr_cat[col].mode()[0]
    missing_count = xTr_cat[col].isnull().sum()

    if missing_count > 0:
        if missing_count / len(xTr_cat) < 0.1:  # Pocos faltantes
            xTr_cat[col] = xTr_cat[col].fillna(most_common_class)
        else:
            class_counts = Counter(xTr_cat[col].dropna())
            if most_common_class in class_counts and class_counts[most_common_class] / len(xTr_cat) > 0.9:
                xTr_cat[col] = xTr_cat[col].fillna(most_common_class)
            else:
                xTr_cat[col] = xTr_cat[col].fillna('Missing')

    most_common_classv = xTr_cat_test[col].mode()[0]
    missing_countv = xTr_cat_test[col].isnull().sum()

    if missing_countv > 0:
        if missing_countv / len(xTr_cat_test) < 0.1:  # Pocos faltantes
            xTr_cat_test[col] = xTr_cat_test[col].fillna(most_common_classv)
        else:
            class_countsv = Counter(xTr_cat_test[col].dropna())
            if most_common_classv in class_countsv and class_countsv[most_common_classv] / len(xTr_cat_test) > 0.9:
                xTr_cat_test[col] = xTr_cat_test[col].fillna(most_common_classv)
            else:
                xTr_cat_test[col] = xTr_cat_test[col].fillna('Missing')
    
    most_common_classt = xTr_test_cat[col].mode()[0]
    missing_countt = xTr_test_cat[col].isnull().sum()

    if missing_countt > 0:
        if missing_countt / len(xTr_test_cat) < 0.1:  # Pocos faltantes
            xTr_test_cat[col] = xTr_test_cat[col].fillna(most_common_classt)
        else:
            class_countst = Counter(xTr_test_cat[col].dropna())
            if most_common_classt in class_countst and class_countst[most_common_classt] / len(xTr_test_cat) > 0.9:
                xTr_test_cat[col] = xTr_test_cat[col].fillna(most_common_classt)
            else:
                xTr_test_cat[col] = xTr_test_cat[col].fillna('Missing')

# One-hot encoding para datos categóricos
xTr_cat = pd.get_dummies(xTr_cat, drop_first=True)
xTr_cat_test = pd.get_dummies(xTr_cat_test, drop_first=True)
xTr_test_cat = pd.get_dummies(xTr_test_cat, drop_first=True)

# Alinear columnas
xTr_cat_test = xTr_cat_test.reindex(columns=xTr_cat.columns, fill_value=0)
xTr_test_cat = xTr_test_cat.reindex(columns=xTr_cat.columns, fill_value=0)

# Combinar los numéricos con categóricos
x_train_processed = pd.concat([xTr_num, xTr_cat], axis=1)
x_val_test_processed = pd.concat([xTr_num_test, xTr_cat_test], axis=1)
x_test_processed = pd.concat([xTr_test_num, xTr_test_cat], axis=1)

# Polynomial Features
# poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
# x_train_poly = poly.fit_transform(x_train_processed)
# x_val_test_poly = poly.transform(x_val_test_processed)

# Selección de features con random forest
selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold="median")
x_train_selected = selector.fit_transform(x_train_processed, y_train)
x_val_test_selected = selector.transform(x_val_test_processed)
x_test_selected = selector.transform(x_test_processed)

# DataFrame de los features
selected_features = selector.get_support(indices=True)
# feature_names_poly = poly.get_feature_names_out(input_features=x_train_processed.columns)
final_feature_names = np.array(x_train_processed.columns)[selected_features]
clean_train_set = pd.DataFrame(x_train_selected, columns=final_feature_names)
clean_test = pd.DataFrame(x_test_selected, columns=final_feature_names)

# clean_train_set.to_csv('clean_train1.csv', index=False)

""" Realizar predicciones y calcular mejor modelo y parámetros """

print(f"Cleaned dataset has {clean_train_set.shape[0]} rows and {clean_train_set.shape[1]} columns.")

param_grid = [
    {'n_estimators': [3, 10, 30, 50, 100], 'max_features': [2, 4, 6, 8, 10], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
    {'bootstrap': [False], 'n_estimators': [3, 10, 30, 50], 'max_features': [2, 3, 4, 6], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
]

# tree_grid = {'max_depth': [3, 5, 10, None],
#              'min_samples_split': [2, 5, 10],
#              'min_samples_leaf': [1, 2, 4]}

forest_reg = RandomForestRegressor()
# tree_reg = DecisionTreeRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search_t = GridSearchCV(tree_reg, tree_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(clean_train_set, y_train)
# grid_search_t.fit(clean_train_set, y_train)

best_model = grid_search.best_estimator_
# best_model_t = grid_search_t.best_estimator_

# Mejores parámetros
print("Mejores parámetros:", grid_search.best_params_)
# Mejor modelo de estimación
print("Mejor modelo de estimación con hiperparámetros:", best_model)

# Predicción del subset del dataset de entrenamiento
y_val_pred = best_model.predict(x_val_test_selected)
# y_val_pred = best_model_t.predict(x_val_test_selected)
# Error de 
mse = mean_squared_error(y_val_test, y_val_pred)
print("MSE del modelo entrenado:", mse)
rmse = np.sqrt(mse)
print("Raíz del MSE:", rmse)

# Porcentage del error absoluto
porcentaje = mean_absolute_percentage_error(y_val_test, y_val_pred)
porcentaje_f = porcentaje * 100

print(f"Porcentaje de error para set de validación: {porcentaje_f:.2f}%")

# Error para el set de testeo "test.csv"
y_pred = best_model.predict(x_test_selected)
print("\n*** Valores del test.csv ***")
mse_test = mean_squared_error(y_test, y_pred)
print("MSE del modelo entrenado:", mse_test)
rmse_test = np.sqrt(mse_test)
print("Raíz del MSE:", rmse_test)

porcentaje_test = mean_absolute_percentage_error(y_test, y_pred)
porcentaje_f_test = porcentaje_test * 100

print(f"Porcentaje de error para set de test: {porcentaje_f_test:.2f}%")