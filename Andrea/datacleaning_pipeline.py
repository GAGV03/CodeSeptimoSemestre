from sys import argv
from collections import Counter
import missingno as msno #type: ignore
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split #type: ignore

"""
Se corre desde la terminal:
Parámetros: - Nombre del archivo para el dataset
            - Nombre del label (target)
"""

# Asignar nombres de parámetros de una
csv_dataset = argv[1]
target_label = str(argv[2])

# Se lee el csv al correr script desde terminal especificando el nombre del archivo
df = pd.read_csv(argv[1])

# Inicializamos las secciones para nuestro dataset (train y test)
x_train = 0; x_test = 0; y_train = 0; y_test = 0

# Se revisa si es que la columna Id existe en el dataset
has_id = "Id" in df

# Asignamos los valores para las secciones de entrenamiento y testeo
if has_id:
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=["Id"]),
        df[target_label],
        test_size=0.2,
        random_state=42)
else:
    x_train, x_test, y_train, y_test = train_test_split(
        df,
        df[target_label],
        test_size=0.2,
        random_state=42)

# Se separan los valores numéricos y categóricos del set de entrenamiento de X
xTr_num = x_train.select_dtypes(include=["int64","float64"]) # Numéricos
xTr_cat = x_train.select_dtypes(include=["object"])          # Categóricos


"""  ********** Variables Numéricas **********  """
""" - Valores vacíos """
# Se guardan los valores numéricos con alto coeficiente de correlación (mayor a 0.7)
matriz_correlacion = xTr_num.corr() # Sacamos matriz
umbral = 0.7                        # Umbral de aceptación
alta_corr = []                      # Nuestros pares

for col in matriz_correlacion.columns:
    correlated_cols = matriz_correlacion.index[matriz_correlacion[col] >= umbral].tolist()
    correlated_cols.remove(col)
    for correlated_col in correlated_cols:
        par = (col, correlated_col)
        alta_corr.append(par)

# Correlacion con la variable de interés. Guardando las 10 más positivas y 10 más negativas:
top_positive_corr = matriz_correlacion[target_label].sort_values(ascending=False).head(10).index
top_negative_corr = matriz_correlacion[target_label].sort_values(ascending=True).head(10).index
selected_columns = (list(top_positive_corr) + list(top_negative_corr))
selected_data = xTr_num[selected_columns]
# selected_corr_matrix = selected_data.corr()

# Eliminar columnas redundantes
for par in alta_corr:
    if not par[0] in selected_columns:
        xTr_num.drop(par[0], axis=1, inplace=True)

# Llenar / eliminar valores para normalizar
total_val_num = xTr_num.shape[0]
for col in xTr_num.columns:
    total_null = xTr_num[col].isnull().sum()
    if total_null >= total_val_num * 0.75:
        xTr_num.drop(col, axis=1, inplace=True)
    elif total_null / total_val_num < 0.1:
        xTr_num[col] = xTr_num[col].fillna(0).astype(int)
    else:
        xTr_num[col] = xTr_num[col].fillna(xTr_num[col].mean())

""" - Valores atípicos """
# Para únicamente eliminar aquellas filas dentro de columnas que contienen 5 o más outliers
import pandas as pd

# Los outliers de cada columna
outliers_per_column = {}

for col in xTr_num.columns:
    Q1 = xTr_num[col].quantile(0.25)
    Q3 = xTr_num[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Encontrar outliers
    outliers = xTr_num[(xTr_num[col] < lower_bound) | (xTr_num[col] > upper_bound)]
    outliers_per_column[col] = outliers

# Se cuentan por columna
outliers_counts = {col: len(outliers) for col, outliers in outliers_per_column.items()}

# Eliminarlos de aquellas columnas con 5 o menos outliers
for col, count in outliers_counts.items():
    if count <= 5:
        Q1 = xTr_num[col].quantile(0.25)
        Q3 = xTr_num[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Eliminar los outliers filtrados
        xTr_num = xTr_num[~((xTr_num[col] < lower_bound) | (xTr_num[col] > upper_bound))]

""" Variables Categóricas"""
# Se inicializan la frecuencia de categorías y datos faltantes como diccionarios
class_counts = {}
missing_values = {}
umbral = 0.01 * len(xTr_cat)
# Se llenan los diccionarios
for col in xTr_cat.columns:
    class_count = Counter(xTr_cat[col].dropna())  # Excluir valores faltantes al contar clases
    class_counts[col] = class_count
    missing_values[col] = xTr_cat[col].isnull().sum()
# Se implementan cambios basado en la información categórica
total_val_cat = xTr_cat.shape[0]
for col, counts in class_counts.items():
    most_common_class, most_common_count = counts.most_common(1)[0]
    total_non_null = sum(counts.values())
    # Si hay muchos datos faltantes, se crea la categría Faltante
    if total_non_null < total_val_cat * 0.15:
        xTr_cat[col] = xTr_cat[col].fillna("Faltante").astype(str)
    # Si hay una clase muy dominante, imputar con moda
    elif most_common_count / total_non_null > 0.9:
        xTr_cat[col] = xTr_cat[col].fillna(xTr_cat[col].mode()[0])
    # Si no hay una clase dominante pero hay pocos valores faltantes, crear una nueva categoría
    elif missing_values[col] / len(xTr_cat) < 0.1:
        xTr_cat[col] = xTr_cat[col].fillna("Faltante").astype(str)
    # Si hay una distribución equilibrada entre varias clases
    elif most_common_count / total_non_null < 0.9:
        # Imputar aleatoriamente basado en la distribución existente
        xTr_cat[col] = xTr_cat[col].apply(lambda x: x if pd.notna(x) else counts.most_common()[np.random.randint(len(counts))][0])
    # Si hay una distribución equilibrada entre varias clases, imputación predictiva
    else:
        rare_categories = [cat for cat, count in class_counts[col].items() if count < umbral]
        xTr_cat[col] = xTr_cat[col].replace(rare_categories, 'Rara')
    # Si aún quedan valores faltantes (caso residual)
    if xTr_cat[col].isnull().sum() > 0:
        xTr_cat[col] = xTr_cat[col].fillna(xTr_cat[col].mode()[0])  # Imputar con la moda como último recurso

# Aplicar one-hot encoding (dummies) después de la imputación
xTr_cat = pd.get_dummies(xTr_cat, drop_first=True)


""" Finalizar uniendo ambos sets """
clean_train_set = pd.concat([xTr_num, xTr_cat], axis=1)

""" Generar CSV  """
clean_train_set.to_csv('clean_train.csv', index=False)
