from sklearn.tree import DecisionTreeRegressor #type: ignore

from sys import argv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.impute import SimpleImputer #type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures #type: ignore
from sklearn.compose import ColumnTransformer #type: ignore
from sklearn.pipeline import Pipeline #type: ignore
from sklearn.ensemble import RandomForestRegressor #type: ignore
from sklearn.feature_selection import SelectFromModel #type: ignore

# Parameters
csv_dataset = argv[1]
target_label = str(argv[2])

# Load the dataset
df = pd.read_csv(csv_dataset)

# Train-test split
has_id = "Id" in df
if has_id:
    df = df.drop(columns=["Id"])

housing_labels = df[target_label].copy()

x_train, x_test, y_train, y_test = train_test_split(
    df,
    df[target_label],
    test_size=0.2,
    random_state=42
)

# 1. Handle Year Columns
year_columns = ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold"]  # Add other year-related columns if any

# Convert years to age
current_year = 2024
for col in year_columns:
    if col in x_train.columns:
        x_train[col + "_Age"] = current_year - x_train[col]
        x_test[col + "_Age"] = current_year - x_test[col]
        x_train.drop(columns=[col], inplace=True)
        x_test.drop(columns=[col], inplace=True)

# 2. Preprocessing Pipelines

# Numerical data preprocessing
xTr_num = x_train.select_dtypes(include=["int64", "float64"])
xTr_num_test = x_test.select_dtypes(include=["int64", "float64"])

# Handle skewness and imputation
for col in xTr_num.columns:
    skewness = xTr_num[col].skew()
    if skewness > 1 or skewness < -1:
        xTr_num[col] = np.log1p(xTr_num[col])
        xTr_num_test[col] = np.log1p(xTr_num_test[col])
    
    if xTr_num[col].isnull().sum() > 0:
        if skewness > 1 or skewness < -1:
            xTr_num[col] = xTr_num[col].fillna(xTr_num[col].median())
            xTr_num_test[col] = xTr_num_test[col].fillna(xTr_num[col].median())
        else:
            xTr_num[col] = xTr_num[col].fillna(xTr_num[col].mean())
            xTr_num_test[col] = xTr_num_test[col].fillna(xTr_num[col].mean())

# Categorical data preprocessing
xTr_cat = x_train.select_dtypes(include=["object"])
xTr_cat_test = x_test.select_dtypes(include=["object"])

for col in xTr_cat.columns:
    most_common_class = xTr_cat[col].mode()[0]
    missing_count = xTr_cat[col].isnull().sum()

    if missing_count > 0:
        if missing_count / len(xTr_cat) < 0.1:  # Few missing values
            xTr_cat[col] = xTr_cat[col].fillna(most_common_class)
            xTr_cat_test[col] = xTr_cat_test[col].fillna(most_common_class)
        else:
            class_counts = Counter(xTr_cat[col].dropna())
            if most_common_class in class_counts and class_counts[most_common_class] / len(xTr_cat) > 0.9:
                xTr_cat[col] = xTr_cat[col].fillna(most_common_class)
                xTr_cat_test[col] = xTr_cat_test[col].fillna(most_common_class)
            else:
                xTr_cat[col] = xTr_cat[col].fillna('Missing')
                xTr_cat_test[col] = xTr_cat_test[col].fillna('Missing')

# Apply one-hot encoding to categorical data
xTr_cat = pd.get_dummies(xTr_cat, drop_first=True)
xTr_cat_test = pd.get_dummies(xTr_cat_test, drop_first=True)

# Ensure the same columns in test set by aligning with train set
xTr_cat_test = xTr_cat_test.reindex(columns=xTr_cat.columns, fill_value=0)

# Combine processed numerical and categorical data
x_train_processed = pd.concat([xTr_num, xTr_cat], axis=1)
x_test_processed = pd.concat([xTr_num_test, xTr_cat_test], axis=1)

# # 3. Feature Engineering: Polynomial Features
# poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
# x_train_poly = poly.fit_transform(x_train_processed)
# x_test_poly = poly.transform(x_test_processed)

# Feature Selection using a model (e.g., RandomForest)
selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold="median")
x_train_selected = selector.fit_transform(x_train_processed, y_train)
x_test_selected = selector.transform(x_test_processed)

# Create a DataFrame from the processed and selected data
selected_features = selector.get_support(indices=True)
# feature_names_poly = poly.get_feature_names_out(input_features=x_train_processed.columns)
final_feature_names = np.array(x_train_processed.columns)[selected_features]
clean_train_set = pd.DataFrame(x_train_selected, columns=final_feature_names)

# Save the cleaned and selected data
clean_train_set.to_csv('clean_train1.csv', index=False)

print(f"Cleaned dataset has {clean_train_set.shape[0]} rows and {clean_train_set.shape[1]} columns.")
