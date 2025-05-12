"""
Module for training a RandomForestRegressor model on the Ames dataset.

This module loads the Ames housing dataset, preprocesses the data by handling
categorical and numerical features differently, and trains a
RandomForestRegressor model. It performs 10-fold cross-validation to evaluate
model performance using RMSE and visualizes feature importances.

Usage:
    python main.py

Dependencies:
    - pandas
    - numpy
    - matplotlib
    - scikit-learn
    - joblib
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# import the dataset
df = pd.read_csv("./raw_data/ames_unprocessed_data.csv")

TARGET_COL = "SalePrice"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# CREATE LISTS WITH NUMERICAL AND CATEGORICAL COLUMNS
# Create a boolean mask for categorical columns
categorical_mask = df.dtypes == object

# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()
print(categorical_columns)

# Create a boolean mask for numerical columns
numerical_mask = [not x for x in categorical_mask]

numerical_columns = df.columns[numerical_mask].tolist()
numerical_columns.remove(TARGET_COL)
feature_columns = numerical_columns
print(feature_columns)

# CREATE PREPROCESSING PIPELINES
# one-hot-encode categorical features
cat_pipeline = Pipeline([("cat", OneHotEncoder(sparse_output=False))])

# impute and scale numerical features
num_pipeline = Pipeline([("imputer", SimpleImputer(fill_value=0)), ("scaler", StandardScaler())])


# CREATE COLUMN TRANSFORMER

# The column transformer preprocesses the numerical and categorical features differently
preprocessor = ColumnTransformer(
    transformers=[("num", num_pipeline, feature_columns), ("cat", cat_pipeline, categorical_columns)],
    remainder="passthrough",
)

# test the column transformer
# num cols have mean=0 and std=1
# x = preprocessor.fit_transform(df)
# df_trans = pd.DataFrame(x, columns=preprocessor.get_feature_names_out())
# display(df_trans.describe())
# print(df_trans.shape)

steps = [("preprocessor", preprocessor), ("clf", RandomForestRegressor(max_depth=5))]

rf_pipeline = Pipeline(steps)

# Cross-validate the model
cross_val_scores = cross_val_score(rf_pipeline, X, y, cv=10, scoring="neg_mean_squared_error")

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))

rf_pipeline.fit(X, y)
# Get feature names after preprocessing
feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
# print(feature_names)
feature_importance = rf_pipeline.named_steps["clf"].feature_importances_


df = pd.DataFrame({"feature": feature_names, "importance": feature_importance}).sort_values(
    by="importance", ascending=False
)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(df["feature"], df["importance"], color="skyblue")
plt.xlabel("Mean Feature Importance")
plt.title("Random Forest Regressor Feature Importances (Cross-Validated)")
plt.gca().invert_yaxis()  # Highest at top
plt.tight_layout()
# plt.show()
plt.savefig("./artifacts/feature_importances.png")  # Save the figure as a PNG image

# Save the fitted pipeline
joblib.dump(rf_pipeline, "./saved_models/model_pipeline.pkl")

# Later: Load it back
loaded_pipeline = joblib.load("./saved_models/model_pipeline.pkl")

# Use it for prediction
# y_pred = loaded_pipeline.predict(X_new)
