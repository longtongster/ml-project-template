import pytest
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Test data loading
def test_data_loading():
    df = pd.read_csv("./raw_data/ames_unprocessed_data.csv")
    assert isinstance(df, pd.DataFrame)
    assert "SalePrice" in df.columns  # Check if the target column exists
    assert df.shape[0] > 0  # Check that the DataFrame is not empty

# Test column identification
def test_column_identification():
    df = pd.read_csv("./raw_data/ames_unprocessed_data.csv")
    categorical_mask = df.dtypes == object
    categorical_columns = df.columns[categorical_mask].tolist()
    numerical_mask = [not x for x in categorical_mask]
    numerical_columns = df.columns[numerical_mask].tolist()

    assert len(categorical_columns) > 0  # Should have at least one categorical column
    assert len(numerical_columns) > 0  # Should have at least one numerical column

# Test column transformer
def test_column_transformer():
    df = pd.read_csv("./raw_data/ames_unprocessed_data.csv")
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = df.select_dtypes(exclude=["object"]).columns.tolist()

    # Define pipelines for preprocessing
    cat_pipeline = Pipeline([("cat", OneHotEncoder(sparse_output=False))])
    num_pipeline = Pipeline([("imputer", SimpleImputer(fill_value=0)), ("scaler", StandardScaler())])

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[("num", num_pipeline, numerical_columns), ("cat", cat_pipeline, categorical_columns)],
        remainder="passthrough",
    )
    
    # Fit the transformer and check the transformation
    X_transformed = preprocessor.fit_transform(df)
    assert X_transformed.shape[1] == 62  # Ensure columns are transformed correctly


