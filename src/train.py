"""
Module for training a RandomForestRegressor model on the Ames housing dataset.

This module loads preprocessed Ames housing data, trains a RandomForestRegressor model,
evaluates it using cross-validation and test set, visualizes feature importances,
and saves the trained model.
"""

from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from utils import get_logger  # pylint: disable=import-error

TARGET_COL = "SalePrice"
TRAIN_PATH = "./processed_data/train_processed.csv"
TEST_PATH = "./processed_data/test_processed.csv"
MAX_DEPTH = 6


def load_dataset(filepath: str, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a dataset from a CSV file and split it into features and target.

    Parameters:
        filepath (str): Path to the CSV file containing the dataset
        target_col (str): Name of the target column to predict

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - X: DataFrame with feature columns
            - y: Series with target values
    """
    df_dataset = pd.read_csv(filepath)
    X = df_dataset.drop(columns=[target_col])
    y = df_dataset[target_col]
    return X, y


if __name__ == "__main__":
    # create a logger
    logger = get_logger(Path(__file__).name)
    logger.info("Start training script")

    # load the training data
    X_train, y_train = load_dataset(TRAIN_PATH, TARGET_COL)
    print(X_train.shape)

    # Load the test data
    X_test, y_test = load_dataset(TEST_PATH, TARGET_COL)
    print(X_test.shape)

    rf = RandomForestRegressor(max_depth=MAX_DEPTH, random_state=42)

    # Get the accuracy on the training data using cross validation
    cross_val_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    # Print the 5-fold RMSE
    print("5-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))

    # Fit the model on the total training data
    rf.fit(X_train, y_train)

    # Determine the accuracy on the test dataset
    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Negative MSE:", rmse)

    data = {"feature": X_train.columns, "importance": rf.feature_importances_}
    df = pd.DataFrame(data).sort_values(by=["importance"])
    plt.figure(figsize=(10, 10))
    plt.barh(y=df["feature"], width=df["importance"])
    plt.xlabel("Mean Feature Importance")
    plt.title("Random Forest Regressor Feature Importances (Cross-Validated)")
    plt.tight_layout()
    plt.savefig("./artifacts/feature_importances.png")  # Save the figure as a PNG image

    # Save the fitted pipeline
    print("save model")
    joblib.dump(rf, "./saved_models/random_forest_model.pkl")
