"""
Module for training a RandomForestRegressor model on the Ames housing dataset.

This module loads preprocessed Ames housing data, trains a RandomForestRegressor model,
evaluates it using cross-validation and test set, visualizes feature importances,
and saves the trained model.
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from utils import get_logger  # pylint: disable=import-error


def load_config(config_path: str = "./config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration parameters from a YAML file.

    Parameters:
        config_path (str): Path to the YAML configuration file

    Returns:
        dict: Dictionary containing configuration parameters organized by sections
            such as 'data', 'model', and 'output'

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the configuration file has invalid YAML syntax
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


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


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, max_depth: int = 5, random_state: int = 42
) -> RandomForestRegressor:
    """
    Train a RandomForest model with cross-validation.

    Parameters:
        X_train (pd.DataFrame): DataFrame containing training features
        y_train (pd.Series): Series containing target values for training

    Returns:
        RandomForestRegressor: Trained RandomForest model ready for predictions
    """
    logger = get_logger(Path(__file__).name)
    rf = RandomForestRegressor(max_depth=max_depth, random_state=random_state)

    # Cross-validation
    cross_val_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    logger.info(f"The RMSE with cv=5 using the training data: {np.mean(np.sqrt(np.abs(cross_val_scores))):.4f}")

    # Train final model
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluate the model on test data using RMSE metric.

    Parameters:
        model (RandomForestRegressor): Trained RandomForest model to evaluate
        X_test (pd.DataFrame): DataFrame containing test features
        y_test (pd.Series): Series containing true target values for testing

    Returns:
        float: Root Mean Squared Error (RMSE) score on test data
    """
    logger = get_logger(Path(__file__).name)
    y_pred = model.predict(X_test)
    rmse: float = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info(f"The RMSE on the hold out (test) dataset: {rmse:.4f}")
    return rmse


def create_feature_importance_plot(model: RandomForestRegressor, X_train: pd.DataFrame) -> None:
    """
    Create and save a horizontal bar chart of feature importances.

    Parameters:
        model (RandomForestRegressor): Trained RandomForest model with feature_importances_ attribute
        X_train (pd.DataFrame): DataFrame containing training features, used to get feature names

    Returns:
        None: The function saves the plot to ./artifacts/feature_importances.png
    """
    logger = get_logger(Path(__file__).name)
    try:
        data = {"feature": X_train.columns, "importance": model.feature_importances_}
        df = pd.DataFrame(data).sort_values(by="importance", ascending=False)
        plt.figure(figsize=(10, 10))
        plt.barh(y=df["feature"], width=df["importance"])
        plt.xlabel("Mean Feature Importance")
        plt.title("Random Forest Regressor Feature Importances")
        plt.tight_layout()
        plt.savefig("./artifacts/feature_importances.png")
        logger.info("Feature importance plot saved to ./artifacts/feature_importances.png")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Error creating feature importance plot: {e}")


def save_model(model: RandomForestRegressor) -> None:
    """
    Save the trained model to disk using joblib serialization.

    Parameters:
        model (RandomForestRegressor): Trained RandomForest model to be saved

    Returns:
        None: The function saves the model to ./saved_models/random_forest_model.pkl
    """
    logger = get_logger(Path(__file__).name)
    try:
        joblib.dump(model, "./saved_models/random_forest_model.pkl")
        logger.info("Model saved to ./saved_models/random_forest_model.pkl")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Error saving model: {e}")


def main() -> None:
    """
    Main function to orchestrate the entire model training process.

    This function coordinates the workflow of:
    1. Loading training and test datasets
    2. Training the RandomForest model with cross-validation
    3. Evaluating model performance on test data
    4. Creating and saving feature importance visualizations
    5. Saving the trained model to disk

    Returns:
        None
    """
    logger = get_logger(Path(__file__).name)
    logger.info("Start training script")

    # In your main function
    config = load_config()
    TARGET_COL = config["model"]["target_column"]
    TRAIN_PATH = config["data"]["train_path"]
    TEST_PATH = config["data"]["test_path"]
    MAX_DEPTH = config["model"]["max_depth"]
    RANDOM_STATE = config["model"]["random_state"]

    try:
        # Load data
        X_train, y_train = load_dataset(TRAIN_PATH, TARGET_COL)
        X_test, y_test = load_dataset(TEST_PATH, TARGET_COL)
        logger.debug(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Train model
        rf = train_model(X_train, y_train, MAX_DEPTH, RANDOM_STATE)

        # Evaluate model
        rmse = evaluate_model(rf, X_test, y_test)
        with open("./artifacts/metrics.json", "w", encoding="utf-8") as f:
            metrics = {"rmse": rmse}
            json.dump(metrics, f)

        # Visualize feature importance
        create_feature_importance_plot(rf, X_train)

        # Save model
        save_model(rf)

        logger.info("Training completed successfully")

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Error during training: {e}")


if __name__ == "__main__":
    main()
