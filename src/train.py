from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

TARGET_COL = "SalePrice"
TRAIN_PATH = "./processed_data/train_processed.csv"
TEST_PATH = "./processed_data/test_processed.csv"
MAX_DEPTH = 6


def load_dataset(filepath: str, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_dataset = pd.read_csv(filepath)
    X = df_dataset.drop(columns=[target_col])
    y = df_dataset[target_col]
    return X, y


if __name__ == "__main__":
    # load the training data
    X_train, y_train = load_dataset(TRAIN_PATH, TARGET_COL)
    print(X_train.shape)

    # Load the test data
    X_test, y_test = load_dataset(TEST_PATH, TARGET_COL)
    print(X_test.shape)

    rf = RandomForestRegressor(max_depth=MAX_DEPTH)

    # Get the accuracy on the training data using cross validation
    cross_val_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    # Print the 10-fold RMSE
    print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))

    # Fit the model on the toal training data
    rf.fit(X_train, y_train)

    # Determine the accuracy on the test dataset
    y_pred = rf.predict(X_test)
    neg_mse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Negative MSE:", neg_mse)

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
    joblib.dump(rf, "./saved_models/model_pipeline.pkl")
