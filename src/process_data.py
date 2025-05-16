from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def read_dataset(filename: str) -> pd.DataFrame:
    """
    Reads the raw data file and returns pandas dataframe.

    Parameters:
        filename (str): Path to the CSV file containing the dataset

    Returns:
        pd.DataFrame: DataFrame containing the loaded dataset
    """
    dataset_df = pd.read_csv(filename)

    return dataset_df


def get_cat_num_cols(dataset_df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """
    Identifies categorical and numerical columns in a DataFrame.

    Parameters:
        dataset_df (pd.DataFrame): DataFrame to analyze for column types
        target_col (str): Name of the target column to exclude from numerical columns

    Returns:
        Tuple[List[str], List[str]]: A tuple containing:
            - categorical_columns: List of categorical column names
            - numerical_columns: List of numerical column names
    """
    categorical_columns = dataset_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_columns = dataset_df.select_dtypes(include=["number"]).columns.tolist()
    numerical_columns.remove(target_col)

    return categorical_columns, numerical_columns


def get_data_preprocess_pipeline(feature_columns: List[str], categorical_columns: List[str]) -> ColumnTransformer:
    """
    Creates a preprocessing pipeline that applies one-hot encoding to categorical
    features and imputation + scaling to numerical features.

    Parameters:
        feature_columns (List[str]): List of numerical feature column names
        categorical_columns (List[str]): List of categorical feature column names

    Returns:
        ColumnTransformer: A transformer that preprocesses numerical and categorical features
        differently using pipelines
    """
    # one-hot-encode categorical features
    cat_pipeline = Pipeline([("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))])

    # impute and scale numerical features
    num_pipeline = Pipeline([("imputer", SimpleImputer(fill_value=0)), ("scaler", StandardScaler())])

    # The column transformer preprocesses the numerical and categorical features differently
    processor = ColumnTransformer(
        transformers=[("num", num_pipeline, feature_columns), ("cat", cat_pipeline, categorical_columns)],
        remainder="passthrough",
    )

    return processor


def process_data(pipeline: ColumnTransformer, data: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the input data using a fitted pipeline and returns a
    DataFrame with processed features and appropriate column names.

    Parameters:
        pipeline (ColumnTransformer): A fitted sklearn ColumnTransformer pipeline
        data (pd.DataFrame): Input dataframe to transform

    Returns:
        pd.DataFrame: Transformed dataframe with updated feature names
    """
    # preprocess the data with fit transform
    X_processed = pipeline.transform(data)

    # Get column names after preprocessing
    column_names = pipeline.get_feature_names_out()

    # Create a DataFrame with the transformed data and new column names
    X_processed_df = pd.DataFrame(X_processed, columns=column_names)

    return X_processed_df


def main() -> None:
    """
    Main function to orchestrate the data preprocessing workflow.

    This function coordinates:
    1. Loading the raw dataset
    2. Splitting into train and test sets
    3. Identifying categorical and numerical columns
    4. Creating and fitting preprocessing pipeline
    5. Transforming train and test data
    6. Saving processed datasets and pipeline

    Returns:
        None
    """
    logger = get_logger(Path(__file__).name)
    logger.info("Starting data preprocessing")

    try:
        # Load configuration
        config = load_config()
        target_col = config["model"]["target_column"]
        train_path = config["data"]["train_path"]
        test_path = config["data"]["test_path"]
        random_state = config["model"]["random_state"]
        filename = config["data"]["raw_data"]
        test_size = config["data"]["test_size"]
        preprocessor_path = config["output"]["preprocessor_path"]

        # Import the dataset
        logger.info(f"Importing dataset from {filename}")
        df = read_dataset(filename)
        logger.debug(f"The raw dataset has the following shape: {df.shape}")

        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logger.debug(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")

        # Get categorical and numerical columns
        categorical_cols, feature_cols = get_cat_num_cols(df, target_col)
        logger.debug(f"The dataframe has the following categorical columns: {categorical_cols}")
        logger.debug(f"The dataframe has the following numerical columns: {feature_cols}")

        # Create and fit preprocessing pipeline
        preprocessor = get_data_preprocess_pipeline(feature_cols, categorical_cols)
        preprocessor.fit(X_train)

        # Transform train and test data
        logger.info("Preprocessing train and test data")
        X_train_processed = process_data(preprocessor, X_train)
        X_test_processed = process_data(preprocessor, X_test)

        # Add target column back to processed data
        y_train_reset = y_train.reset_index(drop=True)
        X_train_processed[target_col] = y_train_reset
        logger.debug(f"X_train_processed shape: {X_train_processed.shape}")

        y_test_reset = y_test.reset_index(drop=True)
        X_test_processed[target_col] = y_test_reset
        logger.debug(f"X_test_processed shape: {X_test_processed.shape}")

        # Save processed data
        logger.info(f"Saving processed train data to {train_path}")
        X_train_processed.to_csv(train_path, index=False)

        logger.info(f"Saving processed test data to {test_path}")
        X_test_processed.to_csv(test_path, index=False)

        # Save the preprocessor pipeline
        logger.info(f"Saving the sklearn preprocessing pipeline to {preprocessor_path}")
        joblib.dump(preprocessor, preprocessor_path)

        logger.info("Preprocessing completed successfully")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Error during preprocessing: {e}")


if __name__ == "__main__":
    main()
