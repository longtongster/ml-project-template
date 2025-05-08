from typing import List, Tuple

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib


def read_dataset(filename: str) -> pd.DataFrame:
    """
    Reads the raw data file and returns pandas dataframe
    Target column values are expected in binary format with Yes/No values

    Parameters:
    filename (str): raw data filename
    drop_columns (List[str]): column names that will be dropped
    target_column (str): name of target column

    Returns:
    pd.Dataframe: Target encoded dataframe
    """
    df = pd.read_csv(filename)

    return df


def get_cat_num_cols(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """
    Takes a dataframe as input and return a list with the categorical
    and numerical columns.

    Parameters:
    df (pd.DataFrame): pandas dataframe
    target_col (str): name of target column

    Returns:
    A list of string with categorical columsn and one with numerical column names
    """
    # Create a boolean mask for categorical columns
    categorical_mask = df.dtypes == object

    # Get list of categorical column names
    categorical_columns = df.columns[categorical_mask].tolist()
    print("The dataframe has the following categorical columns:")
    print(categorical_columns)

    # Create a boolean mask for numerical columns
    numerical_mask = [not x for x in categorical_mask]

    numerical_columns = df.columns[numerical_mask].tolist()
    numerical_columns.remove(target_col)
    print("The dataframe has the following numerical columns")
    print(numerical_columns)
    return categorical_columns, numerical_columns


def get_data_preprocess_pipeline(feature_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    # one-hot-encode categorical features
    print(categorical_cols)
    cat_pipeline = Pipeline([("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))])

    # impute and scale numerical features
    num_pipeline = Pipeline([("imputer", SimpleImputer(fill_value=0)), ("scaler", StandardScaler())])

    # The column transformer preprocesses the numerical and categorical features differently
    preprocessor = ColumnTransformer(
        transformers=[("num", num_pipeline, feature_cols), ("cat", cat_pipeline, categorical_cols)],
        remainder="passthrough",
    )

    return preprocessor


def process_data(pipeline, data):
    # preprocess the data with fit transform
    X_processed = pipeline.transform(data)

    # Get column names after preprocessing
    column_names = pipeline.get_feature_names_out()

    # Create a DataFrame with the transformed data and new column names
    X_processed_df = pd.DataFrame(X_processed, columns=column_names)

    return X_processed_df


if __name__ == "__main__":
    TARGET_COL = "SalePrice"
    FILENAME = "./raw_data/ames_unprocessed_data.csv"
    TEST_SIZE = 0.20

    # import the dataset
    df = read_dataset(FILENAME)

    # remove target from dataframe to keep the features
    X = df.drop(columns=[TARGET_COL])
    # assign target to y
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=54)
    print(X_train.shape, X_test.shape)

    # get categorical and numerical columns
    categorical_columns, feature_columns = get_cat_num_cols(df, TARGET_COL)

    # The preprocessor defines separate steps for categorical and features columns
    preprocessor = get_data_preprocess_pipeline(feature_columns, categorical_columns)

    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)

    # get the processed train data
    X_train_processed = process_data(preprocessor, X_train)

    # get the processed test data
    X_test_processed = process_data(preprocessor, X_test)

    # Optional: inspect the first few rows
    print(X_train_processed.shape)
    print(X_test_processed.shape)

    X_train_processed.insert(loc=0, column=TARGET_COL, value=y)
    print(X_train_processed.shape)

    # Save preprocessed data
    print("Saving processed train and test datat to `processed_data` directory")
    pd.DataFrame(X_train_processed).to_csv("./processed_data/train_processed.csv", index=False)
    pd.DataFrame(X_test_processed).to_csv("./processed_data/test_processed.csv", index=False)

    # Save the preprocessor pipeline
    print("Saving the sklearn preprocessing pipeline to `artificats`")
    joblib.dump(preprocessor, "./artifacts/preprocessor.pkl")
