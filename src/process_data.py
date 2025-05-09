from typing import List, Tuple

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def read_dataset(filename: str) -> pd.DataFrame:
    """
    Reads the raw data file and returns pandas dataframe
    Target column values are expected in binary format with Yes/No values

    Parameters:
    filename (str): raw data filename
    
    Returns:
    pd.Dataframe: Target encoded dataframe
    """
    dataset_df = pd.read_csv(filename)

    return dataset_df


def get_cat_num_cols(dataset_df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """
    Takes a dataframe as input and return a list with the categorical
    and numerical columns.

    Parameters:
    df (pd.DataFrame): pandas dataframe
    target_col (str): name of target column

    Returns:
    A list of string with categorical columsn and one with numerical column names
    """
    categorical_columns = dataset_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_columns = dataset_df.select_dtypes(include=["number"]).columns.tolist()
    numerical_columns.remove(target_col)
    
    # Create a boolean mask for categorical columns
    # categorical_mask = dataset_df.dtypes == object

    # # Get list of categorical column names
    # categorical_columns = dataset_df.columns[categorical_mask].tolist()
    print("The dataframe has the following categorical columns:")
    print(categorical_columns)

    # Create a boolean mask for numerical columns
    # numerical_mask = [not x for x in categorical_mask]

    # numerical_columns = dataset_df.columns[numerical_mask].tolist()
    # numerical_columns.remove(target_col)
    print("The dataframe has the following numerical columns")
    print(numerical_columns)


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
    print(categorical_columns)
    cat_pipeline = Pipeline([("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))])

    # impute and scale numerical features
    num_pipeline = Pipeline([("imputer", SimpleImputer(fill_value=0)), ("scaler", StandardScaler())])

    # The column transformer preprocesses the numerical and categorical features differently
    processor = ColumnTransformer(
        transformers=[("num", num_pipeline, feature_columns), ("cat", cat_pipeline, categorical_columns)],
        remainder="passthrough",
    )

    return processor


def process_data(pipeline: Pipeline, data: pd.DataFrame) -> Pipeline:
    """
    Transforms the input data using a fitted pipeline and returns a
    DataFrame with processed features and appropriate column names.

    Parameters:
    pipeline: A fitted sklearn ColumnTransformer pipeline
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
    categorical_cols, feature_cols = get_cat_num_cols(df, TARGET_COL)

    # The preprocessor defines separate steps for categorical and features columns
    preprocessor = get_data_preprocess_pipeline(feature_cols, categorical_cols)

    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)

    # get the processed train data
    X_train_processed = process_data(preprocessor, X_train)

    # get the processed test data
    X_test_processed = process_data(preprocessor, X_test)

    # Optional: inspect the first few rows
    print(X_train_processed.shape)
    print(X_test_processed.shape)

    X_train_processed.insert(loc=0, column=TARGET_COL, value=y_train)
    print("X_train_processed", X_train_processed.shape)

    X_test_processed.insert(loc=0, column=TARGET_COL, value=y_test)
    print("X_test_processed", X_test_processed.shape)

    # Save preprocessed data
    print("Saving processed train and test datat to `processed_data` directory")
    pd.DataFrame(X_train_processed).to_csv("./processed_data/train_processed.csv", index=False)
    pd.DataFrame(X_test_processed).to_csv("./processed_data/test_processed.csv", index=False)

    # Save the preprocessor pipeline
    print("Saving the sklearn preprocessing pipeline to `artifacts`")
    joblib.dump(preprocessor, "./artifacts/preprocessor.pkl")
