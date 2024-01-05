import os
import pandas as pd
from kaggle import KaggleApi
from sklearn.model_selection import train_test_split


def get_kaggle_data():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('uciml/iris', path='src/data', unzip=True)

    return {"status": "Dataset downloaded successfully"}


def load_iris_dataset():
    """
    Loads the Iris dataset.

    Returns:
        JSON or str : Loaded dataset or error message if not found.
    """
    file_path = 'src/data/Iris.csv'
    try:
        df = pd.read_csv(file_path)
        return df.to_json(orient='records')
    except FileNotFoundError:
        return {"error": "Dataset file not found."}


def processing_dataset():
    """
    Processes the loaded Iris dataset by cleaning the Species column.

    Returns:
        JSON or str : Processed dataset or error message if not found.
    """
    data = load_iris_dataset()
    # file_path = 'services/epf-flower-data-science/src/data/Iris.csv'
    try:
        df = pd.read_json(data)
        df['Species'] = df['Species'].apply(lambda x: x.replace('Iris-', ''))
        return df.to_json(orient='records')
    except FileNotFoundError:
        return {"error": "Dataset file not found."}


def split_dataset():
    """
    Splits the processed dataset into training and testing sets.

    Returns:
        JSON or str: Training and testing datasets or error message if not found.
    """
    dataset_processed = processing_dataset()

    try:
        df = pd.read_json(dataset_processed)
        train_df, test_df = train_test_split(df, test_size=0.2)
        return {
            train_df.to_json(orient='records'),
            test_df.to_json(orient='records')
        }
    except FileNotFoundError:
        return {"error": "Dataset file not found."}