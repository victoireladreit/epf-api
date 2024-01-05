import os
import pandas as pd
from kaggle import KaggleApi


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