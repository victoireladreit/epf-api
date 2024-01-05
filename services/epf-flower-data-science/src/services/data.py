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