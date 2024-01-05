import json
import os

import joblib
import pandas as pd
from kaggle import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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


def train_dataset():
    """
    Trains a machine learning model using the training dataset.

    Returns:
        str : Status message indicating successful model training or error message if not found.
    """
    train, test = split_dataset()

    train_df = pd.read_json(train)

    # Separating X_train and y_train
    X_train = train_df.drop(columns=["Species"])
    y_train = train_df["Species"]

    # Load model parameters from JSON file
    parameters_file_path = "src/config/model_parameters.json"
    with open(parameters_file_path, 'r') as file:
        model_parameters = json.load(file)

    # Initialize and train the model with train data
    model = RandomForestClassifier(**model_parameters)
    model.fit(X_train, y_train)

    if not os.path.exists('src/models'):
        os.makedirs('src/models')

    # Store the model
    model_save_path = 'src/models/random_forest_model.joblib'
    joblib.dump(model, model_save_path)

    return {"status": "Model trained and saved successfully"}