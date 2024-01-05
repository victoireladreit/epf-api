import json
import os

import joblib
import pandas as pd
from kaggle import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from google.cloud import firestore


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


def predict():
    """
    Makes predictions using the trained model and test dataset.

    Returns:
        JSON or str: Predictions or error message if not found.
    """
    # Load the trained model
    model_save_path = 'src/models/random_forest_model.joblib'
    try:
        model = joblib.load(model_save_path)
    except FileNotFoundError:
        return {"error": "Trained model not found."}

    train, test = split_dataset()
    test_df = pd.read_json(test)

    X_test = test_df.drop(columns=["Species"])
    y_pred = pd.DataFrame(model.predict(X_test))

    return y_pred.to_json(orient="records")


def get_firestore_data():
    """
    Retrieves data from a Firestore collection and document.

    Returns:
        str or None: Retrieved data or None if not found.
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "src/config/datasources-401318-1df14cce6bc9.json"
    # Initialize Firestore
    db = firestore.Client()

    # Reference the collection and document
    collection_ref = db.collection("parameters")
    document_ref = collection_ref.document("parameters")

    # Get data from the document
    doc_data = document_ref.get().to_dict()

    # Check if data was retrieved
    if doc_data:
        return doc_data
    else:
        return None


def update_firestore_data(parameter_name, parameter_value):
    """
    Creates or adds a parameter in a Firestore collection.

    Args:
        parameter_name (str): Name of the parameter.
        parameter_value: Value of the parameter.

    Returns:
        str: Status message indicating successful parameter creation.
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "src/config/datasources-401318-1df14cce6bc9.json"
    # Initialize Firestore
    db = firestore.Client()

    # Reference the collection and document
    collection_ref = db.collection("parameters")
    document_ref = collection_ref.document("parameters")

    # Get existing data from the document
    doc_data = document_ref.get().to_dict()

    # Add or update the parameter
    doc_data[parameter_name] = parameter_value
    document_ref.set(doc_data)

    return {"Firestore parameter edited with success"}