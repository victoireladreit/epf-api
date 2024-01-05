import pytest
from src.services.data import (
    get_kaggle_data, load_iris_dataset, processing_dataset,
    split_dataset, train_dataset, predict, get_firestore_data, update_firestore_data
)
import os
import joblib
import pandas as pd


class TestFunctions:
    @pytest.fixture(autouse=True)
    def clean_up_files(self):
        # Fixture to clean up generated files after each test
        yield
        if os.path.exists('src/data/Iris.csv'):
            os.remove('src/data/Iris.csv')
        if os.path.exists('src/models/random_forest_model.joblib'):
            os.remove('src/models/random_forest_model.joblib')

    def test_get_kaggle_data(self):
        output = get_kaggle_data()
        assert output == {"status": "Dataset downloaded successfully"}
        assert os.path.exists('src/data/Iris.csv')

    def test_load_iris_dataset(self):
        # Create a dummy Iris dataset file for testing
        df = pd.DataFrame({'SepalLengthCm': [5.1, 4.9, 4.7],
                           'SepalWidthCm': [3.5, 3.0, 3.2],
                           'PetalLengthCm': [1.4, 1.4, 1.3],
                           'PetalWidthCm': [0.2, 0.2, 0.2],
                           'Species': ['Iris-setosa', 'Iris-setosa', 'Iris-setosa']})
        df.to_csv('src/data/Iris.csv', index=False)

        output = load_iris_dataset()
        assert isinstance(output, str)
        assert "Iris-setosa" in output

    def test_processing_dataset(self):
        # Create a dummy Iris dataset file for testing
        df = pd.DataFrame({'SepalLengthCm': [5.1, 4.9, 4.7],
                           'SepalWidthCm': [3.5, 3.0, 3.2],
                           'PetalLengthCm': [1.4, 1.4, 1.3],
                           'PetalWidthCm': [0.2, 0.2, 0.2],
                           'Species': ['Iris-setosa', 'Iris-setosa', 'Iris-setosa']})
        df.to_csv('src/data/Iris.csv', index=False)

        output = processing_dataset()
        assert isinstance(output, str)
        assert "setosa" in output

    def test_split_dataset(self):
        # Create a dummy Iris dataset file for testing
        df = pd.DataFrame({'SepalLengthCm': [5.1, 4.9, 4.7],
                           'SepalWidthCm': [3.5, 3.0, 3.2],
                           'PetalLengthCm': [1.4, 1.4, 1.3],
                           'PetalWidthCm': [0.2, 0.2, 0.2],
                           'Species': ['Iris-setosa', 'Iris-setosa', 'Iris-setosa']})
        df.to_csv('src/data/Iris.csv', index=False)

        output = split_dataset()
        assert isinstance(output, dict)
        assert "train" in output
        assert "test" in output

    def test_train_dataset(self):
        # Create a dummy Iris dataset file for testing
        df = pd.DataFrame({'SepalLengthCm': [5.1, 4.9, 4.7],
                           'SepalWidthCm': [3.5, 3.0, 3.2],
                           'PetalLengthCm': [1.4, 1.4, 1.3],
                           'PetalWidthCm': [0.2, 0.2, 0.2],
                           'Species': ['Iris-setosa', 'Iris-setosa', 'Iris-setosa']})
        df.to_csv('src/data/Iris.csv', index=False)

        output = train_dataset()
        assert output == {"status": "Model trained and saved successfully"}
        assert os.path.exists('src/models/random_forest_model.joblib')

    def test_predict(self):
        # Create a dummy Iris dataset file for testing
        df = pd.DataFrame({'SepalLengthCm': [5.1, 4.9, 4.7],
                           'SepalWidthCm': [3.5, 3.0, 3.2],
                           'PetalLengthCm': [1.4, 1.4, 1.3],
                           'PetalWidthCm': [0.2, 0.2, 0.2],
                           'Species': ['Iris-setosa', 'Iris-setosa', 'Iris-setosa']})
        df.to_csv('src/data/Iris.csv', index=False)

        train_dataset()  # Train the model before prediction

        output = predict()
        assert isinstance(output, str)
        assert "setosa" in output

    def test_get_firestore_data(self):
        # Assuming Firestore contains some data for testing
        output = get_firestore_data()
        assert isinstance(output, dict)

    def test_update_firestore_data(self):
        # Assuming Firestore contains some data for testing
        parameter_name = "test_parameter"
        parameter_value = "test_value"

        output = update_firestore_data(parameter_name, parameter_value)
        assert output == {"Firestore parameter edited with success"}
