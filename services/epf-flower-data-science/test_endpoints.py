import os
import pytest
from fastapi.testclient import TestClient
from src.app import get_application  # Assuming this is the module where your FastAPI app is defined


class TestEndpointRoutes:
    @pytest.fixture
    def client(self) -> TestClient:
        """
        Test client for integration tests
        """
        app = get_application()
        client = TestClient(app, base_url="http://testserver")
        return client

    def test_get_data_endpoint(self, client: TestClient):
        response = client.get("/data")
        assert response.status_code == 200
        assert response.text == '"Kaggle data downloaded successfully"'

    def test_load_data_as_json_endpoint(self, client: TestClient):
        response = client.get("/data/load")
        assert response.status_code == 200
        assert "SepalLengthCm" in response.json()  # Assuming a key present in the loaded dataset

    def test_get_processed_iris_dataset_endpoint(self, client: TestClient):
        response = client.get("/data/process")
        assert response.status_code == 200
        assert "setosa" in response.json()  # Assuming a value after processing

    def test_split_iris_dataset_endpoint(self, client: TestClient):
        response = client.get("/data/split")
        assert response.status_code == 200
        assert "train" in response.json()
        assert "test" in response.json()

    def test_train_iris_dataset_endpoint(self, client: TestClient):
        response = client.get("/data/train")
        assert response.status_code == 200
        assert "Model trained and saved successfully" in response.json()["status"]
        assert os.path.exists('src/models/random_forest_model.joblib')

    def test_predict_iris_dataset_endpoint(self, client: TestClient):
        response = client.get("/data/predict")
        assert response.status_code == 200
        assert "setosa" in response.json()  # Assuming a value in the prediction

    def test_get_firestore_endpoint(self, client: TestClient):
        response = client.get("/data/get-firestore")
        assert response.status_code == 200
        # Assuming there is data in Firestore
