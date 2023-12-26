import traceback

from src.services.data import get_kaggle_data, load_csv_data_as_json, process_iris_dataset, split_dataset
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/data")
def get_data():
    try:
        get_kaggle_data()
    except:
        return "Error: couldn't get download data."

    return "Kaggle data downloaded successfully"


@ router.get("/data/load")
def load_data_as_json():
    dataset = load_csv_data_as_json()
    if "error" in dataset:
        raise HTTPException(status_code=404, detail=dataset["error"])
    return dataset


@router.get("/data/process")
def process_iris_data():
    try:
        dataset = process_iris_dataset()
    except:
        return "Error: couldn't process the data."
    return dataset


@router.get("/data/split")
def split_iris_dataset(test_size: float = 0.2):
    result = split_dataset(test_size)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result