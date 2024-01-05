import traceback

from src.services.data import get_kaggle_data, load_iris_dataset, processing_dataset
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
    dataset = load_iris_dataset()
    if "error" in dataset:
        raise HTTPException(status_code=404, detail=dataset["error"])
    return dataset


@router.get("/data/process")
def get_processed_iris_dataset():
    processed_dataset = processing_dataset()
    if "error" in processed_dataset:
        raise HTTPException(status_code=404, detail=processed_dataset["error"])
    return processed_dataset