from src.services.data import get_kaggle_data, load_csv_data_as_json, process_species_data
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
def process_data():
    try:
        dataset = process_species_data()
    except:
        return "Error: couldn't process the data."
    return dataset