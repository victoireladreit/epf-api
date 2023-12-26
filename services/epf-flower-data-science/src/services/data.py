import pandas as pd
from kaggle import KaggleApi
from sklearn.preprocessing import StandardScaler


def get_kaggle_data():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('uciml/iris', path='src/data', unzip=True)

    return {"status": "Dataset downloaded successfully"}


def load_csv_data_as_json():
    file_path = 'src/data/iris.csv'
    try:
        df = pd.read_csv(file_path)
        return df.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries
    except FileNotFoundError:
        return {"error": "Dataset file not found."}


def process_iris_dataset():
    file_path = 'src/data/iris.csv'
    try:
        df = pd.read_csv(file_path)

        # Example processing: Standardizing numeric features
        scaler = StandardScaler()
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        return df.to_dict(orient='records')

    except FileNotFoundError:
        return {"error": "Dataset file not found."}
