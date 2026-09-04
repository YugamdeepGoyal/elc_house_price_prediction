import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import kagglehub
import os
from filelock import FileLock


def get_dataset():
    BASE_DIR = Path(__file__).resolve().parent.parent
    lock_path = BASE_DIR / ".dataset_download.lock"
    IMG_DIR = BASE_DIR / "images"
    MODEL_DIR = BASE_DIR / "models"

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    with FileLock(str(lock_path)):
        path = kagglehub.dataset_download("camnugent/california-housing-prices")

    csv_path = os.path.join(path, "housing.csv")
    df = pd.read_csv(csv_path)
    return df