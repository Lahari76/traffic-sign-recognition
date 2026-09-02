from pathlib import Path
import urllib.request
import zipfile

from src.config import RAW_DATA_DIR


BASE_URL = "https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip"
ZIP_PATH = RAW_DATA_DIR / "gtsrb.zip"
EXTRACT_DIR = RAW_DATA_DIR / "gtsrb"


def download_dataset():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_DIR.exists():
        print(f"Dataset already exists: {EXTRACT_DIR}")
        return

    print("Downloading GTSRB dataset...")
    urllib.request.urlretrieve(BASE_URL, ZIP_PATH)

    print(f"Downloaded: {ZIP_PATH}")
    print("Extracting dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)

    ZIP_PATH.unlink()

    print("Dataset downloaded and extracted successfully.")
    print(f"Dataset location: {EXTRACT_DIR}")


if __name__ == "__main__":
    download_dataset()
