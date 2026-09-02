from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3
NUM_CLASSES = 43

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
RANDOM_SEED = 42

MODEL_PATH = MODEL_DIR / "traffic_sign_cnn.keras"

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    REPORT_DIR,
    FIGURE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
