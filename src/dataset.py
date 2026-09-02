from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_DIR, RANDOM_SEED
from src.preprocess import preprocess_image


DATASET_DIR = RAW_DATA_DIR / "gtsrb"


def load_dataset():
    images = []
    labels = []

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

    class_dirs = sorted(
        [p for p in DATASET_DIR.iterdir() if p.is_dir()],
        key=lambda p: int(p.name),
    )

    for class_dir in class_dirs:
        label = int(class_dir.name)

        for image_path in class_dir.glob("*.ppm"):
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            images.append(preprocess_image(image))
            labels.append(label)

    x = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)

    return x, y


def split_dataset(x, y):
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == "__main__":
    x, y = load_dataset()

    print(f"Images: {x.shape}")
    print(f"Labels: {y.shape}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Min pixel: {x.min():.4f}")
    print(f"Max pixel: {x.max():.4f}")
