from pathlib import Path

import cv2
import numpy as np

from src.config import IMAGE_HEIGHT, IMAGE_WIDTH


def preprocess_image(image):
    """
    Preprocess a traffic-sign image for CNN inference.

    Steps:
    1. Validate the image.
    2. Resize to 32x32.
    3. Convert BGR to RGB.
    4. Normalize pixel values to [0, 1].
    """
    if image is None:
        raise ValueError("Image cannot be None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Image cannot be empty.")

    resized = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 4:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
    else:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    normalized = resized.astype(np.float32) / 255.0

    return normalized


def load_and_preprocess_image(image_path):
    """Load an image from disk and preprocess it."""
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    return preprocess_image(image)


def prepare_for_prediction(image):
    """Add the batch dimension required by TensorFlow."""
    processed = preprocess_image(image)
    return np.expand_dims(processed, axis=0)
