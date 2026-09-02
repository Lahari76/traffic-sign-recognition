import argparse

import cv2
import numpy as np
import tensorflow as tf

from src.config import MODEL_PATH
from src.preprocess import prepare_for_prediction


CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}


def predict_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Unable to read image: {image_path}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    processed = prepare_for_prediction(image)

    probabilities = model.predict(processed, verbose=0)[0]

    class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[class_id])

    return class_id, CLASS_NAMES[class_id], confidence


def main():
    parser = argparse.ArgumentParser(
        description="Predict a GTSRB traffic-sign image."
    )
    parser.add_argument(
        "image",
        help="Path to the traffic-sign image",
    )

    args = parser.parse_args()

    class_id, class_name, confidence = predict_image(args.image)

    print(f"Predicted class ID: {class_id}")
    print(f"Traffic sign: {class_name}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
