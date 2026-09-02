import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import FIGURE_DIR, MODEL_PATH, REPORT_DIR
from src.dataset import load_dataset, split_dataset


def main():
    print("Loading dataset...")
    x, y = load_dataset()

    _, _, x_test, _, _, y_test = split_dataset(x, y)

    print(f"Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = float(np.mean(predictions == y_test))

    print(f"Evaluation accuracy: {accuracy:.4f}")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    with open(REPORT_DIR / "classification_report.json", "w") as file:
        json.dump(report, file, indent=2)

    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("GTSRB Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "confusion_matrix.png",
        dpi=150,
    )
    plt.close()

    print(
        "Classification report saved to:",
        REPORT_DIR / "classification_report.json",
    )
    print(
        "Confusion matrix saved to:",
        FIGURE_DIR / "confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
