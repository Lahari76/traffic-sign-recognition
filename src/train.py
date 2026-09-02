import json

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    FIGURE_DIR,
    MODEL_PATH,
    RANDOM_SEED,
    REPORT_DIR,
)
from src.dataset import load_dataset, split_dataset
from src.model import build_model


def plot_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Traffic Sign Classification Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "training_accuracy.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Traffic Sign Classification Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "training_loss.png")
    plt.close()


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    print("Loading GTSRB dataset...")
    x, y = load_dataset()

    print(f"Loaded {len(x)} images.")

    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(x, y)

    print(f"Training samples:   {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples:       {len(x_test)}")

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "training_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
    }

    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_history(history)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {REPORT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
