import tensorflow as tf

from src.config import (
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEARNING_RATE,
    NUM_CLASSES,
)


def build_model():
    """Build and compile a CNN for traffic-sign classification."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
            ),

            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same"
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.20),

            tf.keras.layers.Conv2D(
                64, (3, 3), activation="relu", padding="same"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                64, (3, 3), activation="relu", padding="same"
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(
                128, (3, 3), activation="relu", padding="same"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.30),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.40),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
