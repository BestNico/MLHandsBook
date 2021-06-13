""" create_model.py
"""

from dataclasses import dataclass
from typing import Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist


@dataclass(frozen=True)
class FullData:
    """ FullData data class """
    X_train: Any = None
    X_valid: Any = None
    X_test: Any = None
    y_train: Any = None
    y_valid: Any = None
    y_test: Any = None


@dataclass
class CommonModel:
    """ CommonModel class """

    model = None
    full_data: FullData = None

    def create_model(self):
        X_train, y_train, X_valid, y_valid, X_test, y_test = process_dataset()
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, activation="relu"),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])

        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer="sgd", metrics=["accuracy"])

        self.model = model
        self.full_data = FullData(X_train, X_valid, X_test, y_train, y_valid, y_test)


def process_dataset():
    """ process_dataset """
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    return X_train, y_train, X_valid, y_valid, X_test, y_test



