""" housing_Boston.py
"""

from dataclasses import dataclass
from typing import Any

from tensorflow import keras
from tensorflow.keras.datasets import boston_housing


@dataclass
class BostonHousing:
    """ BostonHousing """

    X_train: Any = None
    X_test: Any = None
    X_valid: Any = None
    y_train: Any = None
    y_test: Any = None
    y_valid: Any = None

    def fill_data(self):
        """ fill_data """
        (X_train_full, y_train_full), (X_test, y_test) = boston_housing.load_data()

        self.X_train = X_train_full[:300]
        self.X_test = X_test
        self.X_valid = X_train_full[300:]
        self.y_train = y_train_full[:300]
        self.y_test = y_test
        self.y_valid = y_train_full[300:]
