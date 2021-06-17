""" housing_california.py
"""

from typing import Any
from dataclasses import dataclass

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class CaliforniaHousing:
    """ CaliforniaHousing """

    X_train: Any = None
    X_valid: Any = None
    X_test: Any = None
    y_train: Any = None
    y_valid: Any = None
    y_test: Any = None

    def fill_data(self):
        """ fill_data """
        housing = fetch_california_housing()
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_valid = scaler.transform(X_valid)
        self.X_test = scaler.transform(X_test)

        self.y_train = y_train
        self.y_test = y_test
        self.y_valid = y_valid
