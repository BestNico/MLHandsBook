""" fine_tuning_hp.py
"""
import numpy as np
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras

from chapter_10.data_model.housing_Boston import BostonHousing
from chapter_10.data_model.housing_california import CaliforniaHousing


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    """ build_model """
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model


def restore_data():
    """ restore_data """
    # boston_housing = BostonHousing()
    # boston_housing.fill_data()

    # return boston_housing.X_train, boston_housing.X_test, \
    #     boston_housing.X_valid, boston_housing.y_train, \
    #     boston_housing.y_test, boston_housing.y_valid

    housing = CaliforniaHousing()
    housing.fill_data()

    return housing.X_train, housing.X_test, housing.X_valid, \
        housing.y_train, housing.y_test, housing.y_valid


def cv_check():
    """ cv_check """
    X_train, X_test, X_valid, y_train, y_test, y_valid = restore_data()
    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
    keras_reg.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    msg_test = keras_reg.score(X_test, y_test)
    print(f"msg_test: {msg_test}")
    y_pred = keras_reg.predict(X_test[:3])
    print(f"predict: {y_pred} ----> {y_test[:3]}")


def randomized_search_cv():
    """ randomized_search_cv """
    X_train, X_test, X_valid, y_train, y_test, y_valid = restore_data()
    param_distribs = {
        "n_hidden": [0, 1, 2, 3],
        "n_neurons": np.arange(1, 100),
        "learning_rate": reciprocal(3e-4, 3e-2),
    }
    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
    rnd_search_cv.fit(X_train, y_train, epochs=100,
                      validation_data=(X_valid, y_valid),
                      callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    print(rnd_search_cv.best_params_)
    print(rnd_search_cv.best_score_)
    model = rnd_search_cv.best_estimator_.model
