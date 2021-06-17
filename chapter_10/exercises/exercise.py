""" exercise.py
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.datasets.mnist import load_data

from chapter_10.exercises.custom_cb import ExponentialLearningRate


(X_train_full, y_train_full), (X_test, y_test) = load_data()

# Split train set and valid set
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.


# Create MLPS
def create_model():
    """ create_model """
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    return model


# find best lr
def find_lr():
    """ find_lr """
    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = create_model()

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-3),
                  metrics=["accuracy"])

    expon_lr = ExponentialLearningRate(factor=1.005)

    history = model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid),
                        callbacks=[expon_lr])

    plt.plot(expon_lr.rates, expon_lr.losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
    plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
    plt.grid()
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.show()
    # best lr=3e-1


def process_model():
    """ process_model """
    model = create_model()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=3e-1),
                  metrics=["accuracy"])

    # Define checkpoint for early stopping
    run_index = 1
    run_logdir = os.path.join(os.curdir, "chapter_10/exercises/my_minst_logs", f"run_{run_index}")
    print(run_logdir)
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "./chapter_10/exercises/my_model.h5", save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),
              callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])


def evaluate_model():
    """ evaluate_model """

    model = keras.models.load_model("./chapter_10/exercises/my_model.h5")
    model.evaluate(X_test, y_test)

    X_new = X_test[:3]
    y_pred = model.predict(X_new)
    print(np.argmax(y_pred, axis=-1))
    print(y_test[:3])
