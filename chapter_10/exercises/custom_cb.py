""" custom_cb.py
"""

from tensorflow import keras

K = keras.backend


class ExponentialLearningRate(keras.callbacks.Callback):
    """ ExponentialLearningRate """

    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)
