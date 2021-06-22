""" l_softmax.py
    practice for softmax layer
"""


import numpy as np
from numpy import ndarray
import tensorflow as tf


# softmax by np
def softmax(x: ndarray):
    """ softmax """

    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


#  softmax layer on keras
def l_softmax():
    """ l_softmax """

    layer = tf.keras.layers.Softmax()
    print(layer(np.asarray([1., 2., 1.])).numpy())


# print(softmax(np.array([[1., 2., 3.]])))
l_softmax()
