""" l_leakyReLU.py
    practice for leakyReLU activation fn
"""

import tensorflow as tf


def l_leaky_relu():
    """ l_leaky_relu """

    # default alpha is equal to 0.3
    layer_1 = tf.keras.layers.LeakyReLU()
    output = layer_1([-3.0, -1.0, 0.0, 2.0])
    print(output.numpy())

    # define alpha
    layer_2 = tf.keras.layers.LeakyReLU(alpha=0.1)
    output = layer_2([-3.0, -1.0, 0.0, 2.0])
    print(output.numpy())


l_leaky_relu()
