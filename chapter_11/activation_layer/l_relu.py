""" l_relu.py
    practice for relu
"""

import tensorflow as tf


def l_relu():
    """ l_rely"""
    layer_1 = tf.keras.layers.ReLU()
    print(layer_1([-3.0, -1.0, 0.0, 1.0, 5.0]).numpy())

    # define max_value
    layer_2 = tf.keras.layers.ReLU(max_value=4.0)
    print(layer_2([-3.0, -1.0, 0.0, 1.0, 5.0]).numpy())

    # define threshold
    layer_3 = tf.keras.layers.ReLU(threshold=1.5)
    print(layer_3([-3.0, -1.0, 0.0, 1.0, 5.0]).numpy())

    # define negative_slope
    layer_4 = tf.keras.layers.ReLU(negative_slope=1.0)
    print(layer_4([-3.0, -1.0, 0.0, 1.0, 5.0]).numpy())

    layer_5 = tf.keras.layers.LeakyReLU(alpha=1.0)
    print(layer_5([-3.0, -1.0, 0.0, 1.0, 5.0]).numpy())  # layer_4 has same results with layer_5


l_relu()
