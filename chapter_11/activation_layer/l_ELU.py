""" l_ELU.py
"""

import tensorflow as tf


def l_elu():
    """ l_elu """
    layer_1 = tf.keras.layers.ELU()
    output = layer_1([-3.0, -2.0, 0.0, 1.0, 4.0])
    print(output.numpy())


l_elu()
