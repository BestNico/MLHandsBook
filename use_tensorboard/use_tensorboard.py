""" use_tensorboard.py
"""

import os
import time
from tensorflow import keras

from save_restore_model.create_model import CommonModel


root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    """ get_run_logdir """
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def use_tensorboard():
    """ use_tensorboard """
    c_model = CommonModel()
    c_model.create_model()

    model = c_model.model
    full_data = c_model.full_data

    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    model.fit(full_data.X_train, full_data.y_train, epochs=30,
              validation_data=(full_data.X_valid, full_data.y_valid),
              callbacks=[tensorboard_cb])

# tensorboard command: tensorboard --logdir=./my_logs --port=6006
