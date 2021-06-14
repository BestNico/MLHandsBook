""" main.py
"""

from use_tensorboard.use_tensorboard import use_tensorboard
from fine_tuning_hps.fine_tuning_hp import restore_data, randomized_search_cv

if __name__ == '__main__':
    # use_tensorboard()
    # restore_data()
    randomized_search_cv()
