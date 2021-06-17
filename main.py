""" main.py
"""

from chapter_10.use_tensorboard.use_tensorboard import use_tensorboard
from chapter_10.fine_tuning_hps.fine_tuning_hp import restore_data, randomized_search_cv
import chapter_10.exercises.exercise as Exercise_model


if __name__ == '__main__':
    # use_tensorboard()
    # restore_data()
    # randomized_search_cv()
    # print(Exercise_model.X_train.shape)
    # print(Exercise_model.X_test.shape)
    # print(Exercise_model.X_valid.shape)
    # print(Exercise_model.y_train.shape)
    # print(Exercise_model.y_test.shape)
    # print(Exercise_model.y_valid.shape)

    # Exercise_model.process_model()
    Exercise_model.evaluate_model()
