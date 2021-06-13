""" use_callbacks.py
"""

import numpy as np
from tensorflow import keras

from create_model import CommonModel

common_model = CommonModel()
common_model.create_model()

model = common_model.model
full_data = common_model.full_data

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_model_cbs.h5", save_best_only=True)

early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True)

history = model.fit(
    full_data.X_train, full_data.y_train, epochs=50,
    validation_data=(full_data.X_valid, full_data.y_valid),
    callbacks=[checkpoint_cb, early_stopping_cb])

new_model = keras.models.load_model("my_model_cbs.h5")

X_new = full_data.X_test[:3]
y_pred = np.argmax(new_model.predict(X_new), axis=-1)
print(y_pred, full_data.y_test[:3])
