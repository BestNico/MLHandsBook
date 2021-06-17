import numpy as np
from tensorflow import keras

from create_model import CommonModel

common_model = CommonModel()
common_model.create_model()

model = common_model.model
full_data = common_model.full_data

model.fit(full_data.X_train, full_data.y_train, epochs=30,
          validation_data=(full_data.X_valid, full_data.y_valid))

# save model
model.save("my_model.h5")

# load model
model = keras.models.load_model("my_model.h5")

model.evaluate(full_data.X_test, full_data.y_test)

X_new = full_data.X_test[:3]
y_new = full_data.y_test[:3]

y_pred = model.predict(X_new)
y_pred = np.argmax(y_pred, axis=-1)
print(y_pred)
print(y_new)
