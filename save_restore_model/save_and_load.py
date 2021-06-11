import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist

# build a classification model

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

# save model
model.save("my_model.h5")

# load model
model = keras.models.load_model("my_model.h5")

model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_new = y_test[:3]

y_pred = model.predict(X_new)
y_pred = np.argmax(y_pred, axis=-1)
print(y_pred)
print(y_new)
