import tensorflow as tf
from keras import layers, models
import os
import numpy as np
import cv2

width = 300
height = 300
train_path = "cats_and_dogs/train/"
predict_path = "cats_and_dogs/predict/"

train_x = []
train_y = []

for i in os.listdir(train_path):
    for j in os.listdir(train_path + i):
        img = cv2.imread(train_path + i + "/" + j)
        resized_img = cv2.resize(img, (width, height))
        train_x.append(resized_img)

        if i == "cats":
            train_y.append([0, 1])
        else:
            train_y.append([1, 0])

x_data = np.array(train_x)
y_data = np.array(train_y)

model = tf.keras.models.Sequential([
    layers.Conv2D(32, 3, 3, input_shape=(width, height, 3)),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, 3, 3),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(128, 3, 3),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(64),
    layers.Activation("relu"),
    layers.Dropout(0.6),
    layers.Dense(2),
    layers.Activation("sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

epochs = 100

model.fit(x_data, y_data, epochs=epochs)

models.save_model(model, "cats_and_dogs.keras")