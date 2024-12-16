from keras import models
import numpy as np
import cv2

model = models.load_model("cats_and_dogs.keras")

my_image = cv2.imread("cats_and_dogs/test/30.jpg")
my_image = cv2.resize(my_image, (300, 300))

my_image = my_image

result = model.predict(np.array([my_image]))

print(result)