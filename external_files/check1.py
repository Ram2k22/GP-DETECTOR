import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r'c:\Users\SATYA\Downloads\1.jpg')

# Resize the image
image = cv2.resize(image, (28, 28))

# Scale the image
image = image / 255.0

# Convert the image to a tensor

image = np.sum(image, axis=2).reshape((28,28,1))
image = np.expand_dims(image, axis=0)


# Load the prediction model
model = tf.keras.models.load_model('saved_models/MNIST_model.h5')

# Get the prediction
prediction = np.argmax(model.predict(image))

# Print the prediction
print(prediction)

plt.imshow(image.reshape((28,28,1)))
