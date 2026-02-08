# mnist dataset (handwritten digits)
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

# import Mnist dataset --> 28X28 pixel images of handwritten digits (0-9)
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

single_image = X_train[0]
# print(single_image)

plt.imshow(single_image, cmap='gray')
# plt.show()

# Normalize pixel values from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# print(X_train[0].max()) verifying

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

model=Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu')) #rectified linear unit - max(0,x) - keeps positive part of linear xfomation
model.add(Dense(units=10, activation='softmax')) #softmax - converts output to probability distribution

model.summary()

from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# optimizer - wts&bias updates --> adam --> adaptive moment estimation --> updates model parameters to minimize loss by taking mean of squared gradients and momentum
# loss - function used to measure error --> sparse_categorical_crossentropy --> used for multi-class classification with integer labels
# metrics - function used to evaluate performance --> accuracy --> ratio of correct predictions to total predictions


model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

model.evaluate(X_test, y_test)

# predictions
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# classification report
print("Classification Report---------------------")
print(classification_report(y_test, predicted_labels))

# confusion matrix
print("Confusion Matrix--------------------------")
print(confusion_matrix(y_test, predicted_labels))
