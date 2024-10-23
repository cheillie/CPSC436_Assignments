# scenario1_image-recognition.py

import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the dataset
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train_data = np.empty((0, 32*32*3))
train_labels = []

for i in range(1, 6):
    batch = unpickle('cifar-10-batches-py/data_batch_' + str(i))
    train_data = np.vstack((train_data, batch[b'data']))
    train_labels += batch[b'labels']
train_labels = np.array(train_labels)
test_data = unpickle('cifar-10-batches-py/test_batch')[b'data']
test_labels = np.array(unpickle('cifar-10-batches-py/test_batch')[b'labels'])

# Reshape and normalize the data
train_data = train_data.reshape(-1, 32, 32, 3) / 255.0
test_data = test_data.reshape(-1, 32, 32, 3) / 255.0

# Define the model
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)