import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, AveragePooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time



# Weights distribution expected
kWeights = np.array([
                     [[1,2,3,4,5],
                      [6,7,8,9,10],
                      [11,12,13,14,15],
                      [16,17,18,19,20],
                      [21,22,23,24,25]],
                     [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]],
                     [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]],
                     [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]],
                     [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]],
                     [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]
                     ])

#Weights distribution we assumed
kWeights_assumed = np.array ([
    [
        [1,1,1,1,1,1], [2,2,2,2,2,2], [3,3,3,3,3,3],[4,4,4,4,4,4], [5,5,5,5,5,5]
    ],
    [
        [6,6,6,6,6,6], [7,7,7,7,7,7], [8,8,8,8,8,8], [9,9,9,9,9,9], [10,10,10,10,10,10]
    ],
    [
        [11,11,11,11,11,11], [12,12,12,12,12,12], [13,13,13,13,13,13], [14,14,14,14,14,14], [15,15,15,15,15,15]
    ],
    [
        [16,16,16,16,16,16], [17,17,17,17,17,17], [18,18,18,18,18,18], [19,19,19,19,19,19], [20,20,20,20,20,20]
    ],
    [
        [21,21,21,21,21,21], [22,22,22,22,22,22], [23,23,23,23,23,23], [24,24,24,24,24,24], [25,25,25,25,25,25]
    ]
])
# after observing the actual pattern, we are now following this pattern
kWeights_assumed_numpy_constraint = np.array ([
     [
        [[1,1,1,1,1,1]], [[2,2,2,2,2,2]], [[3,3,3,3,3,3]],[[4,4,4,4,4,4]], [[5,5,5,5,5,5]]
     ],
    [
        [[6,6,6,6,6,6]], [[7,7,7,7,7,7]], [[8,8,8,8,8,8]], [[9,9,9,9,9,9]], [[10,10,10,10,10,10]]
    ],
    [
        [[11,11,11,11,11,11]], [[12,12,12,12,12,12]], [[13,13,13,13,13,13]], [[14,14,14,14,14,14]], [[15,15,15,15,15,15]]
    ],
    [
        [[16,16,16,16,16,16]], [[17,17,17,17,17,17]], [[18,18,18,18,18,18]], [[19,19,19,19,19,19]], [[20,20,20,20,20,20]]
    ],
    [
        [[21,21,21,21,21,21]], [[22,22,22,22,22,22]], [[23,23,23,23,23,23]], [[24,24,24,24,24,24]], [[25,25,25,25,25,25]]
    ]])

# Convert it into tensor
kw = tf.convert_to_tensor(kWeights_assumed_numpy_constraint)

# Add bias weights
ww = [kWeights_assumed_numpy_constraint, np.random.random((6,))]

# Create Model LENET-5 Traditional
model = Sequential([
    Conv2D(6, kernel_size=(5, 5), activation='tanh',weights =ww, input_shape=(28, 28, 1), padding="same"),
    AveragePooling2D(),  # pool_size=(2, 2),
    Conv2D(filters=16, kernel_size=(5, 5), activation='tanh', input_shape=(10, 10, 1)),
    AveragePooling2D(),
    Flatten(),
    Dense(units=120, activation='tanh'),
    Dense(units=84, activation='tanh'),
    # Softmax function gives probabilities of each output class
    Dense(units=10, activation='softmax')
])


# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
weights = []
for layer in model.layers:
    # Check if its convolutional layer
    if isinstance(layer, keras.layers.Conv2D):
        if layer.name == "conv2d":
            weights = np.array(layer.get_weights())
print(weights)

# After printing weights we proved that our suggested structure of weight tensor is correct and can be accessible by the following formula
# weights[bias][row][column][0][kernel]
