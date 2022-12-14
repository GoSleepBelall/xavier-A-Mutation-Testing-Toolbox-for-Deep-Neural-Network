import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, AveragePooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy



class Lenet5_generator:
    """
    This class will create a LENET-5model, train it on MNIST and save it files as xavier-lenet5.h5
    """
    def generate_model(self):
        # Load Data
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        # Convert into Numpy Arrays
        train_X = np.asarray(train_X)
        train_y = np.asarray(train_y)

        # Create Traditional Model LENET-5
        model = Sequential([
            Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', input_shape=(28, 28, 1), padding="same"),
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
        # Note: I don't exactly remember now (11 Dec 2022) why I used this specific loss function.
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])


        # Train The Model
        # Parameters
        # x = input data (sample data)
        # y = labeled data (target data)
        # epochs = It will run the whole data 10 times before complete training
        # Additional Parameters that can be used
        # batch_size = it will take 10 inputs in an iteration
        # shuffle = shuffle the data in respective order
        # verbose = an option to allow to see the output
        model.fit(x=train_X, y=train_y, epochs=2)
        model.evaluate(test_X, test_y)

        #We already have a working model, so it is commented for the time being
        model.save("../models/xavier-lenet5.h5")
        print("xavier-lenet5 saved successfully")


