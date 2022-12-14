import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, AveragePooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time



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

class alexnet_generator:
    """
    This class will create an ALEXNET-model, train it on cifar10 and save it files as xavier-alexnet.h5
    """

    def process_images(image, label):
        # Normalize images to have a mean of 0 and standard deviation of 1
        image = tf.image.per_image_standardization(image)
        # Resize images from 32x32 to 277x277
        image = tf.image.resize(image, (227, 227))
        return image, label

    def get_run_logdir(self, root_logdir):
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)
    def generate_model(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
        CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        validation_images, validation_labels = train_images[:5000], train_labels[:5000]
        train_images, train_labels = train_images[5000:], train_labels[5000:]

        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        plt.figure(figsize=(20, 20))
        for i, (image, label) in enumerate(train_ds.take(5)):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(image)
            plt.title(CLASS_NAMES[label.numpy()[0]])
            plt.axis('off')

        train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
        test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
        validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
        print("Training data size:", train_ds_size)
        print("Test data size:", test_ds_size)
        print("Validation data size:", validation_ds_size)

        train_ds = (train_ds
                    .map(self.process_images())
                    .shuffle(buffer_size=train_ds_size)
                    .batch(batch_size=32, drop_remainder=True))
        test_ds = (test_ds
                   .map(self.process_images())
                   .shuffle(buffer_size=train_ds_size)
                   .batch(batch_size=32, drop_remainder=True))
        validation_ds = (validation_ds
                         .map(self.process_images())
                         .shuffle(buffer_size=train_ds_size)
                         .batch(batch_size=32, drop_remainder=True))

        model = keras.models.Sequential([
            keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                                input_shape=(227, 227, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])

        root_logdir = os.path.join(os.curdir, "logs\\fit\\")

        run_logdir = self.get_run_logdir(root_logdir)
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001),
                      metrics=['accuracy'])
        model.summary()

        # i have reduced the epochs
        model.fit(train_ds,
                  epochs=15,
                  validation_data=validation_ds,
                  validation_freq=1,
                  callbacks=[tensorboard_cb])

        model.evaluate(test_ds)

        #We already have a working model, so it is commented for the time being
        model.save("../models/xavier-alexnet.h5")
        print("xavier-alexnet saved successfully")



