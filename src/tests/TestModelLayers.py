import unittest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import tensorflow.keras.backend as K

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main")))

from operator_utils import Model_layers

class TestModelLayers(unittest.TestCase):

    def setUp(self):
        K.clear_session()
        # Create a mock model in the format of LeNet-5
        self.model = Sequential([
            Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=120, activation='relu'),
            Dense(units=84, activation='relu'),
            Dense(units=10, activation='softmax')
        ])
        # Compile the model with some random weights
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.train_on_batch(np.zeros((1, 32, 32, 1)), np.zeros((1, 10)))

    def test_get_layer_names(self):
        expected_layer_names = ['conv2d', 'max_pooling2d', 'conv2d_1', 'max_pooling2d_1', 'flatten', 'dense', 'dense_1', 'dense_2']
        self.assertEqual(Model_layers().getLayerNames(self.model), expected_layer_names)

    def test_get_kernel_numbers(self):
        expected_kernel_numbers = 6
        self.assertEqual(Model_layers().getKernelNumbers(self.model, 'conv2d'), expected_kernel_numbers)

    def test_get_neuron_layers(self):
        expected_neuron_layers = ['conv2d', 'conv2d_1']
        self.assertEqual(Model_layers().getNeuronLayers(self.model), expected_neuron_layers)

    def test_get_edge_layers(self):
        expected_edge_layers = ['dense', 'dense_1', 'dense_2']
        self.assertEqual(Model_layers().getEdgeLayers(self.model), expected_edge_layers)

    def test_get_bias_layers(self):
        expected_bias_layers = ['conv2d', 'conv2d_1', 'dense', 'dense_1', 'dense_2']
        self.assertEqual(Model_layers().getBiasLayers(self.model), expected_bias_layers)

if __name__ == '__main__':
    unittest.main()
