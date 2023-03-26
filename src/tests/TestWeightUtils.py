import unittest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow.keras.backend as K

from src.main.operator_utils import WeightUtils


class TestWeightUtils(unittest.TestCase):
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

    def test_GetWeights(self):
        # Test GetWeights function
        weight_utils = WeightUtils()
        layer_name = 'conv2d'
        weights = weight_utils.GetWeights(self.model, layer_name)
        self.assertEqual(len(weights), 2)
        self.assertEqual(weights[0].shape, (5, 5, 1, 6))
        self.assertEqual(weights[1].shape, (6,))

    def test_SetWeights(self):
        # Test SetWeights function
        weight_utils = WeightUtils()
        layer_name = 'conv2d'
        weights = weight_utils.GetWeights(self.model, layer_name)
        new_weights = [np.zeros_like(w) for w in weights]
        weight_utils.SetWeights(self.model, layer_name, new_weights)
        updated_weights = weight_utils.GetWeights(self.model, layer_name)
        self.assertTrue(np.allclose(updated_weights[0], new_weights[0]))
        self.assertTrue(np.allclose(updated_weights[1], new_weights[1]))

    def test_getKernelWeights(self):
        # Test getKernelWeights function
        weight_utils = WeightUtils()
        layer_name = 'conv2d'
        # Check that the layer has trainable weights before getting the kernel weights
        trainable_weights = self.model.get_layer(layer_name).trainable_weights
        if not trainable_weights:
            self.skipTest('No trainable weights in layer {}'.format(layer_name))
        weights = weight_utils.GetWeights(self.model, layer_name)
        kernel_weights = weight_utils.getKernelWeights(weights, 0)
        self.assertEqual(len(kernel_weights), 5)
        self.assertEqual(np.shape(kernel_weights), (5, 5))
        self.assertEqual(len(kernel_weights[0]), 5)

    def test_getBiasWeights(self):
        # Test getBiasWeights function
        weight_utils = WeightUtils()
        layer_name = 'conv2d'
        # Check that the layer has trainable weights before getting the bias weights
        trainable_weights = self.model.get_layer(layer_name).trainable_weights
        if not trainable_weights:
            self.skipTest('No trainable weights in layer {}'.format(layer_name))
        bias_weights = weight_utils.getBiasWeights(self.model, layer_name)
        self.assertEqual(bias_weights.shape, (6,))


if __name__ == '__main__':
    unittest.main()
