import unittest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow.keras.backend as K
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main")))

from mutation_operators import NeuronLevel
from mutation_operators import WeightLevel
from mutation_operators import BiasLevel
from operator_utils import WeightUtils

class TestMutationOperators(unittest.TestCase):
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

    def test_changeNeuron(self):
        op = NeuronLevel()
        w = WeightUtils()

        op.changeNeuron(self.model, 'conv2d', 0, 0, 0, 0.5)
        final_weights = w.GetWeights(self.model, 'conv2d')
        self.assertEqual(final_weights[0][0][0][0][0], 0.5)

        op.changeNeuron(self.model, 'conv2d', 0, 0, 0, 99)
        final_weights = w.GetWeights(self.model, 'conv2d')
        self.assertEqual(final_weights[0][0][0][0][0], 99)

        op.changeNeuron(self.model, 'conv2d_1', 0, 0, 0, 1)
        final_weights = w.GetWeights(self.model, 'conv2d_1')
        self.assertEqual(final_weights[0][0][0][0][0], 1)

    def test_changeEdge(self):
        op = WeightLevel()
        w = WeightUtils()

        op.changeEdge(self.model, 'dense_1', 0, 0, 0.5)
        final_weights = w.GetWeights(self.model, 'dense_1')
        self.assertEqual(final_weights[0][0][0], 0.5)

        op.changeEdge(self.model, 'dense_1', 0, 0, 99)
        final_weights = w.GetWeights(self.model, 'dense_1')
        self.assertEqual(final_weights[0][0][0], 99)

        op.changeEdge(self.model, 'dense_2', 0, 8, 1)
        final_weights = w.GetWeights(self.model, 'dense_2')
        self.assertEqual(final_weights[0][0][8], 1)

    def test_blockEdge(self):
        op = WeightLevel()
        w = WeightUtils()

        op.blockEdge(self.model, 'dense', 0, 0)
        final_weights = w.GetWeights(self.model, 'dense')
        self.assertEqual(final_weights[0][0][0], 0)

        op.blockEdge(self.model, 'dense_1', 0, 0)
        final_weights = w.GetWeights(self.model, 'dense_1')
        self.assertEqual(final_weights[0][0][0], 0)

        op.blockEdge(self.model, 'dense_2', 0, 8)
        final_weights = w.GetWeights(self.model, 'dense_2')
        self.assertEqual(final_weights[0][0][8], 0)

    def test_mul_inverse_edge(self):
            op = WeightLevel()
            w = WeightUtils()

            initial_weights = w.GetWeights(self.model, 'dense')
            op.mul_inverse(self.model, 'dense', 0, 0)
            final_weights = w.GetWeights(self.model, 'dense')
            np.testing.assert_allclose(float(1/initial_weights[0][0][0]), final_weights[0][0][0], atol=0.1)

            initial_weights = w.GetWeights(self.model, 'dense_1')
            op.mul_inverse(self.model, 'dense_1', 0, 0)
            final_weights = w.GetWeights(self.model, 'dense_1')
            np.testing.assert_allclose(float(1/initial_weights[0][0][0]), final_weights[0][0][0], atol=0.1)

            initial_weights = w.GetWeights(self.model, 'dense_2')
            op.mul_inverse(self.model, 'dense_2', 0, 8)
            final_weights = w.GetWeights(self.model, 'dense_2')
            np.testing.assert_allclose(float(1/initial_weights[0][0][8]), final_weights[0][0][8], atol=0.1)


    def test_additive_inverse_edge(self):
                op = WeightLevel()
                w = WeightUtils()

                initial_weights = w.GetWeights(self.model, 'dense')
                op.additive_inverse(self.model, 'dense', 0, 0)
                final_weights = w.GetWeights(self.model, 'dense')
                np.testing.assert_allclose(float(0 - initial_weights[0][0][0]), final_weights[0][0][0], atol=0.1)

                initial_weights = w.GetWeights(self.model, 'dense_1')
                op.additive_inverse(self.model, 'dense_1', 0, 0)
                final_weights = w.GetWeights(self.model, 'dense_1')
                np.testing.assert_allclose(float(0 - initial_weights[0][0][0]), final_weights[0][0][0], atol=0.1)

                initial_weights = w.GetWeights(self.model, 'dense_2')
                op.additive_inverse(self.model, 'dense_2', 0, 8)
                final_weights = w.GetWeights(self.model, 'dense_2')
                np.testing.assert_allclose(float(0 - initial_weights[0][0][8]), final_weights[0][0][8], atol =0.1)

    def test_blockNeuron(self):
        op = NeuronLevel()
        w = WeightUtils()
        op.blockNeuron(self.model, 'conv2d', 0, 0, 0)
        final_weights = w.GetWeights(self.model, 'conv2d')
        np.testing.assert_allclose(0, final_weights[0][0][0][0][0], atol = 0.1)

        op.blockNeuron(self.model, 'conv2d_1', 2, 3, 1)
        final_weights = w.GetWeights(self.model, 'conv2d_1')
        np.testing.assert_allclose(0, final_weights[0][2][3][0][1], atol=0.1)

    def test_mul_inverse(self):
        op = NeuronLevel()
        w = WeightUtils()

        initial_weights = w.GetWeights(self.model, 'conv2d')
        op.mul_inverse(self.model, 'conv2d', 2, 3, 1)
        final_weights = w.GetWeights(self.model, 'conv2d')
        np.testing.assert_allclose(float(1/initial_weights[0][2][3][0][1]), final_weights[0][2][3][0][1], rtol = 1)

        initial_weights = w.GetWeights(self.model, 'conv2d_1')
        op.mul_inverse(self.model, 'conv2d_1', 1, 1, 1)
        final_weights = w.GetWeights(self.model, 'conv2d_1')
        np.testing.assert_allclose(float(1 / initial_weights[0][1][1][0][1]), final_weights[0][1][1][0][1], rtol=1)

    def test_additive_inverse(self):
        op = NeuronLevel()
        w = WeightUtils()
        initial_weights = w.GetWeights(self.model, 'conv2d')
        op.additive_inverse(self.model, 'conv2d', 2, 3, 1)
        final_weights = w.GetWeights(self.model, 'conv2d')
        np.testing.assert_allclose(float(0 - initial_weights[0][2][3][0][1]), final_weights[0][2][3][0][1], rtol = 1)

        initial_weights = w.GetWeights(self.model, 'conv2d_1')
        op.additive_inverse(self.model, 'conv2d_1', 1, 1, 1)
        final_weights = w.GetWeights(self.model, 'conv2d_1')
        np.testing.assert_allclose(float(0 - initial_weights[0][1][1][0][1]), final_weights[0][1][1][0][1], rtol=1)

    def test_changeBiasValue(self):
        op = BiasLevel()
        w = WeightUtils()
        op.changeBiasValue(self.model, 'conv2d', 2, 99)
        final_weights = w.GetWeights(self.model, 'conv2d')
        np.testing.assert_allclose(final_weights[1][2], 99, rtol=1)

        op.changeBiasValue(self.model, 'conv2d_1', 3, 12)
        final_weights = w.GetWeights(self.model, 'conv2d')
        np.testing.assert_allclose(final_weights[1][3], 12, rtol=1)

        op.changeBiasValue(self.model, 'conv2d', 0, 10)
        final_weights = w.GetWeights(self.model, 'conv2d')
        np.testing.assert_allclose(final_weights[1][0], 10, rtol=1)

    def test_blockBiasValue(self):
        op = BiasLevel()
        w = WeightUtils()
        op.blockBiasValue(self.model, 'conv2d', 2)
        final_weights = w.GetWeights(self.model, 'conv2d')
        np.testing.assert_allclose(final_weights[1][2], 0, rtol=1)

        op.blockBiasValue(self.model, 'conv2d', 3)
        final_weights = w.GetWeights(self.model, 'conv2d')
        np.testing.assert_allclose(final_weights[1][3], 0, rtol=1)

        op.blockBiasValue(self.model, 'conv2d_1', 2)
        final_weights = w.GetWeights(self.model, 'conv2d_1')
        np.testing.assert_allclose(final_weights[1][2], 0, rtol=1)

    def test_mulInverseBiasValue(self):
        op = BiasLevel()
        w = WeightUtils()
        initial_weights = w.GetWeights(self.model, 'conv2d')
        op.mulInverseBiasValue(self.model, 'conv2d', 2)
        final_weights = w.GetWeights(self.model, 'conv2d')
        np.testing.assert_allclose(float(1 / initial_weights[1][2]), final_weights[1][2], atol=0.1)

        initial_weights = w.GetWeights(self.model, 'conv2d')
        op.mulInverseBiasValue(self.model, 'conv2d', 3)
        final_weights = w.GetWeights(self.model, 'conv2d')
        np.testing.assert_allclose(float(1 / initial_weights[1][3]), final_weights[1][2], atol=0.1)

        initial_weights = w.GetWeights(self.model, 'conv2d_1')
        op.mulInverseBiasValue(self.model, 'conv2d_1', 13)
        final_weights = w.GetWeights(self.model, 'conv2d_1')
        np.testing.assert_allclose(float(1 / initial_weights[1][13]), final_weights[1][13], atol=0.1)

    def test_additiveInverseBiasValue(self):
        op = BiasLevel()
        w = WeightUtils()

        initial_weights = w.GetWeights(self.model, 'conv2d')
        op.additiveInverseBiasValue(self.model, 'conv2d', 3)
        final_weights = w.GetWeights(self.model, 'conv2d')
        np.testing.assert_allclose(float(0 - initial_weights[1][3]), final_weights[1][3], atol=0.1)

        initial_weights = w.GetWeights(self.model, 'conv2d_1')
        op.additiveInverseBiasValue(self.model, 'conv2d_1', 3)
        final_weights = w.GetWeights(self.model, 'conv2d_1')
        np.testing.assert_allclose(float(0 - initial_weights[1][3]), final_weights[1][3], atol=0.1)

        initial_weights = w.GetWeights(self.model, 'conv2d')
        op.additiveInverseBiasValue(self.model, 'conv2d', 2)
        final_weights = w.GetWeights(self.model, 'conv2d')
        np.testing.assert_allclose(float(0 - initial_weights[1][2]), final_weights[1][2], atol=0.1)


if __name__ == '__main__':
    unittest.main()
