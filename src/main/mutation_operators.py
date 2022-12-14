from tensorflow import keras
from operator_utils import WeightUtils

class NeuronLevel:
    # Composition
    weights = WeightUtils()

    def changeNeuron(self, model, row, column, kernel, value):
        trainable_weights = self.weights.GetWeights(model, keras.layers.Conv2D, "conv2d")
        trainable_weights[0][row][column][0][kernel]  = value
        self.weights.SetWeights(model,keras.layers.Conv2D,"conv2d", trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " successfully changed")

    def blockNeuron(self, model, row, column, kernel):
        trainable_weights = self.weights.GetWeights(model, keras.layers.Conv2D, "conv2d")
        trainable_weights[0][row][column][0][kernel]  = 0
        self.weights.SetWeights(model,keras.layers.Conv2D,"conv2d", trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " is blocked")

    def mul_inverse(self, model, row, column, kernel):
        trainable_weights = self.weights.GetWeights(model, keras.layers.Conv2D, "conv2d")
        trainable_weights[0][row][column][0][kernel]  = float(1/trainable_weights[0][row][column][0][kernel])
        self.weights.SetWeights(model,keras.layers.Conv2D,"conv2d", trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " is changed with it's multiplicative inverse")

    def additive_inverse(self, model, row, column, kernel):
        trainable_weights = self.weights.GetWeights(model, keras.layers.Conv2D, "conv2d")
        trainable_weights[0][row][column][0][kernel]  = float(0 - trainable_weights[0][row][column][0][kernel])
        self.weights.SetWeights(model,keras.layers.Conv2D,"conv2d", trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " is changed with it's additive inverse")

    def invertNeuron(self, model, row, column, kernel):
        trainable_weights = self.weights.GetWeights(model, keras.layers.Conv2D, "conv2d")
        if trainable_weights[0][row][column][0][kernel] > 0:
            trainable_weights[0][row][column][0][kernel]  = -abs(trainable_weights[0][row][column][0][kernel])
        else:
            trainable_weights[0][row][column][0][kernel] = abs(trainable_weights[0][row][column][0][kernel])
        self.weights.SetWeights(model,keras.layers.Conv2D,"conv2d", trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " is inverted")