from tensorflow import keras
from src.main.operator_utils import WeightUtils

class NeuronLevel:
    # Composition
    weights = WeightUtils()

    neuronLevelMutationOperatorslist = ["Change Neuron", "Block Neuron", "Multiplicative Inverse", "Additive Inverse", "Invert Neuron"]
    neuronLevelMutationOperatorsDescription = ["It changes the value of specified neuron to the given value",
                                               "It blocks the effect of Neuron by replacing it with 0",
                                               "It replace the value of a neuron with Multiplicative inverse of it's current value",
                                               "It replaces the value of a neuron with Additive Inverse of it's current value",
                                               "It performs unary negation on the current value of Neuron"]

    def changeNeuron(self, model: object, layerName: str, row: int, column: int, kernel: int, value: float):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[0][row][column][0][kernel] = value
        self.weights.SetWeights(model,layerName, trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " successfully changed")

    def blockNeuron(self, model, layerName, row, column, kernel):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[0][row][column][0][kernel]  = 0                       #Block Effect
        self.weights.SetWeights(model,layerName, trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " is blocked")

    def mul_inverse(self, model,layerName, row, column, kernel):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[0][row][column][0][kernel]  = float(1/trainable_weights[0][row][column][0][kernel])
        self.weights.SetWeights(model,layerName, trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " is changed with it's multiplicative inverse")

    def additive_inverse(self, model,layerName, row, column, kernel):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[0][row][column][0][kernel]  = float(0 - trainable_weights[0][row][column][0][kernel])
        self.weights.SetWeights(model,layerName, trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " is changed with it's additive inverse")

    def invertNeuron(self, model,layerName, row, column, kernel):
        trainable_weights = self.weights.GetWeights(model, layerName)
        if trainable_weights[0][row][column][0][kernel] > 0:
            trainable_weights[0][row][column][0][kernel]  = -abs(trainable_weights[0][row][column][0][kernel])
        else:
            trainable_weights[0][row][column][0][kernel] = abs(trainable_weights[0][row][column][0][kernel])
        self.weights.SetWeights(model, layerName, trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " is inverted")