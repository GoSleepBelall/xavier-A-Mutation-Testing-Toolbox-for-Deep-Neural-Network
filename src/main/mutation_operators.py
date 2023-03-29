from operator_utils import WeightUtils

class BiasLevel:
    # Composition
    weights = WeightUtils()
    biasLevelMutationOperatorhash = {"change-bias-value": "Change Bias Value",
                                     "block-bias-value": "Block Bias Value",
                                     "mul-inverse-bias-value": "Multiplicative Inverse Bias Value",
                                     "additive-inverse-bias-value": "Additive Inverse Bias Value"}
    biasLevelMutationOperatorDescription = ["Change Bias Value (CBV) Changes bias value such that the effect on"
                                            " the subsequent neuron is changed"]

    def changeBiasValue(self, model, layerName, index, value):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[1][index] = value
        self.weights.SetWeights(model, layerName, trainable_weights)
        print("value of bias from kernel number ", index, " successfully changed")

    def blockBiasValue(self, model, layerName, index):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[1][index] = 0             # Block Effect
        self.weights.SetWeights(model, layerName, trainable_weights)
        print("value of bias from kernel number ", index, " successfully blocked")

    def mulInverseBiasValue(self, model, layerName, index):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[1][index] = float(1/trainable_weights[1][index])              # Multiplicative Inverse
        self.weights.SetWeights(model, layerName, trainable_weights)
        print("value of bias from kernel number ", index, " successfully Inversed")

    def additiveInverseBiasValue(self, model, layerName, index):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[1][index] = 0 - (trainable_weights[1][index])                  # Additive Inverse
        self.weights.SetWeights(model, layerName, trainable_weights)
        print("value of bias from kernel number ", index, " successfully Inverted")

class NeuronLevel:
    # Composition
    weights = WeightUtils()

    neuronLevelMutationOperatorslist = ["Change Neuron", "Block Neuron", "Multiplicative Inverse", "Additive Inverse", "Invert Neuron"]
    neuronLevelMutationOperatorshash = {"change-neuron": "Change Neuron", "block-neuron": "Block Neuron", "mul-inverse-neuron": "Multiplicative Inverse", "additive-inverse-neuron": "Additive Inverse"}
    neuronLevelMutationOperatorsDescription = ["It changes the value of specified neuron to the given value",
                                               "It blocks the effect of Neuron by replacing it with 0",
                                               "It replace the value of a neuron with Multiplicative inverse of it's current value",
                                               "It replaces the value of a neuron with Additive Inverse of it's current value",
                                               "It performs unary negation on the current value of Neuron"]

    def changeNeuron(self, model, layerName, row, column, kernel, value):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[0][row][column][0][kernel] = value
        self.weights.SetWeights(model,layerName, trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " successfully changed")

    def blockNeuron(self, model, layerName, row, column, kernel):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[0][row][column][0][kernel] = 0                       # Block Effect
        self.weights.SetWeights(model,layerName, trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " is blocked")

    def mul_inverse(self, model,layerName, row, column, kernel):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[0][row][column][0][kernel]  = float(1/trainable_weights[0][row][column][0][kernel])
        self.weights.SetWeights(model,layerName, trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " is changed with it's multiplicative inverse")

    def additive_inverse(self, model,layerName, row, column, kernel):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[0][row][column][0][kernel] = float(0 - trainable_weights[0][row][column][0][kernel])
        self.weights.SetWeights(model,layerName, trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " is changed with it's additive inverse")

    def invertNeuron(self, model,layerName, row, column, kernel):
        trainable_weights = self.weights.GetWeights(model, layerName)
        if trainable_weights[0][row][column][0][kernel] > 0:
            trainable_weights[0][row][column][0][kernel] = -abs(trainable_weights[0][row][column][0][kernel])
        else:
            trainable_weights[0][row][column][0][kernel] = abs(trainable_weights[0][row][column][0][kernel])
        self.weights.SetWeights(model, layerName, trainable_weights)
        print("value of neuron from kernel number ", kernel, " at row ", row, " and column ", column, " is inverted")


class WeightLevel:
    # Composition
    weights = WeightUtils()

    weightLevelMutationOperatorsList = ["Change Edge", "Block Edge", "Multiplicative Inverse Edge", "Additive Inverse Edge", "Invert Edge"]
    weightLevelMutationOperatorsDescription = ["It changes the value of specified Edge connecting two neurons",
                                               "It blocks the effect of Edge by replacing it with 0",
                                               "It replace the value of a Edge with Multiplicative inverse of it's current value",
                                               "It replaces the value of a Edge with Additive Inverse of it's current value",
                                               "It performs unary negation on the current value of Edge"]

    def changeEdge(self, model, layerName, prevNeuron, currNeuron, value):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[0][prevNeuron][currNeuron] = value
        self.weights.SetWeights(model,layerName, trainable_weights)
        print("value of edge from Neuron # ", prevNeuron, " that joins to Neuron #", currNeuron, " in layer", layerName, " successfully changed")

    def blockEdge(self, model, layerName, prevNeuron, currNeuron):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[0][prevNeuron][currNeuron] = 0                       # Block Effect
        self.weights.SetWeights(model,layerName, trainable_weights)
        print("value of edge from Neuron # ", prevNeuron, " that joins to Neuron #", currNeuron, " in layer", layerName, " is blocked")

    def mul_inverse(self, model,layerName, prevNeuron, currNeuron):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[0][prevNeuron][currNeuron] = float(1/trainable_weights[0][prevNeuron][currNeuron])
        self.weights.SetWeights(model,layerName, trainable_weights)
        print("value of edge from Neuron # ", prevNeuron, " that joins to Neuron #", currNeuron, " in layer", layerName, "  is changed with it's multiplicative inverse")

    def additive_inverse(self, model,layerName,prevNeuron, currNeuron):
        trainable_weights = self.weights.GetWeights(model, layerName)
        trainable_weights[0][prevNeuron][currNeuron] = float(0 - trainable_weights[0][prevNeuron][currNeuron])
        self.weights.SetWeights(model,layerName, trainable_weights)
        print("value of edge from Neuron # ", prevNeuron, " that joins to Neuron #", currNeuron, " in layer", layerName, "  is changed with it's additive inverse")

    def invertEdge(self, model,layerName, prevNeuron, currNeuron):
        trainable_weights = self.weights.GetWeights(model, layerName)
        if trainable_weights[0][prevNeuron][currNeuron] > 0:
            trainable_weights[0][prevNeuron][currNeuron] = -abs(trainable_weights[0][prevNeuron][currNeuron])
        else:
            trainable_weights[0][prevNeuron][currNeuron] = abs(trainable_weights[0][prevNeuron][currNeuron])
        self.weights.SetWeights(model, layerName, trainable_weights)
        print("value of edge from Neuron # ", prevNeuron, " that joins to Neuron #", currNeuron, " in layer", layerName, "  is inverted")