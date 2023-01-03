import numpy as np


class WeightUtils:
    """
    Function to get Weights from a model
    Parameters:
    model: that is loaded from .h5 file
    layertype: layer type can be convolutional or pooling
    layerName: temporary argument, more generic logic would be by index of layer
    """

    def GetWeights(self, model, layerName):
        trainable_weights = np.array([])
        for layer in model.layers:
            if layer.name == layerName:
                trainable_weights = np.array(layer.get_weights(), dtype=object)
        return trainable_weights

    """
    Function to Set Weights of a model 
    Parameters:
    model: that is loaded from .h5 file
    layertype: layer type can be convolutional or pooling
    layername: temporary argument, more generic logic would be by index 
    weight: The manipulated weights array
    """

    def SetWeights(self, model, layerName, trainable_weights):
        for layer in model.layers:
            if layer.name == layerName:
                layer.set_weights(trainable_weights)
        return

    """
    Function to Get weights of a specific kernel
    Parameters:
    trainable_weights: all trainable weights of a model 
    kernel: The index of desired kernel 
    """

    def getKernelWeights(self, trainable_weights, kernel):
        kernel_weights = []
        row = 0
        while row < 5:
            kernel_weights.append([])
            column = 0
            while column < 5:
                num = trainable_weights[0][row][column][0][kernel]
                kernel_weights[row].append(num)
                column = column + 1
            row = row + 1

        return kernel_weights


class Model_layers:
    def find_sameshape_layer(self, model):
        """
        find the layers which can be deleted or duplicated
        :param model: model used
        :return: layer list
        """
        candidate_layer_list = []
        layer_num = len(model.layers)
        # has hidden layers?
        if layer_num > 2:
            for layer_index in range(layer_num):
                # pass input and output layers
                if layer_index == 0 or layer_index == (layer_num - 1):
                    continue
                # last layer's output shape = next layer's input shape
                if model.layers[layer_index].output_shape == model.layers[layer_index].input_shape:
                    candidate_layer_list.append(model.layers[layer_index].name)
        return candidate_layer_list

    def getLayerNames(self, model):
        layer_names = []
        for layer in model.layers:
            layer_names.append(layer.name)
        return layer_names

    def getKernelNumbers(self, model, layerName):
        for layer in model.layers:
            if layer.name == layerName:
                return layer.filters

    def getNeuronLayers(self, model):
        layer_names = []
        for layer in model.layers:
            if hasattr(layer, 'trainable_weights') and len(layer.trainable_weights) > 0:
                layer_names.append(layer.name)
        return layer_names


"""
Additional Information: (DO NOT REMOVE)
We can also import custom_objects while loading model as dictionary
for example: if we have any external loss function
"""
