import tensorflow as tf
import h5py as hf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from keras.utils import np_utils


"""
Function to get Weights from a model 
Parameters:
model: that is loaded from .h5 file
layertype: layer type can be convolutional or pooling
layername: temporary argument, more generic logic would be by index 
"""
def GetWeights(model, layertype, layerName):
    weight = np.array([])
    for layer in model.layers:
        # Check if its convolutional layer
        if isinstance(layer, layertype):
            if layer.name == layerName:
                weights = np.array(layer.get_weights())
    return weights

"""
Function to Set Weights of a model 
Parameters:
model: that is loaded from .h5 file
layertype: layer type can be convolutional or pooling
layername: temporary argument, more generic logic would be by index 
weight: The weights array
factor: temporary argument for increment, more generic logic would be to perform operation externally
"""
def SetWeights(model, layertype, layerName,weight, factor):
    for layer in model.layers:
        # Check if its convolutional layer
        if isinstance(layer, layertype):
            if layer.name == layerName:
                layer.set_weights(weight+factor)
    return
"""
Function to Get weights of a specific kernel
Parameters:
kernel: The index of desired kernel 
"""
def getKernelWeights(kernel):
    kernel_weights = []
    row = 0
    while row < 5:
        kernel_weights.append([])
        column = 0
        while column < 5:
            num = weight[bias][row][column][0][kernel]
            kernel_weights[row].append(num)
            column = column + 1
        row = row + 1

    return kernel_weights

def SetParticularWeight(model, layertype, layerName):
    weight = np.asarray(GetWeights(model, keras.layers.Conv2D, "conv2d"))
    weight[0][0][0][0][0] = 500
    for layer in model.layers:
        # Check if its convolutional layer
        if isinstance(layer, layertype):
            if layer.name == layerName:
                layer.set_weights(weight)
    return

def find_sameshape_layer(model):
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

if __name__ == '__main__':

    model = tf.keras.models.load_model("/home/saad/FYP/xavier/model.h5")
    model.summary()
    weight = np.asarray(GetWeights(model,  keras.layers.Conv2D, "conv2d"))
    # Load Data
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # Convert into Numpy Arrays
    train_X = np.asarray(train_X)
    train_y = np.asarray(train_y)

    #Check Accuracy
    model.evaluate(test_X, test_y)

    #Print Weights
    # for x in weight:
    #     for y in x:
    #         print(y)

    #Change Weights with a factor (currently performing Sum)
    #Paramater List
    # - Model (imported or created)
    # - Type of Layer to be manipulated
    # - Name of Layer to be manipulated
    # - Current Weight Array
    # - Factor to be added
    SetWeights(model, keras.layers.Conv2D, "conv2d", weight, 2)
    print("weights of layer conv2d are increased with a factor of 2")

    # Check Accuracy again
    model.evaluate(test_X, test_y)

    # SetParticularWeight(model, keras.layers.Conv2D, "conv2d")
    #model.evaluate(test_X, test_y)

    print(weight[0][0][0])

#Additional Information: (DO NOT REMOVE)
# We can also import custom_objects while loading model as dictionary
# For example if we have any loss function