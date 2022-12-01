import tensorflow as tf
import h5py as hf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from keras.utils import np_utils


def GetWeights(model, layertype, layerName):
    weight = np.array([])
    for layer in model.layers:
        # Check if its convolutional layer
        if isinstance(layer, layertype):
            if layer.name == layerName:
                weights = np.array(layer.get_weights())
    return weights

def SetWeights(model, layertype, layerName,weight, factor):
    for layer in model.layers:
        # Check if its convolutional layer
        if isinstance(layer, layertype):
            if layer.name == layerName:
                layer.set_weights(weight+factor)
    return


if __name__ == '__main__':

    model = tf.keras.models.load_model("/home/bilal/XAVIER/xavier/model.h5")
    model.summary()
    weight = np.asarray(GetWeights(model,  keras.layers.Conv2D, "conv2d"))

    for x in weight:
        for y in x:
            print(y)

    #SetWeights(model, keras.layers.Conv2D, "conv2d", weight, 2)

    #weight = GetWeights(model, keras.layers.Conv2D, "conv2d")

    #for x in weight:
    #    for y in x:
    #        print(y)

    print()
    print()
    print()
    print()

    print(weight[0][0][0])

# We can also import custom_objects while loading model as dictionary
# For example if we have any loss function