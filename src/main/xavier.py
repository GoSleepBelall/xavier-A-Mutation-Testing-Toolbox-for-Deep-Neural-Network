from mutation_operators import NeuronLevel
from visualizer import VisualKeras
import predictions_analysis as pa
from models_generator import Lenet5Generator
from operator_utils import WeightUtils
from operator_utils import Model_layers
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
import numpy as np



if __name__ == '__main__':
    """ Playground """
    #Lenet5_generator = Lenet5_generator()
    #Lenet5_generator.generate_model()

    """Functionality"""
    print('Xavier started.')
    #Load Model
    model = tf.keras.models.load_model("../models/model.h5")
    model.summary()

    #Create Objects
    layers = Model_layers()
    weights = WeightUtils()
    operator = NeuronLevel()
    VK = VisualKeras()
    VK.visualize_model_using_vk(model)
    print(layers.getKernelNumbers(model, "conv2d"))
    # Get all trainable weights from model
    trainable_weights = np.asarray(weights.GetWeights(model, "conv2d"))
    np.shape(trainable_weights)

    # Load Dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    # Convert into Numpy Arrays
    train_X = np.asarray(train_X)
    train_y = np.asarray(train_y)


    #model.evaluate(test_X, test_y)

    #Generate Classification report of Original Model
    prediction = model.predict(test_X)
    print(pa.generate_classification_report(prediction, test_y))
    pa.printConfusionMatrix(prediction, test_y)
    pa.printClassificationReport(prediction, test_y)

    """
    Good Example
    Inverting row 3 column 3 of kernel 0 have a great impact on letter 7
    """
    #Change Values here
    row = 3
    column = 3
    kernel = 0
    value = 0
    layerName = "conv2d"

    #Un comment Neuron Here
    #operator.changeNeuron(model,layerName, row,column, kernel, value)
    #operator.additive_inverse(model,layerName, row,column, kernel)
    operator.mul_inverse(model,layerName, row,column, kernel)
    #operator.invertNeuron(model,layerName, row,column, kernel)
    #operator.blockNeuron(model,layerName, row,column, kernel)
    #operator.changeNeuron(model,layerName, row,column, kernel)

    # Predict again with the model
    prediction = model.predict(test_X)
    pa.printClassificationReport(prediction, test_y)
    #pa.generate_classification_report(prediction, test_y)








    # Change Weights with a factor (currently performing Sum)
    # Paramater List
    # - Model (imported or created)
    # - Type of Layer to be manipulated
    # - Name of Layer to be manipulated
    # - Current Weight Array
    # - Factor to be added
    #weights.SetWeights(model, keras.layers.Conv2D, "conv2d", trainable_weights)

    # Check Accuracy again
    #model.evaluate(test_X, test_y)




    print('Xavier ended.')

    #Do not remove the terminator
    #"""