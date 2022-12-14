from MutationOperators import NeuronLevel
import Predictions_analysis as pa
from Create_model import Lenet5_generator
from CheckWeights import Weights
from CheckWeights import Model_layers
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
    weights = Weights()
    operator = NeuronLevel()

    # Get all trainable weights from model
    trainable_weights = np.asarray(weights.GetWeights(model, keras.layers.Conv2D, "conv2d"))

    # Load Dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    # Convert into Numpy Arrays
    train_X = np.asarray(train_X)
    train_y = np.asarray(train_y)


    #model.evaluate(test_X, test_y)

    #Generate Classification report of Original Model
    prediction = model.predict(test_X)
    pa.generate_classification_report(prediction, test_y)

    """
    Good Example
    Inverting row 3 column 3 of kernel 0 have a great impact on letter 7
    """
    #Change Values here
    row = 3
    column = 3
    kernel = 0
    value = 0

    #Un comment Neuron Here
    #operator.changeNeuron(model,row,column, kernel, value)
    #operator.additive_inverse(model,row,column, kernel)
    #operator.mul_inverse(model,row,column, kernel)
    operator.invertNeuron(model,row,column, kernel)
    #operator.blockNeuron(model,row,column, kernel)
    #operator.changeNeuron(model,row,column, kernel)

    # Predict again with the model
    prediction = model.predict(test_X)
    pa.generate_classification_report(prediction, test_y)








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