from mutation_operators import NeuronLevel
from mutation_operators import EdgeLevel
from visualizer import VisualKeras
import predictions_analysis as pa
from models_generator import Lenet5Generator
from operator_utils import WeightUtils
from operator_utils import Model_layers
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
import numpy as np
from tensorflow.keras.optimizers import Adam




if __name__ == '__main__':
    #place a hashtag before next line to enter/leave playground
    #""""
    #Playground
    #Lenet5_generator = Lenet5_generator()
    #Lenet5_generator.generate_model()
    model = tf.keras.models.load_model("../models/model.h5")
    model.summary()
    layers = Model_layers()
    print(layers.getNeuronLayers(model))
    print(layers.getEdgeLayers(model))
    """

    """#Functionality
    """
    print('Xavier started.')
    #Load Model
    model = tf.keras.models.load_model("../models/model.h5")
    model.summary()

    #Create Objects
    layers = Model_layers()
    weights = WeightUtils()
    convOperator = NeuronLevel()
    denseOperator = EdgeLevel()
    VK = VisualKeras()

    # Visualization Demo
    VK.visualize_model_using_vk(model)
    layer_names = layers.getLayerNames(model)
    count = 0
    print("select an index: ")
    for x in layer_names:
        print(count , "- ", x)
        count = count+1
    count = int(input("choice: "))
    # Set Layer Name to perform Operation
    layerName = layer_names[count]

    # Get all trainable weights from model
    trainable_weights = np.asarray(weights.GetWeights(model, layerName))

    #The format of Trainable_weights are:
    # Convolution Layer: trainable_weights[bias][row][column][0][kernel]
    # Dense Layer: trainable_weights[bias][prevNeuron][currNeuron]



    # Load Dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    # Convert into Numpy Arrays
    train_X = np.asarray(train_X)
    train_y = np.asarray(train_y)


    #model.evaluate(test_X, test_y)


    # Good Example
    # Inverting row 3 column 3 of kernel 0 have a great impact on letter 7
    
    #Change Values here
    row = 3
    column = 3
    kernel = 0
    value = 0
    if layerName[0] == 'c':
        #Uncomment Neuron Here
        convOperator.changeNeuron(model,layerName, row,column, kernel, value)
        #convOperator.additive_inverse(model,layerName, row,column, kernel)
        #convOperator.mul_inverse(model,layerName, row,column, kernel)
        #convOperator.invertNeuron(model,layerName, row,column, kernel)
        #convOperator.blockNeuron(model,layerName, row,column, kernel)
        #convOperator.changeNeuron(model,layerName, row,column, kernel)
    elif layerName[0] == 'd':
        # Uncomment Neuron Here
        #denseOperator.changeNeuron(model, "dense",0,0, -2)
        #denseOperator.blockNeuron(model, "dense",0,0)
        #denseOperator.invertNeuron(model, "dense",0,0)
        denseOperator.additive_inverse(model, "dense",0,0)
        #denseOperator.mul_inverse(model, "dense",0,0)

    # Generate Classification report of Original Model
    prediction = model.predict(test_X)

    pa.printConfusionMatrix(prediction, test_y)

    pa.printClassificationReport(prediction, test_y, 1.0)

    print("Accuracy: ", pa.getModelAccuracy(prediction, test_y))


    # Check Accuracy again
    #model.evaluate(test_X, test_y)




    print('Xavier ended.')

    #Do not remove the terminator
    #"""