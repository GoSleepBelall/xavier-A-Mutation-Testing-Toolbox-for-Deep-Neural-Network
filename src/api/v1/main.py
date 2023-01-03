from typing import Union
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Response
import sys
import os
import json_tricks as jt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main")))

import json
import tensorflow as tf

from mutation_operators import NeuronLevel
from operator_utils import WeightUtils
from operator_utils import Model_layers

import sys



app = FastAPI()
layers = Model_layers()
weights = WeightUtils()


# Read the models globally

lenet5 = tf.keras.models.load_model("../../models/xavier-lenet5.h5")
alexnet = tf.keras.models.load_model("../../models/xavier-lenet5.h5")       #TODO: this is still reading lenet-5 because i dont have alexnet right now


# GET request to retrieve all the trainable weights of a particular layer in a specific model
@app.get("/weights/{model}/{layerName}")
def get_weights(model: str, layerName: str):
    model_obj = None
    if model == "1":
        model_obj = lenet5
    elif model == "2":
        model_obj = alexnet
    trainable_weights = weights.GetWeights(model_obj, layerName)
    result = trainable_weights.tolist()
    return jt.dumps(result)


# GET request to retrieve all the layers in a specific model
@app.get("/layers/{model}")
def get_layers(model: str):
    if model == "1":
        model = lenet5
    elif model == "2":
        model = alexnet
    layer_names = layers.getLayerNames(model)
    return json.dumps(layer_names)


# GET request to retrieve all the weights of a specific kernel in a specific layer of a specific model
@app.get("/weights/{model}/{layerName}/{kernel}")
def getKernelWeights(model: str, layerName: str, kernel: int):
    if model == "1":
        model = lenet5
    elif model == "2":
        model = alexnet
    trainable_weights = weights.GetWeights(model, layerName)
    kernel_weights = weights.getKernelWeights(trainable_weights, kernel)
    return jt.dumps(kernel_weights)

# GET request to retrieve number of kernels present in a layer
@app.get("/kernel/{model}/{layerName}")
def getKernelNum(model: str, layerName: str):
    return json.dumps(layers.getKernelNumbers(model, layerName))

# GET request to retrieve List of Mutation Operators present
@app.get("/operators-list/{operatortype}")
def getMutationOperatorsList(operatortype: int):
    if operatortype == 1:
        return json.dumps(NeuronLevel.neuronLevelMutationOperatorslist)
    #TODO: add other type of mutation operators when done


# GET request to retrieve Description of Mutation Operators present
@app.get("/operators-description/{type}")
def getMutationOperatorsDescription(operatortype: int):
    if operatortype == 1:
        return jt.dumps(NeuronLevel.neuronLevelMutationOperatorsDescription)
    #TODO: add other type of mutation operators when done


# PUT request to change neuron value using Change Neuron Mutation Operator
@app.put("/change-neuron/{model}/{layerName}/{row}/{column}/{kernel}/{value}")
def change_neuron(model: str, layerName: str, row: int, column: int, kernel: int, value: float, response: Response):
    model_obj = None
    if model == "1":
        model_obj = lenet5
    elif model == "2":
        model_obj = alexnet
    try:
        NeuronLevel.changeNeuron(model_obj, layerName, row, column, kernel, value)
        response.status_code = 200
        return {"message": "Neuron value successfully changed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# PUT request to block neuron value using Block Neuron Mutation Operator
@app.put("/block-neuron/{model}/{layerName}/{row}/{column}/{kernel}")
def block_neuron(model: str, layerName: str, row: int, column: int, kernel: int, response: Response):
    model_obj = None
    if model == "1":
        model_obj = lenet5
    elif model == "2":
        model_obj = alexnet
    try:
        NeuronLevel.blockNeuron(model_obj, layerName, row, column, kernel)
        response.status_code = 200
        return {"message": "Neuron successfully blocked"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# PUT request to change neuron value with its multiplicative inverse
@app.put("/mul-inverse/{model}/{layerName}/{row}/{column}/{kernel}")
def mul_inverse_neuron(model: str, layerName: str, row: int, column: int, kernel: int, response: Response):
    model_obj = None
    if model == "1":
        model_obj = lenet5
    elif model == "2":
        model_obj = alexnet
    try:
        NeuronLevel.mul_inverse(model_obj, layerName, row, column, kernel)
        response.status_code = 200
        return {"message": "Neuron value successfully changed to multiplicative inverse"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# PUT request to replace neuron with its Additive Inverse
@app.put("/additive-inverse/{model}/{layerName}/{row}/{column}/{kernel}")
def additive_inverse_neuron(model: str, layerName: str, row: int, column: int, kernel: int, response: Response):
    try:
        NeuronLevel.additive_inverse(model, layerName, row, column, kernel)
        response.status_code = 200
        return {"message": "Neuron value successfully changed to additive inverse"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# PUT request to invert the value of neuron using Invert Neuron Mutation Operator
@app.put("/invert-neuron/{model}/{layerName}/{row}/{column}/{kernel}")
def invert_neuron(model: str, layerName: str, row: int, column: int, kernel: int, response: Response):
    try:
        NeuronLevel.invertNeuron(model, layerName, row, column, kernel)
        response.status_code = 200
        return {"message": "Neuron value successfully inverted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Index based PUT request for Mutation Operators
@app.put("/mutation-operator/{index}/{model}/{layerName}/{row}/{column}/{kernel}/{value}")
def mutation_operator(index: int, model: str, layerName: str, row: int, column: int, kernel: int, response: Response, value: Union[float, None] = None):
    model_obj = None
    if model == "1":
        model_obj = lenet5
    elif model == "2":
        model_obj = alexnet

    if index == 1:
        result = change_neuron(model_obj, layerName, row, column, kernel, value)
    elif index == 2:
        result = block_neuron(model_obj, layerName, row, column, kernel)
    elif index == 3:
        result = mul_inverse_neuron(model_obj, layerName, row, column, kernel)
    elif index == 4:
        result = additive_inverse_neuron(model_obj, layerName, row, column, kernel)
    elif index == 5:
        result = invert_neuron(model_obj, layerName, row, column, kernel)
    else:
        # If the index is not recognized, return an error message
        raise HTTPException(status_code=400, detail="Invalid Mutation Operator index")

    response.status_code = 200
    return {"message": "Mutation operator successfully applied", "result": result}
