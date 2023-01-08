from typing import Union
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Response
import sys
import os
import json_tricks as jt
import json
from keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import visualkeras as vk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main")))
from mutation_operators import NeuronLevel
from mutation_operators import EdgeLevel
from operator_utils import WeightUtils
from operator_utils import Model_layers
import predictions_analysis as pa

app = FastAPI(title="XAVIER-API", description="A Mutation Testing Toolbox.", version="1.0")
layers = Model_layers()
weights = WeightUtils()
NeuronOperator = NeuronLevel()
EdgeOperator = EdgeLevel()

# Read the models globally

lenet5 = models.load_model("../../models/xavier-lenet5.h5")
alexnet = models.load_model("../../models/xavier-lenet5.h5")
# TODO: this is still reading lenet-5 because i dont have alexnet right now

# Dataset for Lenet
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# A global dictionary mapping model names to model objects
model_dict = {
    'lenet5': lenet5,
    'alexnet': alexnet
}

# GET request to retrieve confusion matirx for a specific model
@app.get("/confusion-matrix/{modelId}")
def getConfusionMatrix(modelId: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    prediction = model_var.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    return json.dumps({str(k): v for k, v in matrix.items()})

# GET request to retrieve accuracy for a specific model clas wise
@app.get("/class-accuracy/{modelId}")
def getAccuracy(modelId: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    prediction = model_var.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    accuracy = pa.getAccuracy(matrix)
    return json.dumps({str(k): v for k, v in accuracy.items()})

# GET request to retrieve accuracy of a specific model
@app.get("/model-accuracy/{modelId}")
def getModelAccuracy(modelId: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    predictions = model_var.predict(test_X)
    matrix = pa.getConfusionMatrix(predictions, test_y)
    accuracy = pa.getModelAccuracy(matrix)
    return json.dumps({"accuracy": accuracy})


# GET request to retrieve specificity for a specific model
@app.get("/specificity/{modelId}")
def getSpecificity(modelId: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    prediction = model_var.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    specificity = pa.getSpecificity(matrix)
    return json.dumps({str(k): v for k, v in specificity.items()})

# GET request to retrieve f1-score for a specific model
@app.get("/f1-score/{modelId}")
def getf1Score(modelId: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    prediction = model_var.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    f1_score = pa.getF1Score(matrix)
    return json.dumps({str(k): v for k, v in f1_score.items()})

# GET request to retrieve recall for a specific model
@app.get("/recall/{modelId}")
def getRecall(modelId: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    prediction = model_var.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    recall = pa.getRecall(matrix)
    return json.dumps({str(k): v for k, v in recall.items()})


# GET request to retrieve precision for a specific model
@app.get("/precision/{modelId}")
def getPrecision(modelId: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    prediction = model_var.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    precision = pa.getPrecision(matrix)
    return json.dumps({str(k): v for k, v in precision.items()})

# GET request to retrieve sensitivity for a specific model
@app.get("/sensitivity/{modelId}")
def get_sensitivity(modelId: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    predictions = model_var.predict(test_X)
    matrix = pa.getConfusionMatrix(predictions, test_y)
    sensitivity = pa.getSensitivity(matrix)
    return json.dumps({str(k): v for k, v in sensitivity.items()})

# GET request to retrieve complete report of a specific model with respect to all classes
@app.get("/report/{modelId}/{beta}")
def getReport(modelId: str, beta: float = 1):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)

    # Generate predictions and labels for the model
    predictions = model_var.predict(test_X)
    matrix = pa.getConfusionMatrix(predictions, test_y)
    class_metrics = pa.getAllMetrics(matrix, beta)
    return json.dumps(class_metrics)

# GET request to retrieve accuracy of a specific model
@app.get("/auc/{modelId}")
def getAuc(modelId: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    predictions = model_var.predict(test_X)
    matrix = pa.getConfusionMatrix(predictions, test_y)
    auc = pa.getAuc(matrix)
    return json.dumps({str(k): v for k, v in auc.items()})

@app.get("/f-beta-score/{modelId}/{beta}")
def getFBetaScore(modelId: str, beta: float = 1.0):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    predictions = model_var.predict(test_X)
    matrix = pa.getConfusionMatrix(predictions, test_y)
    f_beta_score = pa.getFBetaScore(matrix, beta)
    return json.dumps({str(k): v for k, v in f_beta_score.items()})

# GET request to retrieve all the layers in a specific model
@app.get("/layers/{modelId}")
def get_layers(modelId: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())
    layer_names = layers.getLayerNames(model_obj)
    return json.dumps(layer_names)


# GET request to retrieve all the layers on which Neuron level Mutation Operators are applicable
@app.get("/neuron_layers/{modelId}")
def get_neuron_layers(modelId: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)

    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())

    layer_names = layers.getNeuronLayers(model_obj)
    return json.dumps(layer_names)

# GET request to retrieve all the layers on which Edge level Mutation Operators are applicable
@app.get("/edge-layers/{modelId}")
def get_edge_layers(modelId: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)

    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())

    layer_names = layers.getEdgeLayers(model_obj)
    return json.dumps(layer_names)


# GET request to retrieve all the trainable weights of a all layers in a specific model
@app.get("/all-weights/{modelId}/{layerName}")
def get_weights(modelId: str, layerName: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)

    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())
    trainable_weights = weights.GetWeights(model_obj, layerName)
    result = trainable_weights.tolist()
    return jt.dumps(result)


# GET request to retrieve all the weights of a specific kernel in a specific layer of a specific model
@app.get("/kernel-weights/{modelId}/{layerName}/{kernel}")
def getKernelWeights(modelId: str, layerName: str, kernel: int):

    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)

    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model_var)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())
    # Get Trainable Weights
    trainable_weights = weights.GetWeights(model_obj, layerName)

    # Get Kernel wise weights from all weights
    kernel_weights = weights.getKernelWeights(trainable_weights, kernel)
    return jt.dumps(kernel_weights)


# GET request to retrieve all the weights of a specific kernel in a specific layer of a specific model
@app.get("/all-kernel-weights/{modelId}/{layerName}")
def getAllKernelWeights(modelId: str, layerName: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)

    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model_var)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())
    # Get Trainable Weights
    trainable_weights = weights.GetWeights(model_obj, layerName)

    # Get Kernel wise weights from all weights
    kernel_weights = []
    total = layers.getKernelNumbers(model_obj, layerName)
    for kernel in range(total):
        kernel_weights.append(weights.getKernelWeights(trainable_weights, kernel))
    return jt.dumps(kernel_weights)

# GET request to retrieve number of kernels present in a layer
@app.get("/kernel/{modelId}/{layerName}")
def getKernelNum(modelId: str, layerName: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)

    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())

    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model_var)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    return json.dumps(layers.getKernelNumbers(model_obj, layerName))


# GET request to retrieve List of Mutation Operators present
@app.get("/operators-list/{operatortype}")
def getMutationOperatorsList(operatortype: int):
    if operatortype == 1:
        return json.dumps(NeuronLevel.neuronLevelMutationOperatorshash)
    elif operatortype == 2:
        return json.dumps(EdgeLevel.edgeLevelMutationOperatorslist)
    else:
        raise HTTPException(status_code=500, detail=("Invalid Operator Type"))



# GET request to retrieve Description of Mutation Operators present
@app.get("/operators-description/{type}")
def getMutationOperatorsDescription(operatortype: int):
    if operatortype == 1:
        return jt.dumps(NeuronLevel.neuronLevelMutationOperatorsDescription)
    elif operatortype == 2:
        return jt.dumps(EdgeLevel.edgeLevelMutationOperatorsDescription)
    else:
        raise HTTPException(status_code=500, detail=("Invalid Operator Type"))


@app.get("/model-image/{modelId}")
def get_model_image(modelId: str):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)

    image = vk.layered_view(model_var, legend=True)
    image.save("../../supporting files/model.png")
    with open("../../supporting files/model.png", "rb") as f:
        return Response(content=f.read(), media_type="image/png")


# PUT request to change neuron value using Change Neuron Mutation Operator
@app.put("/change-neuron/{modelId}/{layerName}/{row}/{column}/{kernel}/{value}")
def change_neuron(modelId: str, layerName: str, row: int, column: int, kernel: int, value: Union[float,None],
                  response: Response):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)

    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model_var)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    try:
        NeuronOperator.changeNeuron(model_obj, layerName, row, column, kernel, value)
        response.status_code = 200
        # Delete Previous Mutant
        if os.path.exists('../../models/mutant.h5'):
            os.remove('../../models/mutant.h5')
        model_obj.save("../../models/mutant.h5")
        return {"message": "Neuron value successfully changed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/change-edge/{modelId}/{layerName}/{prevNeuron}/{currNeuron}/{value}")
def change_edge(modelId: str, layerName: str, prevNeuron: int, currNeuron: int, value: Union[float,None],
                  response: Response):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model_var)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.changeEdge(model_obj, layerName, prevNeuron, currNeuron, value)
        response.status_code = 200
        # Delete Previous Mutant
        if os.path.exists('../../models/mutant.h5'):
            os.remove('../../models/mutant.h5')
        model_obj.save("../../models/mutant.h5")
        return {"message": "Edge value successfully changed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# PUT request to block neuron value using Block Neuron Mutation Operator
@app.put("/block-neuron/{modelId}/{layerName}/{row}/{column}/{kernel}")
def block_neuron(modelId: str, layerName: str, row: int, column: int, kernel: int, response: Response):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)

    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model_var)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    try:
        NeuronOperator.blockNeuron(model_obj, layerName, row, column, kernel)
        response.status_code = 200
        # Delete Previous Mutant
        if os.path.exists('../../models/mutant.h5'):
            os.remove('../../models/mutant.h5')
        # Save New Mutant
        model_obj.save("../../models/mutant.h5")
        return {"message": "Neuron successfully blocked"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/block-edge/{modelId}/{layerName}/{prevNeuron}/{currNeuron}")
def block_edge(modelId: str, layerName: str, prevNeuron: int, currNeuron: int, response: Response):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model_var)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.blockEdge(model_obj, layerName, prevNeuron, currNeuron)
        response.status_code = 200
        # Delete Previous Mutant
        if os.path.exists('../../models/mutant.h5'):
            os.remove('../../models/mutant.h5')
        model_obj.save("../../models/mutant.h5")
        return {"message": "Edge value successfully blocked"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# PUT request to change neuron value with its multiplicative inverse
@app.put("/mul-inverse-neuron/{modelId}/{layerName}/{row}/{column}/{kernel}")
def mul_inverse_neuron(modelId: str, layerName: str, row: int, column: int, kernel: int, response: Response):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model_var)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())
    try:
        NeuronOperator.mul_inverse(model_obj, layerName, row, column, kernel)
        response.status_code = 200
        # Delete Previous Mutant
        if os.path.exists('../../models/mutant.h5'):
            os.remove('../../models/mutant.h5')
        # Save New Mutant
        model_obj.save("../../models/mutant.h5")
        return {"message": "Neuron value successfully changed to multiplicative inverse"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/mul-inverse-edge/{modelId}/{layerName}/{prevNeuron}/{currNeuron}")
def mul_inverse_edge(modelId: str, layerName: str, prevNeuron: int, currNeuron: int, response: Response):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model_var)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.mul_inverse(model_obj, layerName, prevNeuron, currNeuron)
        response.status_code = 200
        # Delete Previous Mutant
        if os.path.exists('../../models/mutant.h5'):
            os.remove('../../models/mutant.h5')
        model_obj.save("../../models/mutant.h5")
        return {"message": "Edge value successfully changed with multiplicative inverse"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# PUT request to replace neuron with its Additive Inverse
@app.put("/additive-inverse-neuron/{modelId}/{layerName}/{row}/{column}/{kernel}")
def additive_inverse_neuron(modelId: str, layerName: str, row: int, column: int, kernel: int, response: Response):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model_var)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())
    try:
        NeuronOperator.additive_inverse(model_obj, layerName, row, column, kernel)
        response.status_code = 200
        # Delete Previous Mutant
        if os.path.exists('../../models/mutant.h5'):
            os.remove('../../models/mutant.h5')
        # Save New Mutant
        model_obj.save("../../models/mutant.h5")
        return {"message": "Neuron value successfully changed to additive inverse"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/additive-inverse-edge/{modelId}/{layerName}/{prevNeuron}/{currNeuron}")
def additive_inverse_edge(modelId: str, layerName: str, prevNeuron: int, currNeuron: int, response: Response):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model_var)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.additive_inverse(model_obj, layerName, prevNeuron, currNeuron)
        response.status_code = 200
        # Delete Previous Mutant
        if os.path.exists('../../models/mutant.h5'):
            os.remove('../../models/mutant.h5')
        model_obj.save("../../models/mutant.h5")
        return {"message": "Edge value successfully changed with multiplicative inverse"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# PUT request to invert the value of neuron using Invert Neuron Mutation Operator
@app.put("/invert-neuron/{modelId}/{layerName}/{row}/{column}/{kernel}")
def invert_neuron(modelId: str, layerName: str, row: int, column: int, kernel: int, response: Response):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model_var)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())
    try:
        NeuronOperator.invertNeuron(model_obj, layerName, row, column, kernel)
        response.status_code = 200
        # Delete Previous Mutant
        if os.path.exists('../../models/mutant.h5'):
            os.remove('../../models/mutant.h5')
        # Save New Mutant
        model_obj.save("../../models/mutant.h5")
        return {"message": "Neuron value successfully inverted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/invert-edge/{modelId}/{layerName}/{prevNeuron}/{currNeuron}")
def invert_edge(modelId: str, layerName: str, prevNeuron: int, currNeuron: int, response: Response):
    # If modelId is passed as Mutant, we have to load mutant
    if modelId == "Mutant":
        model_var = models.load_model("../../models/mutant.h5")
    else:
        # Get the model object from the dictionary
        model_var = model_dict.get(modelId)
    # Create a deep copy of the model
    model_obj = models.clone_model(model_var)
    model_obj.set_weights(model_var.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model_var)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.invertEdge(model_obj, layerName, prevNeuron, currNeuron)
        response.status_code = 200
        # Delete Previous Mutant
        if os.path.exists('../../models/mutant.h5'):
            os.remove('../../models/mutant.h5')
        model_obj.save("../../models/mutant.h5")
        return {"message": "Edge value successfully changed with multiplicative inverse"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Index based PUT request for Mutation Operators
@app.put("/mutation-operator/{index}/{modelId}/{layerName}/{row}/{column}/{kernel}/{value}")
def mutation_operator(index: int, modelId: str, layerName: str, row: int, column: int, kernel: int, response: Response,
                      value: Union[float, None] = None):
    if index == 1:
        result = change_neuron(modelId, layerName, row, column, kernel, value, response)
    elif index == 2:
        result = block_neuron(modelId, layerName, row, column, kernel, response)
    elif index == 3:
        result = mul_inverse_neuron(modelId, layerName, row, column, kernel, response)
    elif index == 4:
        result = additive_inverse_neuron(modelId, layerName, row, column, kernel, response)
    elif index == 5:
        result = invert_neuron(modelId, layerName, row, column, kernel, response)
    else:
        # If the index is not recognized, return an error message
        raise HTTPException(status_code=400, detail="Invalid Mutation Operator index")

    response.status_code = 200
    return {"message": "Mutation operator successfully applied", "result": result}
