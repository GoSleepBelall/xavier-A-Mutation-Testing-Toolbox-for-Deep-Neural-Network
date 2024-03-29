from typing import Union
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Response
import sys
import os
import json_tricks as jt
import json
from tensorflow import keras
from keras import models
from tensorflow.keras.datasets import mnist
from datetime import datetime
import numpy as np
import yaml
import visualkeras as vk
import pickle
import copy



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main")))

from mutation_operators import NeuronLevel
from mutation_operators import WeightLevel
from mutation_operators import BiasLevel
from mutation_operators import WalkingNeuron
from operator_utils import WeightUtils
from operator_utils import Model_layers
import predictions_analysis as pa
from pg_adapter import PgAdapter
import mutation_killing as mk
from models_generator import Lenet5Generator


app = FastAPI(title="XAVIER-API", description="A Mutation Testing Toolbox.", version="2.0")

#Todo: Integrate your models here

lenet5 = models.load_model("../../models/xavier-lenet5.h5")
# A global dictionary mapping model names to model objects
model_dict = {
    'lenet5': lenet5
}


layers = Model_layers()
weights = WeightUtils()
NeuronOperator = NeuronLevel()
EdgeOperator = WeightLevel()
BiasOperator = BiasLevel()
Walking = WalkingNeuron()
LenetGenerator = Lenet5Generator()

# Constructor for safe load of object coming from database
def int_constructor(loader, node):
    value = loader.construct_scalar(node)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    else:
        return value

yaml.SafeLoader.add_constructor('tag:yaml.org,2002:int', int_constructor)

# Dataset for Lenet
#(train_X, train_y), (test_X, test_y) = mnist.load_data()

conn = PgAdapter.get_instance().connection
cur = conn.cursor()


@app.get("/walking-neuron/{projectId}/{layerName}/{currNeuron}")
def walkingNeuron(projectId, layerName, currNeuron):
    model = LenetGenerator.generate_model()
    # Load Data
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    # Convert into Numpy Arrays
    train_X = np.asarray(train_X)
    train_y = np.asarray(train_y)
    model.fit(x=train_X, y=train_y, epochs=1)

    # Get Accuracies of Model
    prediction1 = model.predict(test_X)
    counters1 = pa.getConfusionMatrix(prediction1, test_y)
    matrices_original = pa.getAllMetrics(counters1, 1.5)
    print("Original model created and evaluated successfully")

    # Save the original model in database
    model_bytes = pickle.dumps(model)
    cur.execute("""
                   INSERT INTO original_models (project_id, name, description, file, matrices,created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   RETURNING id """,
                (projectId, "Original Model",
                 "Original Model",
                 model_bytes, json.dumps(
                    [{str(k): {str(inner_k): inner_v for inner_k, inner_v in v.items()}} for k, v in
                     matrices_original.items()]), datetime.utcnow(), datetime.utcnow(),)
                )
    print("Model inserted in Database")
    # Fetch Original Model Id
    new_model_id = cur.fetchone()[0]
    print("The new model id is ", new_model_id)
    trainable_weights = weights.GetWeights(model, layerName)
    total_neurons = trainable_weights[0].shape[1]
    for i in range(total_neurons):
        if i != int(currNeuron):
            # Create a deep copy of the model
            mutant = copy.deepcopy(model)
            Walking.replaceNeuron(mutant, layerName, int(currNeuron), i)
            print("Mutation operator completed successfully: ", i)
            # Get Accuracies of Mutant
            prediction2 = mutant.predict(test_X)
            counters2 = pa.getConfusionMatrix(prediction2, test_y)
            matrices_mutant = pa.getAllMetrics(counters2, 1.5)
            del (model_bytes)

            model_bytes = pickle.dumps(mutant)
            cur.execute("""
                                                       INSERT INTO mutated_models (original_model_id, name, description, file, matrices,created_at, updated_at)
                                                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                                                       RETURNING id """,
                        (new_model_id, f"Mutant-{new_model_id}",
                         f"Mutated Model with effect of Walking Neuron at layer {layerName} with destination neuron {i}",
                         model_bytes, json.dumps(
                            [{str(k): {str(inner_k): inner_v for inner_k, inner_v in v.items()}} for k, v in
                             matrices_mutant.items()]), datetime.utcnow(), datetime.utcnow(),)
                        )

            print(new_model_id)
            conn.commit()

    return {"message": "Project Successfully Completed"}

@app.get("/mutation_score/projects/{projectId:path}")
def mutation_score(projectId: str):
    flag =1
    k = 0
    isKilled = 0
    allProjectIds = [v for v in projectId.split('/')]
    for x in allProjectIds:
        cur.execute("SELECT * FROM PROJECTS where id = %s", (x,))
        result = cur.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Project Not Found")
        hyper_params = yaml.safe_load(result[4])
        print(hyper_params)
        if flag:
            k = hyper_params['k_value']
            print("k value", k)
            flag = False
        if hyper_params['k_value'] != k:
            raise HTTPException(status_code=400, detail="Your K Values for projects are not equal")
        temp = result[8]
        print(temp)
        mutation = temp['Mutation']
        if mutation == True:
            isKilled+=1

    score = float(isKilled/len(allProjectIds))
    score *= 100
    return json.dumps(score)


@app.get("/run/{projectId}")
def run(projectId: int):
    cur.execute("SELECT * FROM PROJECTS where id = %s", (projectId,))
    result = cur.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Project Not Found")

    elif result[7] == 2:
        raise HTTPException(status_code=200, detail="Project already executed Successfully")

    id = result[0]
    user_id = result[1]
    name = result[2]
    desc = result[3]
    hyper_params = yaml.safe_load(result[4])
    if int(hyper_params['k_value']) < 5:
        raise HTTPException(status_code=400, detail="Value of k must be greater than or equals to 5")
    try:
        # Fetching data
        mk.mutation_killing(projectId, hyper_params)
        return {"message": "Project Successfully Completed"}
    except Exception as e:
        results = {
            'status': e
        }
        cur.execute("""
                UPDATE Projects 
                SET results = %s, status = %s
                WHERE id = %s
                RETURNING id
            """, (json.dumps(results), 2, secondId))
        raise HTTPException(status_code=500, detail=str(e))



# GET request to retrieve confusion matirx for a specific model
@app.get("/confusion-matrix/{tableName}/{modelId}/{secondId}")
def getConfusionMatrix(tableName: str, modelId: int, secondId: int):
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)
    prediction = model.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    return json.dumps({str(k): v for k, v in matrix.items()})

# GET request to retrieve accuracy for a specific model clas wise
@app.get("/class-accuracy/{tableName}/{modelId}/{secondId}")
def getAccuracy(tableName: str, modelId: int, secondId: int):
    # Select the model bytes from the database
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes

    model = pickle.loads(model_bytes)
    prediction = model.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    accuracy = pa.getAccuracy(matrix)
    return json.dumps({str(k): v for k, v in accuracy.items()})

# GET request to retrieve accuracy of a specific model
@app.get("/model-accuracy/{tableName}/{modelId}/{secondId}")
def getModelAccuracy(tableName: str, modelId: int, secondId: int):
    # Select the model bytes from the database
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)
    predictions = model.predict(test_X)
    matrix = pa.getConfusionMatrix(predictions, test_y)
    accuracy = pa.getModelAccuracy(matrix)
    return json.dumps({"accuracy": accuracy})


# GET request to retrieve specificity for a specific model
@app.get("/specificity/{tableName}/{modelId}/{secondId}")
def getSpecificity(tableName: str, modelId: int, secondId: int):
    # Select the model bytes from the database
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)
    prediction = model.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    specificity = pa.getSpecificity(matrix)
    return json.dumps({str(k): v for k, v in specificity.items()})

# GET request to retrieve f1-score for a specific model
@app.get("/f1-score/{tableName}/{modelId}/{secondId}")
def getf1Score(tableName: str, modelId: int, secondId: int):
    # Select the model bytes from the database
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)
    prediction = model.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    f1_score = pa.getF1Score(matrix)
    return json.dumps({str(k): v for k, v in f1_score.items()})

# GET request to retrieve recall for a specific model
@app.get("/recall/{tableName}/{modelId}/{secondId}")
def getRecall(tableName: str, modelId: int, secondId: int):
    # Select the model bytes from the database
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)
    prediction = model.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    recall = pa.getRecall(matrix)
    return json.dumps({str(k): v for k, v in recall.items()})


# GET request to retrieve precision for a specific model
@app.get("/precision/{tableName}/{modelId}/{secondId}")
def getPrecision(tableName: str, modelId: int, secondId: int):
    # Select the model bytes from the database
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query, (modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)
    prediction = model.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    precision = pa.getPrecision(matrix)
    return json.dumps({str(k): v for k, v in precision.items()})

# GET request to retrieve sensitivity for a specific model
@app.get("/sensitivity/{tableName}/{modelId}/{secondId}")
def get_sensitivity(tableName: str, modelId: int, secondId: int):
    # Select the model bytes from the database
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query, (modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)
    predictions = model.predict(test_X)
    matrix = pa.getConfusionMatrix(predictions, test_y)
    sensitivity = pa.getSensitivity(matrix)
    return json.dumps({str(k): v for k, v in sensitivity.items()})

# GET request to retrieve complete report of a specific model with respect to all classes
@app.get("/report/{tableName}/{modelId}/{secondId}/{beta}")
def getReport(tableName: str, modelId: int, secondId: int, beta: float = 1):
    # Select the model bytes from the database
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query, (modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)

    # Generate predictions and labels for the model
    predictions = model.predict(test_X)
    matrix = pa.getConfusionMatrix(predictions, test_y)
    class_metrics = pa.getAllMetrics(matrix, beta)
    return json.dumps([{str(k): {str(inner_k): inner_v for inner_k, inner_v in v.items()}} for k, v in class_metrics.items()])

# GET request to retrieve accuracy of a specific model
@app.get("/auc/{tableName}/{modelId}/{secondId}")
def getAuc(tableName: str, modelId: int, secondId: int):
    # Select the model bytes from the database
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)
    predictions = model.predict(test_X)
    matrix = pa.getConfusionMatrix(predictions, test_y)
    auc = pa.getAuc(matrix)
    return json.dumps({str(k): v for k, v in auc.items()})

@app.get("/f-beta-score/{tableName}/{modelId}/{secondId}/{beta}")
def getFBetaScore(tableName: str, modelId: int, secondId: int, beta: float = 1.0):
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)
    predictions = model.predict(test_X)
    matrix = pa.getConfusionMatrix(predictions, test_y)
    f_beta_score = pa.getFBetaScore(matrix, beta)
    return json.dumps({str(k): v for k, v in f_beta_score.items()})

# GET request to retrieve all the layers in a specific model
@app.get("/layers/{modelId}")
def get_layers(modelId: str):
    # Get the model object from the dictionary
    model_var = model_dict.get(modelId)
    layer_names = layers.getLayerNames(model_var)
    return json.dumps(layer_names)

# GET request to retrieve all the layers on which Neuron level Mutation Operators are applicable
@app.get("/neuron_layers/{modelId}")
def get_neuron_layers(modelId: str):
    # Get the model object from the dictionary
    model_var = model_dict.get(modelId)
    layer_names = layers.getNeuronLayers(model_var)
    return json.dumps(layer_names)


# GET request to retrieve all the layers on which Bias level Mutation Operators are applicable
@app.get("/bias_layers/{modelId}")
def get_bias_layers(modelId: str):
    # Get the model object from the dictionary
    model_var = model_dict.get(modelId)
    layer_names = layers.getBiasLayers(model_var)
    return json.dumps(layer_names)

# GET request to retrieve all the layers on which Walking Neuron Mutation Operators are applicable
@app.get("/walking_neuron_layers/{modelId}")
def get_bias_layers(modelId: str):
    # Get the model object from the dictionary
    model_var = model_dict.get(modelId)
    layer_names = layers.getWalkingLayers(model_var)
    return json.dumps(layer_names)

# GET request to retrieve all the layers on which Edge level Mutation Operators are applicable
@app.get("/edge-layers/{modelId}")
def get_edge_layers(modelId: str):
     # Get the model object from the dictionary
    model_var = model_dict.get(modelId)
    layer_names = layers.getEdgeLayers(model_var)
    return json.dumps(layer_names)
# Todo: >>>>>>>>>>>>>>>>>>    START    >>>>>>>>>>>>>>>>>>>>>>>.
# Todo: Temporarily making these API calls for testing
# Todo: This page will be deleted soon
# Todo: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.


# GET request to retrieve all the trainable weights of a layers in a specific model
@app.get("/all-weights/{modelId}/{layerName}")
def get_weights_temp(modelId: str, layerName: str):
    # Get the model object from the dictionary
    model_var = model_dict.get(modelId)
    trainable_weights = weights.GetWeights(model_var, layerName)
    result = trainable_weights.tolist()
    return jt.dumps(result)

# GET request to retrieve all the trainable weights of a layers in a specific model
@app.get("/bias-weights/{modelId}/{layerName}")
def get_bias_temp(modelId: str, layerName: str):
    # Get the model object from the dictionary
    model_var = model_dict.get(modelId)
    bias_weights = weights.getBiasWeights(model_var, layerName)
    result = bias_weights.tolist()
    return jt.dumps(result)


# GET request to retrieve all the weights of a specific kernel in a specific layer of a specific model
@app.get("/kernel-weights/{modelId}/{layerName}/{kernel}")
def getKernelWeights_temp(modelId: str, layerName: str, kernel: int):

    # Get the model object from the dictionary
    model_var = model_dict.get(modelId)
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model_var)
    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    # Get Trainable Weights
    trainable_weights = weights.GetWeights(model_var, layerName)

    # Get Kernel wise weights from all weights
    kernel_weights = weights.getKernelWeights(trainable_weights, kernel)
    return jt.dumps(kernel_weights)


# GET request to retrieve all the weights of a All kernel in a specific layer of a specific model
@app.get("/all-kernel-weights/{modelId}/{layerName}")
def getAllKernelWeights_temp(modelId: str, layerName: str):

    # Get the model object from the dictionary
    model_var = model_dict.get(modelId)

    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model_var)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    # Get Trainable Weights
    trainable_weights = weights.GetWeights(model_var, layerName)

    # Get Kernel wise weights from all weights
    kernel_weights = []
    total = layers.getKernelNumbers(model_var, layerName)
    for kernel in range(total):
        kernel_weights.append(weights.getKernelWeights(trainable_weights, kernel))
    return jt.dumps(kernel_weights)



# Todo: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
# Todo: Temporarily making these API calls for testing
# Todo: This page will be deleted soon
# Todo: >>>>>>>>>>>>>>>>>>>     END     >>>>>>>>>>>>>>>>>>>>>>>>>.

# GET request to retrieve all the trainable weights of a layers in a specific model
@app.get("/all-weights/{tableName}/{modelId}/{secondId}/{layerName}")
def get_weights(tableName: str, modelId: int, secondId: int, layerName: str):
    # Select the model bytes from the database
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)
    trainable_weights = weights.GetWeights(model, layerName)
    result = trainable_weights.tolist()
    return jt.dumps(result)


# GET request to retrieve all the weights of a specific kernel in a specific layer of a specific model
@app.get("/kernel-weights/{tableName}/{modelId}/{secondId}/{layerName}/{kernel}")
def getKernelWeights(tableName: str, modelId: int, secondId: int, layerName: str, kernel: int):

    # Select the model bytes from the database
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)

    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    # Get Trainable Weights
    trainable_weights = weights.GetWeights(model, layerName)

    # Get Kernel wise weights from all weights
    kernel_weights = weights.getKernelWeights(trainable_weights, kernel)
    return jt.dumps(kernel_weights)


# GET request to retrieve all the weights of a All kernel in a specific layer of a specific model
@app.get("/all-kernel-weights/{tableName}/{modelId}/{secondId}/{layerName}")
def getAllKernelWeights(tableName: str, modelId: int, secondId: int, layerName: str):
    # Select the model bytes from the database
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)

    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    # Get Trainable Weights
    trainable_weights = weights.GetWeights(model, layerName)

    # Get Kernel wise weights from all weights
    kernel_weights = []
    total = layers.getKernelNumbers(model, layerName)
    for kernel in range(total):
        kernel_weights.append(weights.getKernelWeights(trainable_weights, kernel))
    return jt.dumps(kernel_weights)

# GET request to retrieve number of kernels present in a layer
@app.get("/kernel/{modelId}/{layerName}")
def getKernelNum(modelId: str, layerName: str):

    # Get the model object from the dictionary
    model_var = model_dict.get(modelId)
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model_var)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    return json.dumps(layers.getKernelNumbers(model_var, layerName))


# GET request to retrieve List of Mutation Operators present
@app.get("/operators-list/{operatortype}")
def getMutationOperatorsList(operatortype: int):
    if operatortype == 1:
        return json.dumps(NeuronLevel.neuronLevelMutationOperatorshash)
    elif operatortype == 2:
        return json.dumps(WeightLevel.weightLevelMutationOperatorsList)
    elif operatortype == 3:
        return json.dumps(BiasLevel.biasLevelMutationOperatorhash)
    else:
        raise HTTPException(status_code=500, detail=("Invalid Operator Type"))



# GET request to retrieve Description of Mutation Operators present
@app.get("/operators-description/{type}")
def getMutationOperatorsDescription(operatortype: int):
    if operatortype == 1:
        return jt.dumps(NeuronLevel.neuronLevelMutationOperatorsDescription)
    elif operatortype == 2:
        return jt.dumps(WeightLevel.weightLevelMutationOperatorsDescription)
    elif operatortype == 3:
        return jt.dumps(BiasLevel.biasLevelMutationOperatorDescription)
    else:
        raise HTTPException(status_code=500, detail=("Invalid Operator Type"))


@app.get("/model-image/{modelId}")
def get_model_image(modelId: str):

    # Get the model object from the dictionary
    model_var = model_dict.get(modelId)
    image = vk.layered_view(model_var, legend=True)
    image.save("../../supporting files/model.png")
    with open("../../supporting files/model.png", "rb") as f:
        return Response(content=f.read(), media_type="image/png")

# PUT request to change neuron value using Change Neuron Mutation Operator
@app.put("/change-bias/{tableName}/{modelId}/{secondId}/{layerName}/{kernel}/{value}")
def change_bias(tableName: str, modelId: int, secondId: int, layerName: str, kernel: int, value: Union[float,None],
                  response: Response):
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)

    # Create a deep copy of the model
    mutant = models.clone_model(model)
    mutant.set_weights(model.get_weights())

    try:
        BiasOperator.changeBiasValue(mutant,layerName,kernel,value)
        model_bytes = pickle.dumps(mutant)

        # Insert the new model into the database
        cur.execute("""
                    INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                    f"Mutated Model with the effect of Change Bias at"
                    f" Layer: {layerName}, kernel {kernel}",
                    model_bytes, datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "Neuron value successfully changed", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# PUT request to change neuron value using Change Neuron Mutation Operator
@app.put("/change-neuron/{tableName}/{modelId}/{secondId}/{layerName}/{row}/{column}/{kernel}/{value}")
def change_neuron(tableName: str, modelId: int, secondId: int, layerName: str, row: int, column: int, kernel: int, value: Union[float,None],
                  response: Response):
    # Select the model bytes from the database
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)

    # Create a deep copy of the model
    mutant = models.clone_model(model)
    mutant.set_weights(model.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    try:
        NeuronOperator.changeNeuron(mutant, layerName, row, column, kernel, value)
        model_bytes = pickle.dumps(mutant)

        # Insert the new model into the database
        cur.execute("""
                    INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                    f"Mutated Model with the effect of Change Neuron at"
                    f" Layer: {layerName} and [{row}][{column}] of kernel {kernel}",
                    model_bytes, datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "Neuron value successfully changed", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/change-edge/{tableName}/{modelId}/{secondId}/{layerName}/{prevNeuron}/{currNeuron}/{value}")
def change_edge(tableName: str, modelId: int, secondId: int, layerName: str, prevNeuron: int, currNeuron: int, value: Union[float,None],
                  response: Response):
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)
    # Create a deep copy of the model
    mutant = models.clone_model(model)
    mutant.set_weights(model.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.changeEdge(mutant, layerName, prevNeuron, currNeuron, value)
        model_bytes = pickle.dumps(mutant)
        response.status_code = 200
        # Insert the new model into the database
        cur.execute("""
                            INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Change Edge at"
                     f" Layer: {layerName} at {prevNeuron} -> {currNeuron} connection",
                     model_bytes, datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "Edge value successfully changed", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# PUT request to block neuron value using Block Neuron Mutation Operator
@app.put("/block-neuron/{tableName}/{modelId}/{secondId}/{layerName}/{row}/{column}/{kernel}")
def block_neuron(tableName: str, modelId: int, secondId: int, layerName: str, row: int, column: int, kernel: int, response: Response):
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)

    # Create a deep copy of the model
    mutant = models.clone_model(model)
    mutant.set_weights(model.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    try:
        NeuronOperator.blockNeuron(mutant, layerName, row, column, kernel)
        model_bytes = pickle.dumps(mutant)

        # Insert the new model into the database
        cur.execute("""
                            INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Block Neuron at"
                     f" Layer: {layerName} and [{row}][{column}] of kernel {kernel}",
                     model_bytes, datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "Neuron value successfully blocked", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/block-edge/{tableName}/{modelId}/{secondId}/{layerName}/{prevNeuron}/{currNeuron}")
def block_edge(tableName: str, modelId: int, secondId: int, layerName: str, prevNeuron: int, currNeuron: int, response: Response):
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)
    # Create a deep copy of the model
    mutant = models.clone_model(model)
    mutant.set_weights(model.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.blockEdge(mutant, layerName, prevNeuron, currNeuron)
        model_bytes = pickle.dumps(mutant)
        # Insert the new model into the database
        cur.execute("""
                            INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Block Weight at"
                     f" Layer: {layerName} at {prevNeuron} -> {currNeuron} connection",
                     model_bytes, datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "edge value successfully blocked", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# PUT request to change neuron value with its multiplicative inverse
@app.put("/mul-inverse-neuron/{tableName}/{modelId}/{secondId}/{layerName}/{row}/{column}/{kernel}")
def mul_inverse_neuron(tableName: str, modelId: int, secondId: int, layerName: str, row: int, column: int, kernel: int, response: Response):
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    # Create a deep copy of the model
    mutant = models.clone_model(model)
    mutant.set_weights(model.get_weights())
    try:
        NeuronOperator.mul_inverse(mutant, layerName, row, column, kernel)
        model_bytes = pickle.dumps(mutant)

        # Insert the new model into the database
        cur.execute("""
                                    INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Multiplicative Inverse at"
                     f" Layer: {layerName} and [{row}][{column}] of kernel {kernel}",
                     model_bytes, datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]
        response.status_code = 200
        return {"message": "Neuron value successfully inverted", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/mul-inverse-edge/{tableName}/{modelId}/{secondId}/{layerName}/{prevNeuron}/{currNeuron}")
def mul_inverse_edge(tableName: str, modelId: int, secondId: int, layerName: str, prevNeuron: int, currNeuron: int, response: Response):
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)
    # Create a deep copy of the model
    mutant = models.clone_model(model)
    mutant.set_weights(model.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.mul_inverse(mutant, layerName, prevNeuron, currNeuron)
        model_bytes = pickle.dumps(mutant)
        # Insert the new model into the database
        cur.execute("""
                                    INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Multiplicative Inverse at"
                     f" Layer: {layerName} at {prevNeuron} -> {currNeuron} connection",
                     model_bytes, datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "edge value successfully inverted", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# PUT request to replace neuron with its Additive Inverse
@app.put("/additive-inverse-neuron/{tableName}/{modelId}/{secondId}/{layerName}/{row}/{column}/{kernel}")
def additive_inverse_neuron(tableName: str, modelId: int, secondId: int, layerName: str, row: int, column: int, kernel: int, response: Response):
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)

    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    # Create a deep copy of the model
    mutant = models.clone_model(model)
    mutant.set_weights(model.get_weights())
    try:
        NeuronOperator.additive_inverse(mutant, layerName, row, column, kernel)
        model_bytes = pickle.dumps(mutant)
        # Insert the new model into the database
        cur.execute("""
                                           INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                                           VALUES (%s, %s, %s, %s, %s, %s)
                                           RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Additive Inverse at"
                     f" Layer: {layerName} and [{row}][{column}] of kernel {kernel}",
                     model_bytes, datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]
        response.status_code = 200
        return {"message": "Neuron value successfully inverted", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/additive-inverse-edge/{tableName}/{modelId}/{secondId}/{layerName}/{prevNeuron}/{currNeuron}")
def additive_inverse_edge(tableName: str, modelId: int, secondId: int, layerName: str, prevNeuron: int, currNeuron: int, response: Response):
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)

    # Create a deep copy of the model
    mutant = models.clone_model(model)
    mutant.set_weights(model.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.additive_inverse(mutant, layerName, prevNeuron, currNeuron)
        model_bytes = pickle.dumps(mutant)

        # Insert the new model into the database
        cur.execute("""
                                    INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Additive Inverse at"
                     f" Layer: {layerName} at {prevNeuron} -> {currNeuron} connection",
                     model_bytes, datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "edge value successfully inverted", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# PUT request to invert the value of neuron using Invert Neuron Mutation Operator
@app.put("/invert-neuron/{tableName}/{modelId}/{secondId}/{layerName}/{row}/{column}/{kernel}")
def invert_neuron(tableName: str, modelId: int, secondId: int, layerName: str, row: int, column: int, kernel: int, response: Response):
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)

    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    # Create a deep copy of the model
    mutant = models.clone_model(model)
    mutant.set_weights(model.get_weights())
    try:
        NeuronOperator.invertNeuron(mutant, layerName, row, column, kernel)
        model_bytes = pickle.dumps(mutant)
        # Insert the new model into the database
        cur.execute("""
                                           INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                                           VALUES (%s, %s, %s, %s, %s, %s)
                                           RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Invert at"
                     f" Layer: {layerName} and [{row}][{column}] of kernel {kernel}",
                     model_bytes, datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]
        response.status_code = 200
        return {"message": "Neuron value successfully inverted", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/invert-edge/{tableName}/{modelId}/{secondId}/{layerName}/{prevNeuron}/{currNeuron}")
def invert_edge(tableName: str, modelId: int, secondId: int, layerName: str, prevNeuron: int, currNeuron: int, response: Response):
    # Select the model bytes from the database
    if tableName == 'original_models':
        query = "SELECT file FROM {} WHERE id = %s and project_id = %s".format(tableName)
    elif tableName == 'mutated_models':
        query = "SELECT file FROM {} WHERE id = %s and original_model_id = %s".format(tableName)
    else:
        raise HTTPException(status_code=404, detail="Table not found")
    cur.execute(query,(modelId, secondId,))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = pickle.loads(model_bytes)

    # Create a deep copy of the model
    mutant = models.clone_model(model)
    mutant.set_weights(model.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.invertEdge(mutant, layerName, prevNeuron, currNeuron)
        model_bytes = pickle.dumps(mutant)
        # Insert the new model into the database
        cur.execute("""
                                    INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Invert Edge Weight at"
                     f" Layer: {layerName} at {prevNeuron} -> {currNeuron} connection",
                     model_bytes, datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "Edge value successfully inverted", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Index based PUT request for Mutation Operators
@app.put("/mutation-operator/{index}/{tableName}/{modelId}/{secondId}/{layerName}/{row}/{column}/{kernel}/{value}")
def mutation_operator(index: int, tableName: str, modelId: int, secondId: int, layerName: str, row: int, column: int, kernel: int, response: Response,
                      value: Union[float, None] = None):
    if index == 1:
        result = change_neuron(tableName, modelId, secondId, layerName, row, column, kernel, value, response)
    elif index == 2:
        result = block_neuron(tableName, modelId,secondId, layerName, row, column, kernel, response)
    elif index == 3:
        result = mul_inverse_neuron(tableName, modelId,secondId, layerName, row, column, kernel, response)
    elif index == 4:
        result = additive_inverse_neuron(tableName, modelId, secondId, layerName, row, column, kernel, response)
    elif index == 5:
        result = invert_neuron(tableName, modelId,secondId, layerName, row, column, kernel, response)
    else:
        # If the index is not recognized, return an error message
        raise HTTPException(status_code=400, detail="Invalid Mutation Operator index")

    response.status_code = 200
    return {"message": "Mutation operator successfully applied", "result": result}
