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
from datetime import datetime
import numpy as np
import yaml
import visualkeras as vk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main")))

from mutation_operators import NeuronLevel
from mutation_operators import WeightLevel
from operator_utils import WeightUtils
from operator_utils import Model_layers
import predictions_analysis as pa
from pg_adapter import PgAdapter
import mutation_killing as mk


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

# Constructor for safe load of object coming from database
def int_constructor(loader, node):
    value = loader.construct_scalar(node)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    else:
        return value

yaml.SafeLoader.add_constructor('tag:yaml.org,2002:int', int_constructor)

# Dataset for Lenet
(train_X, train_y), (test_X, test_y) = mnist.load_data()

conn = PgAdapter.get_instance().connection
cur = conn.cursor()
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
            """, (json.dumps(results), 2, projectId))
        raise HTTPException(status_code=500, detail=str(e))



# GET request to retrieve confusion matirx for a specific model
@app.get("/confusion-matrix/{tableName}/{modelId}/{projectId}")
def getConfusionMatrix(tableName: str, modelId: int, projectId: int):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    prediction = model.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    return json.dumps({str(k): v for k, v in matrix.items()})

# GET request to retrieve accuracy for a specific model clas wise
@app.get("/class-accuracy/{tableName}/{modelId}/{projectId}")
def getAccuracy(tableName: str, modelId: int, projectId: int):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)

    prediction = model.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    accuracy = pa.getAccuracy(matrix)
    return json.dumps({str(k): v for k, v in accuracy.items()})

# GET request to retrieve accuracy of a specific model
@app.get("/model-accuracy/{tableName}/{modelId}/{projectId}")
def getModelAccuracy(tableName: str, modelId: int, projectId: int):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    predictions = model.predict(test_X)
    matrix = pa.getConfusionMatrix(predictions, test_y)
    accuracy = pa.getModelAccuracy(matrix)
    return json.dumps({"accuracy": accuracy})


# GET request to retrieve specificity for a specific model
@app.get("/specificity/{tableName}/{modelId}/{projectId}")
def getSpecificity(tableName: str, modelId: int, projectId: int):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    prediction = model.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    specificity = pa.getSpecificity(matrix)
    return json.dumps({str(k): v for k, v in specificity.items()})

# GET request to retrieve f1-score for a specific model
@app.get("/f1-score/{tableName}/{modelId}/{projectId}")
def getf1Score(tableName: str, modelId: int, projectId: int):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    prediction = model.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    f1_score = pa.getF1Score(matrix)
    return json.dumps({str(k): v for k, v in f1_score.items()})

# GET request to retrieve recall for a specific model
@app.get("/recall/{tableName}/{modelId}/{projectId}")
def getRecall(tableName: str, modelId: int, projectId: int):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    prediction = model.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    recall = pa.getRecall(matrix)
    return json.dumps({str(k): v for k, v in recall.items()})


# GET request to retrieve precision for a specific model
@app.get("/precision/{tableName}/{modelId}/{projectId}")
def getPrecision(tableName: str, modelId: int, projectId: int):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    prediction = model.predict(test_X)
    matrix = pa.getConfusionMatrix(prediction, test_y)
    precision = pa.getPrecision(matrix)
    return json.dumps({str(k): v for k, v in precision.items()})

# GET request to retrieve sensitivity for a specific model
@app.get("/sensitivity/{tableName}/{modelId}/{projectId}")
def get_sensitivity(tableName: str, modelId: int, projectId: int):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    predictions = model.predict(test_X)
    matrix = pa.getConfusionMatrix(predictions, test_y)
    sensitivity = pa.getSensitivity(matrix)
    return json.dumps({str(k): v for k, v in sensitivity.items()})

# GET request to retrieve complete report of a specific model with respect to all classes
@app.get("/report/{tableName}/{modelId}/{projectId}/{beta}")
def getReport(tableName: str, modelId: int, projectId: int, beta: float = 1):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)

    # Generate predictions and labels for the model
    predictions = model.predict(test_X)
    matrix = pa.getConfusionMatrix(predictions, test_y)
    class_metrics = pa.getAllMetrics(matrix, beta)
    return json.dumps([{str(k): {str(inner_k): inner_v for inner_k, inner_v in v.items()}} for k, v in class_metrics.items()])

# GET request to retrieve accuracy of a specific model
@app.get("/auc/{tableName}/{modelId}/{projectId}")
def getAuc(tableName: str, modelId: int, projectId: int):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    predictions = model.predict(test_X)
    matrix = pa.getConfusionMatrix(predictions, test_y)
    auc = pa.getAuc(matrix)
    return json.dumps({str(k): v for k, v in auc.items()})

@app.get("/f-beta-score/{tableName}/{modelId}/{projectId}/{beta}")
def getFBetaScore(tableName: str, modelId: int, projectId: int, beta: float = 1.0):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
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
@app.get("/all-weights/{tableName}/{modelId}/{projectId}/{layerName}")
def get_weights(tableName: str, modelId: int, projectId: int, layerName: str):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    trainable_weights = weights.GetWeights(model, layerName)
    result = trainable_weights.tolist()
    return jt.dumps(result)


# GET request to retrieve all the weights of a specific kernel in a specific layer of a specific model
@app.get("/kernel-weights/{tableName}/{modelId}/{projectId}/{layerName}/{kernel}")
def getKernelWeights(tableName: str, modelId: int, projectId: int, layerName: str, kernel: int):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")

    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)

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
@app.get("/all-kernel-weights/{tableName}/{modelId}/{projectId}/{layerName}")
def getAllKernelWeights(tableName: str, modelId: int, projectId: int, layerName: str):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)

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
    else:
        raise HTTPException(status_code=500, detail=("Invalid Operator Type"))



# GET request to retrieve Description of Mutation Operators present
@app.get("/operators-description/{type}")
def getMutationOperatorsDescription(operatortype: int):
    if operatortype == 1:
        return jt.dumps(NeuronLevel.neuronLevelMutationOperatorsDescription)
    elif operatortype == 2:
        return jt.dumps(WeightLevel.weightLevelMutationOperatorsDescription)
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
@app.put("/change-neuron/{tableName}/{modelId}/{projectId}/{layerName}/{row}/{column}/{kernel}/{value}")
def change_neuron(tableName: str, modelId: int, projectId: int, layerName: str, row: int, column: int, kernel: int, value: Union[float,None],
                  response: Response):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)

    # Create a deep copy of the model
    model_obj = models.clone_model(model)
    model_obj.set_weights(model.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    try:
        NeuronOperator.changeNeuron(model_obj, layerName, row, column, kernel, value)
        response.status_code = 200
        # Insert the new model into the database
        cur.execute("""
                    INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                    f"Mutated Model with the effect of Change Neuron at"
                    f" Layer: {layerName} and [{row}][{column}] of kernel {kernel}",
                    model_obj.to_bytes(), datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "Neuron value successfully changed", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/change-edge/{tableName}/{modelId}/{projectId}/{layerName}/{prevNeuron}/{currNeuron}/{value}")
def change_edge(tableName: str, modelId: int, projectId: int, layerName: str, prevNeuron: int, currNeuron: int, value: Union[float,None],
                  response: Response):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    # Create a deep copy of the model
    model_obj = models.clone_model(model)
    model_obj.set_weights(model.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.changeEdge(model_obj, layerName, prevNeuron, currNeuron, value)
        response.status_code = 200
        # Insert the new model into the database
        cur.execute("""
                            INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Change Edge at"
                     f" Layer: {layerName} at {prevNeuron} -> {currNeuron} connection",
                     model_obj.to_bytes(), datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "Edge value successfully changed", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# PUT request to block neuron value using Block Neuron Mutation Operator
@app.put("/block-neuron/{tableName}/{modelId}/{projectId}/{layerName}/{row}/{column}/{kernel}")
def block_neuron(tableName: str, modelId: int, projectId: int, layerName: str, row: int, column: int, kernel: int, response: Response):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)

    # Create a deep copy of the model
    model_obj = models.clone_model(model)
    model_obj.set_weights(model.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    try:
        NeuronOperator.blockNeuron(model_obj, layerName, row, column, kernel)
        response.status_code = 200
        # Insert the new model into the database
        cur.execute("""
                            INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Block Neuron at"
                     f" Layer: {layerName} and [{row}][{column}] of kernel {kernel}",
                     model_obj.to_bytes(), datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "Neuron value successfully blocked", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/block-edge/{tableName}/{modelId}/{projectId}/{layerName}/{prevNeuron}/{currNeuron}")
def block_edge(tableName: str, modelId: int, projectId: int, layerName: str, prevNeuron: int, currNeuron: int, response: Response):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    # Create a deep copy of the model
    model_obj = models.clone_model(model)
    model_obj.set_weights(model.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.blockEdge(model_obj, layerName, prevNeuron, currNeuron)
        response.status_code = 200
        # Insert the new model into the database
        cur.execute("""
                            INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Block Weight at"
                     f" Layer: {layerName} at {prevNeuron} -> {currNeuron} connection",
                     model_obj.to_bytes(), datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "Neuron value successfully changed", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# PUT request to change neuron value with its multiplicative inverse
@app.put("/mul-inverse-neuron/{tableName}/{modelId}/{projectId}/{layerName}/{row}/{column}/{kernel}")
def mul_inverse_neuron(tableName: str, modelId: int, projectId: int, layerName: str, row: int, column: int, kernel: int, response: Response):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    # Create a deep copy of the model
    model_obj = models.clone_model(model)
    model_obj.set_weights(model.get_weights())
    try:
        NeuronOperator.mul_inverse(model_obj, layerName, row, column, kernel)
        # Insert the new model into the database
        cur.execute("""
                                    INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Multiplicative Inverse at"
                     f" Layer: {layerName} and [{row}][{column}] of kernel {kernel}",
                     model_obj.to_bytes(), datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]
        response.status_code = 200
        return {"message": "Neuron value successfully blocked", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/mul-inverse-edge/{tableName}/{modelId}/{projectId}/{layerName}/{prevNeuron}/{currNeuron}")
def mul_inverse_edge(tableName: str, modelId: int, projectId: int, layerName: str, prevNeuron: int, currNeuron: int, response: Response):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    # Create a deep copy of the model
    model_obj = models.clone_model(model)
    model_obj.set_weights(model.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.mul_inverse(model_obj, layerName, prevNeuron, currNeuron)
        response.status_code = 200
        # Insert the new model into the database
        cur.execute("""
                                    INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Multiplicative Inverse at"
                     f" Layer: {layerName} at {prevNeuron} -> {currNeuron} connection",
                     model_obj.to_bytes(), datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "Neuron value successfully changed", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# PUT request to replace neuron with its Additive Inverse
@app.put("/additive-inverse-neuron/{tableName}/{modelId}/{projectId}/{layerName}/{row}/{column}/{kernel}")
def additive_inverse_neuron(tableName: str, modelId: int, projectId: int, layerName: str, row: int, column: int, kernel: int, response: Response):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    # Create a deep copy of the model
    model_obj = models.clone_model(model)
    model_obj.set_weights(model.get_weights())
    try:
        NeuronOperator.additive_inverse(model_obj, layerName, row, column, kernel)
        response.status_code = 200
        # Insert the new model into the database
        cur.execute("""
                                           INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                                           VALUES (%s, %s, %s, %s, %s, %s)
                                           RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Additive Inverse at"
                     f" Layer: {layerName} and [{row}][{column}] of kernel {kernel}",
                     model_obj.to_bytes(), datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]
        response.status_code = 200
        return {"message": "Neuron value successfully blocked", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/additive-inverse-edge/{tableName}/{modelId}/{projectId}/{layerName}/{prevNeuron}/{currNeuron}")
def additive_inverse_edge(tableName: str, modelId: int, projectId: int, layerName: str, prevNeuron: int, currNeuron: int, response: Response):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    # Create a deep copy of the model
    model_obj = models.clone_model(model)
    model_obj.set_weights(model.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.additive_inverse(model_obj, layerName, prevNeuron, currNeuron)
        response.status_code = 200
        # Insert the new model into the database
        cur.execute("""
                                    INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Additive Inverse at"
                     f" Layer: {layerName} at {prevNeuron} -> {currNeuron} connection",
                     model_obj.to_bytes(), datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "Neuron value successfully changed", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# PUT request to invert the value of neuron using Invert Neuron Mutation Operator
@app.put("/invert-neuron/{tableName}/{modelId}/{projectId}/{layerName}/{row}/{column}/{kernel}")
def invert_neuron(tableName: str, modelId: int, projectId: int, layerName: str, row: int, column: int, kernel: int, response: Response):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    # Get the names of the layers on which neuron level operators are applicable
    neuron_layers = layers.getNeuronLayers(model)

    if layerName not in neuron_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))
    # Create a deep copy of the model
    model_obj = models.clone_model(model)
    model_obj.set_weights(model.get_weights())
    try:
        NeuronOperator.invertNeuron(model_obj, layerName, row, column, kernel)
        response.status_code = 200
        # Insert the new model into the database
        cur.execute("""
                                           INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                                           VALUES (%s, %s, %s, %s, %s, %s)
                                           RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Invert at"
                     f" Layer: {layerName} and [{row}][{column}] of kernel {kernel}",
                     model_obj.to_bytes(), datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]
        response.status_code = 200
        return {"message": "Neuron value successfully blocked", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/invert-edge/{tableName}/{modelId}/{projectId}/{layerName}/{prevNeuron}/{currNeuron}")
def invert_edge(tableName: str, modelId: int, projectId: int, layerName: str, prevNeuron: int, currNeuron: int, response: Response):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    # Select the model bytes from the database
    cur.execute("SELECT model_bytes FROM %s WHERE id = %d and project_id = %d", (tableName, modelId, projectId))
    model_bytes = cur.fetchone()[0]
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Model not found")
    # Load the model from the bytes
    model = keras.models.model_from_bytes(model_bytes)
    # Create a deep copy of the model
    model_obj = models.clone_model(model)
    model_obj.set_weights(model.get_weights())
    # Get the names of the layers on which neuron level operators are applicable
    edge_layers = layers.getEdgeLayers(model)
    if layerName not in edge_layers:
        raise HTTPException(status_code=500, detail=("Invalid layer"))

    try:
        EdgeOperator.invertEdge(model_obj, layerName, prevNeuron, currNeuron)
        response.status_code = 200
        # Insert the new model into the database
        cur.execute("""
                                    INSERT INTO mutated_models (original_model_id, name, description, file, created_at, updated_at)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    RETURNING id """,
                    (modelId, f"mutant-{modelId}",
                     f"Mutated Model with the effect of Invert Edge Weight at"
                     f" Layer: {layerName} at {prevNeuron} -> {currNeuron} connection",
                     model_obj.to_bytes(), datetime.utcnow(), datetime.utcnow())
                    )
        new_model_id = cur.fetchone()[0]

        response.status_code = 200
        return {"message": "Neuron value successfully changed", "mutated_model_id": new_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Index based PUT request for Mutation Operators
@app.put("/mutation-operator/{index}/{tableName}/{modelId}/{projectId}/{layerName}/{row}/{column}/{kernel}/{value}")
def mutation_operator(index: int, tableName: str, modelId: int, projectId: int, layerName: str, row: int, column: int, kernel: int, response: Response,
                      value: Union[float, None] = None):
    if not tableName.isalnum():
        raise HTTPException(status_code=400, detail="Invalid table name parameter")
    if index == 1:
        result = change_neuron(tableName, modelId, projectId, layerName, row, column, kernel, value, response)
    elif index == 2:
        result = block_neuron(tableName, modelId,projectId, layerName, row, column, kernel, response)
    elif index == 3:
        result = mul_inverse_neuron(tableName, modelId,projectId, layerName, row, column, kernel, response)
    elif index == 4:
        result = additive_inverse_neuron(tableName, modelId, projectId, layerName, row, column, kernel, response)
    elif index == 5:
        result = invert_neuron(tableName, modelId,projectId, layerName, row, column, kernel, response)
    else:
        # If the index is not recognized, return an error message
        raise HTTPException(status_code=400, detail="Invalid Mutation Operator index")

    response.status_code = 200
    return {"message": "Mutation operator successfully applied", "result": result}
