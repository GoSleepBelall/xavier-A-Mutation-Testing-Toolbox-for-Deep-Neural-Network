from typing import Union
from fastapi import FastAPI
import jsonify as json
import tensorflow as tf

from src.main.models_generator import Lenet5Generator
from src.main.operator_utils import WeightUtils
from src.main.operator_utils import Model_layers



app = FastAPI()
layers = Lenet5Generator()
weights = WeightUtils()


# Read the models globally

lenet5 = tf.keras.models.load_model("../../models/xavier-lenet5.h5")
alexnet = tf.keras.models.load_model("../../models/xavier-lenet5.h5")       #TODO: this is still reading lenet-5 because i dont have alexnet right now


@app.get("/weights/{model}/{layerName}")
def get_weights(model: str, layerName: str):
    if model == "1":
        model = lenet5
    elif model == "2":
        model = alexnet
    trainable_weights = weights.GetWeights(model, layerName)
    print(trainable_weights)
    return json(trainable_weights)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
