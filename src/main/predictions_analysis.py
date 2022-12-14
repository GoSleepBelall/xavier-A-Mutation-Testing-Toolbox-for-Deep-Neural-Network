from operator_utils import WeightUtils
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import classification_report

def float_formatter(num):
    return "{:.1f}".format(num)

def matrix_mapper(data, count):
    iteration_number = 0
    print("[", end=" ")
    for x in data:
        print("[", end=" ")
        for y in x:
            print(float_formatter(y), end=" ")
        print("]", end=" ")
        iteration_number = iteration_number+1
        if iteration_number>= count:
            break
        print()
    print("]")

def classify_weights(single_class, classes):
    classified_data = np_utils.to_categorical(single_class, classes)
    return classified_data

def map_manual_predictions(predictions,test_data, classes, count):
    # In order to map predictions with the labels, we can classify data into 10 classes
    classified_test_data = classify_weights(test_data, classes)
    print("predictions")
    matrix_mapper(predictions,count)
    print("labels")
    matrix_mapper(classified_test_data,count)



def generate_classification_report(predictions, labels):
    # Reshaping Predictions (Merging data)
    predictions = np.argmax(predictions, axis=1)
    # Generate Classification Report
    print(classification_report(labels, predictions))






