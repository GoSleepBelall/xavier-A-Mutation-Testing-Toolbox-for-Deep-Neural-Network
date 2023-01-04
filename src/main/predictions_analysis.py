from operator_utils import WeightUtils
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import classification_report
from prettytable import PrettyTable



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

def get_confusion_matrix(predictions, labels):
    # Reshaping Predictions (Merging data)
    predictions = np.argmax(predictions, axis=1)

    # Initialize counters for each class
    counters = {
        0: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        1: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        2: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        3: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        4: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        5: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        6: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        7: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        8: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        9: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
    }
    # Loop over the elements in the predictions and labels arrays
    for i in range(len(predictions)):
        # Get the prediction and label for the current element
        prediction = predictions[i]
        label = labels[i]

        # Increment the appropriate counter based on the prediction and label
        if prediction == label:
            # TRUE-POSITIVE case: Model accurately predicted the label
            counters[prediction]['tp'] += 1
            # TRUE-NEGATIVE case: Model accurately predicted that label does not exist
            for class_label in range(10):
                if class_label != prediction:
                    counters[class_label]['tn'] += 1
        else:
            # Model predicted a wrong label
            counters[prediction]['fp'] += 1
            # Model did not predict the label when it existed
            counters[label]['fn'] += 1
    return counters


def printConfusionMatrix(predictions, labels):
    counters = get_confusion_matrix(predictions, labels)
    table = PrettyTable()
    table.field_names = ['Class', 'True-Positive', 'True-Negative', 'False-Positive', 'False-Negative']

    # Add a row for each class to the table
    for class_label, class_counters in counters.items():
        table.add_row([class_label, class_counters['tp'], class_counters['tn'], class_counters['fp'], class_counters['fn']])
    # Print the table
    print(table)


def generate_classification_report(predictions, labels):
    # Reshaping Predictions (Merging data)
    predictions = np.argmax(predictions, axis=1)
    # Generate Automatic Classification Report
    print(classification_report(labels, predictions))