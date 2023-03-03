"""This file is made for testing purpose of Mutation Killing"""
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
from models_generator import Lenet5Generator
from mutation_operators import NeuronLevel
from operator_utils import Model_layers
import predictions_analysis as pa
from scipy.stats import ttest_ind
from statsmodels.stats.power import TTestIndPower


def create_samples(k):
    # Load Dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    # Convert into Numpy Arrays
    train_X = np.asarray(train_X)
    train_y = np.asarray(train_y)

    total_X = np.concatenate((train_X, test_X))
    total_Y = np.concatenate((train_y, test_y))

    # set 80% of data for training and 20% for testing
    train_size = int(len(total_X) * 0.8)
    test_size = len(total_X) - train_size

    training_data = []
    testing_data = []
    training_data_labels = []
    testing_data_labels = []

    # create k random samples
    for i in range(k):
        # randomly permute the indices of total_X and total_Y
        permuted_indices = np.random.permutation(len(total_X))

        # split the permuted indices into training and testing sets
        train_indices = permuted_indices[:train_size]
        test_indices = permuted_indices[train_size:]

        # split total_X and total_Y into training and testing sets using the permuted indices
        training_data.append(total_X[train_indices])
        training_data_labels.append(total_Y[train_indices])

        testing_data.append(total_X[test_indices])
        testing_data_labels.append(total_Y[test_indices])

    return training_data, training_data_labels, testing_data, testing_data_labels

def train_all_models(k, training_data, training_data_labels):
    m_gnrtr = Lenet5Generator()
    for i in range(k):
        model = m_gnrtr.generate_model()
        model.fit(x=training_data[i], y=training_data_labels[i], batch_size=32, epochs=2)
        filename = "../models/P_{}.h5".format(i)
        model.save(filename)

def create_mutants(k, mutation_operator, layer_name, row, column, kernel):
    convOperator = NeuronLevel()
    layer = Model_layers()
    for i in range(k):
        filename = "../models/P_{}.h5".format(i)
        model = tf.keras.models.load_model(filename)
        layers = layer.getLayerNames(model)
        convOperator.blockNeuron(model,layers[0], row,column, kernel)
        filename = "../models/M_{}.h5".format(i)
        model.save(filename)

def get_accuracies(k, testing_data, testing_data_labels):
    accuracy_model = []
    accuracy_mutant = []
    for i in range(k):
        filename = "../models/P_{}.h5".format(i)
        model = tf.keras.models.load_model(filename)
        prediction = model.predict(testing_data[i])
        counters = pa.getConfusionMatrix(prediction, testing_data_labels[i])
        accuracy_model.append(pa.getModelAccuracy(counters))
    for i in range(k):
        filename = "../models/M_{}.h5".format(i)
        model = tf.keras.models.load_model(filename)
        prediction = model.predict(testing_data[i])
        counters = pa.getConfusionMatrix(prediction, testing_data_labels[i])
        accuracy_mutant.append(pa.getModelAccuracy(counters))
    return accuracy_model, accuracy_mutant

def mutation_killing(k):
    # Firstly We Create Random Samples from union of Testing and Training Data
    training_data, training_data_labels, testing_data, testing_data_labels = create_samples(k)
    # Then we train models for all those samples
    train_all_models(k, training_data, training_data_labels)
    # Then Mutants are created from all those models
    create_mutants(k,"NULL","conv2d", 0,0,1)
    # Then we Got Accuracies for all the models and mutants
    accuracy_model, accuracy_mutant = get_accuracies(k, testing_data, testing_data_labels)
    # Then we performed T-test for the distribution for p value we got
    t_stat, p_value = ttest_ind(accuracy_model, accuracy_mutant)
    # Now to compute Cohen's d for effect size
    d = (np.mean(accuracy_model) - np.mean(accuracy_mutant)) / \
        np.sqrt(((len(accuracy_model) - 1) * np.var(accuracy_model, ddof=1) +
                 (len(accuracy_mutant) - 1) * np.var(accuracy_mutant, ddof=1)) / (
                    len(accuracy_model) + len(accuracy_mutant) - 2))
    # Now to compute beta using the TTestIndPower
    alpha = 0.05
    nobs1 = len(accuracy_model)
    ratio = 1
    power_analysis = TTestIndPower()
    beta = power_analysis.solve_power(effect_size=d, nobs1=nobs1, alpha=alpha, ratio=ratio)

    print("Pvalue: ",p_value)
    print("Beta: ",beta)
    print("Effect Size: ",d)

mutation_killing(5)
