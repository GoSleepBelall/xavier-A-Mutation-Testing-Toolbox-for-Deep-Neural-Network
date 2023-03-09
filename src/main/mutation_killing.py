"""This file is made for testing purpose of Mutation Killing"""
import pickle
from datetime import datetime
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
import numpy as np
from models_generator import Lenet5Generator
from mutation_operators import NeuronLevel
from mutation_operators import WeightLevel
from operator_utils import Model_layers
import predictions_analysis as pa
from scipy.stats import ttest_ind
from statsmodels.stats.power import TTestIndPower
from pg_adapter import PgAdapter
from keras import models
import json
import yaml


conn = PgAdapter.get_instance().connection
cur = conn.cursor()


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
    i = int(k)
    # create k random samples
    for i in range(i):
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


"""
hyper_params = {
"operator_type": "neuron_level",
"model": "lenet5",
"k_value": "2",
"layer": "conv2d", 
"operator_params": {
    "modal_kernel": "0", 
    "modal_row": "2", 
    "modal_col": "0", 
    "operator": "block-neuron", 
    "op_value": ""
    }
}
"""
def train_all_models(projectId, training_data, training_data_labels, testing_data, testing_data_labels, hyper_params ):
    m_gnrtr = Lenet5Generator()
    conv_operator = NeuronLevel()
    den_operator = WeightLevel()
    accuracy_model = []
    accuracy_mutant = []
    k_value = hyper_params['k_value']
    print(hyper_params)
    if k_value.isdigit():
        loop = int(k_value)
    else:
        # handle the case where k_value is not a valid integer
        print("Error: k_value must be a valid integer")
    for i in range(loop):
        matrices_original = {'accuracy': 0}
        matrices_mutant = {'accuracy': 0}
        K.clear_session()
        # Create a Fresh model and train
        model = m_gnrtr.generate_model()
        model.fit(x=training_data[i], y=training_data_labels[i], batch_size=32, epochs=1)

        # Create a deep copy of the model
        mutant = models.clone_model(model)
        mutant.set_weights(model.get_weights())

        # Apply Mutation Operator
        if hyper_params['operator_type'] == 'neuron_level':

            if hyper_params['operator_params']['operator'] == "change-neuron":
                conv_operator.changeNeuron(mutant,
                                          hyper_params['layer'],
                                          int(hyper_params['operator_params']['modal_row']),
                                          int(hyper_params['operator_params']['modal_col']),
                                          int(hyper_params['operator_params']['modal_kernel']),
                                          int(hyper_params['operator_params']['op_value']))

            elif hyper_params['operator_params']['operator'] == "block-neuron":
                conv_operator.blockNeuron(mutant,
                                          hyper_params['layer'],
                                          int(hyper_params['operator_params']['modal_row']),
                                          int(hyper_params['operator_params']['modal_col']),
                                          int(hyper_params['operator_params']['modal_kernel']))

            elif hyper_params['operator_params']['operator'] == "mul-inverse-neuron":
                conv_operator.mul_inverse(mutant,
                                          hyper_params['layer'],
                                          int(hyper_params['operator_params']['modal_row']),
                                          int(hyper_params['operator_params']['modal_col']),
                                          int(hyper_params['operator_params']['modal_kernel']))

            elif hyper_params['operator_params']['operator'] == "additive-inverse-neuron":
                conv_operator.additive_inverse(mutant,
                                          hyper_params['layer'],
                                          int(hyper_params['operator_params']['modal_row']),
                                          int(hyper_params['operator_params']['modal_col']),
                                          int(hyper_params['operator_params']['modal_kernel']))

            elif hyper_params['operator_params']['operator'] == "invert-neuron":
                conv_operator.invertNeuron(mutant,
                                          hyper_params['layer'],
                                          int(hyper_params['operator_params']['modal_row']),
                                          int(hyper_params['operator_params']['modal_col']),
                                          int(hyper_params['operator_params']['modal_kernel']))

        elif hyper_params['operator_type'] == 'weight_level':
            if hyper_params['operator_params']['operator'] == "change-edge":
                den_operator.changeEdge(mutant,hyper_params['layer'],
                                    int(hyper_params['operator_params']['modal_prev']),
                                    int(hyper_params['operator_params']['modal_curr']),
                                    int(hyper_params['operator_params']['value']))

            elif hyper_params['operator_params']['operator'] == "block-edge":
                den_operator.blockEdge(mutant,hyper_params['layer'],
                                        int(hyper_params['operator_params']['modal_prev']),
                                        int(hyper_params['operator_params']['modal_curr']))

            elif hyper_params['operator_params']['operator'] == "mul-inverse-edge":
                den_operator.mul_inverse(mutant,hyper_params['layer'],
                                        int(hyper_params['operator_params']['modal_prev']),
                                        int(hyper_params['operator_params']['modal_curr']))

            elif hyper_params['operator_params']['operator'] == "additive-inverse-edge":
                den_operator.additive_inverse(mutant,hyper_params['layer'],
                                        int(hyper_params['operator_params']['modal_prev']),
                                        int(hyper_params['operator_params']['modal_curr']))

            elif hyper_params['operator_params']['operator'] == "invert-edge":
                den_operator.invertEdge(mutant,hyper_params['layer'],
                                        int(hyper_params['operator_params']['modal_prev']),
                                        int(hyper_params['operator_params']['modal_curr']))

        # Get Accuracies of Model
        prediction = model.predict(testing_data[i])
        counters = pa.getConfusionMatrix(prediction, testing_data_labels[i])
        matrices_original = pa.getAllMetrics(counters,1.5)

        # Also store locally for further calculations
        accuracy_model.append(pa.getModelAccuracy(counters))

        # Get Accuracies of Mutant
        prediction = model.predict(testing_data[i])
        counters = pa.getConfusionMatrix(prediction, testing_data_labels[i])
        matrices_mutant = pa.getAllMetrics(counters,1.5)

        # Also store locally for further calculations
        accuracy_mutant.append(pa.getModelAccuracy(counters))


        # Insert models into database
        # Insert the new model into the database
        model_bytes = pickle.dumps(model)
        cur.execute("""
                                   INSERT INTO original_models (project_id, name, description, file, matrices,created_at, updated_at)
                                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                                   RETURNING id """,
                    (projectId, "Original Model",
                     "Original Model",
                     model_bytes, json.dumps([{str(k): {str(inner_k): inner_v for inner_k, inner_v in v.items()}} for k, v in matrices_original.items()]), datetime.utcnow(), datetime.utcnow(),)
                    )
        new_model_id = cur.fetchone()[0]
        del(model_bytes)

        model_bytes = pickle.dumps(mutant)
        cur.execute("""
                                           INSERT INTO mutated_models (original_model_id, name, description, file, matrices,created_at, updated_at)
                                           VALUES (%s, %s, %s, %s, %s, %s, %s)
                                           RETURNING id """,
                    (new_model_id, f"Mutant-{new_model_id}",
                     f"Mutated Model with effect of {hyper_params['operator_params']['operator']}",
                     model_bytes,  json.dumps([{str(k): {str(inner_k): inner_v for inner_k, inner_v in v.items()}} for k, v in matrices_mutant.items()]), datetime.utcnow(), datetime.utcnow(),)
                    )


        print(new_model_id)
        conn.commit()
        del(model)
        del(mutant)

    return accuracy_model, accuracy_mutant


def mutation_killing(projectId, hyper_params):
    # Firstly We Create Random Samples from union of Testing and Training Data
    training_data, training_data_labels, testing_data, testing_data_labels = create_samples(hyper_params['k_value'])

    # Then we train models for all those samples
    accuracy_model, accuracy_mutant = train_all_models(projectId, training_data, training_data_labels, testing_data, testing_data_labels, hyper_params)
    """
    # Then we Got Accuracies for all the models and mutants
    accuracy_model, accuracy_mutant = get_accuracies(hyper_params['k_value'], testing_data, testing_data_labels)    
    """

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

    results = {
        'P-Value': p_value,
        'Beta': beta,
        'Effect Size': d,
        'status': "Successfully executed"
    }

    cur.execute("""
        UPDATE Projects 
        SET results = %s, status = %s
        WHERE id = %s
        RETURNING id
    """, (json.dumps(results), 2, projectId))

    updated_id = cur.fetchone()[0]
    print(updated_id)
    conn.commit()
    return


