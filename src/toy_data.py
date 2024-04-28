import sys
sys.path.insert(0,'..')

from src.models import *
from src.loss_functions import *
from src.noise import *
from src.metrics import *
from src.plotting import *
from src.generate_data import *

import sklearn
import pandas as pd

from operator import xor

from scipy.stats import bernoulli

import random


def logistic(x):
    return 1 / (1 + np.exp(-x))

def calculate_weights(prob_labels):
    # Create the system of equations
    A = np.array([
        [1, 0, 0],  # For p(0,0)
        [1, 0, 1],  # For p(0,1)
        [1, 1, 0],  # For p(1,0)
        [1, 1, 1]   # For p(1,1)
    ])
    
    # Calculate the logit for each probability
    b = np.array([np.log(p/(1-p)) if p not in [0, 1] else -np.inf if p == 0 else np.inf for p in prob_labels.values()])
    
    # Solve the system of equations
    weights = np.linalg.lstsq(A, b, rcond=None)[0]
    return weights

def generate_probabilistic_labels(features, weights, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)  # Set seed for reproducibility

    # Calculate probabilities using logistic function
    probs = logistic(np.dot(features, weights))
    
    # Sample labels based on probabilities
    labels = np.random.binomial(1, probs)
    return labels


def generate_dataset(true_labels, instances_counts, probabilistic = False, weights=None, seed=None, p_y_x_dict=None):
    """
    Generate a dataset with binary features
    
    :param true_labels: Dictionary with keys as (x1, x2) and values as the deterministic labels.
    :param instances_counts: Dictionary with keys as (x1, x2) and values as the number of instances.
    :param weights: Coefficients for the logistic model (bias, w1, w2).
    :param seed: Seed for the random number generator.
    :return: Shuffled features and probabilistic labels.
    """
    features, labels = [], []
    for (x1, x2), label in true_labels.items():
        n_instances = instances_counts[(x1, x2)]
        #features.extend([(1, x1, x2)] * n_instances)  # Add a 1 for the bias term in logistic regression
        features.extend([(x1, x2)] * n_instances) 
        if not probabilistic:
            labels.extend([label] * n_instances)

    # Convert to numpy array for matrix operations
    features = np.array(features)
    
    if probabilistic:
    # Generate probabilistic labels
        #labels = generate_probabilistic_labels(features, weights, random_seed=seed)
        for i, (x1,x2) in enumerate(features):
            np.random.seed(i)
            p_y_x = p_y_x_dict[(x1,x2)]
            label =  bernoulli.rvs(p=p_y_x[1], size=1)[0]
            labels.append(label)

    # Shuffle the dataset
    #return shuffle(features[:, 1:], labels, random_state=seed)  # Exclude the bias term from the returned features
    return shuffle(features, np.array(labels), random_state=seed)  # Exclude the bias term from the returned features

def generate_noisy_labels(y, random_seeds, noise_transition_matrix= None, instance_dependent=False, X=None, noise_transition_dict=None):
    # Generate noisy labels for given random seeds
    noisy_labels = []
    for seed in random_seeds:
        np.random.seed(seed)
        if instance_dependent:
            noisy_labels.append(add_label_noise(y, instance_dependent=instance_dependent, X=X, noise_transition_dict=noise_transition_dict))
        else:
            noisy_labels.append(add_label_noise(y, noise_transition_matrix=noise_transition_matrix))
    return noisy_labels

def train_models(X, y, noisy_ys, X_test, methods, random_seeds, noise_transition_matrix = None):
    # Train models and collect predictions
    predictions = {}
    for method in methods:
        if method == "clean":
            model = train_LR_no_test(X, y, seed=42, num_epochs=100, correction_type="None", noise_transition_matrix=None)
            predictions[(method, 'clean')] = get_predictions_LR(model, X_test)
        else:
            for ny, seed in zip(noisy_ys, random_seeds):
                model = train_LR_no_test(X, ny, seed=42, num_epochs=100, correction_type=method, noise_transition_matrix=noise_transition_matrix)
                predictions[(method, f'seed={seed}')] = get_predictions_LR(model, X_test)
    return predictions

def compile_predictions(predictions, instances_counts, true_labels):
    # Compile predictions into a DataFrame
    rows = []
    for i,(x1, x2) in enumerate(list(true_labels.keys())):
        row = {
            'n': instances_counts[(x1, x2)],
            'x1': x1,
            'x2': x2,
            'y': true_labels[(x1, x2)]
        }
        for (method, label_type), preds in predictions.items():
            row[f'{method} {label_type}'] = preds[i].item()
        rows.append(row)
    return pd.DataFrame(rows)

def coefs_pm1_to_01(coefs):
    """
    :param coefs: coefficient vector of a linear classifier with features x[j] \in {-1,+1}
    :return: coefficient vector of a linear classifier with features x[j] \in {0,+1}
    """
    coefs = np.array(coefs).flatten()
    t = coefs[0] - sum(coefs[1:])
    w = 2.0 * coefs[1:]
    w = np.insert(w, 0, t)
    return w

def linear_model_batch(coefs, X):
    """
    Apply a linear classifier to a batch of input features X.
    
    :param coefs: Coefficients of the linear model (bias, w1, w2, ..., wd).
    :param X: Input features as a 2D array where each row is a set of features.
    :return: The raw output of the linear model for each input set, before applying the threshold.
    """
    # Add a column of 1s to the beginning of X to account for the bias term
    bias = np.ones((X.shape[0], 1))
    
    X_with_bias = np.hstack((bias, X))
    
    # Compute the linear combination of inputs and weights
    z = np.dot(X_with_bias, coefs)

    return z


def output_01(coefs, X):
    """
    Convert the linear model output to the 0, 1 space for a batch of inputs.
    
    :param coefs: Coefficients of the linear model.
    :param X: Batch of input features as a 2D array.
    :return: Array of 0 or 1 depending on the raw model output for each input set.
    """
    z = linear_model_batch((coefs), X)
    
    return (z > 0).astype(int)

# Function to load the CSV file with the coefficients
def load_coefficients(file_path):
    """
    Load coefficients from a CSV file.

    :param file_path: Path to the CSV file.
    :return: Coefficients as a NumPy array.
    """
    converted = []
    for value in pd.read_csv(file_path, header=None).values:
        converted.append(coefs_pm1_to_01(value))
    return converted


def zero_one_loss(y_true, y_pred):
    """
    Calculate the mean 0-1 loss for a set of predictions.
    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :return: The mean 0-1 loss.
    """
    # Convert y_true and y_pred to numpy arrays if they are not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Calculate the loss
    loss = np.mean(np.where(y_true == y_pred, 0, 1))
    return loss


def backward_zero_one_loss(y_true, y_pred, noise_transition_matrix=None, instance_dependent=False, noise_transition_dict=None, X=None):
    """
    Calculate the noise-corrected 0-1 loss for a set of predictions.

    :param y_true: The observed (noisy) labels.
    :param y_pred: The predicted labels.
    :param noise_transition_matrix: The noise transition matrix.
    :return: The noise-corrected 0-1 loss.
    """
    # Convert y_true and y_pred to numpy arrays if they are not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if instance_dependent:
        
        # Initialize the loss
        corrected_loss = 0
        # Compute the 0-1 loss for each prediction
        for i in range(len(y_true)):
            instance = X[i]
            x0 = instance[0]
            x1 = instance[1]
            tup = (x0,x1)

            # Calculate the inverse noise transition matrix
            noise_transition_matrix = noise_transition_dict[tup]
            inv_noise_transition_matrix = np.linalg.inv(noise_transition_matrix)

            for true_label in range(noise_transition_matrix.shape[0]):
                # Compute the uncorrected 0-1 loss for the current assumed true label
                uncorrected_loss = 1 if y_pred[i] != true_label else 0
                
                # Weight the uncorrected loss by the probability of the true label given the observed label
                corrected_loss += inv_noise_transition_matrix[true_label, y_true[i]] * uncorrected_loss

        # Normalize the corrected loss by the number of samples
        corrected_loss /= len(y_true)
    
    else:

        # Calculate the inverse noise transition matrix
        inv_noise_transition_matrix = np.linalg.inv(noise_transition_matrix)

        # Initialize the loss
        corrected_loss = 0
        # Compute the 0-1 loss for each prediction
        for i in range(len(y_true)):

            for true_label in range(noise_transition_matrix.shape[0]):
                # Compute the uncorrected 0-1 loss for the current assumed true label
                uncorrected_loss = 1 if y_pred[i] != true_label else 0
                
                # Weight the uncorrected loss by the probability of the true label given the observed label
                corrected_loss += inv_noise_transition_matrix[true_label, y_true[i]] * uncorrected_loss

        # Normalize the corrected loss by the number of samples
        corrected_loss /= len(y_true)

    return corrected_loss


def bayes_model(d, X, y, loss_type = "0-1", noise_transition_matrix = None, noise_transition_dict=None):
    
    # Example file path, replace with actual path pattern
    file_path = "/h/snagaraj/noise_multiplicity/src/linear_classifier_coefficients/coefs_complete_sym_d_0"+str(d)+".csv"

    # Load the coefficients from the CSV file
    # The file path will need to be adjusted to the actual path where your file is stored
    coefficients = (load_coefficients(file_path))

    random.shuffle(coefficients) ##Shuffling in case of ties, we have different orders we check
    
    best_loss = 100
    
    for i in range(len(coefficients)):
        coefs = (coefficients[i])
        y_pred = output_01(coefs, X)
        
        if loss_type == "0-1":
            loss = zero_one_loss(y, y_pred)
        else:
            if noise_transition_dict!=None:
                loss = backward_zero_one_loss(y, y_pred, instance_dependent=True, X=X, noise_transition_dict=noise_transition_dict)
            
            else:
                loss = backward_zero_one_loss(y, y_pred, noise_transition_matrix)
            
        if loss < best_loss:
            best_loss = loss
            best_model = coefs
    return best_model, best_loss

def compile_bayes_predictions(d, X, y, noisy_ys, random_seeds, instances_counts, true_labels, noise_transition_matrix):
    X_test = np.array(list(true_labels.keys()))
    
    predictions = {}
    
    methods = ["clean", "noisy"]
    losses = ["0-1", "corrected_0-1"]
    
    for method in methods:
        for loss in losses:
            if (method == "clean" and loss == "corrected_0-1"):
                continue
            elif method == "clean" and loss == "0-1":
                best_model, best_loss = bayes_model(d, X, y, loss_type = "0-1")
                predictions[method+"_"+loss] = output_01((best_model), X_test)
            else: #Noisy
                for ny, seed in zip(noisy_ys, random_seeds):
                    best_model_noisy, best_loss = bayes_model(d, X, ny, loss_type = loss, noise_transition_matrix = noise_transition_matrix)
                    predictions[(method+"_"+loss + "_" + f'seed={seed}')] = output_01((best_model_noisy), X_test)
                
    # Compile predictions into a DataFrame
    rows = []
    for i,(x1, x2) in enumerate(list(true_labels.keys())):
        row = {
            'n': instances_counts[(x1, x2)],
            'x1': x1,
            'x2': x2,
            'y': true_labels[(x1, x2)]
        }
        for (key), preds in predictions.items():
            row[f'{key}'] = int(preds[i].item())
        rows.append(row)
    return pd.DataFrame(rows)

# def generate_metrics_toy(noise_levels, m,d , X, y, true_labels, instances_counts, loss_type, noise_type = "class_independent", fixed_class = None, fixed_noise = None, feature_weights=None):
#     """
#     Generate a dictionary of ambiguity rates across a range of noise levels.
    
#     :param noise_levels: A list or array of noise levels to test.
#     :param X: The input features for the dataset.
#     :param y: The true labels for the dataset.
#     :param true_labels: The true labels dictionary for generating the dataset.
#     :param instances_counts: The instances counts for generating the dataset.
#     :return: A dictionary with keys as instances and values as lists of ambiguity rates.
#     """
    
#     X_test = np.array(list(true_labels.keys()))
#     labels = np.array(list(true_labels.values()))

#     error_rates = {str(list(instance)): [] for instance in X_test}
#     disagreement_rates = {str(list(instance)): [] for instance in X_test}
    
#     for flip_p in tqdm(noise_levels):
#         # Generate noisy labels
#         if noise_type == "class_independent":
#             noise_transition_matrix = np.array([[1-flip_p, flip_p], [flip_p, 1-flip_p]])
#             noisy_ys = generate_noisy_labels(y, range(1, m+1), noise_transition_matrix=noise_transition_matrix)
#         elif noise_type == "class_conditional":
#             if fixed_class == 0:
#                 noise_transition_matrix = np.array([[1-fixed_noise, fixed_noise], [flip_p, 1-flip_p]])
#             elif fixed_class == 1:
#                 noise_transition_matrix = np.array([[1-flip_p, flip_p], [fixed_noise, 1-fixed_noise]])
#             noisy_ys = generate_noisy_labels(y, range(1, m+1), noise_transition_matrix=noise_transition_matrix)
#         elif noise_type == "instance_dependent":

#             # flip_00 = flip_p
#             # flip_01 = 0.1
#             # flip_10 = 0.1
#             # flip_11 = 0.1

#             # noise_transition_dict = {(0,0): np.array([[1-flip_00, flip_00], [flip_00, 1-flip_00]]),
#             #                         (0,1): np.array([[1-flip_01, flip_01], [flip_01, 1-flip_01]]),
#             #                         (1,0): np.array([[1-flip_10, flip_10], [flip_10, 1-flip_10]]),
#             #                         (1,1): np.array([[1-flip_11, flip_11], [flip_11, 1-flip_11]])}

#             noise_transition_dict = {}
            
#             for instance in np.unique(X, axis=0):
#                 flip_instance = instance_noise_level(instance, flip_p, feature_weights)
#                 # Use flip_instance to build the noise transition matrix for this instance
#                 noise_transition_matrix = np.array([[1-flip_instance, flip_instance],
#                                                     [flip_instance, 1-flip_instance]])
#                 noise_transition_dict[tuple(instance)] = noise_transition_matrix

            

#             noisy_ys = generate_noisy_labels(y, range(1, m+1), instance_dependent=True, X=X, noise_transition_dict=noise_transition_dict)
        
        
#         predictions = []

#         for noisy_y in noisy_ys:
#             # Train the model and make predictions
#             if noise_type == "instance_dependent":
#                 best_model_noisy, loss = bayes_model(d, X, noisy_y, loss_type=loss_type, noise_transition_dict=noise_transition_dict)
#             else:
#                 best_model_noisy, loss = bayes_model(d, X, noisy_y, loss_type=loss_type, noise_transition_matrix=noise_transition_matrix)
                
#             preds = output_01(best_model_noisy, X_test)
#             predictions.append(preds)

#         predictions = np.array(predictions)
#         error_rate = calculate_error_rate(predictions, labels)
#         disagreement_rate = estimate_disagreement(predictions)

#         for i, item in enumerate(X_test):
#             error_rates[str(list(item))].append(error_rate[i])
#             disagreement_rates[str(list(item))].append(disagreement_rate[i])

#     return error_rates, disagreement_rates

def generate_metrics_toy(noise_levels, m, max_iter, d , X_train, y_train, X_test, y_test, true_labels, instances_counts, noise_type = "class_independent",  p_y_x_dict = None, probabilistic = False, fixed_class = 0, fixed_noise = 0.3, feature_weights=None):
    """
    Generate a dictionary of ambiguity rates across a range of noise levels.
    
    :param noise_levels: A list or array of noise levels to test.
    :param X: The input features for the dataset.
    :param y: The true labels for the dataset.
    :param true_labels: The true labels dictionary for generating the dataset.
    :param instances_counts: The instances counts for generating the dataset.
    :return: A dictionary with keys as instances and values as lists of ambiguity rates.
    """


    #error_rates = {str(list(instance)): [] for instance in X_test}
    #disagreement_rates = {str(list(instance)): [] for instance in X_test}

    error_rates = {noise_level: [] for noise_level in noise_levels}
    disagreement_rates = {noise_level: [] for noise_level in noise_levels}
    
    for flip_p in tqdm(noise_levels):

        if noise_type == "class_independent":
            yn_train, noise_transition_matrix = generate_class_independent_noise(y_train, flip_p)
        elif noise_type == "class_conditional":
            yn_train, noise_transition_matrix = generate_class_conditional_noise(y_train, flip_p, fixed_class, fixed_noise)
        elif noise_type == "instance_dependent":
            yn_train, noise_transition_dict = generate_instance_dependent_noise(y_train, X_train, flip_p, feature_weights)

        predictions = []
        typical_count = 0
        
        for iteration in range(1, max_iter+1):

            u_vec = []
            for seed, (yn, instance) in enumerate(zip(yn_train, X_train)):
                
                if noise_type == "instance_dependent":
                    noise_transition_matrix = noise_transition_dict[tuple(instance)]

                p_y_x = p_y_x_dict[tuple(instance)]

                u = infer_u(yn, noise_transition_matrix, p_y_x, seed = seed+iteration)
                u_vec.append(u)

            #Check if typical
            
            u_vec = np.array(u_vec)
            
            if noise_type == "instance_dependent":
                bool_flag = True
                for instance in np.unique(X_train, axis=0):
                    p_y_x = p_y_x_dict[tuple(instance)]
                    
                    indices = [idx for idx, elem in enumerate(X_train) if np.array_equal(elem, instance)]
                    
                    if not is_typical(u_vec[indices], noise_transition_matrix, yn_train[indices], p_y_x, noise_type = noise_type):
                        bool_flag = False
                        break
                if not bool_flag:
                    continue
                
            else:
                if not is_typical(u_vec, noise_transition_matrix, yn_train, p_y_x, noise_type = noise_type):
                    continue

            cleaned_labels = flip_labels(yn_train, u_vec)

            best_model_noisy, loss = bayes_model(d, X_train, cleaned_labels, loss_type="0-1")

            preds = output_01(best_model_noisy, X_test)

            predictions.append(preds)
            typical_count+=1

            if typical_count==m:
                break

        predictions = np.array(predictions)

        try:
            error_rate = calculate_error_rate(predictions, y_test)
            disagreement_rate = estimate_disagreement(predictions)
        except:
            continue
        for i, item in enumerate(X_test):
            
            error_rates[flip_p].append(error_rate[i])
            disagreement_rates[flip_p].append(disagreement_rate[i])

    
        print(typical_count/iteration)
    return error_rates, disagreement_rates


def generate_metrics_toy_est_noise(noise_level, m,d , X, y, true_labels, instances_counts, loss_type):
    """
    Generate a dictionary of ambiguity rates across a range of estimated noise levels.
    
    :param noise_levels: A list or array of noise levels to test.
    :param X: The input features for the dataset.
    :param y: The true labels for the dataset.
    :param true_labels: The true labels dictionary for generating the dataset.
    :param instances_counts: The instances counts for generating the dataset.
    :return: A dictionary with keys as instances and values as lists of ambiguity rates.
    """
    
    X_test = np.array(list(true_labels.keys()))
    labels = np.array(list(true_labels.values()))

    error_rates = {str(list(instance)): [] for instance in X_test}
    disagreement_rates = {str(list(instance)): [] for instance in X_test}
    
    flip_p = noise_level
    noise_levels = np.linspace(-noise_level, 0.49-noise_level, num=20)
    
    # Generate noisy labels
    if noise_type == "class_independent":
        noise_transition_matrix = np.array([[1-flip_p, flip_p], [flip_p, 1-flip_p]])

    noisy_ys = generate_noisy_labels(y, noise_transition_matrix, range(1, m+1))
    
    
    for delta in tqdm(noise_levels):
        
        noise_transition_matrix_est = np.array([[1-flip_p-delta, flip_p+delta], [flip_p+delta, 1-flip_p-delta]])
        
        predictions = []
        for noisy_y in noisy_ys:
            # Train the model and make predictions
            best_model_noisy, _ = bayes_model(d, X, noisy_y, loss_type=loss_type, noise_transition_matrix=noise_transition_matrix_est)
            preds = output_01(best_model_noisy, X_test)
            predictions.append(preds)

        predictions = np.array(predictions)
        error_rate = calculate_error_rate(predictions, labels)
        disagreement_rate = estimate_disagreement(predictions)

        for i, item in enumerate(X_test):
            error_rates[str(list(item))].append(error_rate[i])
            disagreement_rates[str(list(item))].append(disagreement_rate[i])

    return error_rates, disagreement_rates




def generate_losses_accuracies(noise_levels, m,d , X, y, true_labels, instances_counts, loss_types, noise_type = "class_independent", fixed_class = None, fixed_noise = None):
    """
    Generate a dictionary of ambiguity rates across a range of noise levels.
    
    :param noise_levels: A list or array of noise levels to test.
    :param X: The input features for the dataset.
    :param y: The true labels for the dataset.
    :param true_labels: The true labels dictionary for generating the dataset.
    :param instances_counts: The instances counts for generating the dataset.
    :return: A dictionary with keys as instances and values as lists of ambiguity rates.
    """
    
    X_test = np.array(list(true_labels.keys()))
    labels = np.array(list(true_labels.values()))
    
    loss_dict = {loss_type: [] for loss_type in loss_types}
    accuracy_dict = {loss_type: [] for loss_type in loss_types}

    for flip_p in tqdm(noise_levels):
        # Generate noisy labels
        if noise_type == "class_independent":
            noise_transition_matrix = np.array([[1-flip_p, flip_p], [flip_p, 1-flip_p]])
            noisy_ys = generate_noisy_labels(y, range(1, m+1), noise_transition_matrix=noise_transition_matrix)
        elif noise_type == "class_conditional":
            if fixed_class == 0:
                noise_transition_matrix = np.array([[1-fixed_noise, fixed_noise], [flip_p, 1-flip_p]])
            elif fixed_class == 1:
                noise_transition_matrix = np.array([[1-flip_p, flip_p], [fixed_noise, 1-fixed_noise]])
            noisy_ys = generate_noisy_labels(y, range(1, m+1), noise_transition_matrix=noise_transition_matrix)
        elif noise_type == "instance_dependent":

            flip_00 = flip_p
            flip_01 = 0.1
            flip_10 = 0.1
            flip_11 = 0.1

            noise_transition_dict = {(0,0): np.array([[1-flip_00, flip_00], [flip_00, 1-flip_00]]),
                                    (0,1): np.array([[1-flip_01, flip_01], [flip_01, 1-flip_01]]),
                                    (1,0): np.array([[1-flip_10, flip_10], [flip_10, 1-flip_10]]),
                                    (1,1): np.array([[1-flip_11, flip_11], [flip_11, 1-flip_11]])}

            noisy_ys = generate_noisy_labels(y, range(1, m+1), instance_dependent=True, X=X, noise_transition_dict=noise_transition_dict)
        
        noisy_y = noisy_ys[0]

        for loss_type in loss_types:
            if loss_type == "0-1 Clean":
                best_model, loss = bayes_model(d, X, y, loss_type="0-1")
                
                predictions = np.array(output_01(best_model, X))
                accuracy = np.mean(predictions == y)
                
                loss_dict[loss_type].append(loss)
                accuracy_dict[loss_type].append(accuracy)
                
                
            elif loss_type == "0-1 Noisy":
                # Train the model and make predictions
                best_model, loss = bayes_model(d, X, noisy_y, loss_type="0-1")
                predictions = np.array(output_01(best_model, X))
                accuracy = np.mean(predictions == y)


                loss_dict[loss_type].append(loss)
                accuracy_dict[loss_type].append(accuracy)
            
            elif loss_type == "Corrected 0-1 Noisy":

                if noise_type == "instance_dependent":
                    # Train the model and make predictions
                    best_model, loss = bayes_model(d, X, noisy_y, loss_type="Corrected 0-1", noise_transition_dict=noise_transition_dict)

                else:
                    # Train the model and make predictions
                    best_model, loss = bayes_model(d, X, noisy_y, loss_type="Corrected 0-1", noise_transition_matrix=noise_transition_matrix)

                predictions = output_01(best_model, X)
                accuracy = np.array(np.mean(predictions == y))


                loss_dict[loss_type].append(loss)
                accuracy_dict[loss_type].append(accuracy)

    return loss_dict, accuracy_dict


def generate_noisy_label(y, noise_transition_matrix= None, instance_dependent=False, X=None, noise_transition_dict=None):
    # Generate a single realization of noisy labels
    np.random.seed(2024)
    if instance_dependent:
        return add_label_noise(y, instance_dependent=instance_dependent, X=X, noise_transition_dict=noise_transition_dict)

    else:
        return add_label_noise(y, noise_transition_matrix=noise_transition_matrix)



def calculate_priors_toy(true_labels, instances_counts):
    """
    Calculate prior probabilities based on the observed frequencies from true_labels and instances_counts.

    Parameters:
    - true_labels: Dictionary mapping (x1, x2) pairs to their deterministic labels.
    - instances_counts: Dictionary mapping (x1, x2) pairs to their instance counts.

    Returns:
    - p_y_x_dict: Dictionary mapping (x1, x2) pairs to numpy arrays representing prior probabilities.
    """
    # Initialize counters for each label
    total_instances = sum(instances_counts.values())
    label_counts = {0: 0, 1: 0}
    
    # Count the number of instances for each label
    for (x1, x2), count in instances_counts.items():
        label = true_labels[(x1, x2)]
        label_counts[label] += count

    # Calculate prior probabilities
    p_y_x_dict = {}
    for (x1, x2), _ in true_labels.items():
        label = true_labels[(x1, x2)]
        # Calculate the prior for the current label and the complementary label
        prior = label_counts[label] / total_instances
        complementary_prior = 1 - prior  # Assuming binary labels (0 and 1)
        p_y_x_dict[(x1, x2)] = np.array([complementary_prior, prior]) if label == 1 else np.array([prior, complementary_prior])
    
    return p_y_x_dict