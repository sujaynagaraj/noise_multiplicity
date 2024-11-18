import numpy as np
from src.noise import *


def get_uncertainty(m, max_iter, preds, yn_train, p_y_x_dict,group_train = None, group_test = None, noise_type="class_independent", model_type="LR", T=None, epsilon=0.25, misspecify=False):
    
    typical_count = 0
    y_vec = yn_train
    all_plausible_labels = []
    
    for seed in (range(1, max_iter+1)):
        u_vec = infer_u(y_vec, group=group_train, noise_type=noise_type, p_y_x_dict=p_y_x_dict, T=T, seed=seed)
        typical_flag, _ = is_typical(u_vec, p_y_x_dict, group=group_train, T=T, y_vec=y_vec, noise_type=noise_type, uncertainty_type="backward", epsilon=epsilon)
        
        if misspecify or noise_type == "group":
            typical_flag = True
            
        if not typical_flag:
            continue
            
        flipped_labels = flip_labels(y_vec, u_vec)
        all_plausible_labels.append(flipped_labels)

        typical_count += 1

        if typical_count == m:
            break
    
    all_plausible_labels = np.array(all_plausible_labels)  # Shape: (k, n)
    
    # Calculate Actual Mistake as a vector of mean values for each instance
    actual_mistakes = np.mean(preds != all_plausible_labels, axis=0)  # Shape: (n,)
    

    # Calculate Unanticipated Mistake as a vector of mean values for each instance
    # Expand preds and yn_train to match dimensions for comparison
    preds_expanded = np.expand_dims(preds, axis=0)  # Shape: (1, n)
    yn_train_expanded = np.expand_dims(yn_train, axis=0)  # Shape: (1, n)

    # Case 1: pred == yn_train but pred != all_plausible_labels
    case_1 = (preds_expanded == yn_train_expanded) & (preds_expanded != all_plausible_labels)
    
    # Case 2: pred != yn_train but pred == all_plausible_labels
    case_2 = (preds_expanded != yn_train_expanded) & (preds_expanded == all_plausible_labels)
    
    # Calculate mean unanticipated mistakes for each instance
    unanticipated_mistakes = np.mean(case_1 | case_2, axis=0)  # Shape: (n,)

    return actual_mistakes, unanticipated_mistakes  
    
def calculate_unanticipated(preds, flipped_labels, yn):
    error_clean = (preds != flipped_labels)
    correct_noisy = (preds == yn)

    error_noisy = preds != yn
    correct_clean = preds == flipped_labels

    unanticipated_mistake = ((error_clean) & (correct_noisy))+ ((correct_clean) & (error_noisy))

    return unanticipated_mistake


def calculate_robustness_rate(predicted_probabilities, epsilon=0.01):
    # Calculate the variance of the predicted probabilities across models for each sample
    variance = np.var(predicted_probabilities, axis=0)
    
    # Determine if the variance is below the threshold (epsilon^2)
    #is_robust = variance[:, 1] < epsilon ** 2
    is_robust = variance < epsilon ** 2
    
    # Calculate the robustness rate
    robustness_rate = np.mean(is_robust)
    return robustness_rate

def estimate_disagreement(predicted_probabilities):
    # Number of models
    m = predicted_probabilities.shape[0]

    predictions = (predicted_probabilities > 0.5).astype(int)
    
    # Calculate the sample mean of predictions for each example
    p_hat = np.mean(predictions, axis=0)
    
    # Calculate the unbiased estimator for disagreement for each example
    mu_hat = 4* (m / (m - 1)) * p_hat * (1 - p_hat)
    
    return mu_hat*100

def disagreement_percentage(disagreement_rates, threshold):
    """
    Calculates the percentage of examples with a disagreement rate greater than threshold.

    :param disagreement_rates: Array or list of disagreement rates.
    :return: Percentage of examples with disagreement rate > threshold
    """
    num_examples = len(disagreement_rates)
    num_high_disagreement = sum(rate >= threshold for rate in disagreement_rates)

    percentage = (num_high_disagreement / num_examples)
    return percentage


def calculate_error_rate(predictions, labels):
    """
    Calculate the error rate per example in predictions across models.
    
    :param predictions: A 2D numpy array where each row contains predictions from a different model.
    :param labels: A 1D numpy array containing the true labels for the examples.
    :return: The error rate per example and the overall error rate.
    """
     # Ensure predictions is a numpy array
    predictions = np.asarray(predictions)

    # Ensure labels is a numpy array
    labels = np.asarray(labels)

    # Count the number of models
    num_models = predictions.shape[0]

    # Calculate the number of models where prediction != label for each example
    incorrect_predictions = predictions != labels.reshape(1, -1)

    # Calculate the error rate per example
    error_rate_per_example = np.mean(incorrect_predictions, axis=0)


    return error_rate_per_example * 100



def iou(array1, array2):
    """
    Compute the Intersection over Union (IoU) based on common indices in two arrays.
    
    Parameters:
    - array1 (np.ndarray): First array of indices.
    - array2 (np.ndarray): Second array of indices.
    
    Returns:
    - float: The IoU of the two arrays.
    """
    # Convert arrays to sets
    set1 = set(array1)
    set2 = set(array2)

    # Calculate intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Compute the IoU
    if len(union) == 0:
        return 0  # To handle the case where there is no union
    else:
        iou = len(intersection) / len(union)
        return iou

def dice(array1, array2):
    """
    Compute the Dice Coefficient of two lists.
    
    Parameters:
    - list1 (list): First list of items.
    - list2 (list): Second list of items.
    
    Returns:
    - float: The Dice Coefficient of the two lists.
    """
    set1 = set(array1)
    set2 = set(array2)
    
    intersection = len(set1.intersection(set2))
    dice = (2 * intersection) / (len(set1) + len(set2))
    return dice

def indices_above_threshold(data, threshold):
    """
    Return the indices of elements in a list or numpy array that are above a given threshold.
    
    Parameters:
    - data (list or np.ndarray): The list or numpy array to search.
    - threshold (float or int): The threshold above which indices should be returned.
    
    Returns:
    - np.ndarray: An array of indices where the corresponding elements are above the threshold.
    """
    # Convert data to a numpy array if it's not already one
    data_array = np.array(data)
    # Find indices where the condition is true
    indices = np.where(data_array >= threshold)[0]

    return indices

def compute_overlap(array1, array2, threshold, overlap_metric="dice"):
    """
    Compute the overlap metric (IoU or Dice) between two arrays after filtering based on a threshold.
    
    Parameters:
    - array1 (list or np.ndarray): First input array.
    - array2 (list or np.ndarray): Second input array.
    - threshold (float): Threshold to filter elements by.
    - metric (str): Metric to compute ('iou' or 'dice').
    
    Returns:
    - float: Computed metric (IoU or Dice) based on the sets of indices above the threshold.
    """
    # Get indices above threshold for both arrays

    indices1 = set(indices_above_threshold(array1, threshold))
    indices2 = set(indices_above_threshold(array2, threshold))
    
    # Calculate the metric based on the selected type
    if overlap_metric.lower() == 'iou':
        return iou(indices1, indices2)
    elif overlap_metric.lower() == 'dice':
        return dice(indices1, indices2)
    else:
        raise ValueError("Unsupported metric specified. Use 'iou' or 'dice'.")


def regret_FPR_FNR(err_true, err_anticipated):
    # Ensure the input arrays are numpy arrays
    err_true = np.array(err_true)
    err_anticipated = np.array(err_anticipated)

    # Calculate False Positives (FP) and False Negatives (FN)
    false_positives = np.where((err_anticipated == 1) & (err_true == 0))[0]
    false_negatives = np.where((err_anticipated == 0) & (err_true == 1))[0]

    # Calculate the rates with checks for division by zero
    #fp_rate = len(false_positives) / (len(false_positives) + len(false_negatives)) if (len(false_positives) + len(false_negatives))  > 0 else 0.0
    #fn_rate = len(false_negatives) / (len(false_positives) + len(false_negatives))  if (len(false_positives) + len(false_negatives))  > 0 else 0.0

    fp_rate = len(false_positives) / len(err_true)
    fn_rate = len(false_negatives) / len(err_true)
    
    return fp_rate, false_positives, fn_rate, false_negatives


