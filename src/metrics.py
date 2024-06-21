import numpy as np

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

    # Calculate the number of true negatives and true positives
    true_negatives = np.sum(1 - err_true)
    true_positives = np.sum(err_true)

    # Calculate the rates with checks for division by zero
    fp_rate = len(false_positives) / true_negatives if true_negatives > 0 else 0
    fn_rate = len(false_negatives) / true_positives if true_positives > 0 else 0

    return fp_rate, false_positives, fn_rate, false_negatives


