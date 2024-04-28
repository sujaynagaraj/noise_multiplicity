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
