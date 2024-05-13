import sys
sys.path.insert(0,'..')

from src.models import *
from src.loss_functions import *
from src.noise import *
from src.metrics import *
from src.plotting import *
from src.generate_data import *
from src.toy_data import *

import sklearn
import pandas as pd

from scipy.stats import bernoulli

from operator import xor

class MetricsStorage:
    def __init__(self, loss_types):
        self.data = {loss:  {metric: [] for metric in self.metrics()} for loss in loss_types}

    def metrics(self):
        return ['regret_train', 'disagreement_train',  'regret_test', 'disagreement_test',  
                'noisy_train_loss', 'clean_train_loss', 'clean_test_loss',
                'noisy_train_acc', 'clean_train_acc', 'clean_test_acc' , 
                'train_loss', 'test_loss', 'train_acc', 'test_acc', 'flip_frequency', 'typical_rate', 'typical_difference',
                "preds_train", "preds_test"]

    def add_metric(self, loss, metric, value):
        self.data[loss][metric].append(value)

    def get_metric(self, loss, metric):
        return self.data[loss][metric]


def dummy_T_dict(group_train, T):
    T_dict = {}
    for key in np.unique(group_train):
        T_dict[key] = T
    return T_dict

def simulate_noise_and_train_model(m, max_iter, X_train, y_train, X_test, y_test, p_y_x_dict, noise_type = "class_independent", uncertainty_type="backward",  model_type = "LR" , fixed_class=0, fixed_noise=0.2, T_true = None, T_est = None, batch_size = 512, base_seed = 2024, epsilon = 0.25):
    
    if uncertainty_type == "forward":
        loss_types = ["BCE", "backward", "forward"]
        y_vec = y_train
    else: # backward
        loss_types = ["BCE"]

        #Initial Noise Draw
        u_vec = get_u(y_train, T = T_true, seed= base_seed, noise_type = noise_type)
        y_vec = flip_labels(y_train, u_vec) #XOR

    metrics = MetricsStorage(loss_types)

    preds_train_dict = {loss: [] for loss in loss_types}
    preds_test_dict = {loss: [] for loss in loss_types}

    typical_count = 0

    
    for seed in tqdm(range(1, max_iter+1)):
        if uncertainty_type == "forward":
            # Using a forward model, so get u directly
            u_vec = get_u(y_vec, T = T_true, seed= seed, noise_type = noise_type)
        else:
            u_vec = infer_u(y_vec, noise_type = noise_type, p_y_x_dict = p_y_x_dict,  T = T_est , seed=seed)

        typical_flag, difference = is_typical(u_vec, p_y_x_dict,  T = T_est, y_vec = y_vec, noise_type = noise_type, uncertainty_type = uncertainty_type, epsilon = epsilon)

        if not typical_flag: 
            continue

        flipped_labels = flip_labels(y_vec, u_vec)

        if uncertainty_type == "forward":
            for loss in loss_types:
                model,  (noisy_train_loss,
                        clean_train_loss, 
                        noisy_train_acc,
                        clean_train_acc,
                        train_probs,
                        clean_test_loss, 
                        clean_test_acc,
                        test_probs
                        ) = train_model(X_train, y_train, flipped_labels,  X_test, y_test,  T = T_est, seed=2024, num_epochs=25, batch_size=batch_size, model_type = model_type, correction_type=loss)

                preds_train = (train_probs > 0.5).astype(int)
                preds_test = (test_probs > 0.5).astype(int)

                preds_train_dict[loss].append(preds_train)
                preds_test_dict[loss].append(preds_test)

                metrics.add_metric(loss, "noisy_train_loss", noisy_train_loss)
                metrics.add_metric(loss, "clean_train_loss", clean_train_loss)
                metrics.add_metric(loss, "noisy_train_acc", noisy_train_acc*100)
                metrics.add_metric(loss, "clean_train_acc", clean_train_acc*100)
                metrics.add_metric(loss, "clean_test_loss", clean_test_loss)
                metrics.add_metric(loss, "clean_test_acc", clean_test_acc*100)
                metrics.add_metric(loss, "flip_frequency", sum(u_vec)/len(u_vec))
                metrics.add_metric(loss, "typical_difference", difference)
                metrics.add_metric(loss, "preds_train", preds_train)
                metrics.add_metric(loss, "preds_test", preds_test)

        else: #backward_sk
            for loss in loss_types:
                model,  (train_acc,
                        test_acc,
                        train_probs,
                        test_probs,
                        train_loss,
                        test_loss,
                        train_preds,
                        test_preds
                        ) = train_model_ours(X_train, flipped_labels, X_test, y_test, seed = 2024, model_type="LR")

                preds_train_dict[loss].append(train_preds)
                preds_test_dict[loss].append(test_preds)

                metrics.add_metric(loss, "train_loss", train_loss)
                metrics.add_metric(loss, "train_acc", train_acc*100)
                metrics.add_metric(loss, "test_loss", test_loss)
                metrics.add_metric(loss, "test_acc", test_acc*100)
                metrics.add_metric(loss, "typical_difference", difference)
                metrics.add_metric(loss, "preds_train", train_preds)
                metrics.add_metric(loss, "preds_test", test_preds)

        typical_count += 1

        if typical_count == m:
            break

    for loss in loss_types:
        typical_rate = typical_count / seed
        print("Typical Rate: ", typical_rate)
        metrics.add_metric(loss, "typical_rate", typical_rate)

        predictions_train = np.array(preds_train_dict[loss])

        predictions_test = np.array(preds_test_dict[loss])

        try:
            regret_train = calculate_error_rate(predictions_train, y_train)
            disagreement_train = estimate_disagreement(predictions_train)

            regret_test = calculate_error_rate(predictions_test, y_test)
            disagreement_test = estimate_disagreement(predictions_test)

        except:
            print("Error: Could not get Disagreement Metrics")
            continue

        for i, item in enumerate(X_train):
            metrics.add_metric(loss, "regret_train", regret_train[i])
            metrics.add_metric(loss, "disagreement_train", disagreement_train[i])

        for i, item in enumerate(X_test):

            metrics.add_metric(loss, "regret_test", regret_test[i])
            metrics.add_metric(loss, "disagreement_test", disagreement_test[i])

    print("DONE")
    return metrics

def abstain(rates, threshold):
    rates = np.where(rates > 100, 100, rates)
    rates = np.where(rates < 0, 0, rates)
    return ((rates >= threshold)).astype(int)

def run_procedure(m, max_iter, X_train, yn_train, X_test, y_test, p_y_x_dict, group_train = None, group_test = None, noise_type = "class_independent", model_type = "LR", T = None, epsilon = 0.25):
    
    typical_count = 0
    preds_test = []
    
    y_vec = yn_train
    
    for seed in tqdm(range(1, max_iter+1)):
        
        u_vec = infer_u(y_vec, group = group_train, noise_type = noise_type, p_y_x_dict = p_y_x_dict,  T = T , seed=seed)
        
        typical_flag, _ = is_typical(u_vec, p_y_x_dict, group = group_train,  T = T, y_vec = y_vec, noise_type = noise_type, uncertainty_type = "backward", epsilon = epsilon)

        if not typical_flag:
            continue
            
        flipped_labels = flip_labels(y_vec, u_vec)
        
        model,  (train_acc,
                test_acc,
                train_probs,
                test_probs,
                train_loss,
                test_loss,
                train_preds,
                test_preds
                ) = train_model_ours(X_train, flipped_labels, X_test, y_test, seed = 2024, model_type="LR")
        
        preds_test.append(test_preds)

        typical_count += 1

        if typical_count == m:
            break
            

    predictions_test = np.array(preds_test)
    disagreement_test = estimate_disagreement(predictions_test)


    return disagreement_test

def train_model_abstain(X_train, y_train, X_test, y_test, model_type="LR"):
    # Set random seed for reproducibility

    seed = 2024
    np.random.seed(seed)
    
    # Choose the model based on the input
    if model_type == "LR":
        model = LR(**DEFAULT_LR_PARAMS, random_state = seed)
    elif model_type == "SVM":
        model = LinearSVC(**DEFAULT_SVM_PARAMS, random_state = seed)
    elif model_type == "NN":
        model = MLPClassifier(**DEFAULT_NN_PARAMS, random_state = seed)
    else:
        raise ValueError("Unsupported model type. Choose 'LR' or 'SVM'.")

    # Train the model using noisy labels (simulating the impact of label noise)
    model.fit(X_train, y_train)

    # Predictions for training and test sets
    test_preds = model.predict(X_test)

    
 
    return model, test_preds

