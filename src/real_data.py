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
                "preds_train", "preds_test", "train_probs", "test_probs"]

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
        loss_types = ["Ours", "BCE", "backward", "forward"]
        y_vec = y_train
    else: # backward
        loss_types = ["Ours"]

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
                if loss == "Ours":
                    model,  (train_acc,
                        test_acc,
                        train_probs,
                        test_probs,
                        train_loss,
                        test_loss,
                        train_preds,
                        test_preds
                        ) = train_model_ours(X_train, flipped_labels, X_test, y_test, seed = 2024, model_type=model_type)

                    preds_train_dict[loss].append(train_preds)
                    preds_test_dict[loss].append(test_preds)

                    metrics.add_metric(loss, "noisy_train_loss", train_loss)
                    metrics.add_metric(loss, "noisy_train_acc", train_acc*100)
                    metrics.add_metric(loss, "clean_test_loss", test_loss)
                    metrics.add_metric(loss, "clean_test_acc", test_acc*100)
                    metrics.add_metric(loss, "typical_difference", difference)
                    metrics.add_metric(loss, "preds_train", train_preds)
                    metrics.add_metric(loss, "preds_test", test_preds)
                    metrics.add_metric(loss, "train_probs", train_probs)
                    metrics.add_metric(loss, "test_probs", test_probs)

                else:
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
                    metrics.add_metric(loss, "train_probs", train_probs)
                    metrics.add_metric(loss, "test_probs", test_probs)

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
                        ) = train_model_ours(X_train, flipped_labels, X_test, y_test, seed = 2024, model_type=model_type)

                preds_train_dict[loss].append(train_preds)
                preds_test_dict[loss].append(test_preds)

                metrics.add_metric(loss, "train_loss", train_loss)
                metrics.add_metric(loss, "train_acc", train_acc*100)
                metrics.add_metric(loss, "test_loss", test_loss)
                metrics.add_metric(loss, "test_acc", test_acc*100)
                metrics.add_metric(loss, "typical_difference", difference)
                metrics.add_metric(loss, "preds_train", train_preds)
                metrics.add_metric(loss, "preds_test", test_preds)
                metrics.add_metric(loss, "train_probs", train_probs)
                metrics.add_metric(loss, "test_probs", test_probs)

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
            regret_train = calculate_error_rate(predictions_train, flipped_labels)
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
    #rates = np.clip(rates, 0, 100)
    return ((rates > threshold)).astype(int)


def run_procedure(m, max_iter, X_train, yn_train, X_test, y_test, p_y_x_dict, group_train = None, group_test = None, noise_type = "class_independent", model_type = "LR", T = None, epsilon = 0.25, misspecify = False):
    
    typical_count = 0
    preds_test = []
    preds_train = []
    errors_train = []
    errors_test = []
    
    y_vec = yn_train
    
    for seed in tqdm(range(1, max_iter+1)):
        
        u_vec = infer_u(y_vec, group = group_train, noise_type = noise_type, p_y_x_dict = p_y_x_dict,  T = T , seed=seed)
        
        typical_flag, _ = is_typical(u_vec, p_y_x_dict, group = group_train,  T = T, y_vec = y_vec, noise_type = noise_type, uncertainty_type = "backward", epsilon = epsilon)
        
        
        if misspecify or noise_type == "group":
            typical_flag = True
            
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
        preds_train.append(train_preds)

        error_train = train_preds != flipped_labels
        error_test = test_preds != y_test

        errors_test.append(error_test)
        errors_train.append(error_train)

        typical_count += 1

        if typical_count == m:
            break
            
    predictions_test = np.array(preds_test)
    disagreement_test = estimate_disagreement(predictions_test)
    ambiguity_test = calculate_error_rate(predictions_test, y_test)
    new_ambiguity_test = np.mean(errors_test, axis=0)*100

    predictions_train = np.array(preds_train)
    disagreement_train = estimate_disagreement(predictions_train)
    ambiguity_train = calculate_error_rate(predictions_train, y_vec)
    new_ambiguity_train = np.mean(errors_train, axis=0)*100

    return disagreement_train, disagreement_test, ambiguity_train,  ambiguity_test, new_ambiguity_train, new_ambiguity_test
    
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
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    
 
    return model, train_preds, test_preds



def run_procedure_regret(m, max_iter, X_train, yn_train, X_test, y_test,  p_y_x_dict,  noise_type = "class_independent", model_type = "LR", T = None, epsilon = 0.1):
    
    typical_count = 0
    preds_train = []
    preds_test = []
    
    y_vec = yn_train
    
    for seed in tqdm(range(1, max_iter+1)):
        
        u_vec = infer_u(y_vec,  noise_type = noise_type, p_y_x_dict = p_y_x_dict,  T = T , seed=seed)
        
        typical_flag, _ = is_typical(u_vec, p_y_x_dict,   T = T, y_vec = y_vec, noise_type = noise_type, uncertainty_type = "backward", epsilon = epsilon)

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
                test_preds)= train_model_ours_regret(X_train, flipped_labels, X_test, y_test, seed = 2024, model_type="LR")
        
        preds_train.append(train_preds)
        preds_test.append(test_preds)

        typical_count += 1

        if typical_count == m:
            break
            
    predictions_train = np.array(preds_train)
    disagreement_train = estimate_disagreement(predictions_train)
    ambiguity_train = calculate_error_rate(predictions_train, y_vec)

    predictions_test = np.array(preds_test)
    disagreement_test = estimate_disagreement(predictions_test)
    ambiguity_test = calculate_error_rate(predictions_test, y_test)

    return predictions_train, predictions_test, disagreement_train, disagreement_test, ambiguity_train, ambiguity_test


class Metrics:
    def __init__(self):
        self.metrics = {}

    def add_metric(self, method, draw_id, metric_name, value):
        if method not in self.metrics:
            self.metrics[method] = {}
        if draw_id not in self.metrics[method]:
            self.metrics[method][draw_id] = {}
        self.metrics[method][draw_id][metric_name] = value

    def get_metric(self, method, draw_id, metric_name):
        return self.metrics.get(method, {}).get(draw_id, {}).get(metric_name, None)

    def get_all_metrics(self, method, draw_id):
        return self.metrics.get(method, {}).get(draw_id, {})

class Vectors:
    def __init__(self):
        self.vectors = {}

    def add_vector(self, method, draw_id, vector_name, value):
        if method not in self.vectors:
            self.vectors[method] = {}
        if draw_id not in self.vectors[method]:
            self.vectors[method][draw_id] = {}
        self.vectors[method][draw_id][vector_name] = value

    def get_vector(self, method, draw_id, vector_name):
        return self.vectors.get(method, {}).get(draw_id, {}).get(vector_name, None)

    def get_all_vectors(self, method, draw_id):
        return self.vectors.get(method, {}).get(draw_id, {})



def run_experiment(dataset, noise_type, model_type, n_models, max_iter, T, training_loss="None", n_draws=5, batch_size=256):
    X_train, X_test, y_train, y_test, group_train, group_test = load_dataset_splits(dataset, group="age")

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    p_y_x_dict = calculate_prior(y_train, noise_type=noise_type, group = group_train)  # Clean prior

    vectors = Vectors()

    for draw_id in range(n_draws):
        u_vec = get_u(y_train, T=T, seed=draw_id, noise_type=noise_type)
        yn_train = flip_labels(y_train, u_vec)

        model, (train_preds, test_preds,
                train_probs, test_probs) = train_model_regret_torch(
            X_train, yn_train, y_train, X_test, y_test, T,
            seed=2024, num_epochs=50, batch_size=batch_size, correction_type=training_loss, model_type=model_type)

        # True Population Error
        pop_err_true_train, instance_err_true_train = instance_01loss(y_train, train_preds)
        pop_err_true_test, instance_err_true_test = instance_01loss(y_test, test_preds)


        (disagreement_train, 
        disagreement_test, 
        ambiguity_train, 
        ambiguity_test, 
        new_ambiguity_train, 
        new_ambiguity_test) = run_procedure(n_models, 
                                            max_iter, 
                                            X_train, 
                                            yn_train, 
                                            X_test, 
                                            y_test, 
                                            p_y_x_dict, 
                                            group_train = None, 
                                            group_test = None, 
                                            noise_type = noise_type, 
                                            model_type = model_type, 
                                            T = T, 
                                            epsilon = 0.1, 
                                            misspecify = "correct")
        
        vectors.add_vector("metadata", draw_id, "dataset", dataset)
        vectors.add_vector("metadata", draw_id, "noise_type", noise_type)
        vectors.add_vector("metadata", draw_id, "model_type", model_type)
        vectors.add_vector("metadata", draw_id, "n_models", n_models)
        vectors.add_vector("metadata", draw_id, "max_iter", max_iter)
        vectors.add_vector("metadata", draw_id, "training_loss", training_loss)
        vectors.add_vector("metadata", draw_id, "n_draws", n_draws)
        vectors.add_vector("metadata", draw_id, "T", T)
        vectors.add_vector("metadata", draw_id, "y_train", y_train)
        vectors.add_vector("metadata", draw_id, "y_test", y_test)
        vectors.add_vector("metadata", draw_id, "train_preds", train_preds)
        vectors.add_vector("metadata", draw_id, "train_probs", train_probs)
        vectors.add_vector("metadata", draw_id, "test_preds", test_preds)
        vectors.add_vector("metadata", draw_id, "test_probs", test_probs)
        vectors.add_vector("metadata", draw_id, "yn_train", yn_train)
        vectors.add_vector("metadata", draw_id, "instance_err_true_train", instance_err_true_train)
        vectors.add_vector("metadata", draw_id, "instance_err_true_test", instance_err_true_test)
        vectors.add_vector("metadata", draw_id, "train_ambiguity", new_ambiguity_train)
        vectors.add_vector("metadata", draw_id, "test_ambiguity", new_ambiguity_test)

    return vectors
