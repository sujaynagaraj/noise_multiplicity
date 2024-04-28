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

from folktables import ACSDataSource, ACSEmployment

class MetricsStorage:
    def __init__(self, loss_types, noise_levels):
        self.data = {loss: {noise_level: {metric: [] for metric in self.metrics()}
                            for noise_level in noise_levels} for loss in loss_types}

    def metrics(self):
        return ['regret_train', 'disagreement_train',  'regret_test', 'disagreement_test',  
                'noisy_train_loss', 'clean_train_loss', 'clean_test_loss',
                'noisy_train_acc', 'clean_train_acc', 'clean_test_acc' ]

    def add_metric(self, loss, noise_level, metric, value):
        self.data[loss][noise_level][metric].append(value)

    def get_metric(self, loss, noise_level, metric):
        return self.data[loss][noise_level][metric]

# Usage
#loss_types = ["loss1", "loss2"]
#noise_levels = ["low", "medium", "high"]
#metrics_store = MetricsStore(loss_types, noise_levels)
#metrics_store.add_metric("loss1", "high", "error_train_rate", 0.02)


def simulate_noise_and_train_model(noise_levels, m, max_iter, d, X_train, y_train, X_test, y_test, p_y_x = np.array([0.5,0.5]), noise_type = "class_independent", uncertainty_type="backward",  model_type = "LR" , fixed_class=0, fixed_noise=0.2, feature_weights=None):
    
    if uncertainty_type == "backward":
        loss_types = ["BCE"]

    else: # forward
        #loss_types = ["BCE", "backward", "forward"]
        loss_types = ["BCE"]
    
    metrics = MetricsStorage(loss_types, noise_levels)
    

    for base_seed, flip_p in (enumerate(tqdm(noise_levels))):

        if noise_type == "class_independent":
            yn_train, noise_transition_matrix = generate_class_independent_noise(y_train, flip_p)
        elif noise_type == "class_conditional":
            yn_train, noise_transition_matrix = generate_class_conditional_noise(y_train, flip_p, fixed_class, fixed_noise)
        elif noise_type == "instance_dependent":
            yn_train, noise_transition_dict = generate_instance_dependent_noise(y_train, X_train, flip_p, feature_weights)

        
        if uncertainty_type == "backward":
            y_vec = yn_train
        else: # forward
            y_vec = y_train


        preds_train_dict = {loss: [] for loss in loss_types}
        preds_test_dict = {loss: [] for loss in loss_types}

        typical_count = 0

        for seed in (range(1, max_iter+1)):

            if uncertainty_type == "backward":
                u_vec = infer_u(y_vec, noise_transition_matrix, p_y_x, seed=base_seed + seed)

            else:
                # Using a forward model, so get u directly
                u_vec = get_u(y_vec, noise_transition_matrix, seed=base_seed + seed)

            if noise_type == "instance_dependent":
                bool_flag = True
                for instance in np.unique(X_train, axis=0):
                    indices = [idx for idx, elem in enumerate(X_train) if np.array_equal(elem, instance)]

                    if not is_typical(u_vec[indices], noise_transition_matrix, y_vec[indices], p_y_x, noise_type=noise_type, uncertainty_type=uncertainty_type):
                        bool_flag = False
                        break
                if not bool_flag:
                    continue

            else:

                if not is_typical(u_vec, noise_transition_matrix, y_vec, p_y_x, noise_type=noise_type, uncertainty_type=uncertainty_type):
                    
                    continue

            flipped_labels = flip_labels(y_vec, u_vec)
            
            for loss in loss_types:
                model,  (noisy_train_loss,
                        clean_train_loss, 
                        noisy_train_acc,
                        clean_train_acc,
                        train_probs,
                        clean_test_loss, 
                        clean_test_acc,
                        test_probs
                        ) = train_model(X_train, y_train, flipped_labels, X_test, y_test, seed=base_seed + seed, num_epochs=50, batch_size=32, model_type = model_type, correction_type=loss, noise_transition_matrix=noise_transition_matrix)
                
                preds_train = (train_probs > 0.5).astype(int)
                preds_test = (test_probs > 0.5).astype(int)

                preds_train_dict[loss].append(preds_train)
                preds_test_dict[loss].append(preds_test)

                metrics.add_metric(loss, flip_p, "noisy_train_loss", noisy_train_loss)
                metrics.add_metric(loss, flip_p, "clean_train_loss", clean_train_loss)
                metrics.add_metric(loss, flip_p, "noisy_train_acc", noisy_train_acc*100)
                metrics.add_metric(loss, flip_p, "clean_train_acc", clean_train_acc*100)
                metrics.add_metric(loss, flip_p, "clean_test_loss", clean_test_loss)
                metrics.add_metric(loss, flip_p, "clean_test_acc", clean_test_acc*100)
            
            typical_count += 1

            if typical_count == m:
                break
        
        for loss in loss_types:
            predictions_train = np.array(preds_train_dict[loss])
            print(predictions_train)
            predictions_test = np.array(preds_test_dict[loss])

            try:
                regret_train = calculate_error_rate(predictions_train, y_train)
                disagreement_train = estimate_disagreement(predictions_train)

                regret_test = calculate_error_rate(predictions_test, y_test)
                disagreement_test = estimate_disagreement(predictions_test)

            except:
                print("Error: Could not get Disagreement Metrics")
                continue

            for i, item in enumerate(X_test):
                 metrics.add_metric(loss, flip_p, "regret_train", regret_train[i])
                 metrics.add_metric(loss, flip_p, "disagreement_train", disagreement_train[i])

            for i, item in enumerate(X_test):
               
                metrics.add_metric(loss, flip_p, "regret_test", regret_test[i])
                metrics.add_metric(loss, flip_p, "disagreement_test", disagreement_test[i])

                #error_rates[loss][flip_p].append(error_rate[i])
                #disagreement_rates[loss][flip_p].append(disagreement_rate[i])

        print("Typical Rate: ", typical_count / seed)

    return metrics



