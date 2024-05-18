import os
import time
import sys
import pickle

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0,'..')

import argparse
from random import SystemRandom


from src.models import *
from src.loss_functions import *
from src.noise import *
from src.metrics import *
from src.plotting import *
from src.generate_data import *
from src.helper import *
from src.real_data import *

from operator import xor

import pickle as pkl

parser = argparse.ArgumentParser('abstain')

parser.add_argument('--n_models', type =int, default=100, help="number of models to train")
parser.add_argument('--noise_type', type=str, default="class_independent", help="specify type of label noise")
parser.add_argument('--max_iter', type =int, default=10000, help="max iterations to check for typical vec")
parser.add_argument('--model_type', type =str, default="LR", help="LR or NN")
parser.add_argument('--dataset', type =str, default="cshock_mimic", help="dataset choice")
parser.add_argument('--epsilon', type =float, default=0.1, help="number of models to train")

args = parser.parse_args()

#####################################################################################################

if __name__ == '__main__':
    start_time = timeit.default_timer()

    print('Starting Abstention')
    print("Noise Type: ", args.noise_type)
    print("Model Type: ", args.model_type)
    print("Dataset: ", args.dataset)

    n_models = args.n_models
    max_iter = args.max_iter
    noise_type = args.noise_type
    model_type = args.model_type
    dataset = args.dataset
    epsilon = args.epsilon

    # Parent directory for saving figures
    #parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    parent_dir = "/scratch/hdd001/home/snagaraj/"
    files_path = os.path.join(parent_dir, "results", "abstain", dataset, args.model_type, args.noise_type)

    if not os.path.exists(files_path):
        os.makedirs(files_path)

    X_train, X_test, y_train, y_test, group_train, group_test = load_dataset_splits(dataset, group = "age")

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    p_y_x_dict =  calculate_prior(y_train, noise_type = noise_type, group=group_train) #Clean prior

    batch_size = 512

    if dataset == "lungcancer":
        n_models = 100
        batch_size = 2048


    if noise_type == "class_independent":
        noise_levels = []
        losses = []
        dis = []
        preds = []

        for noise_level in [0.1, 0.2, 0.3, 0.4, 0.45]:
            
            _, T = generate_class_independent_noise(y_train, noise_level)

            u_vec = get_u(y_train, T = T, seed= 2024, noise_type = noise_type)
            yn_train = flip_labels(y_train, u_vec) #XOR

            disagreement_test = run_procedure(n_models, max_iter, X_train, yn_train, X_test, y_test, p_y_x_dict, group_train = None, group_test = None, noise_type = noise_type, model_type = model_type, T = T, epsilon = epsilon)

            model, test_preds = train_model_abstain(X_train, yn_train, X_test, y_test, model_type=model_type)

            noise_levels.append(noise_level)
            losses.append("BCE")
            dis.append(disagreement_test)
            preds.append(test_preds)

            
            for loss in ["backward", "forward"]:
                model, results = train_model(X_train, y_train, yn_train, X_test, y_test, T,  seed=2024, num_epochs=100, batch_size = 256, correction_type=loss, model_type = model_type)
                
                test_preds = results[-1]

                noise_levels.append(noise_level)
                losses.append(loss)
                dis.append(disagreement_test)
                preds.append(test_preds)

        data = {'noise': noise_levels, 'loss': losses, "disagreement_test":dis, "test_preds":preds}

        path = os.path.join(files_path, f"{epsilon}.pkl")

            # Open a file for writing in binary mode
        with open(path, 'wb') as file:
            # Use pickle to write the dictionary to the file
            pkl.dump(data, file)

        print(timeit.default_timer() - start_time)

    elif noise_type == "class_conditional":

        fixed_classes = [0]
        fixed_noises = [0.0, 0.1]

        noise_levels = []
        fixed_classes = []
        fixed_noises = []
        losses = []
        dis = []
        preds = []

        for fixed_class in fixed_classes:
            for i, fixed_noise in enumerate(fixed_noises):
                

                for noise_level in [0.1, 0.2, 0.3, 0.4, 0.45]:

                    
                    _, T_true = generate_class_conditional_noise(y_train, noise_level, fixed_class, fixed_noise)

                    u_vec = get_u(y_train, T = T, seed= 2024, noise_type = noise_type)
                    yn_train = flip_labels(y_train, u_vec) #XOR

                    disagreement_test = run_procedure(n_models, max_iter, X_train, yn_train, X_test, y_test, p_y_x_dict, group_train = None, group_test = None, noise_type = noise_type, model_type = model_type, T = T, epsilon = epsilon)

                    model, test_preds = train_model_abstain(X_train, yn_train, X_test, y_test, model_type=model_type)

                    noise_levels.append(noise_level)
                    fixed_classes.append(fixed_class)
                    fixed_noises.append(fixed_noise)
                    losses.append("BCE")
                    dis.append(disagreement_test)
                    preds.append(test_preds)

                    
                    for loss in ["backward", "forward"]:
                        model, results = train_model(X_train, y_train, yn_train, X_test, y_test, T,  seed=2024, num_epochs=100, batch_size = batch_size, correction_type=loss, model_type = model_type)
                        
                        test_preds = results[-1]

                        noise_levels.append(noise_level)
                        fixed_classes.append(fixed_class)
                        fixed_noises.append(fixed_noise)
                        losses.append(loss)
                        dis.append(disagreement_test)
                        preds.append(test_preds)

        data = {'noise': noise_levels, 'loss': losses, "disagreement_test":dis, "test_preds":preds, "fixed_noise":fixed_noises, "fixed_class":fixed_classes}

        path = os.path.join(files_path, f"{epsilon}.pkl")

            # Open a file for writing in binary mode
        with open(path, 'wb') as file:
            # Use pickle to write the dictionary to the file
            pkl.dump(data, file)

        print(timeit.default_timer() - start_time)