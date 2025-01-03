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
import timeit

parser = argparse.ArgumentParser('ambiguity real')  

parser.add_argument('--n_models', type =int, default=100, help="number of models to train")
parser.add_argument('--noise_type', type=str, default="class_independent", help="specify type of label noise")
parser.add_argument('--noise_level', type =float, default=0.2, help="noise level")
parser.add_argument('--max_iter', type =int, default=100000, help="max iterations to check for typical vec")
parser.add_argument('--dataset_size', type =int, default=5000, help="max iterations to check for typical vec")
parser.add_argument('--model_type', type =str, default="LR", help="LR or NN")
parser.add_argument('--uncertainty_type', type =str, default="backward", help="backward or forward uncertainty")
parser.add_argument('--dataset', type =str, default="cshock_mimic", help="dataset choice")
parser.add_argument('--epsilon', type =float, default=0.1, help="number of models to train")

# Add a boolean argument that defaults to False, but sets to True when specified
parser.add_argument('--misspecify', type=str, default = "correct" ,help="over or under-estimate T")


args = parser.parse_args()

#####################################################################################################

if __name__ == '__main__':
    
    start_time = timeit.default_timer()

    print('Starting Ambiguity real')
    print("Noise Type: ", args.noise_type)
    print("Noise Level: ", args.noise_level)
    print("Model Type: ", args.model_type)
    print("Uncertainty Type: ", args.uncertainty_type)
    print("Dataset: ", args.dataset)
    print("Models: ", args.n_models)

    n_models = args.n_models
    max_iter = args.max_iter
    noise_type = args.noise_type
    noise_level = args.noise_level
    model_type = args.model_type
    uncertainty_type = args.uncertainty_type
    dataset_size = args.dataset_size
    dataset = args.dataset
    epsilon = args.epsilon
    misspecify = args.misspecify


    if dataset == "cshock_eicu":
        batch_size = 512
    else:
        batch_size = 1024
    if dataset == "lungcancer" and uncertainty_type == "forward":
        n_models = 100
        batch_size = 2048
   

    # Parent directory for saving figures
    #parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    parent_dir = "/scratch/hdd001/home/snagaraj/"
    files_path = os.path.join(parent_dir, "results", "metrics", dataset, args.model_type, args.noise_type, args.misspecify)

    if not os.path.exists(files_path):
        os.makedirs(files_path)

    X_train, X_test, y_train, y_test, group_train, group_test = load_dataset_splits(dataset, group = "age")

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    p_y_x_dict =  calculate_prior(y_train, noise_type = noise_type, group=group_train) #Clean prior

    if noise_type == "class_independent":
        _, T_true = generate_class_independent_noise(y_train, noise_level) #Fixed noise draw
        
        if misspecify == "over": #Estimate more noise than true T
            T_est = adjust_transition_matrix(T_true, 0.1)

        if misspecify == "under": #Estimate less noise than true T
            assert(noise_level!=0)
            T_est = adjust_transition_matrix(T_true, -0.1)

            
        elif misspecify == "correct": #Correct T
            T_est = T_true 

        # Generate error rates
        metrics = simulate_noise_and_train_model(  n_models, 
                                                    max_iter, 
                                                    X_train, 
                                                    y_train,
                                                    X_test, 
                                                    y_test,
                                                    p_y_x_dict, 
                                                    T_true = T_true,
                                                    T_est = T_est,
                                                    noise_type = noise_type, 
                                                    model_type = model_type, 
                                                    uncertainty_type=uncertainty_type,
                                                    batch_size = batch_size,
                                                    epsilon = epsilon)
        
        path = os.path.join(files_path, f"{uncertainty_type}_{noise_level}_{epsilon}_metrics.pkl")

        # Open a file for writing in binary mode
        with open(path, 'wb') as file:
            # Use pickle to write the dictionary to the file
            pkl.dump(metrics, file)
        print(timeit.default_timer() - start_time)

    elif noise_type == "class_conditional":
        fixed_classes = [0]
        fixed_noises = [0.0, 0.1]

        for fixed_class in fixed_classes:
            for i, fixed_noise in enumerate(fixed_noises):
                
                _, T_true = generate_class_conditional_noise(y_train, noise_level, fixed_class, fixed_noise)


                if misspecify == "correct": #Estimate more noise than true T
                    T_est = T_true
                    
                else : 
                    T_est = np.array([[T[1, 1], T[1, 0]], 
                                    [T[0, 1], T[0, 0]]]) 


                # Generate error rates
                metrics = simulate_noise_and_train_model(   n_models, 
                                                            max_iter, 
                                                            X_train,
                                                            y_train,
                                                            X_test, 
                                                            y_test,
                                                            p_y_x_dict,
                                                            T_true = T_true,
                                                            T_est = T_est,
                                                            noise_type = noise_type, 
                                                            model_type = model_type, 
                                                            uncertainty_type=uncertainty_type,  
                                                            fixed_class=fixed_class, 
                                                            fixed_noise=fixed_noise,
                                                            batch_size = batch_size,
                                                            epsilon = epsilon)
            
                

                path = os.path.join(files_path, f"{uncertainty_type}_{noise_level}_{fixed_class}_{fixed_noise}_{epsilon}_metrics.pkl")

                        # Open a file for writing in binary mode
                with open(path, 'wb') as file:
                    # Use pickle to write the dictionary to the file
                    pkl.dump(metrics, file)
                print(timeit.default_timer() - start_time)

elif noise_type == "group" and dataset not in ["support", "saps"]:

    T_dicts = {"case1": {0: np.array([[0.8,0.2],
                            [0.2,0.8]]),
                1: np.array([[0.6,0.4],
                            [0.4,0.6]])}, 
                "case2": { 0: np.array([[0.6,0.4],
                                [0.4,0.6]]),
                    1: np.array([[0.8,0.2],
                                [0.2,0.8]])},
                "case3": { 0: np.array([[0.8,0.2],
                                [0.4,0.6]]),
                    1: np.array([[0.4,0.6],
                                [0.8,0.2]])},              
                "case4": { 0: np.array([[0.8,0.2],
                                [0.4,0.6]]),
                    1: np.array([[0.4,0.6],
                                [0.8,0.2]])}               
    }
    
    for case, T_dict in T_dicts.items():

        yn_train, T_true = generate_group_noise(y_train, group_train, T_dict)

        if uncertainty_type == "forward":
            y_vec = y_train
        else: # backward
            y_vec = yn_train

        if misspecify == True: #Misspecified T
            pass #TODO
        else: #Correct T
            T_est = T_true

        # Generate error rates
        metrics = simulate_noise_and_train_model(   n_models, 
                                                    max_iter, 
                                                    X_train,
                                                    y_train, 
                                                    y_vec, 
                                                    X_test, 
                                                    y_test,
                                                    p_y_x_dict,
                                                    T_true = T_true,
                                                    T_est = T_est,
                                                    group_train = group_train,
                                                    group_test = group_test,
                                                    noise_type = noise_type, 
                                                    model_type = model_type, 
                                                    uncertainty_type=uncertainty_type, 
                                                    T_dict = T_dict,
                                                    batch_size = batch_size,
                                                    epsilon = epsilon)
    

        path = os.path.join(files_path, f"{case}_metrics.pkl")

                # Open a file for writing in binary mode
        with open(path, 'wb') as file:
            # Use pickle to write the dictionary to the file
            pkl.dump(metrics, file)


    print(timeit.default_timer() - start_time)