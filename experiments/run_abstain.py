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

parser.add_argument('--n_models', type =int, default=500, help="number of models to train")
parser.add_argument('--noise_type', type=str, default="class_independent", help="specify type of label noise")
parser.add_argument('--max_iter', type =int, default=10000, help="max iterations to check for typical vec")
parser.add_argument('--model_type', type =str, default="LR", help="LR or NN")
parser.add_argument('--dataset', type =str, default="cshock_mimic", help="dataset choice")
parser.add_argument('--epsilon', type =float, default=0.1, help="number of models to train")

# Add a boolean argument that defaults to False, but sets to True when specified
parser.add_argument('--misspecify', type=str, default = "correct" ,help="over or under-estimate T")

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
    misspecify = args.misspecify

    # Parent directory for saving figures
    #parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    parent_dir = "/scratch/hdd001/home/snagaraj/"
    files_path = os.path.join(parent_dir, "results", "abstain", dataset, args.model_type, args.noise_type, args.misspecify)

    if not os.path.exists(files_path):
        os.makedirs(files_path)

    X_train, X_test, y_train, y_test, group_train, group_test = load_dataset_splits(dataset, group = "age")

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    p_y_x_dict =  calculate_prior(y_train, noise_type = noise_type, group=group_train) #Clean prior

    batch_size = 512

    if dataset == "cshock_eicu":
        batch_size = 512
    elif dataset == "lungcancer":
        batch_size = 2048
    else:
        batch_size = 1024


    if noise_type == "class_independent":
        noise_levels = []
        losses = []
        dis_train = []
        amb_train = []
        new_amb_train = []
        probs_train = []

        dis_test = []
        amb_test = []
        new_amb_test = []
        probs_test = []
        draw_ids = []

        for noise_level in [0.05, 0.2,  0.4]:
            
            _, T_true = generate_class_independent_noise(y_train, noise_level) #Fixed noise draw
        
            if misspecify == "over": #Estimate more noise than true T
                T_est = adjust_transition_matrix(T_true, 0.1)
                misspecify_flag = True
            if misspecify == "under": #Estimate less noise than true T
                assert(noise_level!=0)
                T_est = adjust_transition_matrix(T_true, -0.1)
                misspecify_flag = True
            elif misspecify == "correct": #Correct T
                T_est = T_true
                misspecify_flag = False
            else:
                continue

            for seed in range(5):
                u_vec = get_u(y_train, T = T_true, seed= seed, noise_type = noise_type)
                yn_train = flip_labels(y_train, u_vec) #XOR

                (disagreement_train, 
                    disagreement_test, 
                    ambiguity_train, 
                    ambiguity_test, 
                    new_ambiguity_train, 
                    new_ambiguity_test) = run_procedure(n_models, max_iter, X_train, yn_train, X_test, y_test, p_y_x_dict, group_train = None, group_test = None, noise_type = noise_type, model_type = model_type, T = T_est, epsilon = epsilon, misspecify = misspecify_flag)

                
                model, (train_acc,
                                test_acc,
                                train_probs,
                                test_probs,
                                train_loss,
                                test_loss,
                                train_preds,
                                test_preds) = train_model_ours_regret(X_train, yn_train, X_test, y_test, seed = 2024, model_type=model_type)

                noise_levels.append(noise_level)
                losses.append("BCE")
                dis_train.append(disagreement_train)
                amb_train.append(ambiguity_train)
                new_amb_train.append(new_ambiguity_train)
                probs_train.append(train_probs)

                dis_test.append(disagreement_test)
                amb_test.append(ambiguity_test)
                new_amb_test.append(new_ambiguity_test)
                probs_test.append(test_probs)
                draw_ids.append(seed)

                
                for loss in ["backward", "forward"]:
                    model, results = train_model(X_train, y_train, yn_train, X_test, y_test, T_est,  seed=2024, num_epochs=100, batch_size = 256, correction_type=loss, model_type = model_type)
                    
                    train_probs = results[4]
                    test_probs = results[7]

                    noise_levels.append(noise_level)
                    losses.append(loss)
                    dis_train.append(disagreement_train)
                    amb_train.append(ambiguity_train)
                    new_amb_train.append(new_ambiguity_train)
                    probs_train.append(train_probs)

                    dis_test.append(disagreement_test)
                    amb_test.append(ambiguity_test)
                    new_amb_test.append(new_ambiguity_test)
                    probs_test.append(test_probs)
                    draw_ids.append(seed)

        data = {'noise': noise_levels, 'loss': losses, "new_ambiguity_train":new_amb_train, "new_ambiguity_test":new_amb_test, "disagreement_test":dis_test, "ambiguity_train":amb_train,  "ambiguity_test":amb_test, "disagreement_train":dis_train ,"test_probs":probs_test, "train_probs":probs_train, "draw_id":draw_ids }

        path = os.path.join(files_path, f"{epsilon}.pkl")

            # Open a file for writing in binary mode
        with open(path, 'wb') as file:
            # Use pickle to write the dictionary to the file
            pkl.dump(data, file)

        print(timeit.default_timer() - start_time)

    elif noise_type == "class_conditional":
        classes = [0]
        noises = [0.0, 0.1]

        noise_levels = []
        fixed_classes = []
        fixed_noises = []
        losses = []
        dis_train = []
        amb_train = []
        new_amb_train = []
        probs_train = []

        dis_test = []
        amb_test = []
        new_amb_test = []
        probs_test = []
        draw_ids = []

        for fixed_class in classes:
            for i, fixed_noise in enumerate(noises):
                for noise_level in [0.05, 0.2,  0.4]:

                    _, T_true = generate_class_conditional_noise(y_train, noise_level, fixed_class, fixed_noise)


                    if misspecify == "correct": #Estimate more noise than true T
                        T_est = T_true
                        misspecify_flag = False
                    else : 
                        T_est = np.array([[T_true[1, 1], T_true[1, 0]], 
                                        [T_true[0, 1], T_true[0, 0]]]) 
                        misspecify_flag = True

                    for seed in range(5):
                        u_vec = get_u(y_train, T = T_true, seed= seed, noise_type = noise_type)
                        yn_train = flip_labels(y_train, u_vec) #XOR
                        
                        (disagreement_train, 
                        disagreement_test, 
                        ambiguity_train, 
                        ambiguity_test, 
                        new_ambiguity_train, 
                        new_ambiguity_test) = run_procedure(n_models, max_iter, X_train, yn_train, X_test, y_test, p_y_x_dict, group_train = None, group_test = None, noise_type = noise_type, model_type = model_type, T = T_est, epsilon = epsilon, misspecify = misspecify_flag)

                        model, (train_acc,
                                test_acc,
                                train_probs,
                                test_probs,
                                train_loss,
                                test_loss,
                                train_preds,
                                test_preds) = train_model_ours_regret(X_train, yn_train, X_test, y_test, seed = 2024, model_type=model_type)

                        noise_levels.append(noise_level)
                        fixed_classes.append(fixed_class)
                        fixed_noises.append(fixed_noise)
                        losses.append("BCE")
                        dis_train.append(disagreement_train)
                        amb_train.append(ambiguity_train)
                        new_amb_train.append(new_ambiguity_train)
                        probs_train.append(train_probs)

                        dis_test.append(disagreement_test)
                        amb_test.append(ambiguity_test)
                        new_amb_test.append(new_ambiguity_test)
                        probs_test.append(test_probs)
                        draw_ids.append(seed)

                        
                        for loss in ["backward", "forward"]:
                            model, results = train_model(X_train, y_train, yn_train, X_test, y_test, T_est,  seed=2024, num_epochs=100, batch_size = batch_size, correction_type=loss, model_type = model_type)
                            
                            test_probs = results[7]
                            train_probs = results[4]

                            noise_levels.append(noise_level)
                            fixed_classes.append(fixed_class)
                            fixed_noises.append(fixed_noise)
                            losses.append(loss)
                            dis_train.append(disagreement_train)
                            amb_train.append(ambiguity_train)
                            new_amb_train.append(new_ambiguity_train)
                            probs_train.append(train_probs)

                            dis_test.append(disagreement_test)
                            amb_test.append(ambiguity_test)
                            new_amb_test.append(new_ambiguity_test)
                            probs_test.append(test_probs)
                            draw_ids.append(seed)

        data = {'noise': noise_levels, 'loss': losses, "new_ambiguity_train":new_amb_train, "new_ambiguity_test":new_amb_test, "ambiguity_test":amb_test, "ambiguity_train":amb_train,  "disagreement_test":dis_test, "disagreement_train":dis_train, "test_probs":probs_test, "train_probs":probs_train, "fixed_noise":fixed_noises, "fixed_class":fixed_classes, "draw_id":draw_ids}

        path = os.path.join(files_path, f"{epsilon}.pkl")

            # Open a file for writing in binary mode
        with open(path, 'wb') as file:
            # Use pickle to write the dictionary to the file
            pkl.dump(data, file)

        print(timeit.default_timer() - start_time)