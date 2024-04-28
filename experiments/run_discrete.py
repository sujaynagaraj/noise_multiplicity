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


parser = argparse.ArgumentParser('discrete noise')

parser.add_argument('--dataset', type=str, default='MNIST', help="Dataset to Use")
parser.add_argument('--n_samples', type =int, default=1000, help="number of samples in data")
parser.add_argument('--n_models', type =int, default=1000, help="number of models to train")
parser.add_argument('--batch_size', type=int, default=256, help="specify batch size")
parser.add_argument('--epochs', type=int, default=50, help="specify num epochs")
parser.add_argument('--variance_type', type=str, default="model", help="specify if variance comes from model or noise")
parser.add_argument('--noise_type', type=str, default="class_independent", help="specify type of label noise")

args = parser.parse_args()

#####################################################################################################

if __name__ == '__main__':

    print('Starting Simple Experiment')
    print("Dataset: ", args.dataset)
    print("N Samples: ", args.n_samples)
    print("Variance Type: ", args.variance_type)

    dataset = args.dataset
    n_samples = args.n_samples
    n_models = args.n_models
    variance_type = args.variance_type
    epochs = args.epochs

    

    ##################################################################
    filename = generate_filename(dataset, n_samples)
    path = "/h/snagaraj/noise_multiplicity/data/processed/"

    X_train, X_test, y_train, y_test = load_data(filename, path)

    ##################################################################
    

    #Run Experiments
    for flip_p in ([0.05, 0.2, 0.4]):

        start = time.time()

        noise_transition_matrix = np.array([[1-flip_p, flip_p], [flip_p, 1-flip_p]])
        y_train_noisy = add_label_noise(y_train, noise_transition_matrix)

        correction_types = [("NONE", "Noisy Labels - Uncorrected"), 
                ("forward", "Noisy Labels - Forward"), 
                ("backward", "Noisy Labels - Backward"), 
                ("CLEAN", "Clean Labels")]

        probabilities_dict = {}
        accuracies_dict = {}

        for correction_type, label in correction_types:
            if variance_type == "model":
                y_train_used = y_train_noisy if "Noisy" in label else y_train
                predicted_probabilities, accuracies = train_LR_model_variance(X_train, 
                                                                            y_train_used, 
                                                                            X_test, 
                                                                            y_test, 
                                                                            num_models=n_models, 
                                                                            num_epochs=epochs, 
                                                                            correction_type=correction_type,
                                                                            noise_transition_matrix=noise_transition_matrix if "Noisy" in label else None)
            else:
                predicted_probabilities, accuracies = train_LR_noise_variance(X_train, 
                                                                            y_train, 
                                                                            X_test, 
                                                                            y_test, 
                                                                            num_models=n_models, 
                                                                            num_epochs=epochs, 
                                                                            correction_type=correction_type,
                                                                            noise_transition_matrix = noise_transition_matrix)
            probabilities_dict[label] = predicted_probabilities
            accuracies_dict[label] = accuracies
            
        disagreement_dict = {}

        for label, probabilities in probabilities_dict.items():
            disagreement_rate = estimate_disagreement(probabilities)
            disagreement_dict[label] = disagreement_rate

        print("--- Loop Time %s  ---" % (time.time() - start))
        
        
        ##################################################################
        # Saving Results

        fancy_string = f"n_samples_{args.n_samples}_noise_{str(flip_p)}"

        #results path
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

        if not os.path.exists(os.path.join(parent_dir, "results", "figures","discrete", args.variance_type, args.dataset, args.noise_type)):
            os.makedirs(os.path.join(parent_dir, "results", "figures","discrete", args.variance_type, args.dataset, args.noise_type))

        if not os.path.exists(os.path.join(parent_dir, "results", "dataframes","discrete", args.variance_type, args.dataset,  args.noise_type)):
            os.makedirs(os.path.join(parent_dir, "results", "dataframes","discrete", args.variance_type, args.dataset,  args.noise_type))


        #Save DFs
        df_path = os.path.join(parent_dir, "results","dataframes", "discrete",args.variance_type, args.dataset,  args.noise_type, "acc_prob_" + fancy_string + ".csv")

        acc_prob_df = acc_prob_to_df(accuracies_dict, probabilities_dict)
        acc_prob_df.to_csv(df_path) 

        df_path = os.path.join(parent_dir, "results","dataframes", "discrete",args.variance_type, args.dataset,  args.noise_type, "disagreement_" + fancy_string + ".csv")
        disagreement_df = disagreement_to_df(disagreement_dict)
        disagreement_df.to_csv(df_path)
    
        #Save Figures
        image_path = os.path.join(parent_dir, "results","figures", "discrete",args.variance_type, args.dataset,  args.noise_type, "accuracy_box_"+fancy_string+ ".png")
        plot_boxplot(accuracies_dict, save_path = image_path)

        image_path = os.path.join(parent_dir, "results","figures", "discrete",args.variance_type, args.dataset,  args.noise_type, "robustness_rate_"+fancy_string+ ".png")
        plot_robustness_rates(probabilities_dict, save_path = image_path)
        
        image_path = os.path.join(parent_dir, "results","figures", "discrete",args.variance_type, args.dataset,  args.noise_type, "disagreement_box_"+fancy_string+ ".png")
        plot_boxplot(disagreement_dict, y_range=(0.0, 1), title="Disagreement", save_path = image_path)

        image_path = os.path.join(parent_dir, "results","figures", "discrete",args.variance_type, args.dataset,  args.noise_type, "disagreement_threshold_"+fancy_string+ ".png")
        plot_disagreement_percentage(probabilities_dict, save_path = image_path)
        


       