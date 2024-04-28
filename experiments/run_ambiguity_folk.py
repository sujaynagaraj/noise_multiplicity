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
from sklearn.preprocessing import StandardScaler

from folktables import ACSDataSource, ACSEmployment



parser = argparse.ArgumentParser('ambiguity folk')

parser.add_argument('--n_models', type =int, default=1000, help="number of models to train")
parser.add_argument('--n_dims', type =int, default=16, help="number of dimensions")
parser.add_argument('--noise_type', type=str, default="class_independent", help="specify type of label noise")
parser.add_argument('--max_iter', type =int, default=100000, help="max iterations to check for typical vec")
parser.add_argument('--dataset_size', type =int, default=2500, help="max iterations to check for typical vec")
parser.add_argument('--model_type', type =str, default="LR", help="LR or NN")
parser.add_argument('--uncertainty_type', type =str, default="backward", help="backward or forward uncertainty")

args = parser.parse_args()

#####################################################################################################

if __name__ == '__main__':

    print('Starting Ambiguity Folk')
    print("Noise Type: ", args.noise_type)
    print("Model Type: ", args.model_type)
    print("Uncertainty Type: ", args.uncertainty_type)

    n_models = args.n_models
    max_iter = args.max_iter
    n_dims = args.n_dims
    noise_type = args.noise_type
    model_type = args.model_type
    uncertainty_type = args.uncertainty_type
    dataset_size = args.dataset_size

    noise_levels = np.linspace(0, 0.49, num=5)

     # Parent directory for saving figures
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    figures_path = os.path.join(parent_dir, "results", "figures", "metrics", "folk", args.model_type, args.noise_type)

    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data)


    acs_data_sample = acs_data.sample(n=dataset_size, random_state=2024)
    features, label, group = ACSEmployment.df_to_numpy(acs_data_sample)

    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
                                                                features, 
                                                                label, 
                                                                group, 
                                                                test_size=0.3, 
                                                                random_state=2024)

    X_train = X_train
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    py1 = np.sum(y_train)/len(y_train)
    p_y_x = np.array([1-py1,py1])

    if noise_type == "class_independent":

       
        # Generate error rates
        error_rates, disagreement_rates, accuracy_rates = simulate_noise_and_train_model(noise_levels, 
                                                                                        n_models, 
                                                                                        max_iter, 
                                                                                        n_dims,  
                                                                                        X_train, 
                                                                                        y_train, 
                                                                                        X_test, 
                                                                                        y_test, 
                                                                                        p_y_x = p_y_x,
                                                                                        model_type = model_type,
                                                                                        noise_type = noise_type, 
                                                                                        uncertainty_type=uncertainty_type)
        
        image_path = os.path.join(figures_path, f"{uncertainty_type}_metrics.png")

        # Plot in the appropriate subplot
        plot_metrics_boxplot_real(error_rates,
            disagreement_rates,
            accuracy_rates,
            X_test,
            title=f'Metrics - {uncertainty_type}',
            save_path = image_path)

        image_path2 = os.path.join(figures_path, f"{uncertainty_type}_condition.png")

        # Plot in the appropriate subplot
        plot_boxplots_with_condition(error_rates,
            disagreement_rates,
            X_test,
            group_test,
            y_test = y_test,
            title=f'Condition - {uncertainty_type}',
            save_path = image_path2)

       

    elif noise_type == "class_conditional":
        fixed_classes = [0, 1]
        fixed_noises = [0.0,  0.2,  0.45]
        

        for fixed_class in fixed_classes:

           

            for i, fixed_noise in enumerate(fixed_noises):

                # Generate error rates
                error_rates, disagreement_rates, accuracy_rates = simulate_noise_and_train_model(noise_levels, 
                                                                                                n_models, 
                                                                                                max_iter, 
                                                                                                n_dims,  
                                                                                                X_train, 
                                                                                                y_train, 
                                                                                                X_test, 
                                                                                                y_test, 
                                                                                                p_y_x = p_y_x,
                                                                                                noise_type = noise_type, 
                                                                                                model_type = model_type,
                                                                                                uncertainty_type=uncertainty_type,  
                                                                                                fixed_class=fixed_class, 
                                                                                                fixed_noise=fixed_noise)

                
                image_path = os.path.join(figures_path, f"{fixed_class}_{fixed_noise}_{uncertainty_type}_metrics.png")

                # Plot in the appropriate subplot

                plot_metrics_boxplot_real(error_rates,
                    disagreement_rates,
                    accuracy_rates,
                    X_test,
                    title=f'Metrics - {uncertainty_type}',
                    save_path = image_path)

                image_path2 = os.path.join(figures_path, f"{fixed_class}_{fixed_noise}_{uncertainty_type}_condition.png")

                # Plot in the appropriate subplot

                plot_boxplots_with_condition(error_rates,
                    disagreement_rates,
                    X_test,
                    group_test,
                    y_test = y_test,
                    title=f'Condition - {uncertainty_type}',
                    save_path = image_path2)
