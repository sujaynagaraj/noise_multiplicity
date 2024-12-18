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
from src.toy_data import *

from operator import xor


parser = argparse.ArgumentParser('ambiguity toy prob')

parser.add_argument('--n_models', type =int, default=1000, help="number of models to train")
parser.add_argument('--n_dims', type =int, default=2, help="number of dimensions")
parser.add_argument('--noise_type', type=str, default="class_independent", help="specify type of label noise")

args = parser.parse_args()

#####################################################################################################

if __name__ == '__main__':

    print('Starting Ambiguity Toy Probabilistic')
    print("Noise Type: ", args.noise_type)

    n_models = args.n_models
    n_dims = args.n_dims
    noise_type = args.noise_type

    noise_levels = np.linspace(0, 0.49, num=20)
    loss_types = ["0-1", "Corrected"]

    # Define the deterministic labels for each (x1, x2) pair
    true_labels = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 1
    }

    prob_variations = [
        {'name': 'low_uncertainty', 'probs': {(0, 0): 0.1, (0, 1): 0.9, (1, 0): 0.9, (1, 1): 0.9}},
        {'name': 'high_uncertainty', 'probs': {(0, 0): 0.4, (0, 1): 0.6, (1, 0): 0.6, (1, 1): 0.6}}
    ]

    instance_variations = [
        {'name': 'Equal Samples', 'counts': {(0, 0): 25, (0, 1): 25, (1, 0): 25, (1, 1): 25}},
        {'name': 'Downsampled Minority Class', 'counts': {(0, 0): 10, (0, 1): 25, (1, 0): 25, (1, 1): 25}},
        {'name': 'Balanced Classes', 'counts': {(0, 0): 75, (0, 1): 25, (1, 0): 25, (1, 1): 25}},
        {'name': 'Extreme Imbalance Majority Class', 'counts': {(0, 0): 50, (0, 1): 5, (1, 0): 5, (1, 1): 5}},
        {'name': 'Extreme Imbalance Minority Class', 'counts': {(0, 0): 5, (0, 1): 50, (1, 0): 50, (1, 1): 50}},
        {'name': 'Missing Data (0,1)', 'counts': {(0, 0): 25, (0, 1): 0, (1, 0): 25, (1, 1): 25}},
        {'name': 'Missing Data (1,1)', 'counts': {(0, 0): 25, (0, 1): 25, (1, 0): 25, (1, 1): 0}},
    ]

     # Parent directory for saving figures
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    figures_error_path = os.path.join(parent_dir, "results", "figures", "error_toy", "probabilistic", args.noise_type)
    figures_disagreement_path = os.path.join(parent_dir, "results", "figures", "disagreement_toy", "probabilistic", args.noise_type)

    if not os.path.exists(figures_error_path):
        os.makedirs(figures_error_path)

    if not os.path.exists(figures_disagreement_path):
        os.makedirs(figures_disagreement_path)

    if noise_type == "class_independent":

        # Loop through each variation
        for variation in instance_variations:
            for prob in prob_variations:

                instances_counts = variation['counts']
                condition_name = variation['name']

                prob_name = prob['name']
                prob_labels = prob['probs']

                # Create a figure for the current variation
                fig, axes = plt.subplots(nrows=1, ncols=len(loss_types), figsize=(5 * len(loss_types), 5))

                 # Create a figure for the current variation
                fig2, axes2 = plt.subplots(nrows=1, ncols=len(loss_types), figsize=(5 * len(loss_types), 5))

                weights = calculate_weights(prob_labels) 

                # Generate the dataset
                X, y = generate_dataset(true_labels, instances_counts, probabilistic=True, weights=weights, seed=2023)
                labels = np.array(list(true_labels.values()))

                # Loop through each loss type
                for i, loss_type in enumerate(loss_types):

                    # Generate ambiguity rates
                    error_rates, disagreement_rates = generate_metrics_toy(
                        noise_levels,
                        n_models,
                        n_dims,
                        X,
                        y,
                        true_labels,
                        instances_counts,
                        loss_type=loss_type,
                        noise_type=noise_type
                    )

                    # Plot in the appropriate subplot
                    ax = axes[i] if len(loss_types) > 1 else axes
                    plot_metrics_toy(
                        error_rates,
                        noise_levels,
                        ax=ax,
                        title=f'Loss {loss_type}'
                    )

                    # Plot in the appropriate subplot
                    ax = axes2[i] if len(loss_types) > 1 else axes2
                    plot_metrics_toy(
                        disagreement_rates,
                        noise_levels,
                        ax=ax,
                        title=f'Loss {loss_type}'
                    )

                # Adjust layout and save the plot for the current condition
                fig.suptitle(f'Error Rates for {condition_name} {prob_name}')
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle

                image_path = os.path.join(figures_error_path, f"{condition_name}_{prob_name}.png")
                fig.savefig(image_path)
                plt.close(fig)  # Close the figure to free memory

                # Adjust layout and save the plot for the current condition
                fig2.suptitle(f'Disagreement Rates for {condition_name} {prob_name}')
                fig2.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle

                image_path = os.path.join(figures_disagreement_path, f"{condition_name}_{prob_name}.png")
                fig2.savefig(image_path)
                plt.close(fig2)  # Close the figure to free memory


    elif noise_type == "class_conditional":
        fixed_classes = [0, 1]
        fixed_noises = [0.0,  0.2,  0.45]
        
        # Loop through each condition and fixed_class
        for variation in instance_variations:
            for prob in prob_variations:
                for fixed_class in fixed_classes:
                    # Create a figure for the current condition and fixed_class
                    fig, axs = plt.subplots(nrows=len(fixed_noises), ncols=len(loss_types), figsize=(5 * len(loss_types), 5 * len(fixed_noises)))

                     # Create a figure for the current condition and fixed_class
                    fig2, axs2 = plt.subplots(nrows=len(fixed_noises), ncols=len(loss_types), figsize=(5 * len(loss_types), 5 * len(fixed_noises)))

                    instances_counts = variation['counts']
                    condition_name = variation['name']

                    prob_name = prob['name']
                    prob_labels = prob['probs']

                    weights = calculate_weights(prob_labels)

                     # Generate the dataset
                    X, y = generate_dataset(true_labels, instances_counts, probabilistic=True, weights=weights, seed=2023)
                    labels = np.array(list(true_labels.values()))


                    for i, fixed_noise in enumerate(fixed_noises):
                        for j, loss_type in enumerate(loss_types):
                           
                            # Generate ambiguity rates
                            error_rates, disagreement_rates = generate_metrics_toy(
                                noise_levels,
                                n_models,
                                n_dims,
                                X,
                                y,
                                true_labels,
                                instances_counts,
                                loss_type=loss_type,
                                noise_type=noise_type,
                                fixed_class=fixed_class,
                                fixed_noise=fixed_noise
                            )

                            # Determine the correct subplot
                            ax = axs[i, j] if len(fixed_noises) > 1 and len(loss_types) > 1 else axs[max(i, j)]

                            # Plot the ambiguity rates
                            plot_metrics_toy(
                                error_rates,
                                noise_levels,
                                ax=ax,
                                title=f'Fixed Noise {fixed_noise}, Loss {loss_type}'
                            )

                            # Determine the correct subplot
                            ax = axs2[i, j] if len(fixed_noises) > 1 and len(loss_types) > 1 else axs2[max(i, j)]

                            # Plot the ambiguity rates
                            plot_metrics_toy(
                                disagreement_rates,
                                noise_levels,
                                ax=ax,
                                title=f'Fixed Noise {fixed_noise}, Loss {loss_type}'
                            )
                    
                    # Adjust layout and save the plot for the current condition and fixed_class
                    fig.suptitle(f'Error Rates for {variation["name"]} - Fixed Class {fixed_class} {prob_name}')
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle

                    image_path = os.path.join(figures_error_path, f"{variation['name']}_class_{fixed_class}_{prob_name}.png")
                    fig.savefig(image_path)
                    plt.close(fig)  # Close the figure to free memory

                    # Adjust layout and save the plot for the current condition and fixed_class
                    fig2.suptitle(f'Disagreement Rates for {variation["name"]} - Fixed Class {fixed_class} {prob_name}')
                    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle

                    image_path = os.path.join(figures_disagreement_path, f"{variation['name']}_class_{fixed_class} {prob_name}.png")
                    fig2.savefig(image_path)
                    plt.close(fig2)  # Close the figure to free memory