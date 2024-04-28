import torch
import torch.nn as nn

import numpy as np

def forward_loss(outputs, labels, noise_transition_matrix, device):
    softmax = nn.Softmax(dim=1)
    criterion = nn.NLLLoss(reduction="mean").to(device)
    
    # Move outputs to the correct device and ensure it's the same data type as the noise transition matrix
    outputs = outputs.to(device).to(dtype=torch.float32)
    labels = labels.to(device)
    clean_posterior = softmax(outputs)

    # Move noise transition matrix to the correct device and ensure correct data type
    noise_transition_matrix = noise_transition_matrix.to(device).to(dtype=torch.float32)
    noise_transition_matrix_T = torch.transpose(noise_transition_matrix, 0, 1)
    
    # Adjust the outputs based on the noise transition matrix
    noisy_posterior = torch.matmul(noise_transition_matrix_T, clean_posterior.unsqueeze(-1)).squeeze()
    
    # Calculate loss
    loss = criterion(noisy_posterior.log(), labels)
    
    return loss


def backward_loss(outputs, labels, noise_transition_matrix, device):
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss(reduction="none").to(device)
    
    # Move outputs to the correct device and ensure correct data type
    outputs = outputs.to(device).to(dtype=torch.float32)
    labels = labels.to(device)
    
    # Move noise transition matrix to the correct device and ensure correct data type
    noise_transition_matrix = noise_transition_matrix.to(device).to(dtype=torch.float32)
    
    # Evaluate loss on all possible labels for each example then concatenate
    all_label_loss = []
    for c in range(outputs.shape[-1]):
        # Generate labels filled with class c for comparison
        class_labels = torch.full(labels.shape, c, device=device).long()
        loss = criterion(outputs, class_labels)
        all_label_loss.append(loss)

    all_label_loss = torch.stack(all_label_loss, dim=-1).unsqueeze(-1)

    # Compute backward corrected loss
    backward_loss_values = torch.matmul(torch.inverse(noise_transition_matrix), all_label_loss)
    backward_loss = backward_loss_values[range(labels.size(0)), labels.long(), 0].mean()

    return backward_loss
