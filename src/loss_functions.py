import torch
import torch.nn as nn

import numpy as np

import timeit


def backward_loss(outputs, labels, T, device):
    #start_time = timeit.default_timer()

    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss(reduction="none").to(device)
    
    # Prepare outputs
    outputs = outputs.to(device).to(dtype=torch.float32)
    labels = labels.to(device)

    # Use the provided single noise transition matrix
    T_inv = torch.linalg.inv(T)

    all_label_loss = []
    for c in range(outputs.shape[-1]):
        class_labels = torch.full(labels.shape, c, device=device).long()
        loss = criterion(outputs, class_labels)
        all_label_loss.append(loss)

    all_label_loss = torch.stack(all_label_loss, dim=-1).unsqueeze(-1)

    # Compute backward corrected loss
    backward_loss_values = torch.matmul(T_inv.float(), all_label_loss.float())
    backward_loss = backward_loss_values[range(labels.size(0)), labels.long(), 0].mean()

    # code you want to evaluate
    #elapsed = timeit.default_timer() - start_time
    #print("backward ", elapsed)

    return backward_loss


def forward_loss(outputs, labels, T, device, noise_type = "class_independent"):
    start_time = timeit.default_timer()

    softmax = nn.Softmax(dim=1)
    criterion = nn.NLLLoss(reduction="mean").to(device)
    
    # Prepare outputs and labels
    outputs = outputs.to(device).to(dtype=torch.float32)
    labels = labels.to(device)
    clean_posterior = softmax(outputs)

   
    T = T.to(device).to(dtype=torch.float32)
    

    T_T = torch.transpose(T, 0, 1)


    # Adjust the outputs based on the noise transition matrix
    noisy_posterior = torch.matmul(T_T, clean_posterior.unsqueeze(-1)).squeeze()
    
    # Calculate loss
    loss = criterion(noisy_posterior.log(), labels)
    
    return loss
