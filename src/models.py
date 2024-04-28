import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import numpy as np

from src.loss_functions import *
from src.metrics import *
from src.noise import *

# Define the Logistic Regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)

class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 5)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(5, 2)
        self.output = nn.Linear(2, 2)  # Assuming binary classification for simplicity

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x

def train_model(X_train, y_train, y_train_noisy, X_test, y_test,  seed, num_epochs=100, batch_size = 256, correction_type="forward", model_type = "LR", noise_transition_matrix=None):
    # Check if GPU is available and set the default device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if correction_type in ['backward', 'forward']:
        # Define noise transition matrix (Example)
        # Convert it to the correct device
        noise_transition_matrix = torch.tensor(noise_transition_matrix, dtype=torch.float32).to(device)
    
    torch.manual_seed(seed)

    # Convert to PyTorch tensors and move them to the device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_train_noisy = torch.tensor(y_train_noisy, dtype=torch.long).to(device)

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # Create DataLoader for mini-batch SGD
    train_data = TensorDataset(X_train, y_train_noisy, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    if model_type == "LR":
        # Initialize the model and move it to the device
        model = LogisticRegression(X_train.shape[1]).to(device)
    else:
        # Initialize the model and move it to the device
        model = NeuralNet(X_train.shape[1]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    #optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    for epoch in (range(num_epochs)):
        for features, noisy_labels, clean_labels in train_loader:
            # Move features and labels to the device
            features, noisy_labels, clean_labels = features.to(device), noisy_labels.to(device), clean_labels.to(device)
            
            # Forward pass
            outputs = model(features)

            if correction_type == 'forward':
                noisy_loss = forward_loss(outputs, noisy_labels, noise_transition_matrix, device)
            elif correction_type == 'backward':
                noisy_loss = backward_loss(outputs, noisy_labels, noise_transition_matrix, device)
            else:
                noisy_loss = criterion(outputs, noisy_labels)

            #print(correction_type, loss)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            noisy_loss.backward()
            optimizer.step()
    
    train_outputs = model(X_train)
    test_outputs = model(X_test)

    #Get final train losses
    if correction_type == 'forward':
        noisy_train_loss = forward_loss(train_outputs, y_train_noisy, noise_transition_matrix, device)

        clean_train_loss = forward_loss(train_outputs, y_train, noise_transition_matrix, device)
        clean_test_loss = forward_loss(test_outputs, y_test, noise_transition_matrix, device)
    elif correction_type == 'backward':
        noisy_train_loss = backward_loss(train_outputs, y_train_noisy, noise_transition_matrix, device)

        clean_train_loss = backward_loss(train_outputs, y_train, noise_transition_matrix, device)
        clean_test_loss = backward_loss(test_outputs, y_test, noise_transition_matrix, device)
    else:
        noisy_train_loss = criterion(train_outputs, y_train_noisy)

        clean_train_loss = criterion(train_outputs, y_train)
        clean_test_loss = criterion(test_outputs, y_test)


    # Evaluate the model
    with torch.no_grad():

        _, predicted = torch.max(test_outputs.data, 1)
        # Move the predictions back to the CPU for sklearn accuracy calculation
        clean_test_acc = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
        test_probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()

        _, predicted = torch.max(train_outputs.data, 1)
        # Move the predictions back to the CPU for sklearn accuracy calculation
        clean_train_acc = accuracy_score(y_train.cpu().numpy(), predicted.cpu().numpy())
        noisy_train_acc = accuracy_score(y_train_noisy.cpu().numpy(), predicted.cpu().numpy())
        train_probs = torch.softmax(train_outputs, dim=1)[:, 1].cpu().numpy()

    
    results = (noisy_train_loss,
                clean_train_loss, 
                noisy_train_acc,
                clean_train_acc,
                train_probs,
                clean_test_loss, 
                clean_test_acc,
                test_probs
                )

    return model, results

def train_LR_no_test(X_train, y_train,  seed, num_epochs=50, correction_type="forward", noise_transition_matrix=None):
    # Check if GPU is available and set the default device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if correction_type in ['backward', 'forward']:
        # Define noise transition matrix (Example)
        # Convert it to the correct device
        noise_transition_matrix = torch.tensor(noise_transition_matrix, dtype=torch.float32).to(device)
    
    torch.manual_seed(seed)

    # Convert to PyTorch tensors and move them to the device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    # Create DataLoader for mini-batch SGD
    batch_size = 256
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # Initialize the model and move it to the device
    model = LogisticRegression(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    # Train the model
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            # Move features and labels to the device
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            if correction_type == 'forward':
                loss = forward_loss(outputs, labels, noise_transition_matrix, device)
            elif correction_type == 'backward':
                loss = backward_loss(outputs, labels, noise_transition_matrix, device)
            else:
                loss = criterion(outputs, labels)
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

def get_predictions_LR(model, X_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert X_test to a torch tensor and move to the same device as the model
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():  # Disable gradient calculations
        # Get the raw output from the model
        raw_output = model(X_test_tensor)
        
        # Apply sigmoid function to get probabilities
        probabilities = torch.sigmoid(raw_output)

        # Use argmax to get the class with the highest probability
        predictions = torch.argmax(raw_output, dim=1)

    return predictions



def train_LR_model_variance(X_train, y_train, X_test, y_test, num_models=1000, num_epochs = 1000, correction_type = "None", noise_transition_matrix = None):
    accuracies = []
    predicted_probabilities = []
    for seed in tqdm(range(num_models)):
        probabilities, accuracy, _ = train_LR(X_train, 
                                              y_train, 
                                              X_test, 
                                              y_test, 
                                              seed, 
                                              num_epochs = num_epochs, 
                                              correction_type = correction_type, 
                                              noise_transition_matrix = noise_transition_matrix)
        accuracies.append(accuracy)
        predicted_probabilities.append(probabilities)
    return np.array(predicted_probabilities), np.array(accuracies)

def train_LR_noise_variance(X_train, y_train, X_test, y_test, num_models=1000, num_epochs = 100, correction_type = "None", noise_transition_matrix = None):
    accuracies = []
    predicted_probabilities = []
    for seed in tqdm(range(num_models)):
        
        #VARY THE NP SEED FOR NOISE INJECTION
        np.random.seed(seed)
        
        if correction_type == "CLEAN": #Clean Labels
            probabilities, accuracy, _ = train_LR(X_train, 
                                                  y_train, 
                                                  X_test, 
                                                  y_test, 
                                                  seed=42, #FIX THE TORCH SEED
                                                  num_epochs = num_epochs, 
                                                  correction_type = correction_type, 
                                                  noise_transition_matrix = noise_transition_matrix)
        else: #Noisy Labels
            y_train_noisy = add_label_noise(y_train, noise_transition_matrix)
            probabilities, accuracy, _ = train_LR(X_train, 
                                                  y_train_noisy, 
                                                  X_test, 
                                                  y_test, 
                                                  seed=42, #FIX THE TORCH SEED
                                                  num_epochs = num_epochs, 
                                                  correction_type = correction_type, 
                                                  noise_transition_matrix = noise_transition_matrix)
            
        accuracies.append(accuracy)
        predicted_probabilities.append(probabilities)
    return np.array(predicted_probabilities), np.array(accuracies)