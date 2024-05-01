
import numpy as np
import pickle as pkl

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, resample

from sklearn.preprocessing import StandardScaler, LabelEncoder

import pandas as pd

import os




def save_data(X_train, X_test, y_train, y_test, filename, path):
    with open(path+filename+".pkl",'wb') as f:
        pkl.dump([X_train, X_test, y_train, y_test], f)

def load_data(filename, path):
    with open(path+filename+".pkl",'rb') as f:
        X_train, X_test, y_train, y_test = pkl.load(f)
    return X_train, X_test, y_train, y_test

def generate_filename(dataset, n_samples):
    return f"{dataset}_n_samples_{n_samples}"

def load_dataset(dataset, include_groups = False):
    
    #dataset = "cshock_eicu"

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_path = os.path.join(parent_dir, "data", dataset, dataset+"_data.csv")
    
    df = pd.read_csv(data_path)
    
    if dataset in ["cshock_eicu", "cshock_mimic"]:
        # Labels
        labels = df['hospital_mortality'].values
        
        # Encode 'sex' using LabelEncoder
        label_encoder = LabelEncoder()
        df['age'] = label_encoder.fit_transform(df['age'])
        df['sex'] = label_encoder.fit_transform(df['sex'])

        # Groups
        groups = {
            'age': df['age'].values,
            'sex': df['sex'].values
        }
        
        if include_groups:
            # Features including all except label
            features = pd.get_dummies(df.drop('hospital_mortality', axis=1)).values
        else:
            # Features including all except group and label
            features = pd.get_dummies(df.drop(['hospital_mortality', 'age', 'sex'], axis=1)).values


    elif dataset == "support":

        # Labels
        df['Death_in_5Yr'] =  (df['Death_in_5Yr'] + 1) // 2
        labels = df['Death_in_5Yr'].values.astype(int)
        
        # Encode 'sex' using LabelEncoder
        label_encoder = LabelEncoder()
        

        df['age'] = label_encoder.fit_transform(df['age'])
        df['sex'] = label_encoder.fit_transform(df['sex'])

        # Groups
        groups = {
            'age': df['age'].values,
            'sex': df['sex'].values
        }
        
        if include_groups:
            # Features including all except label
            features = pd.get_dummies(df.drop('Death_in_5Yr', axis=1)).values.astype(int)
        else:
            # Features including all except group and label
            features = pd.get_dummies(df.drop(['Death_in_5Yr', 'age', 'sex'], axis=1)).values.astype(int)
        
    elif dataset == "saps":
        # Labels
        labels = df['DeadAtDischarge'].values
        
        # Groups
        groups = {
            
        }
        #No groups

            # Features including all except label
        features = pd.get_dummies(df.drop('DeadAtDischarge', axis=1)).values
        
    elif dataset == "lungcancer":
        # Labels
        labels = df['Malignant'].values.astype(int)
        
        # Encode 'sex' using LabelEncoder
        label_encoder = LabelEncoder()
        df['Age'] = label_encoder.fit_transform(df['Age'])
        df['Gender'] = label_encoder.fit_transform(df['Gender'])

        # Groups
        groups = {
            'age': df['Age'].values,
            'sex': df['Gender'].values
        }
        
        if include_groups:
            # Features including all except label
            features = pd.get_dummies(df.drop('Malignant', axis=1)).values
        else:
            # Features including all except group and label
            features = pd.get_dummies(df.drop(['Malignant', 'Age', 'Gender'], axis=1)).values


    # Create a StandardScaler object
    scaler = StandardScaler()

    # Fit and transform the data
    features = scaler.fit_transform(features)
            
    return features, labels, groups


def balance_data(features, labels):
        # Suppose this function loads your dataset
    #features, labels, groups = load_dataset("cshock_eicu", include_groups=True)
    np.random.seed(2024)
    
    # Find the unique classes and the frequency of each class
    class_counts = np.bincount(labels)
    minority_class = np.argmin(class_counts)
    majority_class = np.argmax(class_counts)

    # Separate the majority and minority classes
    features_minority = features[labels == minority_class]
    features_majority = features[labels == majority_class]
    labels_minority = labels[labels == minority_class]
    labels_majority = labels[labels == majority_class]

    # Upsample the minority class
    features_minority_upsampled, labels_minority_upsampled = resample(
        features_minority,
        labels_minority,
        replace=True,  # Sample with replacement
        n_samples=len(features_majority),  # Match number in majority class
        random_state=2024)  # Reproducible results

    # Combine the majority class with the upsampled minority class
    features_balanced = np.vstack((features_majority, features_minority_upsampled))
    labels_balanced = np.hstack((labels_majority, labels_minority_upsampled))

    # Shuffle the dataset to mix up minority and majority samples
    indices = np.arange(len(labels_balanced))
    np.random.shuffle(indices)
    features_balanced = features_balanced[indices]
    labels_balanced = labels_balanced[indices]
    
    return features_balanced, labels_balanced


def load_MNIST(n_samples, random_state = 42):
    np.random.seed(random_state)
    # Load MNIST dataset
    mnist = fetch_openml(name='mnist_784', version=1)

    # Filter the dataset for digits 1 and 7
    mask = (mnist.target == '1') | (mnist.target == '7')
    X = mnist.data[mask]
    Y = mnist.target[mask]

    # Convert labels to binary: 1 for digit 1, and 0 for digit 7
    Y = (Y == '1').astype(int)

    # Select a random subset of the data
    subset_size = 1000  # for example, 500 samples
    subset_indices = np.random.choice(np.arange(X.shape[0]), size=subset_size, replace=False)
    X = X.iloc[subset_indices].values
    Y = Y.iloc[subset_indices].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, Y

if __name__ == "__main__":

    samples = [1000]
    datasets = ["MNIST"]

    random_state = 42

    for n_samples in samples:
        for dataset in datasets:
            
            if dataset == "MNIST":
                X_scaled, Y = load_MNIST(dataset, n_samples)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=random_state)

            print("Loaded Data!")

            filename = generate_filename(dataset, n_samples)
            path = "/h/snagaraj/noise_multiplicity/data/processed/"

            save_data(X_train, X_test, y_train, y_test, filename, path)
            
            print("Saved Data!")