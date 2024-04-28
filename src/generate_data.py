
import numpy as np
import pickle as pkl

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

def save_data(X_train, X_test, y_train, y_test, filename, path):
    with open(path+filename+".pkl",'wb') as f:
        pkl.dump([X_train, X_test, y_train, y_test], f)

def load_data(filename, path):
    with open(path+filename+".pkl",'rb') as f:
        X_train, X_test, y_train, y_test = pkl.load(f)
    return X_train, X_test, y_train, y_test

def generate_filename(dataset, n_samples):
    return f"{dataset}_n_samples_{n_samples}"

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