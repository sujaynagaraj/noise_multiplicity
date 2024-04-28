import numpy as np
from scipy.stats import bernoulli


def generate_class_independent_noise(y, flip_p):
    noise_transition_matrix = np.array([[1-flip_p, flip_p], [flip_p, 1-flip_p]])

    return add_label_noise(y, noise_transition_matrix=noise_transition_matrix), noise_transition_matrix

def generate_class_conditional_noise(y, flip_p, fixed_class, fixed_noise):
    if fixed_class == 0:
        noise_transition_matrix = np.array([[1-fixed_noise, fixed_noise], [flip_p, 1-flip_p]])
    else:
        noise_transition_matrix = np.array([[1-flip_p, flip_p], [fixed_noise, 1-fixed_noise]])
    return add_label_noise(y, noise_transition_matrix=noise_transition_matrix), noise_transition_matrix

def generate_instance_dependent_noise(y, X, flip_p, feature_weights):
    noise_transition_dict = {}
    for instance in np.unique(X, axis=0):
        flip_instance = instance_noise_level(instance, flip_p, feature_weights)
        noise_transition_dict[tuple(instance)] = np.array([[1-flip_instance, flip_instance],
                                                           [flip_instance, 1-flip_instance]])
    return add_label_noise(y, instance_dependent=True, X=X, noise_transition_dict=noise_transition_dict), noise_transition_dict


def add_label_noise(labels, noise_transition_matrix=None, instance_dependent=False, X= None, noise_transition_dict=None):
    noisy_labels = np.copy(labels)
    
    if instance_dependent:
        for i, label in enumerate(labels):
            x0 =  X[i][0]
            x1 =  X[i][1]
            
            noise_transition_matrix_instance = noise_transition_dict[(x0,x1)]
            noisy_labels[i] = np.random.choice([0, 1], p=noise_transition_matrix_instance[label])
    else:
    
        for i, label in enumerate(labels):
            noisy_labels[i] = np.random.choice([0, 1], p=noise_transition_matrix[label])
    return noisy_labels


def instance_noise_level(instance_features, base_noise_level, feature_weights):
    """
    Calculate noise levels for an instance based on its features and given parameters.
    
    :param instance_features: The features of the instance.
    :param base_noise_level: The base noise level.
    :param feature_weights: Weights to apply to each feature to determine its influence on the noise level.
    :return: The noise level for the instance.
    """
    noise_level = base_noise_level + np.dot(instance_features, feature_weights)
    # Ensure the noise level is within valid probability bounds [0, 0.49]
    noise_level = min(max(noise_level, 0), 0.49)
    return noise_level

#u | y
# def get_u(y, T, seed):
#     np.random.seed(seed)

#     if y == 0:
#         noise_rate = T[0,1]
#     else:
#         noise_rate = T[1,0]
    
#     sampled_u = bernoulli.rvs(p=noise_rate, size=1)
    
#     return sampled_u[0]


def get_u(y, T, seed=None):
    np.random.seed(seed)
    # Define noise rates based on the label
    noise_rates = np.where(y == 0, T[0, 1], T[1, 0])
    # Sample u for all labels at once
    sampled_u = bernoulli.rvs(p=noise_rates)
    return sampled_u

#u | yn
# def infer_u(yn, T, p_y_x, seed):
#     np.random.seed(seed)

#     posterior = calculate_posterior(yn, T, p_y_x)
    
#     sampled_u = bernoulli.rvs(p=posterior, size=1)
    
#     return sampled_u[0]


def infer_u(yn, T, p_y_x, seed):
    np.random.seed(seed)  # Set seed for reproducibility

    # Calculate posterior probabilities for the whole vector of yn
    posterior = calculate_posterior(yn, T, p_y_x)
    
    # Generate random samples from a Bernoulli distribution for the entire vector
    sampled_u = bernoulli.rvs(p=posterior)
    
    return sampled_u


# def calculate_posterior(yn, T, p_y_x):
    
#     if yn == 0:
#         opp_class = 1
#     else:
#         opp_class = 0
    
    
#     p_u_opp = T[opp_class, yn]
    
#     p_u = T[yn,opp_class]

#     numerator = p_u_opp * p_y_x[opp_class]
    
    
#     # Sum the resulting vector to get P(y_tilde = observed_y_tilde | x = x)
#     denominator = (1-p_u)*p_y_x[yn] + p_u_opp*p_y_x[opp_class]
    
#     # Divide the element-wise product by the sum to get P(u=1 | y_tilde = observed_y_tilde, x = x)
#     p_u_given_yn_x = numerator / denominator
    
#     return p_u_given_yn_x

def calculate_posterior(yn, T, p_y_x):
    # Create arrays of opposite class indices
    opp_class = 1 - yn  # Flips 0 to 1 and 1 to 0

    # Index T for probabilities of unobserved true class and observed noisy class
    p_u_opp = T[opp_class, yn]
    p_u = T[yn, opp_class]

    # Indexing p_y_x for class probabilities
    p_y_given_x_yn = p_y_x[yn]           # P(Y=yn|X)
    p_y_given_x_opp = p_y_x[opp_class]   # P(Y=opp_class|X)

    # Calculate the numerator of Bayes' rule
    numerator = p_u_opp * p_y_given_x_opp

    # Calculate the denominator of Bayes' rule
    denominator = (1 - p_u) * p_y_given_x_yn + p_u_opp * p_y_given_x_opp

    # Element-wise division to find posterior probabilities
    p_u_given_yn_x = numerator / denominator

    return p_u_given_yn_x

def flip_labels(y, u):
    """
    Takes two binary numpy arrays, u and y, and returns a new array noisy_y
    which is the element-wise XOR of u and y.
    """

    # Ensure the inputs are numpy arrays in case they're not already
    u = np.array(u)
    y = np.array(y)

    # Perform element-wise XOR operation
    noisy_y = np.logical_xor(u, y).astype(int)
    
    return noisy_y





def is_typical(u_vec, T, y_vec = None, p_y_x = None, epsilon=0.25, noise_type = "class_independent", uncertainty_type = "backward"):
    """
    Checks if the observed flips (u_vec) are typical given the noise model (T) and, optionally,
    the class distributions p_y_x.

    Parameters:
    - u_vec: Array of observed flips.
    - T: Noise transition matrix or dict.
    - y_vec: True labels, required for instance-dependent noise.
    - p_y_x: Class probabilities given features, required for instance-dependent noise.
    - epsilon: Tolerance level for deviation from expected noise.
    - noise_type: Type of noise, affects how typicality is assessed.

    Returns:
    - bool: True if the flips are typical, False otherwise.
    """

    
    if uncertainty_type == "forward" and noise_type=="class_independent":


        noise_rate = T[0,1]
        
        #print(sum(u_vec)/len(u_vec), noise_rate)
    
        if abs(sum(u_vec)/len(u_vec) - noise_rate) <= epsilon*noise_rate:
            return True
        else:
            return False
    else:
        bool_flag = True

        for y in [0,1]:

            # Create a boolean mask where both conditions are true
            mask = (u_vec == 1) & (y_vec == y) #(U=u,Y=y)
            
            #print(yn, u_vec, y_vec, mask)
            # Count the number of True values in the mask
            count = np.sum(mask)

            freq = count/len(u_vec)

            #freq = count/np.sum(y_vec == y)
            
            #true_freq = #p(u,yn) = p(u = 1 | Yn = yn) p(Yn = yn)
            
            opp_class = 1-y

            p_u_opp = T[opp_class, y]

            p_u = T[y,opp_class]

            # Sum the resulting vector to get P(y_tilde = observed_y_tilde | x = x)
            p_yn = (1-p_u)*p_y_x[y] + p_u_opp*p_y_x[opp_class]
            
            if uncertainty_type == "forward":
                true_freq = p_u*p_y_x[y]
            else:
                true_freq = calculate_posterior(y, T, p_y_x)*p_yn

            #print(y, p_u, p_y_x[y], freq, true_freq)
            if (abs(freq - true_freq) > epsilon*true_freq): #atypical
                bool_flag = False
                break
        return bool_flag
                