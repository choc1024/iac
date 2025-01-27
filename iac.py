'''
Import the necessary libraries
'''

# Import OS for file operations
import os
# Re: Regular expressions matching for function parameters
import re
# Numpy: Matrices, maths, etc
import numpy as np
# Log: Logging for the program (I made this lib btw)
from log import log as lg # pylint: disable=W
# Also import the log entirely
import log

# Suppress the log file missing warning
log.lf_suppress = True

lg('QuickTux Initialized', 1)

# A function that creates a neural network with the specified parameters
def make_network(input_neurons: int, output_neurons: int, dense_layers: int,
            dense_neurons: int, dense_activation: str = "relu",
            output_activation: str = "linear",
            weight_init: str = "he", bias_init: float = 0.0,
            batchnorm: bool = False):
    '''
    Create a neural network with the specified parameters

    Returns: list
    '''

    # Define the weight initialization function that NumPy should use according
    # to the weight_init parameter

    ## He initialization
    if weight_init == "he":
        lg('Weight Initialization is He', 0)
        def w_init():
            return np.random.uniform(0, np.sqrt(2 / input_neurons))

    ## Xavier initialization
    elif weight_init == "xavier":
        lg('Weight Initialization is Xavier', 0)
        def w_init():
            x = np.sqrt(6 / (input_neurons + output_neurons))
            return np.random.uniform(-x, x)

    ## Random initialization
    elif re.match(r"rand, (-?\d+(\.\d+)?)", weight_init):
        lg('Random Weight Initialization', 0)
        match = re.match(r"rand, (-?\d+(\.\d+)?)", weight_init)
        match_range = float(match.group(1))
        def w_init():
            return np.random.uniform(-match_range, match_range)

    ## Constant initialization
    elif re.match(r"const, (-?\d+(\.\d+)?)", weight_init):
        lg('Constant Weight Initialization', 0)
        match = re.match(r"const, (-?\d+(\.\d+)?)", weight_init)
        match_const = float(match.group(1))
        def w_init():
            return match_const
    ## No match, weights are initialized to 0
    else:
        lg('Weight Initialization Parameter Unrecognized, Zero Weight Initialization', 2)
        def w_init():
            return 0

    net = []

    # List of weight matrices
    weights = []

    # List of bias matrices
    biases = []

    # Weight matrix from input layer to first dense layer
    weights.append(
        np.array(
            [[w_init() for _ in range(dense_neurons)]
            for _ in range(input_neurons)]))
    for _ in range(dense_layers-1):
        # Weight matrices from dense layer to dense layer
        weights.append(
            np.array(
                [[w_init() for _ in range(dense_neurons)]
                for _ in range(dense_neurons)]))
    # Weight matrix from last dense layer to output layer
    weights.append(
        np.array(
            [[w_init() for _ in range(output_neurons)]
            for _ in range(dense_neurons)]))
    lg('Weight Matrices Created', 1)

    # Bias matrices for each dense layer
    for _ in range(dense_layers):
        biases.append(np.full((1, dense_neurons), bias_init))

    # Bias matrix for output layer
    biases.append(np.full((1, output_neurons), bias_init))
    lg('Bias Matrices Created', 1)

    net.append(weights)
    net.append(biases)
    net.append(dense_activation)
    net.append(output_activation)
    net.append(batchnorm)

    # Add some extra info to make my life easier in training functions
    net.append([input_neurons, output_neurons, dense_layers, dense_neurons])
    return net

def forward(network, inputs):
    '''
    Forward propagate the inputs through the network
    Returns: array of outputs
    '''

    h_act = network[2]
    o_act = network[3]

    # Define Hidden Activation Function
    if h_act == "relu":
        def hidden_activation(layer):
            return np.maximum(layer, 0, layer)
    elif h_act == "sigmoid":
        def hidden_activation(layer):
            return 1 / (1 + np.exp(-layer))
    elif h_act == "tanh":
        def hidden_activation(layer):
            return np.tanh(layer)
    else:
        def hidden_activation(layer):
            return layer

    # Define Output Activation Function
    if o_act == "relu":
        def output_activation(layer):
            return np.maximum(layer, 0, layer)
    elif o_act == "sigmoid":
        def output_activation(layer):
            return 1 / (1 + np.exp(-layer))
    elif o_act == "tanh":
        def output_activation(layer):
            return np.tanh(layer)
    else:
        def output_activation(layer):
            return layer

    # If batchnorm is enabled, normalize the inputs
    if network[4]:
        def batchnorm(inputs):
            return (inputs - np.mean(inputs)) / np.std(inputs)
    else:
        def batchnorm(inputs):
            return inputs

    weights = network[0]
    biases = network[1]

    # First layer
    layer = np.dot(inputs, weights[0]) + biases[0]

    # Hidden layers
    for i in range(1, len(weights)-1):
        layer = batchnorm(hidden_activation(np.dot(layer, weights[i]) + biases[i]))

    # Output layer
    return output_activation(np.dot(layer, weights[-1]) + biases[-1])


#Format of the dataset:
#[array([[input1, input2], [input1, input2], [input1, input2]]),
# array([output1, output2, #output3], ...)]

def av(x): # I didn't bother to use the numpy average function
    '''
    Average function
    Returns: average of x
    '''
    return sum(x) / len(x)

def mse(network, dataset, l1: float = 0.0, l2: float = 0.0):
    '''
    Mean Squared Error function
    Returns: MSE of the network
    '''
    outputs = forward(network, dataset[0])
    return av(av((dataset[1] - outputs)**2)) + (l1 * sum(np.sum(np.abs(w) for w in network[0]))) + (l2 * sum(np.sum(w**2) for w in network[0]))


def simulated_annealing(network, dataset, t0: float = 5.0, limit: float = 0.001,
                        alpha: float = None, a: float = None, b: float = None,
                        k: int = 1, l1: float = 0.0, l2: float = 0.0):
    '''
    Simulated Annealing for optimizing the network
    Returns: optimized network
    '''
    if alpha is None and (a is None or b is None):
        raise ValueError("A decay function must be specified")
    elif alpha is not None and (a is not None or b is not None):
        raise ValueError("Only one decay function can be specified")
    elif alpha is not None and (a is None and b is None):
        def decay(init_t, t, cycle): # pylint: disable=W
            return t*alpha
    elif alpha is None and (a is not None and b is not None):
        def decay(init_t, t, cycle): # pylint: disable=W
            return init_t / (1 + np.e**(a*(cycle-b)))
    else:
        raise ValueError("Invalid decay function")

    t = t0
    cycle = 0
    while t > limit:
        cycle += 1
        initial_error = mse(network, dataset, l1, l2)
        # Change the weights in the input layer randomly
        for neuron in range(len(network[0][0])):
            for weight in range(len(network[0][0][neuron])):
                change = np.random.uniform(-(k*t), (k*t))
                network[0][0][neuron][weight] += change
                new_error = mse(network, dataset, l1, l2)
                if new_error > initial_error:
                    prob = np.e**-((initial_error - new_error) / t)
                    if np.random.uniform(0, 1) > prob:
                        network[0][0][neuron][weight] -= change

        # Change the weights and biases in the hidden layers randomly
        for layer in range(1, len(network[0])-1):
            for neuron in range(len(network[0][layer])):
                for weight in range(len(network[0][layer][neuron])):
                    change = np.random.uniform(-(k*t), (k*t))
                    network[0][layer][neuron][weight] += change
                    new_error = mse(network, dataset, l1, l2)
                    if new_error > initial_error:
                        prob = np.e**-((initial_error - new_error) / t)
                        if np.random.uniform(0, 1) > prob:
                            network[0][layer][neuron][weight] -= change
                change = np.random.uniform(-(k*t), (k*t))
                network[1][layer][neuron] += change
                new_error = mse(network, dataset, l1, l2)
                if new_error > initial_error:
                    prob = np.e**-((initial_error - new_error) / t)
                    if np.random.uniform(0, 1) > prob:
                        network[1][layer][neuron] -= change
        # Change the biases in the output layer randomly
        for neuron in range(network[5][1]):
            change = np.random.uniform(-(k*t), (k*t))
            network[1][-1][neuron] += change
            new_error = mse(network, dataset, l1, l2)
            if new_error > initial_error:
                prob = np.e**-((initial_error - new_error) / t)
                if np.random.uniform(0, 1) > prob:
                    network[1][-1][neuron] -= change
        t = decay(t0, t, cycle)
        after_error = mse(network, dataset, l1, l2)
        lg(f"Cycle {cycle}: Error: {after_error}, Temperature: {t}")
    return network

def save_network(network, filename: str):
    '''
    Save the network to a file
    '''
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(str(network))
    # Change the file extension to .tux
    base = os.path.splitext(filename)[0]  # Get the file name without extension
    new_filename = f"{base}.tux"
    os.rename(filename, new_filename)
