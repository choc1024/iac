import os
import pickle
import numpy as np
from typing import List, Tuple, Any, Callable


################################################################################
# Utility Functions
################################################################################

def dense(*layers: Tuple[int, int]) -> List[int]:
    """
    Generate a list of dense layer sizes from tuples of (num_layers, neurons_per_layer).
    Example: dense((2,64), (3,32)) -> [64, 64, 32, 32, 32]
    """
    return [neurons for num, neurons in layers for _ in range(num)]


def get_activation_fn(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Retrieve the activation function based on the provided name.
    """
    activations = {
        "relu": lambda x: np.maximum(x, 0),
        "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
        "tanh": np.tanh,
        "leaky_relu": lambda x: np.maximum(0.01 * x, x),
        "softmax": lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True),
        "linear": lambda x: x
    }
    return activations.get(name, lambda x: x)


def get_activation_derivative(name: str, activated: np.ndarray) -> np.ndarray:
    """
    Retrieve the derivative of the activation function with respect to its output.
    """
    if name == "relu":
        return (activated > 0).astype(float)
    elif name == "sigmoid":
        return activated * (1.0 - activated)
    elif name == "tanh":
        return 1.0 - activated ** 2
    elif name == "leaky_relu":
        return np.where(activated > 0, 1.0, 0.01)
    elif name == "linear":
        return np.ones_like(activated)
    else:
        return np.ones_like(activated)


def batchnorm(inputs: np.ndarray, use_batchnorm: bool = True, epsilon: float = 1e-5) -> np.ndarray:
    """
    Apply batch normalization if enabled.
    Uses per-batch statistics; note that for a production system, you'd typically
    maintain running averages for inference.
    """
    if not use_batchnorm:
        return inputs
    batch_mean = np.mean(inputs, axis=0)
    batch_var = np.var(inputs, axis=0)
    return (inputs - batch_mean) / np.sqrt(batch_var + epsilon)


def initialize_weights(shape: Tuple[int, int], method: str = "he") -> np.ndarray:
    """
    Initialize weights based on the selected method (He or Xavier).
    """
    if method == "he":
        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
    elif method == "xavier":
        return np.random.randn(*shape) * np.sqrt(1.0 / shape[0])
    return np.random.randn(*shape)


################################################################################
# Network Creation and Forward Pass
################################################################################

def make_network(input_size: int,
                 output_size: int,
                 layers: List[int],
                 hidden_activation: str = "relu",
                 output_activation: str = "linear",
                 use_batchnorm: bool = False,
                 weight_init: str = "he",
                 bias_init: float = 0.0) -> List[Any]:
    """
    Create a neural network with specified properties.
    Returns [weights, biases, hidden_activation, output_activation, use_batchnorm].
    """
    # Initialize weights
    weights = [initialize_weights((input_size, layers[0]), weight_init)]
    for i in range(len(layers) - 1):
        weights.append(initialize_weights((layers[i], layers[i + 1]), weight_init))
    weights.append(initialize_weights((layers[-1], output_size), weight_init))

    # Initialize biases
    biases = [np.full((1, l), bias_init) for l in layers]
    biases.append(np.full((1, output_size), bias_init))

    return [weights, biases, hidden_activation, output_activation, use_batchnorm]


def forward(network: List[Any], inputs: np.ndarray) -> np.ndarray:
    """
    Perform a forward pass through the neural network.
    """
    weights, biases, hidden_activation, output_activation, use_batchnorm = network

    hidden_activation_fn = get_activation_fn(hidden_activation)
    output_activation_fn = get_activation_fn(output_activation)

    for i in range(len(weights) - 1):
        z = np.dot(inputs, weights[i]) + biases[i]
        a = batchnorm(hidden_activation_fn(z), use_batchnorm)
        inputs = a

    z = np.dot(inputs, weights[-1]) + biases[-1]
    outputs = output_activation_fn(z)
    return outputs


################################################################################
# MSE Function (Network + Dataset)
################################################################################

def mse(network: List[Any], dataset: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Compute Mean Squared Error (MSE) for the given network and dataset.
    Dataset format: (X, Y), where X is input data and Y is labels/target outputs.
    """
    X, Y = dataset
    preds = forward(network, X)
    return np.mean((preds - Y) ** 2)


################################################################################
# Training Function with SGD, Adam, AdamW, Batchnorm, L1 and L2 Regularization
################################################################################

def train(network: List[Any],
          dataset: Tuple[np.ndarray, np.ndarray],
          optimizer: str = "adamw",
          lr: float = 0.001,
          epochs: int = 1000,
          beta1: float = 0.9,
          beta2: float = 0.999,
          epsilon: float = 1e-8,
          weight_decay: float = 1e-2,
          l1: float = 0.0,
          l2: float = 0.0) -> List[Any]:
    """
    Train the neural network using backpropagation with options for:
      - 'sgd'
      - 'adam'
      - 'adamw'

    Additionally, this function now includes L1 and L2 regularization.
    The loss printed includes the regularization penalty.
    """
    X, Y = dataset
    weights, biases, hidden_activation, output_activation, use_batchnorm = network

    # For Adam/AdamW, initialize moments for each weight and bias
    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]

    hidden_activation_fn = get_activation_fn(hidden_activation)
    output_activation_fn = get_activation_fn(output_activation)

    for epoch in range(epochs):
        ############################################################################
        # Forward pass
        ############################################################################
        z_values = []
        a_values = [X]

        for i in range(len(weights) - 1):
            z = np.dot(a_values[-1], weights[i]) + biases[i]
            z_values.append(z)
            a = batchnorm(hidden_activation_fn(z), use_batchnorm)
            a_values.append(a)

        z_out = np.dot(a_values[-1], weights[-1]) + biases[-1]
        z_values.append(z_out)
        y_pred = output_activation_fn(z_out)
        a_values.append(y_pred)

        ############################################################################
        # Loss calculation (MSE + regularization penalties)
        ############################################################################
        mse_loss = np.mean((y_pred - Y) ** 2)
        reg_loss = 0.0
        for w in weights:
            reg_loss += l1 * np.sum(np.abs(w)) + l2 * np.sum(w ** 2)
        loss = mse_loss + reg_loss

        ############################################################################
        # Backward pass
        ############################################################################
        dA = (y_pred - Y) / Y.shape[0]

        for i in reversed(range(len(weights))):
            # Select activation function derivative for output vs. hidden layers
            act_name = output_activation if i == len(weights) - 1 else hidden_activation
            dZ = dA * get_activation_derivative(act_name, a_values[i + 1])
            grad_w = np.dot(a_values[i].T, dZ)
            grad_b = np.sum(dZ, axis=0, keepdims=True)

            # Add regularization derivatives to the weight gradients
            grad_w += l1 * np.sign(weights[i]) + 2 * l2 * weights[i]

            # Optimizer updates
            if optimizer in ["adam", "adamw"]:
                m_w[i] = beta1 * m_w[i] + (1 - beta1) * grad_w
                v_w[i] = beta2 * v_w[i] + (1 - beta2) * (grad_w ** 2)

                m_b[i] = beta1 * m_b[i] + (1 - beta1) * grad_b
                v_b[i] = beta2 * v_b[i] + (1 - beta2) * (grad_b ** 2)

                m_w_corr = m_w[i] / (1 - beta1 ** (epoch + 1))
                v_w_corr = v_w[i] / (1 - beta2 ** (epoch + 1))
                m_b_corr = m_b[i] / (1 - beta1 ** (epoch + 1))
                v_b_corr = v_b[i] / (1 - beta2 ** (epoch + 1))

                if optimizer == "adam":
                    weights[i] -= lr * (m_w_corr / (np.sqrt(v_w_corr) + epsilon))
                    biases[i] -= lr * (m_b_corr / (np.sqrt(v_b_corr) + epsilon))
                else:  # AdamW
                    weights[i] -= (lr * (m_w_corr / (np.sqrt(v_w_corr) + epsilon))
                                   + lr * weight_decay * weights[i])
                    biases[i] -= lr * (m_b_corr / (np.sqrt(v_b_corr) + epsilon))
            else:
                # Vanilla SGD update
                weights[i] -= lr * grad_w
                biases[i] -= lr * grad_b

            if i > 0:
                dA = np.dot(dZ, weights[i].T)

        ############################################################################
        # Update progress bar
        ############################################################################
        try:
            old_loss += 1
        except:
            old_loss = loss
        progress = (epoch + 1) / epochs
        bar_length = 50
        block = int(round(bar_length * progress))
        bar = '#' * block + ' ' * (bar_length - block)
        if loss > old_loss:
            print(f"\rEpoch {epoch + 1}/{epochs} [{bar}] - {loss:.10f} Loss ▲", end='')
        elif loss < old_loss:
            print(f"\rEpoch {epoch + 1}/{epochs} [{bar}] - {loss:.10f} Loss ▼", end='')
        else:
            print(f"\rEpoch {epoch + 1}/{epochs} [{bar}] - {loss:.10f} Loss ✔️", end='')
        old_loss = loss
        if epoch == epochs - 1:
            print()

    return network


################################################################################
# Save/Load Functions
################################################################################

def save_network(network: List[Any], filename: str) -> None:
    """
    Save the network to a .iac file using pickle.
    """
    filename = os.path.splitext(filename)[0] + ".iac"
    with open(filename, "wb") as file:
        pickle.dump(network, file)


def load_network(filename: str) -> List[Any]:
    """
    Load a network from a .iac file.
    """
    with open(filename, "rb") as file:
        return pickle.load(file)
