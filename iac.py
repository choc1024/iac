import os
import pickle
import numpy as np
from typing import List, Tuple, Any, Callable, Dict
import time

################################################################################
# Utility Functions
################################################################################

def format_time(seconds):
    days, seconds = divmod(int(seconds), 86400)  # 86400 seconds in a day
    hours, seconds = divmod(seconds, 3600)       # 3600 seconds in an hour
    minutes, seconds = divmod(seconds, 60)       # 60 seconds in a minute
    return f"{days}d {hours}h {minutes}m {seconds}s"

def dense(*layers: Tuple[int, int]) -> List[int]:
    """
    Generate a list of dense layer sizes from tuples of (num_layers, neurons_per_layer).
    Example: dense((2,64), (3,32)) -> [64, 64, 32, 32, 32]
    """
    return [neurons for num, neurons in layers for _ in range(num)]

def default_activations():
    """
    Return the built-in activation functions dictionary.
    """
    return {
        "relu": (lambda x: np.maximum(x, 0),
                 lambda out: (out > 0).astype(float)),
        "sigmoid": (lambda x: 1 / (1 + np.exp(-x)),
                    lambda out: out * (1.0 - out)),
        "tanh": (np.tanh,
                 lambda out: 1.0 - out ** 2),
        "leaky_relu": (lambda x: np.maximum(0.01 * x, x),
                       lambda out: np.where(out > 0, 1.0, 0.01)),
        "softmax": (lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True),
                    # out here is the result of softmax, so derivative wrt out
                    lambda out: out * (1.0 - out)), 
        "linear": (lambda x: x,
                   lambda out: np.ones_like(out)),
    }

def get_activation_fn(name: str,
                      custom_activations: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray],
                                                        Callable[[np.ndarray], np.ndarray]]] = None
                      ) -> Callable[[np.ndarray], np.ndarray]:
    """
    Retrieve the forward activation function based on the provided name.
    If custom_activations is provided and has an entry matching 'name',
    use that instead of the default.
    """
    # Merge defaults with custom, custom takes precedence
    all_acts = default_activations()
    if custom_activations:
        for k, v in custom_activations.items():
            all_acts[k] = v

    if name not in all_acts:
        # If unknown, default to linear as fallback
        return lambda x: x
    return all_acts[name][0]

def get_activation_derivative(name: str,
                              custom_activations: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray],
                                                                Callable[[np.ndarray], np.ndarray]]] = None
                              ) -> Callable[[np.ndarray], np.ndarray]:
    """
    Retrieve the derivative function of the activation based on the provided name.
    If custom_activations is provided and has an entry matching 'name',
    use that derivative instead of the default.
    """
    # Merge defaults with custom, custom takes precedence
    all_acts = default_activations()
    if custom_activations:
        for k, v in custom_activations.items():
            all_acts[k] = v

    if name not in all_acts:
        # If unknown, default derivative is 1
        return lambda x: np.ones_like(x)
    return all_acts[name][1]

def batchnorm(inputs: np.ndarray, use_batchnorm: bool = True, epsilon: float = 1e-5) -> np.ndarray:
    """
    Apply batch normalization if enabled.
    Uses per-batch statistics; for production, you'd typically maintain running
    averages for inference.
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
# Network Creation
################################################################################

def make_network(input_size: int,
                 output_size: int,
                 layers: List[int],
                 hidden_activation: str = "relu",
                 output_activation: str = "linear",
                 use_batchnorm: bool = False,
                 weight_init: str = "he",
                 bias_init: float = 0.0,
                 dropout_rate: float = 0.0,
                 custom_activations: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray],
                                                  Callable[[np.ndarray], np.ndarray]]] = None
                 ) -> List[Any]:
    """
    Create a neural network with specified properties.
    Returns [
        weights, 
        biases, 
        hidden_activation, 
        output_activation, 
        use_batchnorm,
        dropout_rate,
        custom_activations
    ].
    """
    # Initialize weights
    weights = [initialize_weights((input_size, layers[0]), weight_init)]
    for i in range(len(layers) - 1):
        weights.append(initialize_weights((layers[i], layers[i + 1]), weight_init))
    weights.append(initialize_weights((layers[-1], output_size), weight_init))

    # Initialize biases
    biases = [np.full((1, l), bias_init) for l in layers]
    biases.append(np.full((1, output_size), bias_init))

    return [
        weights,
        biases,
        hidden_activation,
        output_activation,
        use_batchnorm,
        dropout_rate,
        custom_activations
    ]

################################################################################
# Forward Pass (for quick eval without training)
################################################################################

def forward(network: List[Any], inputs: np.ndarray, training: bool = False) -> np.ndarray:
    """
    Perform a forward pass through the neural network. By default (training=False),
    dropout is not applied. For quick evaluation or inference, call this directly.
    """
    (weights, 
     biases, 
     hidden_activation, 
     output_activation, 
     use_batchnorm,
     dropout_rate,
     custom_activations) = network

    hidden_activation_fn = get_activation_fn(hidden_activation, custom_activations)
    output_activation_fn = get_activation_fn(output_activation, custom_activations)

    # Forward through hidden layers
    for i in range(len(weights) - 1):
        z = np.dot(inputs, weights[i]) + biases[i]
        a = hidden_activation_fn(z)
        a = batchnorm(a, use_batchnorm=use_batchnorm)
        # If training, apply dropout
        if training and dropout_rate > 0.0:
            mask = (np.random.rand(*a.shape) < (1.0 - dropout_rate))
            a *= mask / (1.0 - dropout_rate)
        inputs = a

    # Forward through output layer
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
    preds = forward(network, X, training=False)
    return np.mean((preds - Y) ** 2)

################################################################################
# Training Function with Dropout, SGD, Adam, AdamW, BN, L1/L2 Regularization
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
          l2: float = 0.0,
          loss_limit: float = 0.001) -> List[Any]:
    """
    Train the neural network using backpropagation with options for:
      - 'sgd'
      - 'adam'
      - 'adamw'

    Also includes:
      - L1 and L2 regularization
      - Dropout support
      - Batchnorm usage if 'use_batchnorm' is True in network
    """
    X, Y = dataset
    (weights, 
     biases, 
     hidden_activation, 
     output_activation, 
     use_batchnorm,
     dropout_rate,
     custom_activations) = network

    hidden_activation_fn = get_activation_fn(hidden_activation, custom_activations)
    output_activation_fn = get_activation_fn(output_activation, custom_activations)
    hidden_activation_deriv = get_activation_derivative(hidden_activation, custom_activations)
    output_activation_deriv = get_activation_derivative(output_activation, custom_activations)

    # For Adam/AdamW, initialize moments for each weight and bias
    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]

    # Keep track of best loss for progress arrow
    old_loss = np.inf
    init_time = time.time()

    for epoch in range(epochs):
        ############################################################################
        # Forward pass (storing all intermediate values for backprop)
        ############################################################################
        z_values = []
        a_values = [X]  # a_values[i] is the output (post-activation) of layer i-1
        dropout_masks = []  # track dropout for each hidden layer

        # Hidden layers
        for i in range(len(weights) - 1):
            z = np.dot(a_values[-1], weights[i]) + biases[i]
            z_values.append(z)
            a = hidden_activation_fn(z)
            a = batchnorm(a, use_batchnorm=use_batchnorm)

            # Apply dropout if needed
            if dropout_rate > 0.0:
                mask = (np.random.rand(*a.shape) < (1.0 - dropout_rate))
                a *= mask / (1.0 - dropout_rate)
                dropout_masks.append(mask)
            else:
                dropout_masks.append(None)

            a_values.append(a)

        # Output layer
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
        dA = (y_pred - Y) / Y.shape[0]  # derivative of MSE wrt predictions

        for i in reversed(range(len(weights))):
            # Decide which activation derivative to use
            if i == len(weights) - 1:
                # Output layer
                dZ = dA * output_activation_deriv(a_values[i + 1])
            else:
                # Hidden layer
                dZ = dA * hidden_activation_deriv(a_values[i + 1])
                # For dropout: zero out dropped neurons
                if dropout_masks[i] is not None:
                    dZ *= dropout_masks[i] / (1.0 - dropout_rate)

            grad_w = np.dot(a_values[i].T, dZ)
            grad_b = np.sum(dZ, axis=0, keepdims=True)

            # L1 and L2 regularization terms for weight gradients
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
                else:
                    # AdamW
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
        # Progress bar & early stopping
        ############################################################################
        elapsed = time.time() - init_time
        if epoch > 0:
            eta = (elapsed / epoch) * (epochs - epoch)
        else:
            eta = 0

        progress = float(epoch + 1) / float(epochs)
        bar_length = 50
        block = int(round(bar_length * progress))
        bar = '#' * block + ' ' * (bar_length - block)

        if loss > old_loss:
            arrow = '▲'
        elif loss < old_loss:
            arrow = '▼'
        else:
            arrow = '○'

        print(f"\rEpoch {epoch + 1}/{epochs} [{bar}] - {loss:.6f} Loss {arrow} ETA {format_time(eta)}", end='')
        
        old_loss = loss

        if epoch == epochs - 1:
            print()
        if loss < loss_limit:
            print("\nBreaking Ahead of Time\n")
            break

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
