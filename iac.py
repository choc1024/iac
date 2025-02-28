import numpy as np
import os
import pickle
import time
from typing import List, Tuple, Any, Callable, Dict

###############################################################################
# Utility Functions (same as before)
###############################################################################

def format_time(seconds):
    days, seconds = divmod(int(seconds), 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days}d {hours}h {minutes}m {seconds}s"

def batchnorm(inputs: np.ndarray, use_batchnorm: bool = True, epsilon: float = 1e-5) -> np.ndarray:
    if not use_batchnorm:
        return inputs
    batch_mean = np.mean(inputs, axis=0)
    batch_var = np.var(inputs, axis=0)
    return (inputs - batch_mean) / np.sqrt(batch_var + epsilon)

def initialize_weights(shape: Tuple[int, ...], method: str = "he") -> np.ndarray:
    if method == "he":
        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
    elif method == "xavier":
        return np.random.randn(*shape) * np.sqrt(1.0 / shape[0])
    return np.random.randn(*shape)

###############################################################################
# Activation Functions (with custom support)
###############################################################################

def default_activations():
    return {
        "relu": (lambda x: np.maximum(x, 0),
                 lambda out: (out > 0).astype(float)),
        "sigmoid": (lambda x: 1 / (1 + np.exp(-x)),
                    lambda out: out * (1 - out)),
        "tanh": (np.tanh,
                 lambda out: 1.0 - out**2),
        "linear": (lambda x: x,
                   lambda out: np.ones_like(out))
    }

def get_activation_fn(name: str,
                      custom_activations: Dict[str, Tuple[Callable, Callable]] = None
                      ) -> Callable[[np.ndarray], np.ndarray]:
    all_acts = default_activations()
    if custom_activations:
        all_acts.update(custom_activations)
    return all_acts.get(name, all_acts["linear"])[0]

def get_activation_derivative(name: str,
                              custom_activations: Dict[str, Tuple[Callable, Callable]] = None
                              ) -> Callable[[np.ndarray], np.ndarray]:
    all_acts = default_activations()
    if custom_activations:
        all_acts.update(custom_activations)
    return all_acts.get(name, all_acts["linear"])[1]

###############################################################################
# Convolution Forward and Backward (naïve implementations)
###############################################################################

def conv_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: Tuple[int, int]=(1,1)) -> np.ndarray:
    # x: (N, H, W, C)
    # w: (kH, kW, C, F)
    # b: (F,)
    N, H, W, C = x.shape
    kH, kW, _, F = w.shape
    sH, sW = stride
    out_H = (H - kH) // sH + 1
    out_W = (W - kW) // sW + 1
    out = np.zeros((N, out_H, out_W, F))
    for n in range(N):
        for i in range(out_H):
            for j in range(out_W):
                window = x[n, i*sH:i*sH+kH, j*sW:j*sW+kW, :]
                for f in range(F):
                    out[n, i, j, f] = np.sum(window * w[:, :, :, f]) + b[f]
    return out

def conv_backward(dout: np.ndarray, x: np.ndarray, w: np.ndarray, stride: Tuple[int, int]=(1,1)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # dout: gradient of output, shape (N, out_H, out_W, F)
    # x: input, shape (N, H, W, C)
    N, H, W, C = x.shape
    kH, kW, _, F = w.shape
    sH, sW = stride
    out_H = (H - kH) // sH + 1
    out_W = (W - kW) // sW + 1
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros((F,))
    for n in range(N):
        for i in range(out_H):
            for j in range(out_W):
                for f in range(F):
                    window = x[n, i*sH:i*sH+kH, j*sW:j*sW+kW, :]
                    dw[:, :, :, f] += dout[n, i, j, f] * window
                    dx[n, i*sH:i*sH+kH, j*sW:j*sW+kW, :] += dout[n, i, j, f] * w[:, :, :, f]
                    db[f] += dout[n, i, j, f]
    return dx, dw, db

###############################################################################
# Network Construction using an Architecture List
###############################################################################

def make_network_architecture(input_shape: Any,
                              architecture: List[Dict],
                              weight_init: str = "he",
                              bias_init: float = 0.0,
                              custom_activations: Dict[str, Tuple[Callable, Callable]] = None
                              ) -> List[Dict]:
    """
    Build a network as a list of layer dictionaries.
    
    For a convolutional layer, specify:
      {
         "type": "conv",
         "filters": <number of filters>,       # e.g., 1 for one output per row
         "kernel_size": (kH, kW),               # e.g., (1, 19)
         "stride": (sH, sW),                    # default is (1,1)
         "activation": "<activation name>",     # e.g., "relu"
         "batchnorm": <True/False>,             # optional
         "dropout": <dropout rate>,             # optional
      }
      
    For a dense layer, specify:
      {
         "type": "dense",
         "units": <number of neurons>,
         "activation": "<activation name>",
         "batchnorm": <True/False>,             # optional
         "dropout": <dropout rate>,             # optional
      }
      
    If a dense layer follows a conv layer, the conv output is automatically flattened.
    """
    network = []
    current_shape = input_shape  # could be a tuple (for conv) or an int (for dense)
    for layer_spec in architecture:
        if layer_spec["type"] == "conv":
            # Expect current_shape as (H, W, C)
            if not isinstance(current_shape, tuple) or len(current_shape) != 3:
                raise ValueError("Convolutional layers require an input shape tuple of (H, W, C).")
            H, W, C = current_shape
            kH, kW = layer_spec["kernel_size"]
            stride = layer_spec.get("stride", (1,1))
            filters = layer_spec["filters"]
            out_H = (H - kH) // stride[0] + 1
            out_W = (W - kW) // stride[1] + 1
            new_shape = (out_H, out_W, filters)
            weight_shape = (kH, kW, C, filters)
            weights = initialize_weights(weight_shape, method=weight_init)
            biases = np.full((filters,), bias_init)
            layer = {
                "type": "conv",
                "weights": weights,
                "biases": biases,
                "activation": layer_spec.get("activation", "linear"),
                "stride": stride,
                "batchnorm": layer_spec.get("batchnorm", False),
                "dropout": layer_spec.get("dropout", 0.0),
                "custom_activations": custom_activations,
            }
            network.append(layer)
            current_shape = new_shape
        elif layer_spec["type"] == "dense":
            # Flatten if coming from a conv layer
            if isinstance(current_shape, tuple):
                flattened_dim = np.prod(current_shape)
            else:
                flattened_dim = current_shape
            units = layer_spec["units"]
            weight_shape = (flattened_dim, units)
            weights = initialize_weights(weight_shape, method=weight_init)
            biases = np.full((units,), bias_init)
            layer = {
                "type": "dense",
                "weights": weights,
                "biases": biases,
                "activation": layer_spec.get("activation", "linear"),
                "batchnorm": layer_spec.get("batchnorm", False),
                "dropout": layer_spec.get("dropout", 0.0),
                "custom_activations": custom_activations,
            }
            network.append(layer)
            current_shape = units
        else:
            raise ValueError(f"Unknown layer type: {layer_spec['type']}")
    return network

###############################################################################
# Forward Pass for the New Architecture
###############################################################################

def forward_network(network: List[Dict], inputs: np.ndarray, training: bool = False) -> np.ndarray:
    """
    Process inputs through all layers in the network.
    For conv layers, inputs should have shape (N, H, W, C).
    For dense layers, inputs should be 2D.
    """
    a = inputs
    for layer in network:
        if layer["type"] == "conv":
            a = conv_forward(a, layer["weights"], layer["biases"], layer["stride"])
            if layer["batchnorm"]:
                # For conv layers, apply batchnorm on the flattened channels (simplified)
                N = a.shape[0]
                a = batchnorm(a.reshape(N, -1)).reshape(a.shape)
            act_fn = get_activation_fn(layer["activation"], layer.get("custom_activations"))
            a = act_fn(a)
            if training and layer["dropout"] > 0.0:
                mask = (np.random.rand(*a.shape) < (1.0 - layer["dropout"]))
                a *= mask / (1.0 - layer["dropout"])
        elif layer["type"] == "dense":
            if len(a.shape) > 2:
                a = a.reshape(a.shape[0], -1)
            a = np.dot(a, layer["weights"]) + layer["biases"]
            if layer["batchnorm"]:
                a = batchnorm(a)
            act_fn = get_activation_fn(layer["activation"], layer.get("custom_activations"))
            a = act_fn(a)
            if training and layer["dropout"] > 0.0:
                mask = (np.random.rand(*a.shape) < (1.0 - layer["dropout"]))
                a *= mask / (1.0 - layer["dropout"])
        else:
            raise ValueError(f"Unknown layer type: {layer['type']}")
    return a
