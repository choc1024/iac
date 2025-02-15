# IAC Neural Network Framework Wiki

Welcome to the IAC Neural Network Framework Wiki! This guide will take you from having zero knowledge about neural networks to being able to create, train, and deploy all sorts of AI models using the IAC framework.

# ⚠️ Warning: This Wiki has been written by ChatGPT. Careful. A human-made Wiki is currently still in progress. ⚠️

---

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Utility Functions Overview](#utility-functions-overview)
- [Creating a Neural Network](#creating-a-neural-network)
- [Performing a Forward Pass](#performing-a-forward-pass)
- [Evaluating with MSE](#evaluating-with-mse)
- [Training the Network](#training-the-network)
- [Saving and Loading Models](#saving-and-loading-models)
- [Example Workflow](#example-workflow)
- [Conclusion](#conclusion)

---

## Introduction

The IAC framework provides a modular, from-scratch approach to building neural networks in Python using **NumPy**. It covers:
- **Network creation** with customizable layer architectures.
- **Activation functions** with derivatives for backpropagation.
- **Forward propagation** for inference.
- **Loss computation** (Mean Squared Error - MSE).
- **Training routines** with multiple optimizers (SGD, Adam, AdamW), along with support for batch normalization and regularization (L1 & L2).
- **Model persistence** through saving and loading with pickle.

Whether you're a beginner or an experienced AI enthusiast, this wiki will help you understand and harness the power of these tools.

---

## Requirements

Before getting started, ensure you have the following installed:
- **Python 3.x**
- **NumPy**: For numerical operations.
- **pickle** (standard with Python): For saving and loading models.

You can install NumPy using pip:

    pip install numpy

---

## Utility Functions Overview

### 1. `dense(*layers: Tuple[int, int]) -> List[int]`
- **Purpose**: Create a list of neuron counts for dense layers.
- **Usage Example**: 
  ```python
  # Creates two layers of 64 neurons and three layers of 32 neurons.
  layer_sizes = dense((2, 64), (3, 32))
  # Result: [64, 64, 32, 32, 32]
  ```

### 2. Activation Functions and Their Derivatives
- **`get_activation_fn(name: str)`**: Returns an activation function (e.g., ReLU, Sigmoid, Tanh, etc.).
- **`get_activation_derivative(name: str, activated: np.ndarray)`**: Returns the derivative of the specified activation function.

### 3. `batchnorm(inputs: np.ndarray, use_batchnorm: bool, epsilon: float)`
- **Purpose**: Apply batch normalization to input data.
- **Note**: In production, consider using running averages for inference.

### 4. `initialize_weights(shape: Tuple[int, int], method: str)`
- **Purpose**: Initialize weights using methods like "he" or "xavier".
- **Usage**: Automatically called during network creation.

---

## Creating a Neural Network

The `make_network` function helps create a neural network with your desired architecture.

### Function Signature

```python
make_network(input_size: int,
             output_size: int,
             layers: List[int],
             hidden_activation: str = "relu",
             output_activation: str = "linear",
             use_batchnorm: bool = False,
             weight_init: str = "he",
             bias_init: float = 0.0) -> List[Any]
```

### Steps:
1. **Define Your Architecture**: Use the `dense` function to create your layer configuration.
2. **Set Activations**: Choose activation functions for hidden and output layers.
3. **Batch Normalization**: Enable if needed.
4. **Weight Initialization**: Pick "he" or "xavier" based on your network's needs.

### Example:

```python
# Import the necessary utility function
layers = dense((2, 64), (3, 32))  # Define your layer sizes

# Create a network with 10 inputs and 2 outputs
network = make_network(input_size=10,
                       output_size=2,
                       layers=layers,
                       hidden_activation="relu",
                       output_activation="softmax",
                       use_batchnorm=True,
                       weight_init="he")
```

---

## Performing a Forward Pass

The `forward` function computes the output of the network for given input data.

### Function Signature

```python
forward(network: List[Any], inputs: np.ndarray) -> np.ndarray
```

### Example:

```python
import numpy as np

# Dummy input data (e.g., 5 examples with 10 features each)
X = np.random.rand(5, 10)

# Get predictions
predictions = forward(network, X)
print("Network Predictions:", predictions)
```

---

## Evaluating with MSE

The `mse` function calculates the Mean Squared Error between the network's predictions and the actual labels.

### Function Signature

```python
mse(network: List[Any], dataset: Tuple[np.ndarray, np.ndarray]) -> float
```

### Example:

```python
# Dummy dataset: 5 samples, 10 features, and corresponding labels (5 samples, 2 outputs)
Y = np.random.rand(5, 2)
dataset = (X, Y)

# Calculate MSE
error = mse(network, dataset)
print("Mean Squared Error:", error)
```

---

## Training the Network

The `train` function allows you to optimize your network using different optimizers and regularization techniques.

### Function Signature

```python
train(network: List[Any],
      dataset: Tuple[np.ndarray, np.ndarray],
      optimizer: str = "adamw",
      lr: float = 0.001,
      epochs: int = 1000,
      beta1: float = 0.9,
      beta2: float = 0.999,
      epsilon: float = 1e-8,
      weight_decay: float = 1e-2,
      l1: float = 0.0,
      l2: float = 0.0) -> List[Any]
```

### Key Features:
- **Optimizers**: Choose between "sgd", "adam", and "adamw".
- **Regularization**: L1 and L2 regularization to prevent overfitting.
- **Batch Normalization**: Integrated into the training process if enabled.

### Example:

```python
# Train the network for 500 epochs using AdamW optimizer
trained_network = train(network,
                        dataset,
                        optimizer="adamw",
                        lr=0.001,
                        epochs=500,
                        weight_decay=1e-2,
                        l1=0.0,
                        l2=0.001)
```

*Watch the training progress on your console via the progress bar and loss output.*

---

## Saving and Loading Models

### Saving a Network

The `save_network` function allows you to save your trained network to a file.

```python
save_network(network, "my_model.iac")
```

### Loading a Network

Load a saved network with the `load_network` function:

```python
loaded_network = load_network("my_model.iac")
```

---

## Example Workflow

Here's a complete example to tie everything together:

```python
import numpy as np

# 1. Define the network architecture using dense
layers = dense((2, 64), (3, 32))

# 2. Create the network (e.g., for a dataset with 10 features and 2 output classes)
network = make_network(input_size=10,
                       output_size=2,
                       layers=layers,
                       hidden_activation="relu",
                       output_activation="softmax",
                       use_batchnorm=True,
                       weight_init="he")

# 3. Create a dummy dataset
X = np.random.rand(100, 10)  # 100 samples, 10 features each
Y = np.random.rand(100, 2)   # 100 samples, 2 outputs each
dataset = (X, Y)

# 4. Train the network
network = train(network,
                dataset,
                optimizer="adamw",
                lr=0.001,
                epochs=500,
                weight_decay=1e-2,
                l2=0.001)

# 5. Evaluate the network using MSE
error = mse(network, dataset)
print("Final Mean Squared Error:", error)

# 6. Save the trained network
save_network(network, "trained_model.iac")

# 7. (Optional) Load the network later
# loaded_network = load_network("trained_model.iac")
# predictions = forward(loaded_network, X)
```
