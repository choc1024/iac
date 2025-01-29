# Instant AI Cooker
This is my personal attempt at creating a Neural Network from scratch. It aims to be quick, lightweight, and easy to use.

# Usage
Currently I have yet to upload it to PyPI, but before that, the only way is to download the iac.py file, put it in your working directory, and do `import iac`. If you want, `from iac import *` can also work. Please note that it requires NumPy for matrix operations, and [my custom logging library](https://github.com/choc1024/log). To be honest, it is actually just a bunch of functions. 

## make_network()
Parameters:

### Required
- `input_neurons` (INT): The number of input neurons
- `output_neurons` (INT): The number of output neurons
- `dense_layers` (INT): The number of dense layers
- `dense_neurons` (INT): The number of neurons per dense layer

### Optional
- `dense_activation` (STR): The activation function for dense layers. Defaults to ReLU. Valid Options: `"relu"`, `"sigmoid"`, `"tanh"`. For any invalid option, the network will use none.
- `output_activation` (STR): The activation function for the output layer. Defaults to Sigmoid. Valid Options: `"relu"`, `"sigmoid"`, `"tanh"`, "`round"` (uses np.round()). For any invalid option, the network will use none.
- `weight_init` (STR): The weight initialization method. Defaults to He. Valid Options: `"he"`, `"xavier"`, `"rand, X"` (will generate a uniform random number from `-X` to `X`), `"const, X"` (all weights initialized to `X`). For any invalid option, the network will initialize all weights to `0`.
- `bias_init` (FLOAT): The bias initialization float. Defaults to `0.0`. All biases will be initialized to this value.
- `batchnorm` (BOOL): Whether to use Batch Normalization or not. Defaults to `False`.

### Returns
Returns a list with all weight and bias matrices along with additional information about the network.

## `forward()`
Parameters:

### Required
- `network` (LIST): The network, made with `make_network()`
- `inputs` (NUMPY.ARRAY): The 1 row numpy array of inputs.

### Returns
Returns a numpy array of the output layer outputs

## `av()`
Parameters:

### Required
- `x` (LIST): A list of numbers

### Returns
A float which is the sum of x divided by `len(x)`

# To be continued...
