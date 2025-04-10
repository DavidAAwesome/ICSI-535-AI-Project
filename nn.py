"""
Module: neural_network
======================

This modual provides a barebones, modular implementation of a neural network using numpy.

Classes:
-------
- Layer_Dense:
    A layer in our network, provides a method to calculate its output.
    Output is represented by an array containing wX = b for each neuron.

- Activation_ReLU:
    Takes the output of a layer as input and returns the 0 if the layers is<0

- Activation_Softmax:
    Returns a probability distribution over the predicted classes

- Loss:
    A base class for loss funcitons.

- Loss_CategoricalCrossentropy:
    Performs this type of loss.


Usage Example:
-------------
dense = Layer_Dense(2,3)
activation = Activation_ReLU()

X, y = some_data(samples=100, classes=3)

dense.forward(X)
activation.forward(dense1.output)

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output,y)
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from matplotlib.colors import same_color
import numpy as np
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt


class Layer_Dense:
    """
    Represents a layer with m neurons that takes n features.

    Instance variables:
        weights -- n x m Weight vector, W
        biases -- bias', b
        output -- z = Wx + b, calculated with a forward pass, where x is the inputs
    """

    def __init__(self, n_features, m_neurons):
        self.weights = 0.10 * np.random.randn(n_features, m_neurons)
        # self.biases = np.zeros((1, m_neurons))
        self.biases = np.zeros(m_neurons)

    def forward(self, inputs):
        """ Calculates layers output, in batch form, z = Wx + b """
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, output_loss, learning_rate):
        if output_loss.ndim == 1:
            output_loss = output_loss.reshape(1, -1)

        # What is the ground truth neuron

        # how do we change neurons to increase the confidence of ground truth
        # how do we change weight and bias to decrease wrong neurons


# ACTIVATION FUNCTIONS

class Activation_ReLU:
    """
    Rectified linear activation function
    - computationaly efficient
    - large and meaningful gradients, mitigating vanishing gradient for training
        deeper networks
        - this means the network will learn faster as weights are not affected
            by saturated neurons

    Instance variables:
        output -- a = maximum(0, inputs)
    """

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def prime(self, inputs):
        # print(inputs.shape) # (10, 3)
        inputs[inputs>0] = 1
        inputs[inputs<0] = 0
        return inputs

class Activation_Softmax:
    """
    This activation function gives a probability distribution over the outputs of a layer.
    - meaningful negative predictions 

    Works best on the output layer
    """
    def forward(self, inputs):
        """ Takes a batch of outputs from a layer and applies the softmax activation function """
        # exponentials get very large, prevent overflow by subtracting the max input
        # gives us exp_values between 0 and 1
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # e^x
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # normalize
        self.output = probabilities

    # Compute derivative

    def prime(self, inputs):

        return

class Loss(ABC):
    """
    Measures the cost of predicted and true values in batch form
    """
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def derivative(self, output, y):
        """
        rate of change of C with respect to the output activations

        output -- activations from the last layer
        y      -- ground truth values
        """
        sample_losses = self.backward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    @abstractmethod
    def forward(self, y_pred, y_true) -> float:
        pass

    @abstractmethod
    def backward(self, y_pred, y_true) -> list:
        pass

class Loss_CategoricalCrossentropy(Loss):
    """
    Measures the distance between the predicted probability and the ground truth probability.
    - Probabalistic output (between 0 and 1)
    - Only one class has a value of 1, the rest are 0

    -sum(yi*log(y_hati))
    """

    def forward(self, y_pred, y_true) -> float:
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)


        if len(y_true.shape) == 1: # not one-hot encoded
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        else: return -1

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, y_pred, y_true) -> list:
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)


        if len(y_true.shape) == 1: # not one-hot encoded
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        else: return [-1]

        # print(y_pred)
        # print(y_pred_clipped)
        # print(correct_confidences)

        # account for all predictions
        y_pred_clipped = y_pred_clipped*-1
        y_pred_clipped[range(samples), y_true] = 1+y_pred_clipped[range(samples), y_true]

        # y_pred_clipped[range(samples), y_true] = 1-y_pred_clipped[range(samples), y_true]

        print('Predicted loss:\n', y_pred_clipped)

        average_loss = np.mean(y_pred_clipped, axis=0, keepdims=True)
        # geometric mean
        return average_loss


# INITIALIZE
CLASSES = 3
SAMPLES = 3
X, y = spiral_data(samples=SAMPLES, classes=CLASSES)

layers = [Layer_Dense(2,5), Layer_Dense(5,CLASSES)]
activations = [Activation_ReLU(), Activation_Softmax()]
CCE = Loss_CategoricalCrossentropy()


# FORWARD PASS
for layer, activation in zip(layers, activations):
    layer.forward(X)
    activation.forward(layer.output)
    X = activation.output
print('Forward Pass - classifying {} datapoints from {} classes: \n'.format(SAMPLES*CLASSES, CLASSES), activations[-1].output)


# BACKPROPIGATION

# BP1) find error for the last layer
# nabla_b = [print(b.shape) for b in [layer.biases for layer in layers] ]
nabla_b = [np.zeros(b.shape) for b in (layer.biases for layer in layers) ]
nabla_w = [np.zeros(w.shape) for w in (layer.weights for layer in layers) ]


# delta defines the error in layer l
delta = CCE.backward(activations[-1].output, y)
print('Calculating average Loss for {} examples:\n'.format(SAMPLES*CLASSES), delta, end='\n\n')

# WHEN NOT USING SOFTMAX AND CCE
# sigma_prime = activations[-1].prime
# print('cost', cost.backward(activations[-1].output, y))

# delta = cost.backward(activations[-1].output, y) * sigma_prime(layers[-1].output)

# BP3,BP4)
mean = np.mean(activations[-2].output, axis=0, keepdims=True)

nabla_b[-1][:] = delta
nabla_w[-1][:] = np.dot(mean.transpose(), delta)


# BP2) move the error backward throught the layers
for l in range(2, len(layers)+1):
    z = layers[-l].output
    sigma_prime = np.mean(activations[-l].prime(z), axis=0)
    # print(sigma_prime)

    delta = np.array(delta).flatten()
    delta = np.dot(layers[-l+1].weights, delta) * sigma_prime
    # print('delta', delta)

    nabla_b[-l][:] = delta
    mean = np.mean(activations[-l].output, axis=0)
    nabla_w[-l][:] = np.dot(mean.transpose(), delta)
print('nabla_b', nabla_b)
print('nabla_w', nabla_w)

# print("data points: \n", X)
# print("z_l1: \n", dense1.output)
# print("a_l1: \n", activation1.output)



# PLOT data
# plt.scatter(X[:,0], X[:,1], c=y)
# plt.show()
