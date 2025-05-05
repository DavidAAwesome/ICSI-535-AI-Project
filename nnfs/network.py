from numpy import void
from nn import *
import sys

class Network:
    def __init__(self, inputs, hidden_layers, classes):
        """
        hidden_layers -- tuple representing the amount of neurons in each hidden layer
        classes -- how many classes to clasify
        """
        # INPUT LAYER
        self.layers = [Layer_Dense(inputs, hidden_layers[0])]
        self.activations: list = [Activation_ReLU()]
        # HIDDEN LAYERS
        for i in range(1, len(hidden_layers)):
            self.layers.append(Layer_Dense(hidden_layers[i-1], hidden_layers[i]))
            self.activations.append(Activation_ReLU())
        # OUTPUT LAYER
        self.layers.append(Layer_Dense(hidden_layers[-1], classes))
        self.activations.append(Activation_Softmax())
        # LOSS
        self.CCE = Loss_CategoricalCrossentropy()
        self.loss = 1

    def forward(self, X):
        for layer, activation in zip(self.layers, self.activations):
            layer.forward(X)
            activation.forward(layer.output)
            X = activation.output
        self.output = self.activations[-1].output

    def train(self, X, y, batch_size, axloss=void):
        """updates the weights and biases for all these sweet sweet training examples"""
        Xlen, ylen = len(X), len(y)
        if Xlen != ylen:
            return -1

        for e in np.arange(0, Xlen-batch_size, batch_size):
            self.forward(X[e:e+batch_size])
            self.update(y[e:e+batch_size], -1*self.loss)
            if (e % 100 == 0):
                print('LOSS', self.loss)
                sys.stdout.flush()

    def update(self, y_true, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of ground truth values 'y', and "eta"
        is the learning rate"""

        weights = [layer.weights for layer in self.layers]
        biases = [layer.biases for layer in self.layers]

        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]

        delta_nabla_b, delta_nabla_w = self.backprop(self.output, y_true)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        for i in range(len(weights)):
            weights[i] -= eta/len(y_true) * nabla_w[i]
        for i in range(len(biases)):
            biases[i] -= eta/len(y_true) * nabla_b[i]

        # weights = [w-(eta/len(batch))*nw
        #                 for w, nw in zip(weights, nabla_w)]
        # biases = [b-(eta/len(batch))*nb
        #                 for b, nb in zip(biases, nabla_b)]

    def backprop(self, x, y):
        """Calculates the gradient of the loss function aka, how much each weight and bias in the network
        impacts the final prediction"""
        # BP1) find error for the last layer
        # nabla_b = [print(b.shape) for b in [layer.biases for layer in layers] ]
        nabla_b = [np.zeros(b.shape) for b in (layer.biases for layer in self.layers) ]
        nabla_w = [np.zeros(w.shape) for w in (layer.weights for layer in self.layers) ]


        # delta defines the error in layer l
        delta = self.CCE.derivative(x, y)


        # WHEN NOT USING SOFTMAX AND CCE
        # sigma_prime = activations[-1].prime
        # print('cost', cost.backward(activations[-1].output, y))

        # delta = cost.backward(activations[-1].output, y) * sigma_prime(layers[-1].output)

        # BP3,BP4)
        mean = np.mean(self.activations[-2].output, axis=0, keepdims=True)

        nabla_b[-1][:] = delta
        nabla_w[-1][:] = np.dot(mean.transpose(), delta)


        # BP2) move the error backward through the layers
        for l in range(2, len(self.layers)+1):
            z = self.layers[-l].output
            sigma_prime = np.mean(self.activations[-l].prime(z), axis=0)
            # print(sigma_prime)

            delta = np.array(delta).flatten()
            delta = np.dot(self.layers[-l+1].weights, delta) * sigma_prime
            # print('delta', delta)

            nabla_b[-l][:] = delta
            mean = np.mean(self.activations[-l].output, axis=0)
            nabla_w[-l][:] = np.dot(mean.transpose(), delta)
        # print('nabla_b', nabla_b)
        # print('nabla_w', nabla_w)

        self.loss = self.CCE.calculate(self.output, y)
        return (nabla_b, nabla_w)

    def get_weights(self):
        """returns a list of the networks weights"""
        flattened = []
        for layer in self.layers:
            for weight_matrix in layer.weights:
                for weight in np.ravel(weight_matrix):
                    flattened.append(weight)
        return np.array(flattened)
    def get_biases(self):
        """returns a list of the networks biases"""
        flattened = []
        for layer in self.layers:
            for bias in layer.biases:
                flattened.append(bias)
        return np.array(flattened)



