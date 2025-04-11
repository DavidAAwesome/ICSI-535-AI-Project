from nn import *


class Network:
    def __init__(self, hidden_layers, classes):
        """
        hidden_layers -- tuple representing the amount of neurons in each hidden layer
        classes -- how many classes to clasify
        """
        # input layer
        self.layers = [Layer_Dense(2, hidden_layers[0])]
        self.activations: list = [Activation_ReLU()]
        # hidden layers
        for i in range(1, len(hidden_layers) -1): # different activation on the last layer
            self.layers.append(Layer_Dense(hidden_layers[i-1], hidden_layers[i]))
            self.activations.append(Activation_ReLU())
        # output layer
        # probablility distribution over the number of classes
        self.layers.append(Layer_Dense(hidden_layers[-1], classes))
        self.activations.append(Activation_Softmax())

        self.CCE = Loss_CategoricalCrossentropy()

        print(self.layers)

    def forward(self, X):
        for layer, activation in zip(self.layers, self.activations):
            layer.forward(X)
            activation.forward(layer.output)
            X = activation.output
        self.output = self.activations[-1].output

    def backprop(self):
        # BP1) find error for the last layer
        # nabla_b = [print(b.shape) for b in [layer.biases for layer in layers] ]
        nabla_b = [np.zeros(b.shape) for b in (layer.biases for layer in self.layers) ]
        nabla_w = [np.zeros(w.shape) for w in (layer.weights for layer in self.layers) ]


        # delta defines the error in layer l
        delta = self.CCE.backward(self.output, y)
        print('Calculating average Loss for {} examples:\n'.format(SAMPLES*CLASSES), delta, end='\n\n')

        # WHEN NOT USING SOFTMAX AND CCE
        # sigma_prime = activations[-1].prime
        # print('cost', cost.backward(activations[-1].output, y))

        # delta = cost.backward(activations[-1].output, y) * sigma_prime(layers[-1].output)

        # BP3,BP4)
        mean = np.mean(self.activations[-2].output, axis=0, keepdims=True)

        nabla_b[-1][:] = delta
        nabla_w[-1][:] = np.dot(mean.transpose(), delta)


        # BP2) move the error backward throught the layers
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
        return (nabla_b, nabla_w)



# INITIALIZE
CLASSES = 3
SAMPLES = 3
X, y = spiral_data(samples=SAMPLES, classes=CLASSES)

network = Network([5], CLASSES)


network.forward(X)
print('Forward Pass - classifying {} datapoints from {} classes: \n'.format(SAMPLES*CLASSES, CLASSES), network.output)

network.backprop()
