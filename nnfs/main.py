from network import Network
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":
    #DEFINE NETWORK
    CLASSES = 3
    SAMPLES = 32
    network = Network(2, [5], CLASSES)

    figbias, axbias = plt.subplots()
    figweight, axweight = plt.subplots()
    datafig, dataax = plt.subplots()
    figloss, axloss = plt.subplots()

    axbias.set_title('calculating bias over {} classes {} datapoints'.format(CLASSES, CLASSES*SAMPLES))
    axweight.set_title('calculating weights over {} classes {} datapoints'.format(CLASSES, CLASSES*SAMPLES))

    # for graph
    start_weights = network.get_weights()
    start_biases = network.get_biases()
    start_loss = 0

    # INITIALIZE TRAINING EXAMPLES
    epochs = 10000; last = 0
    X, y = spiral_data(epochs*SAMPLES, CLASSES)


    print('training on {} datapoints from {} classes: \n'.format(SAMPLES*CLASSES, CLASSES))
    # network.train(X, y, epochs)

    for e in np.arange(0, epochs*CLASSES, SAMPLES):
        network.forward(X[e:e+SAMPLES])
        network.update(y[e:e+SAMPLES], -1*network.loss)
        if (e % 100 == 0):
            print('LOSS', network.loss)
            sys.stdout.flush()
            #update values
            weights = network.get_weights()
            biases = network.get_biases()
            loss = network.loss
            #plot
            axweight.plot([last, e], np.array([start_weights, weights]))
            axbias.plot([last, e], np.array([start_biases, biases]))
            axloss.plot([last, e], np.array([start_loss, loss]))
            plt.pause(0.005)
            start_weights = weights
            start_biases = biases
            start_loss = loss
            last = e

    sys.stdout.flush()
    plt.show()
