from network import Network
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":
    # DEFINE NETWORK
    CLASSES = 3
    network = Network(2, [5], CLASSES)
    # INITIALIZE TRAINING EXAMPLES
    epochs = 10000; prev = 0
    BATCH_SIZE = 32
    X, y = spiral_data(epochs*BATCH_SIZE, CLASSES)
    #DEFINE PLOTS
    _, axweight = plt.subplots()
    _, axbias = plt.subplots()
    _, axloss = plt.subplots()
    _, ax = plt.subplots()

    #TRAIN
    start_weights = network.get_weights(); start_biases = network.get_biases(); start_loss = 0;

    prev=0
    for e in np.arange(0, epochs*BATCH_SIZE*CLASSES, BATCH_SIZE):
            network.forward(X[e:e+BATCH_SIZE])
            network.update(y[e:e+BATCH_SIZE], -1*network.loss)
            if (e % 100 == 0):
                print('LOSS', network.loss)
                sys.stdout.flush()
                #fetch status
                weights = network.get_weights()
                biases = network.get_biases()
                loss = network.loss
                #plot
                axweight.plot([prev, e], np.array([start_weights, weights]))
                axbias.plot([prev, e], np.array([start_biases, biases]))
                axloss.plot([prev, e], np.array([start_loss, loss]))
                plt.pause(0.005)
                start_weights = weights
                start_biases = biases
                start_loss = loss
                prev = e


    print('training on {} datapoints from {} classes: \n'.format(BATCH_SIZE*CLASSES, CLASSES))

    axbias.set_title('calculating bias'.format(CLASSES, CLASSES*BATCH_SIZE))
    axweight.set_title('calculating weights'.format(CLASSES, CLASSES*BATCH_SIZE))
    plt.show()
