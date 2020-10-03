import src.library.net.components.Neuron as Neuron
import src.library.net.components.Activation as Activation
import src.library.interface.cli.ProgressBar as ProgressBar

import numpy as np
import os
import matplotlib.pyplot as plt

class Network:

    """
    Neural Network class
    """

    def __init__(self, layers=[2,8,1], learning_rate=0.001, iterations=100):

        self.neuron = Neuron.Neuron(Activation.Activation.Sigmoid)
        self.progress_bar = ProgressBar.ProgressBar()

        self.params = {}
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []


    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        for i in range(1, len(self.layers)):
            self.params["W" + str(i)] = np.random.uniform(-1, 1, size=(self.layers[i - 1], self.layers[i])) 
            self.params['b' + str(i)] = np.random.uniform(-1, 1, size=(self.layers[i])) 
#            print("W" + str(i) + ": " + str(self.params["W" + str(i)]))
#            print("b" + str(i) + ": " + str(self.params["b" + str(i)]))

    def mathematical_forward_propagation(self):
        '''
        Performs the forward propagation
        '''
        layer_values = self.data

        for i in range(1, len(self.layers)):
            layer_values, activation_input = self.neuron.activate(self.params['W' + str(i)], layer_values, self.params['b' + str(i)])
            self.params['Z' + str(i)] = activation_input
            self.params['A' + str(i)] = layer_values

        loss = self.entropy_loss(self.results, layer_values)

        return layer_values, loss


    def mathematical_back_propagation(self, prediction):
        '''
        Computes the derivatives and update weights and bias according.
        '''
#        error = -(np.divide(self.results, prediction) - np.divide((1 - self.results),(1 - prediction))) # causes a divide by zero error if prediction is 1
        self.params['A0'] = self.data
        self.params['dl_wrt_a' + str(len(self.layers) - 1)] = prediction - self.results

        for a in range(1, len(self.layers)):
            i = len(self.layers) - a
            self.params['dl_wrt_z' + str(i)] = self.params['dl_wrt_a' + str(i)] * Activation.sigmoid_derivate(self.params['A' + str(i)])
            self.params['dl_wrt_a' + str(i - 1)] = self.params['dl_wrt_z' + str(i)].dot(self.params['W' + str(i)].T)
            self.params['dl_wrt_w' + str(i)] = self.params['A' + str(i - 1)].T.dot(self.params['dl_wrt_z' + str(i)])
            self.params['dl_wrt_b' + str(i)] = np.sum(self.params['dl_wrt_z' + str(i)], axis=0)


        for i in range(1, len(self.layers)):
            self.params['W' + str(i)] = self.params['W' + str(i)] - self.learning_rate * self.params['dl_wrt_w' + str(i)]
            self.params['b' + str(i)] = self.params['b' + str(i)] - self.learning_rate * self.params['dl_wrt_b' + str(i)]



    def entropy_loss(self, results, prediction):
        nsample = len(results)
        loss = -1 / nsample * (np.sum(np.multiply(np.log(prediction), results) + np.multiply((1 - results), np.log(1 - prediction))))
        return loss



    def train(self, data, results):
        '''
        Trains the neural network
        '''
        self.data = data
        self.results = results
        self.init_weights()

        col, row = os.get_terminal_size()
        self.progress_bar.progressBar(self.iterations, prefix='Progress:', suffix='Complete', length=col-30)

        for i in range(self.iterations):
            prediction, loss = self.mathematical_forward_propagation()
            self.mathematical_back_propagation(prediction)
            self.loss.append(loss)

            self.progress_bar.printProgressBar(i + 1)


    def predict(self, data):
        '''
        Predicts based on the trained values
        '''
        layer_values = data

        for i in range(1, len(self.layers)):
            layer_values, activation_input = self.neuron.activate(self.params['W' + str(i)], layer_values, self.params['b' + str(i)])

        return np.round(layer_values, decimals=4)


    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()