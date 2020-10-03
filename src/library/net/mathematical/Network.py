import src.library.net.components.Neuron as Neuron
import src.library.net.components.Activation as Activation
import src.library.interface.cli.ProgressBar as ProgressBar
import src.library.util.Logger as Logger

import numpy as np
import os
import matplotlib.pyplot as plt
import datetime

class Network:

    """
    Neural Network class
    """

    def __init__(self, layers=[2,8,1], learning_rate=0.001, iterations=100):

        self.neuron = Neuron.Neuron(Activation.Activation.Gaussian)
        self.progress_bar = ProgressBar.ProgressBar()
        self.logger = Logger.Logger("NeuralNetwork")

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
            self.params['W' + str(i)] = np.random.uniform(-1, 1, size=(self.layers[i - 1], self.layers[i])) 
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
            self.params['NetInput' + str(i)] = activation_input
            self.params['Output' + str(i)] = layer_values

        loss = self.results - layer_values

        return layer_values, loss
       

    def mathematical_back_propagation(self, prediction):
        '''
        Computes the derivatives and update weights and bias according.
        '''
        self.params['Output0'] = self.data
        self.params['weight_error' + str(len(self.layers) - 1)] = self.results - prediction

        for a in range(1, len(self.layers)):
            i = len(self.layers) - a
            self.params['out_error' + str(i)] = self.params['weight_error' + str(i)] * Activation.gaussian_derivate(self.params['NetInput' + str(i)])
            self.params['weight_error' + str(i - 1)] = self.multiply(self.params['out_error' + str(i)], self.params['W' + str(i)])
            self.params['d_weight' + str(i)] = np.multiply.outer(self.params['Output' + str(i - 1)], self.params['out_error' + str(i)])
            self.params['d_bias' + str(i)] = self.params['out_error' + str(i)]



        for i in range(1, len(self.layers)):
            self.params['W' + str(i)] = self.params['W' + str(i)] + self.learning_rate * self.params['d_weight' + str(i)]
            self.params['b' + str(i)] = self.params['b' + str(i)] + self.learning_rate * self.params['d_bias' + str(i)]


    def entropy_loss(self, results, prediction):
        nsample = len(results)
        loss = -1 / nsample * (np.sum(np.multiply(np.log(prediction), results) + np.multiply((1 - results), np.log(1 - prediction))))
        return loss



    def train(self, data, results):
        '''
        Trains the neural network
        '''

        start_time = datetime.datetime.now().timestamp()

        self.init_weights()

        if self.iterations == -1:
        
            i = 0

            while True:
                losssum = 0
                for a in range(len(data)):
                    self.data = data[a]
                    self.results = results[a]

                    prediction, loss = self.mathematical_forward_propagation()
                    self.mathematical_back_propagation(prediction)

                    self.log_data(i * len(data) + a + 2)
                
                    losssum += np.abs(loss)
                self.loss.append(losssum / 4)

                i += 1

                if (losssum / 4) < 0.0005:
                    break

        elif self.iterations == 0:
            print("Programm must train at least once")

        else:

            col, row = os.get_terminal_size()
            self.progress_bar.progressBar(self.iterations, prefix='Progress:', suffix='Complete', length=col-30)

            for i in range(self.iterations):
                losssum = 0
                for a in range(len(data)):
                    self.data = data[a]
                    self.results = results[a]

                    prediction, loss = self.mathematical_forward_propagation()
                    self.mathematical_back_propagation(prediction)
                    
                    self.log_data(i * len(data) + a + 2)
                    
                    losssum += np.abs(loss)

                self.loss.append(losssum / 4)
                self.progress_bar.printProgressBar(i + 1)

        print("Iterations: " + str(len(self.loss) * 4))

        end_time = datetime.datetime.now().timestamp()
        print("Calculation Time: " + str(end_time - start_time) + " Seconds")

        self.logger.save_file()

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

    def log_data(self, i):
        weight_count = 0
        weights = np.array([])
        for layer in range(len(self.layers) - 1):
            weights = np.append(weights, self.params['W' + str(layer + 1)].ravel())
            weight_count += self.layers[layer] * self.layers[layer + 1]

        for weight in range(1, weight_count + 1):
            self.logger.write_data(1, weight, "weight_" + str(weight))
            self.logger.write_data(i, weight, weights[weight - 1])

    def multiply(self, error, weights):
        out = np.array([])
        for a in range(len(weights)):
            errorsum = 0
            for i in range(len(error)):
                errorsum += error[i] * weights[a][i]
            out = np.append(out, errorsum)
        return out