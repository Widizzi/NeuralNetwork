import src.library.net.components.Neuron as Neuron
import src.library.net.components.Activation as Activation

import numpy as np

class Network:
    """
    A simple genetic Network
    """

    def __init__(self, layers=[2,8,1]):

        self.neuron = Neuron.Neuron(Activation.Activation.Sigmoid)

        self.params = {}
        self.layers = layers

        self.init_weights()

    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        for i in range(1, len(self.layers)):
            self.params['Weights' + str(i)] = np.random.uniform(-1, 1, size=(self.layers[i - 1], self.layers[i])) 
            self.params['Biases' + str(i)] = np.random.uniform(-1, 1, size=(self.layers[i])) 
#            print("Weights" + str(i) + ": " + str(self.params["Weights" + str(i)]))
#            print("Biases" + str(i) + ": " + str(self.params["Biases" + str(i)]))

    def genetic_forward_propagation(self):
        '''
        Performs the forward Propagation
        '''
        layer_values = self.data

        for i in range(1, len(self.layers)):
            layer_values, activation_input = self.neuron.activate(self.params['Weights' + str(i)], layer_values, self.params['Biases' + str(i)])

        return layer_values

    def iterate(self, data, ratefunction):
        '''
        iterates once through the network
        '''
        self.data = data

        prediction = self.genetic_forward_propagation()
        rating = ratefunction(prediction, data)
    
        return rating


    def predict(self, data):
        '''
        Predicts based on the trained values
        '''
        self.data = data

        return self.genetic_forward_propagation()

    def getWeights(self):
        '''
        get the Weights of the Network
        '''
        weights = np.array([])
        for i in range(1, len(self.layers)):
            weights = np.append(weights, self.params['Weights' + str(i)]).ravel()
        return weights

    def setWeights(self, weights):
        '''
        set the Weights of the Network
        '''
        next = 0
        for i in range(1, len(self.layers)):
            for w in range(len(self.params['Weights' + str(i)])):
                self.params['Weights' + str(i)][w] = weights[next]
                next += 1

    def getBiases(self):
        '''
        get the Biases of the Network
        '''
        biases = np.array([])
        for i in range(1, len(self.layers)):
            biases = np.append(biases, self.params['Biases' + str(i)]).ravel()
        return biases

    def setBiases(self, biases):
        '''
        set the Biases of the Network
        '''
        next = 0
        for i in range(1, len(self.layers)):
            for w in range(len(self.params['Biases' + str(i)])):
                self.params['Biases' + str(i)][w] = biases[next]
                next += 1
