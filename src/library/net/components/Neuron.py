import src.library.net.components.Activation as Activation

import math
import numpy as np

class Neuron:

    '''
    A single Neuron
    '''

    def __init__(self, activationFunction=Activation.Activation.ReLu):
        '''
        Initializes the Neuron with a given activation function
        '''
        self.activationFunction = activationFunction


    def activate(self, weight, value, bias):
        '''
        activates the neuron to calculate the sum of a matrix multiplication of values and weigths plus the bias
        and puts it into the activation function
        '''
        activation_input = value.dot(weight) + bias

        return Activation.switch(self.activationFunction, activation_input), activation_input