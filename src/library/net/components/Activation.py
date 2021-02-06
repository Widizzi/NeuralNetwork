import numpy as np
from enum import Enum

class Activation(Enum):
    ReLu = 1
    Sigmoid = 2
    Gaussian = 3

def switch(function, input):
    ActivationFunctions = {
        Activation.ReLu: relu(input),
        Activation.Sigmoid: sigmoid(input),
        Activation.Gaussian: gaussian(input)
    }
    return ActivationFunctions.get(function, "Invalid Function")


''' ACTIVATION FUNCTIONS '''

def relu(input):
    '''
    The ReLu activation function is to perform a threshold
    operation to each input element where values less 
    than zero are set to zero.
    '''
    return np.maximum(0, input)

def sigmoid(input):
    '''
    The sigmoid function takes in real numbers in any range and 
    squashes it to a real-valued output between 0 and 1.
    '''
    return 2.0 / (1.0 + np.exp(-input)) - 1

def tanh(input):
    '''
    The tanh function take real number in any range and
    sqashes it to a real-valued output between 0 and 1.
    '''
    return np.tanh(input)

def gaussian(input):
    '''
    The gaussian function
    '''
    return 2 * np.power(np.e, -np.power(input, 2)) - 1


''' DERIVATES '''

def relu_derivate(input):
    '''
    ReLu derivate
    '''
    input[input<=0] = 0
    input[input>0] = 1
    return input

def sigmoid_derivate(input):
    '''
    Sigmoid derivate
    '''
    return 0.5 / np.power(np.cosh(0.5 * input), 2)

def tanh_derivate(input):
    '''
    Tanh derivate
    '''
    return (1 - (input ** 2))

def gaussian_derivate(input):
    '''
    Gaussian derivate
    '''
    return -4 * np.power(np.e, -np.power(input, 2)) * input

