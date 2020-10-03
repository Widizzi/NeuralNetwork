import src.library.net.mathematical.Network as Network

import numpy as np

def run():

    ''' initializes the neural network '''
    neural_network = Network.Network(layers=[2,8,1], learning_rate=0.05, iterations=-1)

    ''' OR-Gate Training Data '''
    training_or_data = np.array([[0,0],[0,1],[1,0],[1,1]])
    training_or_results = np.array([[0],[1],[1],[1]])

    ''' NOR-Gate Training Data '''
    training_nor_data = np.array([[0,0],[0,1],[1,0],[1,1]])
    training_nor_results = np.array([[1],[0],[0],[0]])

    ''' AND-Gate Training Data '''
    training_and_data = np.array([[0,0],[0,1],[1,0],[1,1]])
    training_and_results = np.array([[0],[0],[0],[1]])

    ''' NAND-Gate Training Data '''
    training_nand_data = np.array([[0,0],[0,1],[1,0],[1,1]])
    training_nand_results = np.array([[1],[1],[1],[0]])

    ''' XOR-Gate Training Data '''
    training_xor_data = np.array([[0,0],[0,1],[1,0],[1,1]])
    training_xor_results = np.array([[0],[1],[1],[0]])

    ''' XNOR-Gate Training Data '''
    training_xnor_data = np.array([[0,0],[0,1],[1,0],[1,1]])
    training_xnor_results = np.array([[1],[0],[0],[1]])

    ''' trains the neural network '''
    neural_network.train(training_xor_data, training_xor_results)

    ''' tests the network and prints the predicted values '''
    print(neural_network.predict(training_xor_data))

    ''' plots the loss curve '''
    neural_network.plot_loss()