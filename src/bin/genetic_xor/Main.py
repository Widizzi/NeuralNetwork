import src.library.net.genetic.NetworkHandler as NetworkHandler

import numpy as np

def run():

    ''' initializes the network handler '''
    network_handler = NetworkHandler.NetworkHandler(ratefunction, layers=[2,1], network_count=100, mutation_rate=0.08, iterations=2000)

    data = np.array([[0,0],[0,1],[1,0],[1,1]])

    network_handler.train(data)

    network_handler.predict(np.array([0,0]))
    network_handler.predict(np.array([0,1]))
    network_handler.predict(np.array([1,0]))
    network_handler.predict(np.array([1,1]))

    network_handler.plot_prediction()

def ratefunction(prediction, data):
    return int((1 / np.abs(np.power(data[0] + data[1], 2) - prediction)) * 10)