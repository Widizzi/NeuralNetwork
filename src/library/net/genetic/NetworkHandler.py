import src.library.net.genetic.Network as Network
import src.library.net.genetic.GeneticLearning as GeneticLearning
import src.library.interface.cli.ProgressBar as ProgressBar

import numpy as np
import os
import matplotlib.pyplot as plt

class NetworkHandler:
    """
    The NetworkHandler runs multiple networks in parallel
    """

    def __init__(self, ratefunction, layers=[2,8,1], network_count=12, mutation_rate=0.05, iterations=100):

        self.genetic_learning = GeneticLearning.GeneticLearning(mutation_rate)
        self.progress_bar = ProgressBar.ProgressBar()
        self.ratefunction = ratefunction

        self.average = []

        self.params = {}
        self.layers = layers
        self.network_count = network_count
        self.iterations = iterations

    def init_networks(self):
        """
        Initialize the networks
        """
        self.params['Networks'] = np.array([Network.Network(layers=self.layers) for i in range(self.network_count)], dtype=object)

    def train(self, data):
        ''' 
        Trains the Networks
        '''
        self.init_networks()
        
        col, row = os.get_terminal_size()
        self.progress_bar.progressBar(self.iterations, prefix='Progress:', suffix='Complete', length=col-30)

        for iteration in range(self.iterations):

            average_prediction = 0

            for i in range(len(data)):
                self.data = data[i]

                """
                Average over iterations
                """
                average_net_prediction = 0
                for i in range(len(self.params['Networks'])):
                    average_net_prediction += self.params['Networks'][i].predict(self.data)
                average_prediction += average_net_prediction / len(self.params['Networks'])



                self.params['Rating'] = np.array([], dtype=object)

                for network in self.params['Networks']:
                    self.params['Rating'] = np.append(self.params['Rating'], network.iterate(self.data, self.ratefunction))

                post_weights = []
                post_biases = []
                
                for i in range(self.network_count // 2):
                    selected_weights = []
                    selected_biases = []
                    rating = np.array(self.params['Rating'], copy=True)
                    selected_rating = self.genetic_learning.select(rating)
                    for a in range(len(selected_rating)):
                        selected_weights.append(self.params['Networks'][selected_rating[a]].getWeights())
                        selected_biases.append(self.params['Networks'][selected_rating[a]].getBiases())
                    selected_weights = np.array(selected_weights)
                    selected_biases = np.array(selected_biases)
                    changed_weights = self.genetic_learning.mix(selected_weights)
                    changed_biases = self.genetic_learning.mix(selected_biases)
                    for weights in range(len(changed_weights)):
                        post_weights.append(changed_weights[weights])
                        post_biases.append(changed_biases[weights])

                for i in range(len(post_weights)):
                    post_weights[i] = self.genetic_learning.mutate(post_weights[i])
                    post_biases[i] = self.genetic_learning.mutate(post_biases[i])
                    self.params['Networks'][i].setWeights(post_weights[i])
                    self.params['Networks'][i].setBiases(post_biases[i])
                
            self.progress_bar.printProgressBar(iteration + 1)
            self.average.append(average_prediction / len(data))


    def predict(self, data):
        prediction_data = []
        ratingsum = 0
        average_prediction = 0
        for i in range(len(self.params['Networks'])):
            ratingsum += self.params['Networks'][i].iterate(data, self.ratefunction)
            prediction_data.append(self.params['Networks'][i].predict(data))
            average_prediction += prediction_data[i]
        
        print(average_prediction / len(self.params['Networks']))

        negative_prediction = 0
        positive_prediction = 0
        for data in prediction_data:
            if data < 0:
                negative_prediction += 1
            else:
                positive_prediction += 1

        print("Negative: " + str(negative_prediction))
        print("Positive: " + str(positive_prediction))
        print("RatingSum: " + str(ratingsum))

    def plot_prediction(self):
        '''
        Plots the prediction curve
        '''
        plt.plot(self.average)
        plt.xlabel("Iteration")
        plt.ylabel("Prediction")
        plt.title("Prediction curve for training")
        plt.show()