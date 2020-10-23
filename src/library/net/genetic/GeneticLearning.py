import random
import numpy as np

class GeneticLearning:
    """
    This class combines the weights from different nets to learn in a genetic way
    """
    
    def __init__(self, mutation_rate=0.05):

        self.mutation_rate = mutation_rate

    def select(self, ratings):
        elected_rating = np.array([], dtype=int)

        for i in range(2):
            ratingsum = 0
            for i in ratings:
                ratingsum += i
        
            randint = random.randint(1, ratingsum)

            count = 0
            for a in range(len(ratings)):
                count += ratings[a]
                if randint <= count:
                    elected_rating = np.append(elected_rating, a)
                    ratings[a] = 0
                    break

        return elected_rating
    
    def mix(self, weights):

        '''
        takes a random amount of weights to change and saves the random indexes to an array
        '''
        
        exchange_count = random.randint(0, len(weights[0]))
#        print("count: " + str(exchange_count))

        exchange_indexes = []
        for i in range(exchange_count):
            index = random.randint(0, len(weights[0]) - 1 - i)
#            print("index: " + str(index))
            for a in range(len(exchange_indexes)):
                if exchange_indexes[a] == index:
                    index += 1
            exchange_indexes.append(index)
            exchange_indexes.sort()
        exchange_indexes = np.array(exchange_indexes)
#        print("exchange" + str(exchange_indexes))

        '''
        swiches the random indexes between the arrays
        '''
        placeholder = []
        for net in range(len(weights)):
            placeholder.insert(0, weights[net])

        placeholder = np.array(placeholder)
        for net in range(len(weights)):
            for index in exchange_indexes:
                weights[net][index] = placeholder[net][index]
        
        return weights


    def mutate(self, weight):

        '''
        takes a random amount of weights to mutate and saves the random indexes to an array
        '''
        
        mutate_count = random.randint(0, len(weight))
#        print("count: " + str(mutate_count))

        mutate_indexes = []
        for i in range(mutate_count):
            index = random.randint(0, len(weight) - 1 - i)
#            print("index: " + str(index))
            for a in range(len(mutate_indexes)):
                if mutate_indexes[a] == index:
                    index += 1
            mutate_indexes.append(index)
            mutate_indexes.sort()
        mutate_indexes = np.array(mutate_indexes)

        '''
        mutates the random indexes
        '''

        for index in mutate_indexes:
            weight[index] += random.uniform(-self.mutation_rate, self.mutation_rate)
        
        return weight