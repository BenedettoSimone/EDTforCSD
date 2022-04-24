import numpy as np
from sklearn.metrics import accuracy_score

from EDT.generate_random_tree import generate_random_tree


class GeneticAlgorithm:
    # constructor
    def __init__(self, population_size, n_epochs, min_depth, max_depth):
        self._population_size = population_size
        self._n_epochs = n_epochs
        self._X_train = None
        self._y_train = None
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._num_of_classes = None
        self._num_features = None
        self._min_max = None

    def fit(self, X_train, y_train, stop_after_no_progress):
        self._X_train = X_train.to_numpy()
        self._y_train = y_train.to_numpy()

        # get useful information from dataset
        self.__process_data()

        # generate starting population
        population = self.__create_starting_population()
        print(population)
        for pop in population:
            print(pop)
            print(pop.get_height_tree())
            print(pop.get_score())
            print("===================================")

        # find best individual to see progress
        '''
        repeat for num_epochs
            generate new population with selection, crossover and mutation
            find best individual
            compare it with current best individuals
            exit with stopping condition (>num_epoch, >epoch_no_progress
        return all best individuals 
        '''

    # function to get useful information from dataset
    def __process_data(self):
        # get min value for each feature
        X_min = self._X_train.min(axis=0)

        # reshape in one column
        X_min = X_min.reshape(len(X_min), 1)

        # get max value for each feature
        X_max = self._X_train.max(axis=0)
        X_max = X_max.reshape(len(X_max), 1)

        # concatenate min e max for each feature
        self._min_max = np.concatenate((X_min, X_max), axis=1)

        # number of classes
        self._num_of_classes = len(np.unique(self._y_train))

        # number of features
        self._num_features = self._X_train.shape[1]

    # function for create the starting population
    def __create_starting_population(self):
        population = []

        for i in range(self._population_size):
            # create new random tree
            new_tree = generate_random_tree(self._min_depth, self._max_depth, self._num_of_classes, self._num_features,
                                            self._min_max)
            # Evaulating the tree
            self.__evaluate_tree(new_tree)
            # Adding the tree to the population
            population.append(new_tree)
        return population

    # function that allow us to evaluate a tree
    def __evaluate_tree(self, tree):
        # compute a fitness score by approriately weighting accuracy and height

        alpha1 = 0.99  # penalty for misclassification
        alpha2 = 0.01  # penalty for large trees

        # predicted label
        Y_pred_train = np.apply_along_axis(tree.get_result, axis=1, arr=self._X_train)

        accuracy = accuracy_score(self._y_train, Y_pred_train)
        print("accuracy",accuracy)
        #define height score
        height_score = 1 - tree.get_height_tree()

        fitness_score = (alpha1 * accuracy) + (alpha2 * height_score)

        tree.set_score(fitness_score)
        print(self._y_train)
        print(Y_pred_train)
