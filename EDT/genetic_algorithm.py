import math

from copy import deepcopy
from random import randrange
import random

import numpy as np
from tabulate import tabulate
from sklearn.metrics import accuracy_score

from EDT.generate_random_tree import generate_random_tree
from tree import Leaf, Decision


data_to_print = []
SIZE_MATING_POOL = 0.2


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
        print("\u2501" * 50)
        print("STARTING POPULATION")
        population = self.__create_starting_population()

        best_trees = []

        # find best individual to see progress
        best_tree = self.__find_best(population)
        print("BEST TREE with fitness of:", self.__get_tree_score(best_tree))

        epoch_without_progress = 0

        for epoch in range(self._n_epochs):

            # execute genetic operators
            population = self.__execute_genetic_operators(population)

            epoch_best = self.__find_best(population)
            if self.__get_tree_score(epoch_best) < self.__get_tree_score(best_tree):
                epoch_without_progress += 1
                if epoch_without_progress >= stop_after_no_progress:
                    break
            else:
                best_tree = epoch_best

            best_trees.append(epoch_best)

            print('Epoch {} - best fitness: {}'.format(epoch, self.__get_tree_score(epoch_best)))

        return best_trees

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
        print(tabulate(data_to_print, headers=["Fitness", "Accuracy", "Height Score"]))

        return population

    # function that allow us to evaluate a tree
    def __evaluate_tree(self, tree):
        # compute a fitness score by approriately weighting accuracy and height

        alpha1 = 0.99  # penalty for misclassification
        alpha2 = 0.01  # penalty for large trees

        # predicted label
        Y_pred_train = np.apply_along_axis(tree.get_result, axis=1, arr=self._X_train)

        accuracy = accuracy_score(self._y_train, Y_pred_train)
        # print("accuracy", accuracy)
        # define height score
        height_score = 1 - tree.get_height_tree()

        fitness_score = (alpha1 * accuracy) + (alpha2 * height_score)
        if fitness_score < 0:
            fitness_score = 0

        tree.set_score(fitness_score)
        # print(self._y_train)
        # print(Y_pred_train)
        # print("FITNESS", fitness_score, "ACCURACY:", accuracy)
        data_to_print.append([fitness_score, accuracy, height_score])

    def __find_best(self, population):
        population.sort(key=self.__get_tree_score, reverse=True)
        return population[0]

    def __get_tree_score(self, tree):
        return tree.get_score()

    def __execute_genetic_operators(self, population):
        global data_to_print
        offsprings = []

        # Selection with roulette wheel
        mating_pool = self.__roulette_wheel(population)
        data_to_print = []

        # crossover
        print("\u2501" * 50)
        print("CROSSOVER")

        for i in range(self._population_size):
            new_child = self.__crossover(mating_pool)
            offsprings.append(new_child)

        print(tabulate(data_to_print, headers=["Fitness", "Accuracy", "Height Score"]))
        print("\u2501" * 50)

        print("MUTATION")

        # mutation
        data_to_print = []
        mutated_population = self.__mutation(offsprings)
        print(tabulate(data_to_print, headers=["Fitness", "Accuracy", "Height Score"]))
        print("\u2501" * 50)
        return mutated_population

    def __roulette_wheel(self, population):

        rotation = math.ceil(len(population) * SIZE_MATING_POOL)
        mating_pool = []
        for i in range(rotation):
            # Computes the totality of the population fitness
            population_fitness = sum([tree.get_score() for tree in population])
            # Computes for each individual the probability
            selection_probabilities = [tree.get_score() / population_fitness for tree in population]
            # Select one individual based on probabilities
            mating_pool.append(population[np.random.choice(len(population), p=selection_probabilities)])
        return mating_pool

    def __crossover(self, mating_pool):

        first_tree = None
        second_tree = None

        if len(mating_pool) < 2:
            first_tree = deepcopy(mating_pool[0])
            second_tree = deepcopy(first_tree)

        else:
            index_first_tree = randrange(len(mating_pool))

            # exclude index_first_tree from generation
            index_second_tree = random.choice(
                list(set([x for x in range(0, len(mating_pool))]) - set([index_first_tree])))

            first_tree = deepcopy(mating_pool[index_first_tree])
            second_tree = deepcopy(mating_pool[index_second_tree])

        subtree = second_tree.copy_subtree()
        first_tree.paste_subtree(subtree)

        new_height = self.__compute_height(first_tree)
        first_tree.set_height_tree(new_height)
        self.__evaluate_tree(first_tree)

        return first_tree

    def __mutation(self, offsprings):

        mutated_population = []
        # mutation

        for tree in offsprings:
            tree.mutate(self._num_features, self._min_max)
            self.__evaluate_tree(tree)
            mutated_population.append(tree)

        return mutated_population

    def __compute_height(self, tree):

        if isinstance(tree, Leaf):
            return 1
        elif isinstance(tree, Decision):
            children = tree.get_children()
            height_sx = self.__compute_height(children[0])
            height_dx = self.__compute_height(children[1])
            return max(height_sx, height_dx) + 1
