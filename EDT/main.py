import os
import shutil

import colorama
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from genetic_algorithm import GeneticAlgorithm


def execute_task(dataset_path, log_file):
    dataset = pd.read_csv(dataset_path)

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    print("\u2501" * 50, colorama.Style.BRIGHT, colorama.Fore.YELLOW)
    print(log_file.name[17:-4])

    print("Train size:", len(X_train), ", Test size:", len(X_test), colorama.Style.NORMAL, colorama.Fore.RESET)
    print("\u2501" * 50)

    # create Genetic algorithm

    genetic_algorithm = GeneticAlgorithm(population_size=15, n_epochs=10, min_depth=3, max_depth=10)

    # find population of best individuals
    best_individuals = genetic_algorithm.fit(X_train, y_train, 10)
    # choose best individual
    print("\u2501" * 50)
    counter = 1
    for i in best_individuals:
        print("Best individual", counter, ":", get_fitness(i))
        counter = counter + 1

    best_individuals.sort(key=get_fitness, reverse=True)
    best_tree = best_individuals[0]
    print("\u2500" * 50)
    print("\u2500" * 50)
    print(colorama.Fore.GREEN, "BEST TREE with fitness", get_fitness(best_tree), colorama.Fore.RESET)
    print("\u2500" * 50)
    print("\u2500" * 50)

    # make prediction on train set
    Y_pred_train = np.apply_along_axis(best_tree.get_result, axis=1, arr=X_train)
    accuracy_train = accuracy_score(y_train, Y_pred_train)
    print(colorama.Fore.LIGHTGREEN_EX, "Accuracy score in TRAIN set ", accuracy_train, colorama.Fore.RESET)

    # make prediction on test set
    Y_pred_test = np.apply_along_axis(best_tree.get_result, axis=1, arr=X_test)
    accuracy_test = accuracy_score(y_test, Y_pred_test)
    print(colorama.Fore.LIGHTGREEN_EX, "Accuracy score in TEST set ", accuracy_test, colorama.Fore.RESET)

    # precision, recall, f-measure
    per_class_precision = precision_score(Y_pred_test, y_test, average=None)
    print(colorama.Fore.LIGHTGREEN_EX, 'Precision score:', per_class_precision, colorama.Fore.RESET)

    recall = recall_score(Y_pred_test, y_test, average=None)
    print(colorama.Fore.LIGHTGREEN_EX, 'Recall score:', recall, colorama.Fore.RESET)

    F1_measure = f1_score(Y_pred_test, y_test, average=None)
    print(colorama.Fore.LIGHTGREEN_EX, 'F1 score:', F1_measure, colorama.Fore.RESET)

    if log_file is not None:
        log_file.write("=" * 200)
        log_file.write("\nFITNESS {}\n".format(get_fitness(best_tree)))
        log_file.write("Accuracy score in TRAIN set: {}\n".format(accuracy_train))
        log_file.write("Accuracy score in TEST set: {}\n".format(accuracy_test))
        log_file.write("Precision: {}\n".format(per_class_precision))
        log_file.write("Recall: {}\n".format(recall))
        log_file.write("F-measure: {}\n".format(F1_measure))
        log_file.write("=" * 200)
        log_file.write(best_tree.__str__(feature_names=X_train.columns.values.tolist(),
                                         class_names=["NOT " + y.columns.values.tolist()[0],
                                                      y.columns.values.tolist()[0]]))


def get_fitness(tree):
    return tree.get_score()


if __name__ == '__main__':

    path_directory = 'result'
    isExist = os.path.exists(path_directory)
    if isExist:
        shutil.rmtree(path_directory)
    os.mkdir(path_directory)

    # execute task
    os.chdir("../datasets/oversampling/")
    for path in os.listdir():
        with open("../../EDT/result/" + path[13:-4] + '.txt', 'w+') as file:
            execute_task(path, file)
    os.chdir("../..")
