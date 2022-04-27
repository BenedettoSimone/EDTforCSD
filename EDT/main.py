import colorama
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from EDT.genetic_algorithm import GeneticAlgorithm


def execute_task(dataset_path, log_file):
    dataset = pd.read_csv(dataset_path)

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    print("Train size:", len(X_train), ", Test size:", len(X_test))

    # create Genetic algorithm

    genetic_algorithm = GeneticAlgorithm(10, 10, 5, 10)

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
    print(colorama.Fore.MAGENTA, "Accuracy score in TRAIN set ", accuracy_train, colorama.Fore.RESET)

    # make prediction on test set
    Y_pred_test = np.apply_along_axis(best_tree.get_result, axis=1, arr=X_test)
    accuracy_test = accuracy_score(y_test, Y_pred_test)
    print(colorama.Fore.YELLOW, "Accuracy score in TEST set ", accuracy_test, colorama.Fore.RESET)


    if log_file is not None:
        log_file.write("=" * 200)
        log_file.write("\nFITNESS {}\n".format(get_fitness(best_tree)))
        log_file.write("Accuracy score in TRAIN set {}\n".format(accuracy_train))
        log_file.write("Accuracy score in TEST set {}\n".format(accuracy_test))
        log_file.write("=" * 200)
        log_file.write(best_tree.__str__(feature_names=X_train.columns.values.tolist(),class_names=["NO complex class", "Complex class"]))


def get_fitness(tree):
    return tree.get_score()


if __name__ == '__main__':
    with open('log.txt', 'w+') as file:
        # execute task
        execute_task('../datasets/oversampling/oversampling_ComplexClass.csv', file)
