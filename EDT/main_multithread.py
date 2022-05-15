import os
import shutil
import threading

import colorama
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statistics import mean
from genetic_algorithm import GeneticAlgorithm


def execute_task(log_file, path, dir):
    train_set_over = pd.read_csv("datasets/stratifiedKFold/" + path + "/" + dir + "/" + "train_set_oversampled.csv")
    X_train = train_set_over.iloc[:, :-1]
    y_train = train_set_over.iloc[:, -1:]

    test_set = pd.read_csv("datasets/stratifiedKFold/" + path + "/" + dir + "/" + "test_set.csv")
    X_test = test_set.iloc[:, 1:-1]
    y_test = test_set.iloc[:, -1:]

    print("\u2501" * 50, colorama.Style.BRIGHT, colorama.Fore.YELLOW)
    print(path + ' - Fold ' + dir)

    print("Train size:", len(X_train), ", Test size:", len(X_test), colorama.Style.NORMAL, colorama.Fore.RESET)
    print("\u2501" * 50)

    # create Genetic algorithm
    genetic_algorithm = GeneticAlgorithm(population_size=4, n_epochs=1, min_depth=3, max_depth=5)

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

    F_measure = f1_score(Y_pred_test, y_test, average=None)
    print(colorama.Fore.LIGHTGREEN_EX, 'F-Measure:', F_measure, colorama.Fore.RESET)

    if log_file is not None:
        log_file.write("=" * 200)
        log_file.write("\nFITNESS {}\n".format(get_fitness(best_tree)))
        log_file.write("Accuracy score in TRAIN set: {}\n".format(accuracy_train))
        log_file.write("Accuracy score in TEST set: {}\n".format(accuracy_test))
        log_file.write("Precision: {}\n".format(per_class_precision))
        log_file.write("Recall: {}\n".format(recall))
        log_file.write("F-measure: {}\n".format(F_measure))
        log_file.write("=" * 200)
        log_file.write(best_tree.__str__(feature_names=X_train.columns.values.tolist(),
                                         class_names=["NOT " + y_train.columns.values.tolist()[0],
                                                      y_train.columns.values.tolist()[0]]))


def get_fitness(tree):
    return tree.get_score()


def get_f_measure(report_file):
    os.chdir("EDT/results")
    for dir in os.listdir():
        if '.txt' not in dir:
            f_smell = []
            f_not_smell = []
            os.chdir(dir)
            for file in os.listdir():

                file = open(file, "r")
                lines = file.readlines()
                file.close()

                for line in lines:
                    if "F-measure" in line:
                        line_to_save = line

                splitted = line_to_save.split()
                first_value = splitted[1][1:]

                if first_value == "0.":
                    first_value = first_value[:-1]

                second_value = splitted[2]

                if "]" in second_value:
                    second_value = second_value[:-1]

                if second_value == "0.":
                    second_value = second_value[:-1]

                f_not_smell.append(float(first_value))
                f_smell.append(float(second_value))

            print("=================================")
            print(dir)
            print("F-Measure: " + "[" + str(mean(f_not_smell)) + "," + str(mean(f_smell)) + "]")

            if report_file is not None:
                report_file.write("\nSmell type: {}\n".format(dir))
                report_file.write(
                    "F-Measure: {}\n".format("[" + str(mean(f_not_smell)) + "," + str(mean(f_smell)) + "]"))
                report_file.write("=" * 200)
            os.chdir("..")
    os.chdir("../..")
    print("================================")


def execute_thread(paths, type):
    path_directory = 'EDT/results/' + type
    isExist = os.path.exists(path_directory)
    if isExist:
        shutil.rmtree(path_directory)
    os.mkdir(path_directory)

    for path in paths:
        split_path = path.split("/", 1)
        substring_1 = split_path[0]
        substring_2 = split_path[1]

        # execute GA for each fold
        with open("EDT/results/" + substring_1 + '/' + substring_2 + '.txt', 'w+') as file:
            execute_task(file, substring_1, substring_2)


if __name__ == '__main__':

    path_directory = 'results'
    isExist = os.path.exists(path_directory)
    if isExist:
        shutil.rmtree(path_directory)
    os.mkdir(path_directory)

    # -------copy path of folds for each code smell folder---#
    paths_complex_class = []
    paths_large_class = []
    paths_lazy_class = []
    paths_refused_bequest = []
    paths_spaghetti_code = []

    os.chdir("../datasets/stratifiedKfold/")
    # folder smell
    for path in os.listdir():
        if '.DS_Store' not in path:
            os.chdir(path)
            # folder num fold
            for dir in os.listdir():
                if '.DS_Store' not in dir:
                    if path == "ComplexClass":
                        paths_complex_class.append(path + '/' + dir)
                    elif path == "LargeClass":
                        paths_large_class.append(path + '/' + dir)
                    elif path == "LazyClass":
                        paths_lazy_class.append(path + '/' + dir)
                    elif path == "RefusedBequest":
                        paths_refused_bequest.append(path + '/' + dir)
                    elif path == "SpaghettiCode":
                        paths_spaghetti_code.append(path + '/' + dir)
            os.chdir("..")
    os.chdir("../..")

    # creating threads
    t1 = threading.Thread(target=execute_thread, args=(paths_complex_class, "ComplexClass",))
    t2 = threading.Thread(target=execute_thread, args=(paths_large_class, "LargeClass",))
    t3 = threading.Thread(target=execute_thread, args=(paths_lazy_class, "LazyClass"))
    t4 = threading.Thread(target=execute_thread, args=(paths_refused_bequest, "RefusedBequest",))
    t5 = threading.Thread(target=execute_thread, args=(paths_spaghetti_code, "SpaghettiCode",))

    # starting threads
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()

    # wait until threads are completely executed
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()

    # threads completely executed
    print("Done!")

    with open("EDT/results/final_results.txt", 'w+') as report_file:
        get_f_measure(report_file)
