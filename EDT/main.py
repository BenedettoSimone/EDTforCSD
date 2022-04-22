import pandas as pd
from sklearn.model_selection import train_test_split


def execute_task(dataset_path, log_file):
    dataset = pd.read_csv(dataset_path)

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    print("Train size:", len(X_train), ", Test size:", len(X_test))

    # create Genetic algorithm
        
    # find population of best individuals

    # choose best individual

    # make prediction on train set

    # make prediction on test set

    # show results


if __name__ == '__main__':
    with open('log.txt', 'w+') as file:
        # execute task
        execute_task('../datasets/oversampling/oversampling_ComplexClass.csv', file)
