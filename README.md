# EDTforCSD
Evolutionary Decision Tree for Code Smell Detection.

## Requirements
* Python 3.9
* pandas 1.4.2
* impyute 0.0.8
* matplotlib 3.5.1
* imbalanced-learn 0.9.0
* scikit-learn 1.0.2
* seaborn 0.11.2
* numpy 1.22.3
* colorama 0.4.4
* tabulate 0.8.9

## Usage
To replicate our experiments, you need to clone the repository:
```
git clone https://github.com/DanieleSalerno/EDTforCSD
```

## Dataset
The path of original dataset is:
```
datasets/complete_dataset.csv
```

### Data imputation
Since we had some features with NaN values we applied *KNN imputation*. The imputed dataset is available in:
```
datasets/imputed_dataset.csv
```
If you want to perform other experiments use ``data_imputation.ipynb``


### Data balancing
Because the dataset in unbalanced, we need to apply the oversampling technique. 



Since our goal is to build ad-hoc classifiers for each type of code smell we used 5 datasets considering one smell for each dataset.

Before applying oversampling, we performed Stratified K-Fold obtaining 10 folds for each dataset.
Then, we applied oversampling on the generated training sets while leaving the test sets intact. To create new artificial instances, we used the Synthetic Minority Oversampling Technique (SMOTE). 

You can find the five datasets, splitted in 10 folds, in:
```
datasets/stratifiedKfold/
```
To perform data balancing use ``data_balancing.ipynb``

## Evolutionary Decision Tree

### Tree structure
Tree is composed by classes ``Leaf`` and ``Decision``.

 * ``Leaf``: represent the class to which the objects belongs (are the nodes that give us the classification).
 * ``Decision``: represent the decision node and use a ``Rule`` to choose which of the two children will have the decision.
   * It can have two types of children : Decision or Leaf.

The structure of tree is available in ``EDT/tree.py``



### Genetic Algorithm
``GeneticAlgorithm`` class represents the Genetic Algorithm, available in ``EDT/genetic_algorithm.py`` 

>#### 1. <u>Generate starting population</u>
>The method ``generate_random_tree()`` perform the random creation of first population. First generate a ``Decision`` node as root, and then choose random if the next node will be a Decision or a Leaf. <br><br>
>The last step will be repeated until the tree has reached the maximum depth or when all the nodes of the last level are leaves.


>#### 2. <u>Fitness function</u>
>In order to evaluate each tree, we defined the following fitness function:
>```
>fitness_score = (alpha1 * accuracy) + (alpha2 * height_score)
> ```
>* *alpha1* is penalty for misclassification
>* *alpha2* is penalty for large trees


>#### 3. <u>Selection</u>
> To select parents to create next generation we used *Roulette Wheel Selection*, assigning at each individual a probability of selection based on their fitness score. <br><br>
> The size of mating pool is 20% of population size.

>#### 4. <u>Crossover</u>
> To perform the crossover we combined pairs of parents.
> To obtain a new child:
>* we select a random node of first parent;
>* we copy the entire subtree starting from this node;
>* we randomly select a leaf of second parent;
>* we replace the leaf with subtree.

>#### 5. <u>Mutation</u>
> To perform the mutation we randomly select a node:
>* if the type of the node is a **Leaf** we change the ``result_class``;
>* if the type is a **Decision** we change the ``Rule``.


>#### 6. <u>Termination</u>
> The algorithm stops when it arrives to the last iteration, or when for a determined number of iterations do not happen improvements.


### Run it
To execute the program use the follow instruction in command line:
```
python3 main.py
```

## Developed by
[Salerno Daniele](https://github.com/DanieleSalerno) <br>
[Simone Benedetto](https://github.com/BenedettoSimone)
