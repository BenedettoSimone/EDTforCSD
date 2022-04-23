"""
    This class represent the decision tree
    Is composed By Leaf and Decision
    LEAF: represent the class to which the objects belongs ( are the nodes that give us the classification)
    DECISION: represent the decision node and use a rules to choose which of the two children will have the decision
              - it can have 2 type of childrens : Decision or Leafs
"""
import random


class Leaf:
    # constructor#
    def __init__(self, num_of_classes):
        self._score = 0.0
        self._result_class = random.randrange(num_of_classes)
        self._num_of_classes = num_of_classes

    # Getter and Setter Method
    def set_score(self, score):
        self._score = score

    def set_result(self, result):
        self._result = result

    def get_score(self):
        return self._score

    def get_result(self):
        return self._result
