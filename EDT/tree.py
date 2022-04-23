"""
    This class represent the decision tree
    Is composed By Leaf and Decision
    LEAF: represent the class to which the objects belongs ( are the nodes that give us the classification)
    DECISION: represent the decision node and use a rules to choose which of the two children will have the decision
              - it can have 2 type of childrens : Decision or Leafs
"""
import random
from enum import Enum


class NodesTypes(Enum):
    LEAF = 0
    DECISION = 1

LEFT_CHILD = 0
RIGH_CHILD = 1




class Leaf:
    # constructor#
    def __init__(self, num_of_classes):
        self._score = 0.0
        self._result_class = random.randrange(num_of_classes)
        self._num_of_classes = num_of_classes
        self._depthTree = 0

    # Getter and Setter Method
    def set_score(self, score):
        self._score = score

    def set_result(self, result):
        self._result = result

    def get_score(self):
        return self._score

    def get_result(self):
        return self._result

    def set_depthTree(self, depthTree):
        self._depthTree = depthTree

    def get_depthTree(self):
        return self._depthTree


#rule is the class used to chose how we do the decision#
class Rule:
    def __init__(self,num_features,min_max):
        #chose a random number in number of feature to make a decision on this feature
        self._index=random.randrange(num_features)

        #computing the treshold
        self._treshold=random.uniform(min_max[self._index][0],min_max[self._index][1])
        self._num_features=num_features
        self._min_max=min_max

    #Getter and Setter
    def set_index(self,index):
        self._index=index

    def set_treshold(self,treshold):
        self._treshold=treshold

    def get_index(self):
        return self._index

    def get_treshold(self):
        return self._treshold


class Decision():
    #contructor#
    def __init__(self,rule):
        #contain the childs of the node
        self._children = []
        self._rule=rule
        self._score=0.0
        self._depthTree = 0

    #Getter and Setter
    def set_score(self,score):
        self._score = score

    def get_score(self):
        return self._score

    def set_depthTree(self, depthTree):
        self._depthTree = depthTree

    def get_depthTree(self):
        return self._depthTree

    def add_child(self, new_child):
        self._children.append(new_child)



