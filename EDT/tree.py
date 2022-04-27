"""
    This class represent the decision tree
    Is composed By Leaf and Decision
    LEAF: represent the class to which the objects belongs ( are the nodes that give us the classification)
    DECISION: represent the decision node and use a rules to choose which of the two children will have the decision
              - it can have 2 type of childrens : Decision or Leafs
"""
import random
from enum import Enum
from copy import deepcopy


class NodesTypes(Enum):
    LEAF = 0
    DECISION = 1


LEFT_CHILD = 0
RIGHT_CHILD = 1


class Leaf:
    # constructor#
    def __init__(self, num_of_classes):
        self._score = 0.0
        self._result_class = random.randrange(num_of_classes)
        self._num_of_classes = num_of_classes
        self._height_tree = 0

    # Getter and Setter Method
    def set_score(self, score):
        self._score = score

    def set_result(self, result):
        self._result_class = result

    def get_score(self):
        return self._score

    def get_result(self, data):
        return self._result_class

    def set_height_tree(self, height_tree):
        self._height_tree = height_tree

    def get_height_tree(self):
        return self._height_tree

    def copy_subtree(self):
        return self

    def mutate(self, num_features, min_max):
        self._result_class = random.randrange(self._num_of_classes)

    def __str__(self, level=0, feature_names=None, class_names=None):
        return '{}{}{} Class: {}\n'.format('|', '\t|' * (level - 1), '--$ ', class_names[self._result_class])

    def __deepcopy__(self, memodict={}):
        leaf = Leaf(deepcopy(self._num_of_classes))
        leaf.set_result(deepcopy(self._result_class))
        leaf.set_score(deepcopy(self._score))
        return leaf


# rule is the class used to chose how we do the decision#
class Rule:
    def __init__(self, num_features, min_max):
        # chose a random number in number of feature to make a decision on this feature
        self._index = random.randrange(num_features)

        # computing the treshold
        self._treshold = random.uniform(min_max[self._index][0], min_max[self._index][1])
        self._num_features = num_features
        self._min_max = min_max

    # Getter and Setter
    def set_index(self, index):
        self._index = index

    def set_treshold(self, treshold):
        self._treshold = treshold

    def get_index(self):
        return self._index

    def get_treshold(self):
        return self._treshold

    def pass_rule(self, feature_test):
        try:
            # check if the feature respect the rule
            return feature_test[self._index] >= self._treshold
        except RecursionError:
            return 0


class Decision:
    # contructor#
    def __init__(self, rule):
        # contain the childs of the node
        self._children = []
        self._rule = rule
        self._score = 0.0
        self._height_tree = 0

    # Getter and Setter
    def set_score(self, score):
        self._score = score

    def get_score(self):
        return self._score

    def set_height_tree(self, height_tree):
        self._height_tree = height_tree

    def get_height_tree(self):
        return self._height_tree

    def add_child(self, new_child):
        self._children.append(new_child)

    def get_children(self):
        return self._children

    def get_result(self, data):
        if self._rule.pass_rule(data):
            return self._children[LEFT_CHILD].get_result(data)
        else:
            return self._children[RIGHT_CHILD].get_result(data)

    def copy_subtree(self):
        random_num = random.randrange(1)

        if random_num == 0:
            return self._children[LEFT_CHILD].copy_subtree()
        else:
            return self._children[RIGHT_CHILD].copy_subtree()

    def paste_subtree(self, sub_tree):

        if isinstance(self._children[LEFT_CHILD], Leaf) and isinstance(self._children[RIGHT_CHILD], Leaf):
            child_type = random.choice([LEFT_CHILD, RIGHT_CHILD])
            self._children[child_type] = sub_tree

        elif isinstance(self._children[LEFT_CHILD], Decision) and isinstance(self._children[RIGHT_CHILD], Leaf):
            self._children[RIGHT_CHILD] = sub_tree

        elif isinstance(self._children[LEFT_CHILD], Leaf) and isinstance(self._children[RIGHT_CHILD], Decision):
            self._children[LEFT_CHILD] = sub_tree

        else:
            random_num = random.randrange(1)

            if random_num == 0:
                return self._children[LEFT_CHILD].paste_subtree(sub_tree)
            else:
                return self._children[RIGHT_CHILD].paste_subtree(sub_tree)

    def mutate(self, num_features, min_max):

        random_num = random.randrange(1)
        if random_num == 0:
            # change rule
            new_rule = Rule(num_features, min_max)
            self._rule = new_rule
        else:
            # mutate random child
            self._children[random.choice([LEFT_CHILD, RIGHT_CHILD])].mutate()

    def __str__(self, level=0, feature_names=None, class_names=None):
        index = self._rule.get_index()
        threshold = self._rule.get_treshold()
        text = '{}{}{}{}>={}\n'.format('|', '\t|' * (level - 1), '--- ', feature_names[index], threshold)
        text += self._children[LEFT_CHILD].__str__(level + 1, feature_names, class_names) + self._children[
            RIGHT_CHILD].__str__(level + 1, feature_names, class_names)
        return text

    def __deepcopy__(self, memodict={}):
        decision = Decision(deepcopy(self._rule))
        decision.add_child(deepcopy(self._children[LEFT_CHILD]))
        decision.add_child(deepcopy(self._children[RIGHT_CHILD]))
        decision.set_score(deepcopy(self._score))
        return decision
