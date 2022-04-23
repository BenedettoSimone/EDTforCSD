import random
from tree import NodesTypes, Leaf, Rule, Decision

'''
This class is used to generate random tree
'''


def generate_random_tree(min_depth, max_depth, num_of_classes, num_features, min_max):
    if min_depth <= 0:
        raise ValueError('min_depth must be at least 1')
    else:
        # choose random if the next node will be a Decision or a Leaf
        node_type = random.choices(list(NodesTypes))[0]

        if node_type is NodesTypes.LEAF:
            root = Leaf(num_of_classes)

        elif node_type is NodesTypes.DECISION:
            rule = Rule(num_features, min_max)
            root = Decision(rule)

            # add child to root
            root.add_child(tree_node(1, min_depth, max_depth, num_of_classes, num_features, min_max))
            root.add_child(tree_node(1, min_depth, max_depth, num_of_classes, num_features, min_max))

        else:
            raise ValueError('Enum type error: {}'.format(type))
    return root


def tree_node(depth, min_depth, max_depth, num_of_classes, num_features, min_max):
    node = None

    if depth == max_depth:
        node = Leaf(num_of_classes)
    elif min_depth <= depth <= max_depth:

        # choose random if the next node will be a Decision or a Leaf
        node_type = random.choices(list(NodesTypes))[0]

        if node_type is NodesTypes.LEAF:
            node = Leaf(num_of_classes)

        elif node_type is NodesTypes.DECISION:
            rule = Rule(num_features, min_max)
            node = Decision(rule)
            node.add_child(tree_node(depth + 1, min_depth, max_depth, num_of_classes, num_features, min_max))
            node.add_child(tree_node(depth + 1, min_depth, max_depth, num_of_classes, num_features, min_max))

        else:
            raise ValueError('Enum type error: {}'.format(type))

    return node
