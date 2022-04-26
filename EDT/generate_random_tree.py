import random
from tree import NodesTypes, Leaf, Rule, Decision

'''
This class is used to generate random tree
'''


def generate_random_tree(min_depth, max_depth, num_of_classes, num_features, min_max):
    if min_depth <= 0:
        raise ValueError('min_depth must be at least 1')
    else:
            rule = Rule(num_features, min_max)
            root = Decision(rule)

            # add child to root
            root.add_child(tree_node(1, min_depth, max_depth, num_of_classes, num_features, min_max))
            root.add_child(tree_node(1, min_depth, max_depth, num_of_classes, num_features, min_max))

            # get children to compute height of tree
            children = root.get_children()
            height = max(children[0].get_height_tree(), children[1].get_height_tree())

            #print("Compute max", children[0].get_height_tree(), children[1].get_height_tree())
            root.set_height_tree(height)
            #print("Tree height", root.get_height_tree())

    return root


def tree_node(depth, min_depth, max_depth, num_of_classes, num_features, min_max):
    node = None

    if depth == max_depth:
        node = Leaf(num_of_classes)
        node.set_height_tree(1)
        #print("1-Leaf height", node.get_height_tree())

    elif depth < min_depth:

        rule = Rule(num_features, min_max)
        node = Decision(rule)
        depth = depth + 1
        node.add_child(tree_node(depth, min_depth, max_depth, num_of_classes, num_features, min_max))
        node.add_child(tree_node(depth, min_depth, max_depth, num_of_classes, num_features, min_max))
        children = node.get_children()
        height = 1 + max(children[0].get_height_tree(), children[1].get_height_tree())

        #print("1-Compute max", children[0].get_height_tree(), children[1].get_height_tree() )
        node.set_height_tree(height)
        #print("1-Decision height", node.get_height_tree())

    elif min_depth <= depth < max_depth:

        # choose random if the next node will be a Decision or a Leaf
        node_type = random.choices(list(NodesTypes))[0]

        if node_type is NodesTypes.LEAF:
            node = Leaf(num_of_classes)
            node.set_height_tree(1)
            #print("2-Leaf height", node.get_height_tree())

        elif node_type is NodesTypes.DECISION:
            rule = Rule(num_features, min_max)
            node = Decision(rule)
            depth = depth + 1
            node.add_child(tree_node(depth, min_depth, max_depth, num_of_classes, num_features, min_max))
            node.add_child(tree_node(depth, min_depth, max_depth, num_of_classes, num_features, min_max))

            children = node.get_children()
            height = 1 + max(children[0].get_height_tree(), children[1].get_height_tree())
            #print("2-Compute max", children[0].get_height_tree(), children[1].get_height_tree())
            node.set_height_tree(height)
            #print("2-Decision height", node.get_height_tree())

        else:
            raise ValueError('Enum type error: {}'.format(type))

    return node
