#!/usr/bin/env python3
"""
This module implements a decision tree
and contains the class Node, Leaf and Decision Tree.
"""

import numpy as np


class Node:
    """This class defines a nodes of a decision tree"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialization of Node instance."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Recursively calculates the maximum of the depths of the nodes."""

        if self.left_child is not None:
            max_depth_left = self.left_child.max_depth_below()

        if self.right_child is not None:
            max_depth_right = self.right_child.max_depth_below()

        if max_depth_left > max_depth_right:
            return max_depth_left
        return max_depth_right

    def count_nodes_below(self, only_leaves=False):
        """Recursively counts the number of nodes."""

        left_nodes = self.left_child.count_nodes_below(only_leaves)
        right_nodes = self.right_child.count_nodes_below(only_leaves)
        if only_leaves:
            return left_nodes + right_nodes
        return 1 + left_nodes + right_nodes

    def __str__(self):
        """Returns string representation of nodes"""
        def left_child_add_prefix(text):
            """Adds prefix +---> or  | to left branch lines."""
            lines = text.split("\n")
            new_text = "    +--"+lines[0]+"\n"
            for x in lines[1:]:
                new_text += ("    |  "+x)+"\n"
            return (new_text)

        def right_child_add_prefix(text):
            """Adds prefix +---> or  | to right branch lines."""
            lines = text.split("\n")
            new_text = "    +--" + lines[0] + "\n"
            for x in lines[1:]:
                if x.strip():
                    new_text += "           " + x + "\n"
            return new_text

        left_branch = self.left_child.__str__()
        res_left = left_child_add_prefix(left_branch)
        right_branch = self.right_child.__str__()
        res_right = right_child_add_prefix(right_branch)

        if self.is_root:
            return (
                f"root [feature={self.feature}, threshold={self.threshold}]\n"
                f"{res_left}{res_right}"
            )
        return (
            f"node [feature={self.feature}, threshold={self.threshold}]\n"
            f"{res_left}{res_right}"
        )

    def get_leaves_below(self):
        """Returns list of leaves."""
        left_branch = self.left_child.get_leaves_below()
        right_branch = self.right_child.get_leaves_below()

        return left_branch + right_branch

    def update_bounds_below(self):
        """generates dictionaries containing lower and upper bounds"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()

            if child == self.left_child:
                child.lower[self.feature] = self.threshold
            elif child == self.right_child:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Updates the indicator for Node class"""
        def is_large_enough(x):
            """Checks if all features are > than the lower bounds"""
            feature_conditions = np.array(
                [x[:, key] >= self.lower[key] for key in self.lower.keys()]
                )
            return np.all(feature_conditions, axis=0)

        def is_small_enough(x):
            """Checks if all features are < than the upper bounds"""
            feature_conditions = np.array(
                [x[:, key] <= self.upper[key] for key in self.upper.keys()]
                )
            return np.all(feature_conditions, axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0
            )

    def pred(self, x):
        """Generates a prediction using recursion"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """ This class defines a Leaf of a decision tree
    and inherits from Node."""
    def __init__(self, value, depth=None):
        """Initialization of leaf instance."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """calculates  the depth of the leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Counts the number of nodes below for a leaf."""
        return 1

    def __str__(self):
        """Returns string representation of the leaf"""
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """Returns list of leaves."""
        return [self]

    def update_bounds_below(self):
        """generates dictionaries containing lower and upper bounds"""
        pass

    def pred(self, x):
        """Returns value of leaf."""
        return self.value


class Decision_Tree():
    """This class defines a decision tree. """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialization of decision tree instance."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """calculates the maximum depths of the decision tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the number of nodes of the decision tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Returns string representation of the decision tree"""
        return self.root.__str__()

    def get_leaves(self):
        """Returns list of leaves."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """generates dictionaries containing lower and upper bounds"""
        self.root.update_bounds_below()

    def update_predict(self):
        """Defines the prediction function for the decision tree
        after updating bounds and updating indicator function for each leaf."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves],
            axis=0
            )

    def pred(self, x):
        """Generates a prediction using recursion."""
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """Fits the decision tree to explanatory variables/features
        and target variables or data."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
- Depth                     : { self.depth()       }
- Number of nodes           : { self.count_nodes() }
- Number of leaves          : { self.count_nodes(only_leaves=True) }
- Accuracy on training data : { self.accuracy(
    self.explanatory,self.target
    )    }""")

    def np_extrema(self, arr):
        """Returns max and min values."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Given a node to split,
        it returns a feature and threshold value to split data"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
                )
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1-x)*feature_min + x*feature_max
        return feature, threshold

    def is_leaf(self, sub_population):
        """Checks if a node should be leaf."""
        target_values = self.target[sub_population]
        return any([
            np.sum(sub_population) < self.min_pop,
            len(np.unique(target_values)) == 1,
            self.root.depth >= self.max_depth
        ])

    def fit_node(self, node):
        """Fits decision tree, splits data on a given node."""
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold
            )
        right_population = node.sub_population & (
            self.explanatory[:, node.feature] <= node.threshold
            )

        # Is left node a leaf ?
        is_left_leaf = self.is_leaf(left_population)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = self.is_leaf(right_population)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Returns a leaf child based on parent node and sub_population."""
        target_values = self.target[sub_population]
        unique_values = np.unique(target_values)
        counts = np.array(
            [np.sum(target_values == value) for value in unique_values]
            )
        value = unique_values[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth+1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates child node."""
        n = Node()
        n.depth = node.depth+1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Calculates decision tree accuracy."""
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
            )/test_target.size
