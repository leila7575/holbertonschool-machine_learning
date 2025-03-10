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
