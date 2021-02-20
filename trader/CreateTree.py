from __future__ import annotations
from typing import List,Dict
import pandas as pd
import numpy as np
from copy import deepcopy

from DataStructures.Node import Node
from DataStructures.Terminal import Terminal


def recursive_tree_creation(
        parent:Node,
        terminals:List,
        vars_available:List[Dict],
        depth:int,
        max_depth:int):
    """Recursively constructs a tree. Number of children determined by unique
    number of terminals available.
    Parameters:
        parent (Node): The node which descended into this one.
        terminals (List): Set of terminals allowed.
        vars_available (Dict[str]): The list of column variables which are used
            in intermediate node creation mapped to initial thresholds
        depth (int): The current depth of creation.
        max_depth (int): The maximum depth allowed for this tree."""

    # Add the leaf children
    if depth == max_depth-1:
        for t in terminals:
            parent.add_child(Terminal(var_name=t))
        return
    
    # Add the nonleaf children
    for _ in terminals:
        node = Node(deepcopy(np.random.choice(vars_available)))
        parent.add_child(node)

        recursive_tree_creation(parent=node,
                                terminals=terminals,
                                vars_available=vars_available,
                                depth=depth+1,
                                max_depth=max_depth)


def create_tree(
        terminals:List,
        variables:List[Dict],
        depth:int)->Node:
    """Create a program tree using the data and terminals provided.
    Parameters:
        terminals (List): The set of terminals to use as tree leaves. These are
            sampled by replacement.
        variables (Dict): Map of variable name to mean value (initial threshold)
        depth (int): The maximum depth to build this tree to.
    Returns the root node with a chain to all of its children.
    """
    if depth <= 1:
        raise RuntimeError("Minimum tree depth is 2 for stumps.")

    selection = np.random.choice(variables)
    root = Node(variable=deepcopy(selection))

    recursive_tree_creation(
        parent=root,
        terminals=terminals,
        vars_available=variables,
        depth=1,
        max_depth=depth)

    return root


def create_stump(variable:Dict,terminals:List[str])->Node:
    """Create a node with two terminal children. This is the simplest valid
    tree structure.
    Parameters:
        variable (str): The variable to assign to the root node of this stump.
        terminals (List[str]): The terminals we are allowed to choose from in
            this stump creation.
    Returns the root node of the stump."""
    stump = Node(deepcopy(variable))
    for t in terminals:
        stump.add_child(Terminal(t))

    return stump


def create_indicator_tree(variables:Dict,
                          terminals:List[str],
                          popid:int,
                          depth:int=2)->Dict:
    """Create a pop from the variables and terminals given. Initializes a
    random tree for it.
    Parameters:
        variables (Dict): Valid variables mapped to initial thresholds to
            choose from when creating a tree.
        Terminals (List[str]): The valid terminal values allowed to be assigned.
        popid (int): The unique identifier to assign to this tree on creation.
        depth (int): The initial depth of the tree being created.
    Returns a dictionary containng the tree and some other data about the tree.
    """
    return {
        "popid": popid,
        "tree": create_tree(terminals=terminals,
                            variables=variables,
                            depth=depth),
        "fitness": None,
        "balance": None,
        "cluster": None,
        "coordinate": None}