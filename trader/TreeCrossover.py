from __future__ import annotations
from typing import List,Dict

import numpy as np

from Common import random_choice
from DataStructures.Node import Node
from DataStructures.Terminal import Terminal
from CreateTree import create_stump
from TreeActions import (
    list_tree_variables,
    list_tree_terminals,
    get_random_node,
    replace_node,
    clone_node)

def crossover_reproduction(treeA:Node,treeB:Node,probability:float=0.5)->Node:
    """Creates a pair of child nodes from the reproduction of two nodes 
    provided with the probability of a crossover being determined by the 
    probability value passed in. If no crossover takes place, then the 
    trees are simply copied."""
    if not random_choice(prob_true=probability):
        return clone_node(treeA),clone_node(treeB)

    child_a = clone_node(treeA)
    child_b = clone_node(treeB)
    
    cross_a = get_random_node(child_a)
    cross_b = get_random_node(child_b)
    
    parent_a = cross_a.get_parent()
    parent_b = cross_b.get_parent()
    
    posA = parent_a.remove_child(cross_a)
    posB = parent_b.remove_child(cross_b)

    parent_a.add_child(cross_b,posA)
    parent_b.add_child(cross_a,posB)

    return child_a,child_b

