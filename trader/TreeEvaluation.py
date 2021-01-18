from __future__ import annotations
from typing import List,Union
import pandas as pd
import numpy as np
import uuid

from Common import random_choice
from DataStructures.Node import Node
from DataStructures.Terminal import Terminal
from TreeActions import tree_depth,get_node,replace_node


def evaluate_tree(node:Node,substitutions:Dict)->str:
    """Evaluates a single instance of substitution data on the tree specified.
    Returns the single decision result of the evaluation."""
    if isinstance(node,Terminal):
        return str(node)
    
    next_node = node.evaluate(substitutions)
    return evaluate_tree(next_node,substitutions)


def step_thresholds(tree:Node,generation:int,step_percent:float):
    """Changes the tree thresholds based on the step_percent anealed by the
    generation count.
    Parameters:
        tree (Node): The root node of the tree we are adjusting
        generation (int): The current generation we are at to use to simulate 
            our annealing steps.
        step_percent (float): The initial base step used expressed as a decimal
            percent"""
    if isinstance(tree,Node):
        tree.step_threshold(base_step=step_percent,generation=generation)
        first,second = tree.children()

        step_thresholds(first,generation,step_percent)
        step_thresholds(second,generation,step_percent)
