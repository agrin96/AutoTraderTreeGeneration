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
    replace_node)


def mutate(pop:Dict,
           variables:Dict[str],
           terminals:List[str],
           unique:bool=True,
           probability:float=0.5,
           mutation_types:Dict={
               "replace":0.50,
               "insert_node":0.25,
               "insert_terminal":0.25
           })->Dict:
    """Execute a point mutation on the pop with the given probability of a 
    mutation occuring at all. Three types of mutations can take place. 
    The first replaces a node or adds a stump (addition). The second replaces 
    a node with a terminal (deletion). And the third modifies a threshold or
    changes a terminal type (modification). Note the thid doesn't change the
    tree structure.
    Parameters:
        unique (bool): Whether to make sure that all variables in the tree are
            unique (sampled without replacement) when mutating.
        probability (float): Liklihood of mutation occuring at all.
        mutation_types (Dict): A dictionary indicating the likelihood of each
            variant of mutation described above occuring. Setting to 0 means
            that mutation type will never occur.
    """
    pop["tree"] = point_mutate(
        pop["tree"],
        variables,
        terminals,
        unique,
        probability,
        mutation_types)
    return pop


def point_mutate(
        node:Node,
        variables:Dict[str],
        terminals:List[str],
        unique:bool=True,
        probability:float=0.5,
        mutation_types:Dict={
            "replace":0.50,
            "insert_node":0.25,
            "insert_terminal":0.25
        })->Node:
    """Execute a point mutation on the tree with the given probability of a 
    mutation occuring at all. Three types of mutations can take place. 
    The first replaces a node or adds a stump (addition). The second replaces 
    a node with a terminal (deletion). And the third modifies a threshold or
    changes a terminal type (modification). Note the thid doesn't change the
    tree structure.
    Parameters:
        unique (bool): Whether to make sure that all variables in the tree are
            unique (sampled without replacement) when mutating.
        probability (float): Liklihood of mutation occuring at all.
        mutation_types (Dict): A dictionary indicating the likelihood of each
            variant of mutation described above occuring. Setting to 0 means
            that mutation type will never occur.
    """
    if not random_choice(prob_true=probability):
        return node
    
    unused_variables = variables.copy()
    if unique:
        # Essentially the same as set(A)-set(B), but for dictionaries.
        used_variables = list_tree_variables(node,with_threshold=True)
        all(map(unused_variables.pop,[*used_variables]))

        if len(list(used_variables.keys())) <= 1:
            return node

    chosen = get_random_node(node)

    if chosen.is_fixed():
        insert_node(chosen,unused_variables,terminals)
        return node
    
    if random_choice(prob_true=mutation_types["replace"]):
        replace_with_node(chosen,unused_variables,terminals)
        return node

    if random_choice(prob_true=mutation_types["insert_node"]):
        insert_node(chosen,unused_variables,terminals)
        return node

    if random_choice(prob_true=mutation_types["insert_terminal"]):
        replace_with_terminal(chosen,terminals)
        return node
    
    if isinstance(chosen,Terminal):
        replace_with_terminal(chosen,terminals)
    else:
        threshold = chosen.get_threshold()
        threshold = threshold + threshold*(np.random.rand()-0.5)
        chosen.set_threshold(threshold)
    return node


def replace_with_terminal(node:Node,terminals:List[str]):
    """Replace a node with a terminal. This is a tree reduction operation since
    it simplifies the tree."""
    old = node.get_variable()
    choices = [t for t in terminals if t != old]
    replacement = Terminal(np.random.choice(choices))
    replace_node(node,new_node=replacement)


def replace_with_node(node:Node,variables:Dict[str],terminals:List[str]):
    """Replace a node in the tree with a node inplace or replace a terminal
    with a node stump. This is an expansion operation since it generally leads
    to growing the tree."""
    selection = np.random.choice([*variables])
    if isinstance(node,Terminal):
        replacement = create_stump(selection,variables[selection],terminals)
        replace_node(node,new_node=replacement)
    else:
        replacement = Node(selection,initial_threshold=variables[selection])
        replace_node(node,new_node=replacement,include_children=True)


def insert_node(node:Node,variables:Dict[str],terminals:List[str]):
    """Inserts a node between the node specified and its parent. Node must
    not be a root."""
    selection = np.random.choice([*variables])
    parent = node.get_parent()

    insertion = Node(selection,initial_threshold=variables[selection])
    idx = parent.remove_child(node)
    parent.add_child(insertion,idx)

    insertion.add_child(node)
    insertion.add_child(Terminal(np.random.choice(terminals)))