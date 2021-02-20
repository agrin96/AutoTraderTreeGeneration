from __future__ import annotations
from typing import List,Dict

import numpy as np
from copy import deepcopy

from Common import random_choice
from DataStructures.Node import Node
from DataStructures.Terminal import Terminal

from CreateTree import create_stump
    
from TreeActions import (
    get_random_node,
    replace_node,
    is_left_child,
    node_depth)


def point_mutate(
        pop:Dict,
        variables:Dict[str],
        terminals:List[str],
        probability:float=0.5,
        mutation_types:Dict={
            "replace":0.50,
            "insert_node":0.25,
            "insert_terminal":0.25
        })->Dict:
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
        return pop
    
    tree = pop["tree"]
    unused_variables = deepcopy(variables)
    chosen = get_random_node(tree,include_root=True)
    
    if random_choice(prob_true=mutation_types["replace"]):
        chosen = get_random_node(tree,include_root=True)
        new_node = replace_with_node(chosen,unused_variables,terminals)
        tree = new_node if new_node else tree

    if random_choice(prob_true=mutation_types["insert_node"]):
        chosen = get_random_node(tree)
        insert_node(chosen,unused_variables,terminals)

    if random_choice(prob_true=mutation_types["insert_terminal"]):
        chosen = get_random_node(tree)
        replace_with_terminal(chosen,terminals)

    if random_choice(prob_true=mutation_types["parameter"]):
        chosen = get_random_node(tree,include_root=True,include_terminals=False)
        mutate_indicator_parameters(chosen)
    
    pop["tree"] = tree
    return pop


def mutate_indicator_parameters(node:Node):
    """Mutate one of the parameter variables of the indicator stored in the 
    node."""
    indicator = node.get_variable()
    to_mutate = np.random.choice(list(indicator["variables"].keys()))
    
    low = indicator["variables"][to_mutate]["range"]["lower"]
    high = indicator["variables"][to_mutate]["range"]["upper"]

    modulator = 1
    is_float = False
    if high - low == 2 or high - low == 1:
        modulator = 100
        is_float = True
    
    # This is cleaner than using the random floats which can be ridiculous
    # in their length.
    new_value = np.random.randint(low*modulator,high*modulator) / modulator
    new_value = float(new_value) if is_float else int(new_value)

    indicator["variables"][to_mutate]["value"] = new_value
    
    node.set_variable(indicator)


def replace_with_terminal(node:Node,terminals:List[str]):
    """Replace a node with a terminal. This is a tree reduction operation since
    it simplifies the tree."""
    parent = node.get_parent()
    for term,child in zip(terminals,parent.children()):
        if child.node_id() == node.node_id():
            replacement = Terminal(term)    
            replace_node(node,new_node=replacement)
            return


def replace_with_node(node:Node,variables:List[Dict],terminals:List[str])->Node:
    """Replace a node in the tree with a node inplace or replace a terminal
    with a node stump. This is an expansion operation since it generally leads
    to growing the tree."""
    selection = np.random.choice(variables)
    if isinstance(node,Terminal):
        replacement = create_stump(selection,terminals)
        return replace_node(node,new_node=replacement)
    else:
        replacement = Node(deepcopy(selection))
        return replace_node(node,new_node=replacement)


def insert_node(node:Node,variables:List[Dict],terminals:List[str]):
    """Inserts a node between the node specified and its parent. Node must
    not be a root."""
    parent = node.get_parent()

    insertion = Node(deepcopy(np.random.choice(variables)))
    idx = parent.remove_child(node)
    parent.add_child(insertion,idx)

    insertion.add_child(node)
    insertion.add_child(Terminal(terminals[1]))
    insertion.add_child(Terminal(terminals[2]))