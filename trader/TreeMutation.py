from __future__ import annotations
from typing import List,Dict

import numpy as np

from Common import random_choice
from DataStructures.Node import Node
from DataStructures.Terminal import Terminal

from CreateTree import (
    create_stump,
    threshold_at_depth)
    
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
    unused_variables = variables.copy()
    chosen = get_random_node(tree,include_root=True)

    if chosen.is_fixed():
        insert_node(chosen,unused_variables,terminals)
    
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

    chosen = get_random_node(tree)
    if isinstance(chosen,Terminal):
        replace_with_terminal(chosen,terminals)
    
    pop["tree"] = tree
    return pop


def replace_with_terminal(node:Node,terminals:List[str]):
    """Replace a node with a terminal. This is a tree reduction operation since
    it simplifies the tree."""
    old = node.get_variable()
    choices = [t for t in terminals if t != old]
    replacement = Terminal(np.random.choice(choices))
    replace_node(node,new_node=replacement)


def replace_with_node(node:Node,variables:Dict[str],terminals:List[str])->Node:
    """Replace a node in the tree with a node inplace or replace a terminal
    with a node stump. This is an expansion operation since it generally leads
    to growing the tree."""
    selection = np.random.choice([*variables])
    threshold = threshold_at_depth(node_depth(node),
                                   variables[selection],
                                   is_left_child(node))
    if isinstance(node,Terminal):
        replacement = create_stump(selection,threshold,terminals)
        return replace_node(node,new_node=replacement)
    else:
        replacement = Node(selection,initial_threshold=threshold)
        return replace_node(node,new_node=replacement)


def insert_node(node:Node,variables:Dict[str],terminals:List[str]):
    """Inserts a node between the node specified and its parent. Node must
    not be a root."""
    selection = np.random.choice([*variables])
    threshold = threshold_at_depth(node_depth(node),
                                   variables[selection],
                                   is_left_child(node))
    parent = node.get_parent()

    insertion = Node(selection,initial_threshold=threshold)
    idx = parent.remove_child(node)
    parent.add_child(insertion,idx)

    insertion.add_child(node)
    insertion.add_child(Terminal(np.random.choice(terminals)))