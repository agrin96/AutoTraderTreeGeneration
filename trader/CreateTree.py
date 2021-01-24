from __future__ import annotations
from typing import List,Union,Dict
import pandas as pd
import numpy as np
import uuid

from Common import random_choice
from DataStructures.Node import Node
from DataStructures.Terminal import Terminal
from TreeActions import tree_depth,get_node,replace_node

def create_initial_variables(data:pd.DataFrame)->Dict[str]:
    """Creates a mapping of column names of the data to their mean values which
    will be used as the initial threshold."""
    variables = data.columns.tolist()
    values = [data[v].mean() for v in variables]
    return dict(zip(variables,values))


def recursive_tree_creation(
        parent:Node,
        terminals:List,
        vars_available:Dict[str],
        depth:int,
        unique_nodes:bool,
        max_depth:int,
        full:bool):
    """Recursively constructs a tree.
    Parameters:
        parent (Node): The node which descended into this one.
        terminals (List): Set of terminals allowed.
        vars_available (Dict[str]): The list of column variables which are used
            in intermediate node creation mapped to initial thresholds
        depth (int): The current depth of creation.
        unique_nodes (bool): Whether the variables are sampled by replacement
        max_depth (int): The maximum depth allowed for this tree.
        full (bool): Whether the tree is a full tree."""
    kwargs = {
        "terminals":terminals,
        "depth":depth+1,
        "unique_nodes":unique_nodes,
        "max_depth":max_depth,
        "full":full}

    if depth == max_depth-1:
        # Add the two leaf children
        parent.add_child(Terminal(var_name=np.random.choice(terminals)))
        parent.add_child(Terminal(var_name=np.random.choice(terminals)))
        return
    
    for _ in range(2):
        if full:
            selection = np.random.choice([*vars_available])
            node = Node(selection,initial_threshold=vars_available[selection])
            parent.add_child(node)

            if unique_nodes:
                vars_available.pop(selection)
            recursive_tree_creation(
                parent=node,
                vars_available=vars_available,**kwargs)
        else:
            if random_choice():
                selection = np.random.choice([*vars_available])
                node = Node(selection,initial_threshold=vars_available[selection])
                parent.add_child(node)

                if unique_nodes:
                    vars_available.pop(selection)
                recursive_tree_creation(
                    parent=node,
                    vars_available=vars_available,**kwargs)
            else:
                parent.add_child(Terminal(var_name=np.random.choice(terminals)))


def create_tree(
        terminals:List,
        data:pd.DataFrame,
        depth:int,
        unique_nodes:bool=True,
        fixed_part:Optional[Node]=None,
        full:bool=True)->Node:
    """Create a program tree using the data and terminals provided.
    Parameters:
        terminals (List): The set of terminals to use as tree leaves. These are
            sampled by replacement.
        data (pd.DataFrame): Columns of data to use in the intermediate nodes.
        depth (int): The maximum depth to build this tree to.
        unique_nodes (bool): Decides whether the data columns are sampled by
            replacement or not.
        fixed_part (node): A subtree root which will be added as a fixed
            element to this tree.
        full (bool): Whether the tree constructed is a full tree or not.
    Returns the root node with a chain to all of its children.
    """
    if depth <= 1:
        raise RuntimeError("Minimum tree depth is 2 for stumps.")
    
    if fixed_part:
        if tree_depth(fixed_part) >= depth:
            raise RuntimeError(
            F"""The depth of your fixed_part: {tree_depth(fixed_part)} must be
            less than the maximum depth of this tree {depth}""")

    variables = create_initial_variables(data)

    selection = np.random.choice([*variables])
    root = Node(var_name=selection,initial_threshold=variables[selection])
    if unique_nodes:
        variables.pop(selection)

    recursive_tree_creation(
        parent=root,
        terminals=terminals,
        vars_available=variables,
        depth=1,
        unique_nodes=unique_nodes,
        max_depth=depth,
        full=full)

    if fixed_part:
        needed_depth = depth - tree_depth(fixed_part) + 1
        node = get_node(root,of_depth=needed_depth)
        replace_node(node,fixed_part)

    return root


def create_stump(variable:str,initial_threshold:str,terminals:List[str])->Node:
    stump = Node(variable,initial_threshold=initial_threshold)
    stump.add_child(Terminal(np.random.choice(terminals)))
    stump.add_child(Terminal(np.random.choice(terminals)))

    return stump


def create_fixed_tree(definition:Dict,_depth:int=0)->Node:
    """Create a fixed tree using the defenition dictionary which is formatted
    the same way as our serialization. Fixed trees are structures which are
    static and genetic operations cannot be performed on their submembers.
    Parameters:
        json_data (str): json serialized tree representation. 
        depth (int): Default to 0, shouldn't be modified since its used for
            recursion."""
    if "parent" not in definition:
        return Terminal.terminal_from_dict(definition,fixed=True)
    node = Node.node_from_dict(definition["parent"],fixed=True)
    
    # Our tree trees are binary trees thus guaranteed 2 children
    first,second = definition["children"]
    node.add_child(create_fixed_tree(first,_depth=_depth+1))
    node.add_child(create_fixed_tree(second,_depth=_depth+1))

    if node.is_root():
        return node


def create_buy_tree(data:pd.DataFrame,depth:int=2)->Node:
    return create_tree(terminals=["BUY","HOLD"],data=data,depth=depth)


def create_sell_tree(data:pd.DataFrame,depth:int=3)->Node:
    definition = {
        "parent": {
            "type": "NODE",
            "variable": "bought_price",
            "threshold": 20000,
        },
        "children":[
            {
                "type": "TERMINAL",
                "variable":"SELL"
            },
            {
                "type": "TERMINAL",
                "variable":"HOLD"
            }
        ]
    }
    fixed = create_fixed_tree(definition)
    return create_tree(
        terminals=["SELL","HOLD"],data=data,fixed_part=fixed,depth=depth)