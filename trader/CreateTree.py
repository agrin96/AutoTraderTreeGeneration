from __future__ import annotations
from typing import List,Dict,Optional
import pandas as pd
import numpy as np

from DataStructures.Node import Node
from DataStructures.Terminal import Terminal
from TreeActions import tree_depth,get_node,replace_node_with_fixed


def create_initialization_variables(data:pd.DataFrame)->Dict[str]:
    """Creates a mapping of column names of the data to their mean values which
    will be used as the initial threshold."""
    variables = data.columns.tolist()
    values = [np.max(data[v])+np.min(data[v]) for v in variables]
    return dict(zip(variables,values))


def recursive_tree_creation(
        parent:Node,
        terminals:List,
        vars_available:Dict[str],
        depth:int,
        max_depth:int):
    """Recursively constructs a tree.
    Parameters:
        parent (Node): The node which descended into this one.
        terminals (List): Set of terminals allowed.
        vars_available (Dict[str]): The list of column variables which are used
            in intermediate node creation mapped to initial thresholds
        depth (int): The current depth of creation.
        max_depth (int): The maximum depth allowed for this tree."""

    if depth == max_depth-1:
        # Add the two leaf children
        parent.add_child(Terminal(var_name=np.random.choice(terminals)))
        parent.add_child(Terminal(var_name=np.random.choice(terminals)))
        return
    
    # Left sub tree -----------------------------------------------
    selection = np.random.choice([*vars_available])
    threshold = threshold_at_depth(depth,vars_available[selection],left=True)

    node = Node(selection,initial_threshold=threshold)
    parent.add_child(node)

    recursive_tree_creation(parent=node,
                            terminals=terminals,
                            vars_available=vars_available,
                            depth=depth+1,
                            max_depth=max_depth)
    
    # Right sub tree -----------------------------------------------
    selection = np.random.choice([*vars_available])
    threshold = threshold_at_depth(depth,vars_available[selection],left=False)

    node = Node(selection,initial_threshold=threshold)
    parent.add_child(node)

    recursive_tree_creation(parent=node,
                            terminals=terminals,
                            vars_available=vars_available,
                            depth=depth+1,
                            max_depth=max_depth)


def create_tree(
        terminals:List,
        variables:Dict,
        depth:int,
        fixed_part:Optional[Node]=None)->Node:
    """Create a program tree using the data and terminals provided.
    Parameters:
        terminals (List): The set of terminals to use as tree leaves. These are
            sampled by replacement.
        variables (Dict): Map of variable name to mean value (initial threshold)
        depth (int): The maximum depth to build this tree to.
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

    selection = np.random.choice([*variables])
    threshold = threshold_at_depth(0,variables[selection])
    root = Node(var_name=selection,initial_threshold=threshold)

    recursive_tree_creation(
        parent=root,
        terminals=terminals,
        vars_available=variables,
        depth=1,
        max_depth=depth)

    if fixed_part:
        needed_depth = depth - tree_depth(fixed_part) + 1
        node = get_node(root,of_depth=needed_depth)
        root = replace_node_with_fixed(node,fixed_part)

    return root


def threshold_at_depth(depth:int,datarange:float,left:bool=True)->float:
    """Returns the threshold value to be used on this node. For the left tree
    we add the previous and for the right tree we subtract. This essentially
    behaves like a binary search tree where left is always larger and right
    is smaller.
    Parameters:
        depth (int): The current depth of the threshold being set.
        datarange (float): The max+min value of the particular variable
        left (bool): Whether this is a left tree or a right tree."""
    direction = 1 if left else -1
    if depth == 0:
        return datarange / 2
    else:
        return (datarange / np.power(2,depth-1))\
            + ((datarange / np.power(2,depth)) * direction)


def create_stump(variable:str,initial_threshold:str,terminals:List[str])->Node:
    """Create a node with two terminal children. This is the simplest valid
    tree structure.
    Parameters:
        variable (str): The variable to assign to the root node of this stump.
        initial_threshold (str): The initial threshold value of the decision
            made by the variable in the root node.
        terminals (List[str]): The terminals we are allowed to choose from in
            this stump creation.
    Returns the root node of the stump."""
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
        return Terminal.terminal_from_dict(definition)
    node = Node.node_from_dict(definition["parent"])
    
    # Our tree trees are binary trees thus guaranteed 2 children
    first,second = definition["children"]
    node.add_child(create_fixed_tree(first,_depth=_depth+1))
    node.add_child(create_fixed_tree(second,_depth=_depth+1))

    if node.is_root():
        return node


def create_buy_tree(variables:Dict,popid:int,depth:int=2)->Dict:
    """Create a buy tree using the specified options.
    Parameters:
        variables (Dict): Valid variables mapped to initial thresholds to
            choose from when creating a tree.
        popid (int): The unique identifier to assign to this tree on creation.
        depth (int): The initial depth of the tree being created.
    Returns a dictionary containng the tree and some other data about the tree.
    """
    return {
        "popid": popid,
        "tree": create_tree(terminals=["BUY","HOLD"],
                            variables=variables,
                            depth=depth),
        "fitness": None,
        "balance": None,
        "cluster": None,
        "coordinate": None}


def create_sell_tree(variables:Dict,popid:int,depth:int=3)->Node:
    """Create a sell tree using the specified options.
    Parameters:
        variables (Dict): Valid variables mapped to initial thresholds to
            choose from when creating a tree.
        popid (int): The unique identifier to assign to this tree on creation.
        depth (int): The initial depth of the tree being created.
    Returns a dictionary containng the tree and some other data about the tree.
    """
    definition = {
        "parent": {
            "type": "NODE",
            "variable": "bought_price",
            "threshold": 20000,
            "fixed": True
        },
        "children":[
            {
                "type": "TERMINAL",
                "variable":"SELL",
                "fixed": True
            },
            {
                "type": "TERMINAL",
                "variable":"HOLD",
                "fixed": True
            }
        ]
    }
    fixed = create_fixed_tree(definition)
    return {
        "popid": popid,
        "tree": create_tree(terminals=["SELL","HOLD"],
                            variables=variables,
                            fixed_part=fixed,
                            depth=depth),
        "fitness": None,
        "balance": None,
        "cluster": None,
        "coordinate": None}