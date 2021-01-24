from __future__ import annotations
from typing import List,Union
import json
import numpy as np

from Common import random_choice
from DataStructures.Node import Node
from DataStructures.Terminal import Terminal


def stringify_tree(node:Node,previous:str="",_depth:int=0)->str:
    """Prints a tree like structure to the terminal when passed a node in a 
    tree.
    Parameters:
        previous (str): Can be used to prefix the printed tree with some text.
    Returns the tree structure string representation of the node.
    """
    output = ""
    if node.is_root() or _depth == 0:
        output += F"\n[{node}]"
        if isinstance(node,Terminal):
            return output
    first,second = node.children()

    if isinstance(first,Terminal):
        output += F"\n{previous}├───{first}"
    else:
        output += F"\n{previous}├───[{first}]"
        output += stringify_tree(first,previous=previous+"│   ",_depth=_depth+1)

    if isinstance(second,Terminal):
        output += F"\n{previous}└───{second}"
    else:
        output += F"\n{previous}└───[{second}]"
        output += stringify_tree(second,previous=previous+"    ",_depth=_depth+1)
    
    return output


def pprint_tree(node:Node):
    """Convenience method for using stringify to print trees for debugging."""
    print(stringify_tree(node))


def tree_depth(node:Node)->int:
    """Returns the maximum depth of the tree. The root node counts as 1."""
    if isinstance(node,Terminal):
        return 1
    else:
        first,second = node.children()
        return max(1 + tree_depth(first),1 + tree_depth(second))


def count_nodes(node:Node,with_terminals:bool=True)->int:
    """Returns the node count of the tree. Terminals are generally considered
    nodes as well."""
    if isinstance(node,Terminal):
        if with_terminals:
            return 1
        return 0
    else:
        first,second = node.children()
        return 1 + count_nodes(first,with_terminals)\
                 + count_nodes(second,with_terminals)


def get_node(node:Node,of_depth:int=1,_depth:int=1,_nodes:List[Node]=[])->Node:
    """Returns a node from the specified depth. Randomly chooses from the list 
    of nodes at that depth."""
    if of_depth < 1:
        raise RuntimeError(
            "The 'of_depth' must be at least 1 which will return the root.")

    if _depth == 1:
        _nodes = []
        if of_depth > tree_depth(node):
            raise RuntimeError(
            "The depth you specified is greater than the tree depth")
    
    if _depth == of_depth:
        _nodes.append(node)
    else:
        if isinstance(node,Terminal):
            return
        first,second = node.children()
        get_node(first,of_depth=of_depth,_depth=_depth+1,_nodes=_nodes)
        get_node(second,of_depth=of_depth,_depth=_depth+1,_nodes=_nodes)

    if _depth == 1:
        return np.random.choice(_nodes)


def get_random_node(node:Node,_depth:int=1,_nodes:List[Node]=[])->Node:
    """Chose a random node from all nodes in this tree excluding the root. If 
    the random node is a fixed node, then it will traverse up and return the 
    root of the fixed part so that we only ever change the entire fixed section.
    """
    if _depth == 1:
        _nodes = []
    
    if not node.is_root():
        _nodes.append(node)

    if isinstance(node,Node):
        first,second = node.children()
        get_random_node(first,_depth=_depth+1,_nodes=_nodes)
        get_random_node(second,_depth=_depth+1,_nodes=_nodes)
    
    if _depth == 1:
        # _nodes = [n for n in _nodes if not n.is_fixed()]
        choice = np.random.choice(_nodes)
        if choice.is_fixed():
            return root_of_fixed(choice)
        return choice


def root_of_fixed(node:Node)->Node:
    """Traverse the fixed section up and find the root of the fixed section."""
    parent = node.get_parent()
    if parent.is_fixed():
        return root_of_fixed(node.get_parent()) 
    else:
        return node 


def replace_node(original:Node,new_node:Node,include_children:bool=False):
    """Replace the original node with a new node by swapping the children
    of the parent element.
    Parameters:
        include_children (bool): Whether to reassign children as well. If true
            then the new node should not have any children of its own."""
    if original.is_root():
        raise RuntimeError("Attempting to replace root. This is not allowed.")
    parent = original.get_parent()
    idx = parent.remove_child(original)
    
    if include_children and isinstance(original,Node):
        first,second = original.children()
        new_node.add_child(first)
        new_node.add_child(second)

    parent.add_child(new_node,index=idx)


def list_tree_variables(node:Node,with_threshold:bool=False,_depth:int=1,_variables:List[str]=[])->[str]:
    """Get a list of variables used in this tree."""
    if _depth == 1:
        _variables = []
        if with_threshold:
            _variables = {}
    
    if isinstance(node,Node):
        if with_threshold:
            _variables[node.get_variable()] = node.get_threshold()
        else:
            _variables.append(node.get_variable())

        first,second = node.children()
        list_tree_variables(
            first,
            with_threshold=with_threshold,
            _depth=_depth+1,
            _variables=_variables)
        list_tree_variables(
            second,
            with_threshold=with_threshold,
            _depth=_depth+1,
            _variables=_variables)
    
    if _depth == 1:
        return _variables


def list_tree_terminals(node:Node,_depth:int=1,_terminals:List[str]=[])->[str]:
    """Get a list of terminals used in this tree."""
    if _depth == 1:
        _terminals = []
    
    if isinstance(node,Terminal):
        _terminals.append(node.get_variable())
    else:
        first,second = node.children()
        list_tree_terminals(first,_depth=_depth+1,_terminals=_terminals)
        list_tree_terminals(second,_depth=_depth+1,_terminals=_terminals)
    
    if _depth == 1:
        return list(set(_terminals))


def clone_node(node:Union[Node,Terminal],deep:bool=True)->Node:
    """Clone the provided node traversing down its children if deep enabled.
    Parameters:
        deep (bool): If False will only copy the node without any children"""
    copy = None
    if isinstance(node,Node):
        copy = Node.node_from_dict(node.node_as_dict())
    else:
        copy = Terminal.terminal_from_dict(node.node_as_dict())
    
    if not deep:
        return copy

    if isinstance(node,Node):
        for child in node.children():
            copy.add_child(clone_node(child,deep))

    return copy
