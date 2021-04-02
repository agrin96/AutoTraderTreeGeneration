from __future__ import annotations
from typing import List,Union
import json
import numpy as np

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
    
    children,last = node.children()[:-1],node.children()[-1]
    for child in children:
        if isinstance(child,Terminal):
            output += F"\n{previous}├───{child}"
        else:
            output += F"\n{previous}├───[{child}]"
            output += stringify_tree(node=child,
                                     previous=previous+"│   ",
                                     _depth=_depth+1)

    if isinstance(last,Terminal):
        output += F"\n{previous}└───{last}"
    else:
        output += F"\n{previous}└───[{last}]"
        output += stringify_tree(last,previous=previous+"    ",_depth=_depth+1)
    
    return output


def pprint_tree(node:Node):
    """Convenience method for using stringify to print trees for debugging."""
    print(stringify_tree(node))


def tree_depth(node:Node)->int:
    """Returns the maximum depth of the tree. The root node counts as 1."""
    if isinstance(node,Terminal):
        return 1
    else:
        return max([1 + tree_depth(child) for child in node.children()])


def is_left_child(node:Node)->bool:
    """Returns whether this node is a left chilf of its parent or a right child.
    On root it return false, but this is meaningless."""
    if node.is_root():
        return False
    
    left,right = node.get_parent().children()
    if left.node_id() == node.node_id():
        return True
    return False


def node_depth(node:Node)->int:
    """Returns the depth of this node in its tree. Root node is 0."""
    if node.is_root():
        return 0
    else:
        return 1 + node_depth(node.get_parent())


def count_nodes(node:Node,with_terminals:bool=True)->int:
    """Returns the node count of the tree. Terminals are considered nodes as 
    well."""
    if isinstance(node,Terminal):
        if with_terminals:
            return 1
        return 0
    else:
        return 1 + np.sum(
            [count_nodes(child,with_terminals) for child in node.children()])


def count_terminals(node:Node)->int:
    """Count only the terminal nodes in this tree."""
    if isinstance(node,Terminal):
        return 1
    else:
        return np.sum([count_terminals(child) for child in node.children()])


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

        for child in node.children():
            get_node(child,of_depth=of_depth,_depth=_depth+1,_nodes=_nodes)

    if _depth == 1:
        return np.random.choice(_nodes)


def get_random_node(node:Node,
                    include_root:bool=False,
                    include_terminals:bool=True,
                    _depth:int=1,
                    _nodes:List[Node]=[])->Node:
    """Chose a random node from all nodes in this tree excluding the root. If 
    the random node is a fixed node, then it will traverse up and return the 
    root of the fixed part so that we only ever change the entire fixed section.
    """
    if _depth == 1:
        _nodes = []
    
    if not node.is_root():
        if isinstance(node,Terminal):
            if include_terminals:
                _nodes.append(node)
        if isinstance(node,Node):
            _nodes.append(node)

    if include_root and node.is_root():
        _nodes.append(node)

    if isinstance(node,Node):
        for child in node.children():
            get_random_node(node=child,
                            _depth=_depth+1,
                            _nodes=_nodes,
                            include_terminals=include_terminals)

    if _depth == 1:
        # _nodes = [n for n in _nodes if not n.is_fixed()]
        if len(_nodes) == 0:
            return None
        return np.random.choice(_nodes)


def replace_node(original:Node,new_node:Node)->Node:
    """Replace the original node with a new node by swapping the children
    of the parent element.
    Returns the original root of the tree if a node was replaced in the middle
    or returns the new root if the root was replaced."""
    if isinstance(original,Terminal) or isinstance(new_node,Terminal):
        parent = original.get_parent()
        idx = parent.remove_child(original)
        parent.add_child(new_node,idx)
        return root_of_tree(new_node)

    for child in original.children():
        new_node.add_child(child)

    if not original.is_root():
        parent = original.get_parent()
        idx = parent.remove_child(original)
        parent.add_child(new_node,idx)

        # Return the root
        return root_of_tree(new_node)

    # Return new root
    return new_node


def list_tree_variables(node:Node,
                        with_threshold:bool=False,
                        _depth:int=1,
                        _variables:List[str]=[])->[str]:
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

        for child in node.children():
            list_tree_variables(
                child,
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
        for child in node.children():
            list_tree_terminals(child,_depth=_depth+1,_terminals=_terminals)
    
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


def root_of_tree(node:Node)->Node:
    """Traverse the tree to get the root and return it."""
    if node.is_root():
        return node
    else:
        return root_of_tree(node.get_parent())