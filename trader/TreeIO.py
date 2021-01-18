from __future__ import annotations
from typing import List
import json

from Common import random_choice
from DataStructures.Node import Node
from DataStructures.Terminal import Terminal

def serialize_tree(node:Node,_depth:int=0)->str:
    """Convert node tree into a json serialized string. Used to export
    trees with their internal data for reuse.
    Parameters:
        node (Node): Root node of tree to serialize.
        depth (int): Default to 0, shouldn't be modified since its used for
            recursion."""
    if isinstance(node,Terminal):
        return node.node_as_dict()
    
    output = {"parent":node.node_as_dict()}
    first,second = node.children()

    output["children"] = [
        serialize_tree(first,_depth=_depth+1),
        serialize_tree(second,_depth=_depth+1)]

    if _depth == 0:
        return json.dumps(output,ensure_ascii=False,indent=2)
    else:
        return output


def deserialize_tree(json_data:str,_depth:int=0)->Node:
    """Read in a serialized tree and convert it into a node tree. Returns the
    root node.
    Parameters:
        json_data (str): json serialized tree representation. 
        depth (int): Default to 0, shouldn't be modified since its used for
            recursion."""
    tree = json_data
    if _depth == 0:
        tree = json.loads(json_data)

    if "parent" not in tree:
        return Terminal.terminal_from_dict(tree)
    node = Node.node_from_dict(tree["parent"])
    
    # Our tree trees are binary trees thus guaranteed 2 children
    first,second = tree["children"]
    node.add_child(deserialize_tree(first,_depth=_depth+1))
    node.add_child(deserialize_tree(second,_depth=_depth+1))

    if node.is_root():
        return node



    
