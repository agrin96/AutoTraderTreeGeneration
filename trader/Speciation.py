from __future__ import annotations
from typing import List,Union,Dict
import json
import numpy as np

from Common import random_choice,kdistance
from DataStructures.Node import Node
from DataStructures.Terminal import Terminal
from TreeActions import (
    count_nodes,
    count_terminals,
    tree_depth,
    list_tree_variables,
    pprint_tree)
from multiprocessing import Pool

def speciate(population:List[Dict],process_pool:Pool)->List[List]:
    if not process_pool:
        raise RuntimeError("Process pool was not initialized at speciation.")
    
    buy_trees = [pop["buy"] for pop in population]
    sell_trees = [pop["sell"] for pop in population]

    buy_coordinates = process_pool.map(generate_coordinate_from_tree,buy_trees)
    sell_coordinates = process_pool.map(generate_coordinate_from_tree,sell_trees)

    for pop,bcoord,scoord in zip(population,buy_coordinates,sell_coordinates):
        pop["bpoint"] = bcoord
        pop["spoint"] = scoord

    

    
def structural_similarity(treeA:Node,treeB:Node)->float:
    """Traverses two trees simultaneously to determine their structural 
    similarity. For every node that is in the same position for two trees,
    the similarity is incremented by one. The result is then the number of
    similar nodes over the total number of nodes in both trees."""
    def get_similarity(left,right):
        if isinstance(treeA,Terminal):
            if isinstance(treeB,Terminal):
                return 1
            else:
                first,second = treeB.children()
                return structural_similarity(None,first)\
                    + structural_similarity(None,second)
        else:
            if isinstance(treeB,Terminal):
                first,second = treeA.children()
                return structural_similarity(first,None)\
                    + structural_similarity(second,None)
            else:
                first,second = treeA.children()
                firstb,secondb = treeB.children()
                return 1 + structural_similarity(first,firstb)\
                        + structural_similarity(second,secondb)
    
    total_nodes = count_nodes(treeA) + count_nodes(treeB)
    return get_similarity(treeA,treeB) / total_nodes


def generate_coordinate_from_tree(node:Node)->List:
    """Generates a k-dimensional coordinate point using some meta data about
    the specific tree. returns a list of integers indicating the specific
    coordinates and can be used as a similarity comparison to cluster nodes."""
    return [
        count_nodes(node),
        count_terminals(node),
        tree_depth(node),
        len(set(list_tree_variables(node)))]


def generate_coordinate_from_buysell(buy:Node,sell:Node)->List:
    """Generates a k-dimensional coordinate point using metadata from the buy
    and sell trees respectively. In this way we get a list of integers which
    can be used to compare buy_sell trees directly."""
    return [*generate_coordinate_from_tree(buy),
            *generate_coordinate_from_tree(sell)]


def evenstep_clustering(values:np.array,max_distance:float):
    """Executes a clustering based on distance values."""
    clusters = []
    centers = np.arange(values.min(),values.max(),max_distance)
    if (centers[-1] + max_distance/2) < values.max():
        centers = np.array([*centers,centers[-1]+max_distance])

    for center in centers:
        distances = np.abs(np.subtract(np.repeat(center,values.shape[0]),values))

        # if 1 then we selected if 0 then we didnt
        mask = np.where(distances<=max_distance/2,True,False)
        clusters.append(values[mask])

    if not np.array_equal(np.sort(values),np.sort(np.array([v for clus in clusters for v in clus]))):
        raise RuntimeError("The two arrays are different")

    if values.shape[0] != len([v for clus in clusters for v in clus]):
        print("Original values: ",values.shape[0])
        print("Clustered: ",len([v for clus in clusters for v in clus]))
        raise RuntimeError("Different lengths") 
    return clusters