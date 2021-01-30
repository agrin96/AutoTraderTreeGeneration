from __future__ import annotations
from typing import List,Union,Dict,Tuple
import json
import numpy as np
from multiprocessing import Pool

from Common import random_choice
from DataStructures.Node import Node
from DataStructures.Terminal import Terminal
from TreeActions import (
    count_nodes,
    count_terminals,
    tree_depth,
    list_tree_variables,
    pprint_tree)
from Clustering.kmeans import (
    kmeans_clustering,
    find_kmeans_k)


def speciate_by_kmeans(population:List[Dict],process_pool:Pool)->List:
    if not process_pool:
        raise RuntimeError("Process pool was not initialized at speciation.")
    
    population = process_pool.map(populate_points_for_pop,population)

    args = [[population,"buy_point"],[population,"sell_point"]]
    buy_k,sell_k = process_pool.starmap(find_kmeans_k,args)

    population = kmeans_clustering(population,"buy_point","buy_cluster",buy_k)
    population = kmeans_clustering(population,"sell_point","sell_cluster",sell_k)

    return population

    
def populate_points_for_pop(pop:Dict)->Dict:
    """Generate coordinate points using internal metrics for both the buy
    and the sell trees. The updated pop is returned to allow multiprocessing,
    otherwise the pop is updated in place and return can be ignored."""
    pop["buy_point"] = generate_coordinate_from_tree(pop["buy"])
    pop["sell_point"] = generate_coordinate_from_tree(pop["sell"])
    
    return pop


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


def generate_coordinate_from_pop(buy:Node,sell:Node)->List:
    """Generates a k-dimensional coordinate point using metadata from the buy
    and sell trees respectively. In this way we get a list of integers which
    can be used to compare buy_sell trees directly."""
    return [*generate_coordinate_from_tree(buy),
            *generate_coordinate_from_tree(sell)]