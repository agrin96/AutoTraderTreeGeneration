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


def speciate_by_coordinate(population:List[Dict],
                           search_distance_modifier:float,
                           process_pool:Pool)->List:
    """For speciation we use k means clustering to assign cluster ids to each
    member of the population. Returns the population with assigned clusters."""
    if not process_pool:
        raise RuntimeError("Process pool was not initialized at speciation.")
    
    population = process_pool.map(populate_points_for_pop,population)
    k_value = find_kmeans_k(population,search_distance_modifier)

    return kmeans_clustering(population,k_value)


def speciate_by_structure(population:List[Dict])->List:
    """For speciation we use k means clustering to assign cluster ids to each
    member of the population. Returns the population with assigned clusters."""
    return structural_similarity_clustering(population)

    
def populate_points_for_pop(pop:Dict)->Dict:
    """Generate coordinate points using internal metrics for both the buy
    and the sell trees. The updated pop is returned to allow multiprocessing,
    otherwise the pop is updated in place and return can be ignored."""
    pop["coordinate"] = generate_coordinate_from_tree(pop["tree"])
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
                return 0
        else:
            if isinstance(treeB,Terminal):
                return 0
            else:
                count = 1
                for childA,childB in zip(treeA.children(),treeB.children()):
                    count += structural_similarity(childA,childB)
                return count
    
    total_nodes = count_nodes(treeA) + count_nodes(treeB)
    return get_similarity(treeA,treeB) / total_nodes


def structural_similarity_clustering(population:List[Dict])->List[Dict]:
    """Cluster based on structural similarity measures. We select N number of
    pops for our cluster seeds and then do a greedy search on each of the seed
    pops to find the most similar members until all pops are accounted for.
    Returns the population with the cluster attribute assigned"""
    max_population = len(population)
    total_clusters = max(len(population)//50,4)
    clustered = []

    # choose some pops randomly as cluster centers and remove them from the pool
    centers = np.random.choice(population,total_clusters,replace=False)

    for cluster,center in enumerate(centers):
        population.remove(center)
        center["cluster"] = cluster
        clustered.append(center)

    while len(clustered) < max_population:
        for cluster,center in enumerate(centers):
            closest = (None,np.inf)
            for pop in population:
                measure = structural_similarity(center["tree"],pop["tree"])
                if measure < closest[1]:
                    closest = (pop,measure)
            
            # Assign cluster to the closest pop and remove it from the running.
            population.remove(closest[0])
            closest[0]["cluster"] = cluster
            clustered.append(closest[0])

            if len(clustered) >= max_population:
                return clustered
        
    return clustered


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