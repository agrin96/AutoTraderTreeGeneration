from __future__ import annotations
from typing import List,Dict,Tuple

import numpy as np

from Common import random_choice
from DataStructures.Node import Node
from DataStructures.Terminal import Terminal
from CreateTree import create_stump
from TreeActions import (
    list_tree_variables,
    list_tree_terminals,
    get_random_node,
    replace_node,
    clone_node)

def double_crossover_reproduction(treeA:Node,
                                  treeB:Node,
                                  probability:float=0.5)->Tuple:
    """Creates a pair of child nodes from the reproduction of two nodes 
    provided with the probability of a crossover being determined by the 
    probability value passed in. If no crossover takes place, then the 
    trees are simply copied."""
    if not random_choice(prob_true=probability):
        return clone_node(treeA),clone_node(treeB)

    child_a = clone_node(treeA)
    child_b = clone_node(treeB)
    
    cross_a = get_random_node(child_a)
    cross_b = get_random_node(child_b)
    
    parent_a = cross_a.get_parent()
    parent_b = cross_b.get_parent()
    
    posA = parent_a.remove_child(cross_a)
    posB = parent_b.remove_child(cross_b)

    parent_a.add_child(cross_b,posA)
    parent_b.add_child(cross_a,posB)

    return child_a,child_b


def single_crossover_reproduction(treeA:Node,
                                  treeB:Node,
                                  probability:float=0.5)->Node:
    """Reproduction on two parents where only one child is returned. The child
    returned is a random selection between a child of A or child of B. If
    no crossover occers we randomly return either a copy of A or B."""
    if not random_choice(prob_true=probability):
        if random_choice():
            return clone_node(treeA)
        return clone_node(treeB)

    childA,childB = double_crossover_reproduction(treeA,treeB,1.0)
    if random_choice():
        return childA
    return childB


def repopulate(population:List[Dict],max_population:int,crossover_p:float)->List:
    """Generates new population members through crossover reproduction within
    clusters. Will iteratively add one member to each cluster until there are
    enought members to reach the max_population.
    Parameters:
        crossover_p (float): The probability of crossover occuring.
    Returns the full new population."""
    clusters = max(population,key=lambda k: k["cluster"])["cluster"]
    lastid = max(population,key=lambda k: k["popid"])["popid"]
    clusters += 1

    while len(population) < max_population:
        for i in range(clusters):
            current_cluster = [p for p in population if p["cluster"] == i]
            if len(current_cluster) == 0:
                continue
            
            lastid += 1
            if len(current_cluster) == 1:
                population.append({
                    "popid":lastid,
                    "tree":clone_node(current_cluster[0]["tree"]),
                    "fitness": None,
                    "cluster": None,
                    "coordinate": None})
            else:
                parentA,parentB = np.random.choice(current_cluster,
                                                   size=2,
                                                   replace=False)
                child = single_crossover_reproduction(
                            treeA=parentA["tree"],
                            treeB=parentB["tree"],
                            probability=crossover_p)
                population.append({
                    "popid":lastid,
                    "tree":child,
                    "fitness": None,
                    "cluster": None,
                    "coordinate": None})

    return population



def clone_pop(pop:Dict)->Dict:
    """Copies a population member including a deep copy of its tree attribute.
    """
    new_pop = pop.copy()
    new_pop["tree"] = clone_node(pop["tree"])
    return new_pop