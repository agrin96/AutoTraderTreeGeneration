from __future__ import annotations
from typing import List,Dict,Tuple
import numpy as np

def tournament_selection(population:List,
                         tournament_size:int,
                         survivors:float)->List:
    """Execute simple tournament selection on the population without 
    replacement. Selects the best member until it reaches the desired remaining
    members in each cluster.
    Parameters:
        tournament_size (int): The number of members selected for each 
            tournament.
        survivors (float): The percent of each cluster that should survive into
            the next generation.
    Returns the surviving members."""
    cluster_count = max(population,key=lambda k: k["cluster"])["cluster"]
    cluster_count += 1

    survivors = int(len(population)*survivors)

    next_generation = []
    while len(next_generation) < survivors:
        for cid in range(cluster_count):
            cluster = [pop for pop in population if pop["cluster"] == cid]
            if len(cluster) == 0:
                continue
            try:
                contestants = np.random.choice(cluster,tournament_size)
                winner = max(contestants,key=lambda c: c["fitness"])
                population.remove(winner)
                next_generation.append(winner)
            except Exception as e:
                print("cluster", cluster)
                print("contestants",contestants)
                raise

    return next_generation


def match_hanging_trees(treesA:List[Dict],treesB:List[Dict])->Tuple:
    """After executing selection we may be left with `hanging trees` meaning
    trees which do not have a buy or sell partner because they were not selected
    in tournament. Because we can guarantee that the lengths of the selected
    trees is still the same, we can use this function to match up hanging trees
    so that everyone has a pair."""
    for atree in treesA:
        idA = atree["popid"]
        
        # Check if this id exists in the second set
        if len([t for t in treesB if t["popid"] == idA]):
            continue
        else:
            #Find an id in B that isnt assigned in A and the B[id] to the Aid
            for btree in treesB:
                idB = btree["popid"]
                if len([t for t in treesA if t["popid"] == idB]):
                    continue
                else:
                    btree["popid"] = idA
                    break

    return treesA,treesB
