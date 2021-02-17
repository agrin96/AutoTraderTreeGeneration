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

            contestants = np.random.choice(cluster,tournament_size)
            winner = max(contestants,key=lambda c: c["fitness"])
            population.remove(winner)
            next_generation.append(winner)

            if len(next_generation) >= survivors:
                return next_generation

    return next_generation
