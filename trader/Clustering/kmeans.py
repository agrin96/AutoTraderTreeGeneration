from __future__ import annotations
from typing import List,Dict,Any,Tuple,Callable
import numpy as np


def kdistance(kpointA:List[Any],kpointB:List[Any])->float:
    """Return a euclidian distance between 2 k dimensional points in space.""" 
    if len(kpointA) != len(kpointB):
        raise RuntimeError(
        "The dimensionality k-points must be the same, but encountered"\
        F" {kpointA} and {kpointB}")
    
    return np.sqrt(np.sum(np.power(np.subtract(kpointA,kpointB),2)))


def find_kmeans_k(population:List[Dict],
                  search_distance_modifier:float=0.25,
                  distance_func:Callable=kdistance)->int:
    """Calculate the k value to be used in k means clustering for the specific
    attribute in the population.
    Parameters:
        search_distance_modifier (float): Our base search distance is just the
            std deviation. This is the value it is multiplied by to adjust.
    Returns the number of clusters to create for the set of points under this
        attribute"""
    memo = {}
    for i in range(len(population)):
        for j in range(len(population)):
            if i == j:
                continue
            if F"{i}-{j}" in memo:
                continue
            elif F"{j}-{i}" in memo:
                continue
            else:
                memo[F"{i}-{j}"] = distance_func(population[i]["coordinate"],
                                                 population[j]["coordinate"])
    
    distances = list(memo.values())
    search_distance = np.std(distances)*search_distance_modifier
    if search_distance == 0:
        return 0
    
    centers = np.arange(np.min(distances),np.max(distances),search_distance)
    output = centers.shape[0]

    if centers[-1] + search_distance / 2 < np.max(distances):
        return output+1
    return output


def kmeans_clustering(population:List,
                      k:int,
                      distance_func:Callable=kdistance)->List:
    """Cluster the population using k means and return the population to allow
    for multiprocessing otherwise it is donein place.
    Parameters:
        k (str): The number of clusters in the kmeans."""
    did_change = True
    centroids = np.random.randint(0,len(population),size=k)
    centroids = [population[c]["coordinate"] for c in centroids]

    while did_change:
        for i in range(len(population)):
            min_distance = (0,np.inf)
            for idx,point in enumerate(centroids):
                current_distance = kdistance(population[i]["coordinate"],point)
                
                if current_distance < min_distance[1]:
                    min_distance = (idx,current_distance)
            
            if "cluster" in population[i]:
                if population[i]["cluster"] == min_distance[0]:
                    did_change = False
                    continue
            population[i]["cluster"] = min_distance[0]

        new_centroids = []
        for c in range(k):
            cluster = [p["coordinate"] for p in population 
                      if p["cluster"] == c]
            
            if len(cluster) == 0:
                continue

            new_point = []
            for dim in range(len(cluster[0])):
                new_point.append(np.median([point[dim] for point in cluster]))
            new_centroids.append(new_point)

        centroids = new_centroids
    return population
