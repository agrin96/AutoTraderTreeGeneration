import numpy as np
from typing import Dict,List
import matplotlib.pyplot as plt
from TreeActions import count_nodes,tree_depth

def pprint_generation_statistics(pops:List[Dict],
                                 rolling_balances:List[float]):
    """Prints key statistics of the current generation. Shows the best buy
    and sell tree as well as the mean fitness and mean balance of the 
    generation population. This step occurs before mutation and selection."""
    print("\n\tCurrent Best Tree")
    best = max(pops,key=lambda k: k["fitness"])
    print("\t"+str(best))
    
    print("\n\tAverage Balance and Fitness of Trees")
    mean_fitness = np.mean(list(map(lambda k: k["fitness"],pops)))
    mean_balance = np.mean(list(map(lambda k: k["balance"],pops)))
    print(F"\t\tMean Fitness: {mean_fitness}")
    print(F"\t\tMean Balance: {mean_balance}")

    print("\n\tAverage Tree Depth")
    depth = np.mean(list(map(lambda t: tree_depth(t["tree"]),pops)))
    print(F"\t\tMean Depth: {depth}")

    print("\n\tAverage Node Counts")
    count = np.mean(list(map(lambda t: count_nodes(t["tree"]),pops)))
    print(F"\t\tMean Node Count: {count}")

    print("\n\tRolling Mean of Best Member Balance")
    rolling_balances.append(best["balance"])
    rolling_mean = np.mean(rolling_balances)
    print(F"\t\tCurrent rolling mean balance: {rolling_mean}\n")

