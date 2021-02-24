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
    
    output = F"POPID: {best['popid']} Fitness: {best['fitness']}"
    output += F" Balance: {best['balance']} Trades: {best['trades']}"
    print("\t"+str(output))
    
    print("\n\tPopulation Statistics")
    mean_fitness = np.mean(list(map(lambda k: k["fitness"],pops)))
    mean_balance = np.mean(list(map(lambda k: k["balance"],pops)))
    print(F"\t\tMean Fitness: {mean_fitness}")
    print(F"\t\tMean Balance: {mean_balance}")
    lowestf = min(pops,key=lambda k: k["fitness"])["fitness"]
    lowestb = min(pops,key=lambda k: k["balance"])["balance"]
    print(F"\t\tLowest Fitness: {lowestf}")
    print(F"\t\tLowest Balance: {lowestb}")
    f_var = np.var(list(map(lambda k: k["fitness"],pops)))
    b_var = np.var(list(map(lambda k: k["balance"],pops)))
    print(F"\t\tFitness Variance: {f_var}")
    print(F"\t\tBalance Variance: {b_var}")

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

