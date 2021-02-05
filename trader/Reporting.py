import numpy as np
from typing import Dict,List
import matplotlib.pyplot as plt
from TreeActions import count_nodes,tree_depth

def pprint_generation_statistics(buy_trees:List[Dict],
                                 sell_trees:List[Dict],
                                 rolling_balances:List[float]):
    """Prints key statistics of the current generation. Shows the best buy
    and sell tree as well as the mean fitness and mean balance of the 
    generation population. This step occurs before mutation and selection."""
    print("\n\tCurrent Best Buy and Sell Trees")
    best_buy = max(buy_trees,key=lambda k: k["fitness"])
    best_sell = max(sell_trees,key=lambda k: k["fitness"])
    print("\t"+str(best_buy))
    print("\t"+str(best_sell))
    
    print("\n\tAverage Balance and Fitness of Buy Trees")
    mean_fitness = np.mean(list(map(lambda k: k["fitness"],buy_trees)))
    mean_balance = np.mean(list(map(lambda k: k["balance"],buy_trees)))
    print(F"\t\tMean Fitness: {mean_fitness}")
    print(F"\t\tMean Balance: {mean_balance}")

    print("\n\tAverage Balance and Fitness of Sell Trees")
    mean_fitness = np.mean(list(map(lambda k: k["fitness"],sell_trees)))
    mean_balance = np.mean(list(map(lambda k: k["balance"],sell_trees)))
    print(F"\t\tMean Fitness: {mean_fitness}")
    print(F"\t\tMean Balance: {mean_balance}")

    print("\n\tAverage Tree Depth")
    buy_depth = np.mean(list(map(lambda t: tree_depth(t["tree"]),buy_trees)))
    sell_depth = np.mean(list(map(lambda t: tree_depth(t["tree"]),sell_trees)))
    print(F"\t\tMean Buy Depth: {buy_depth}")
    print(F"\t\tMean Sell Depth: {sell_depth}")

    print("\n\tAverage Node Counts")
    buy_count = np.mean(list(map(lambda t: count_nodes(t["tree"]),buy_trees)))
    sell_count = np.mean(list(map(lambda t: count_nodes(t["tree"]),sell_trees)))
    print(F"\t\tMean Buy Node Count: {buy_count}")
    print(F"\t\tMean Sell Node Count: {sell_count}")

    print("\n\tRolling Mean of Best Member Balances")
    rolling_balances.append(best_buy["balance"])
    rolling_balances.append(best_sell["balance"])
    rolling_mean = np.mean(rolling_balances)
    print(F"\t\tCurrent rolling mean balance: {rolling_mean}\n")

