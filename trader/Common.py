import numpy as np
from typing import Dict,Any,List
import json
import os
import matplotlib.pyplot as plt


def random_choice(prob_true:float=0.5)->bool:
    return np.random.choice([True,False],p=[prob_true,1-prob_true])


def parse_configuration(path:str)->Dict:
    """Read in a json configuration file and return it as a dictionary."""
    if not os.path.exists(path):
        raise RuntimeError(
        "The configuration path you specified doesn't exist.")  
    
    with open(path,"r") as file:
        return json.loads(file.read())


def pprint_generation_statistics(buy_trees:List[Dict],sell_trees:List[Dict]):
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
    print(F"\t\tMean Balance: {mean_balance}\n")
