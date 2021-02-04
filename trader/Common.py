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


def store_serialized_pop(serial_buy:str,
                         serial_sell:str,
                         output_folder:str="SerializedTrees/"):
    """Write the serialized population members to the root directory to store
    them. Does not retain information in the pop dictionary just the trees."""
    current_dir = os.getcwd()
    for idx,folder in enumerate(current_dir.split("/")):
        if folder == "AutoTrader":
            current_dir = "/".join(current_dir.split("/")[:idx+1])
            break
    
    output_path = os.path.join(current_dir,output_folder)
    popid = 0
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        # Get the current largest file id and iterate by one
        contents = os.listdir(output_path)
        files = [c for c in contents if "popfile" in c]
        files = [c.split('.')[0] for c in contents]
        fileids = [c.split("-")[-1] for c in files]
        popid = max(fileids)+1
    
    filename = F"buy-popfile-{popid}.json"
    with open(os.path.join(output_path,filename),"w+") as out:
        out.write(serial_buy)

    filename = F"sell-popfile-{popid}.json"
    with open(os.path.join(output_path,filename),"w+") as out:
        out.write(serial_sell)
    