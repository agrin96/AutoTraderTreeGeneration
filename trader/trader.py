from __future__ import annotations
from typing import List,Union,Dict
import pandas as pd
import numpy as np
import uuid

from Common import random_choice,parse_configuration
from DataStructures.Node import Node
from DataStructures.Terminal import Terminal
from CreateTree import (
    create_initialization_variables,
    create_tree,
    create_buy_tree,
    create_sell_tree,
    create_stump)

from TreeActions import (
    pprint_tree,
    get_node,
    tree_depth,
    count_nodes,
    list_tree_variables,
    list_tree_terminals,
    get_random_node,
    replace_node)

from TreeIO import serialize_tree,deserialize_tree
from TreeEvaluation import make_decision
from TreeMutation import point_mutate
from TreeCrossover import crossover_reproduction

import argparse
import os
from multiprocessing import Pool

reusable_pool = None

def start_pool(config_data:Dict)->int:
    """Initializes a process pool with the number of processes specified in the
    config.
    Returns the number or processes initialized"""
    global reusable_pool
    procs = config_data["process_pool_size"]\
            if "process_pool_size" in config_data\
            else 1

    reusable_pool = Pool(processes=procs)
    return procs


def prepare_raw_data(data_path:str,config_data:Dict)->pd.DataFrame:
    """Read in the specified data and select only the variables used in the
    in the creation of trees. Also clean up the index column if it exists
    Parameters:
        config_data (Dict): Configuration data for this run."""
    df = pd.read_csv(data_path)
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0",inplace=True,axis=1)

    if "variables" in config_data:
        return df[config_data["variables"]]
    else:
        return df


def initialize_buysell_trees(config_data:Dict,data:pd.DataFrame)->List[Dict]:
    """Initialize a population of trees. Note that a combination of a buy and
    a sell tree is considered a sinlge population member since both actions
    depend on eachother. An initial population of 100 results in 200 trees."""
    global reusable_pool
    if not reusable_pool:
        raise RuntimeError("The process pool was not initialized properly.")

    initial_population = config_data["initial_population"]\
                         if "initial_population" in config_data\
                         else 1
    buy_node_depth = config_data["initial_buy_node_depth"]\
                    if "initial_buy_node_depth" in config_data\
                    else 3
    sell_node_depth = config_data["initial_sell_node_depth"]\
                    if "initial_sell_node_depth" in config_data\
                    else 3

    variables = create_initialization_variables(data)
    args = [[variables.copy(),buy_node_depth] 
            for _ in range(initial_population)]
    buy_trees = reusable_pool.starmap(create_buy_tree,args)

    args = [[variables.copy(),sell_node_depth] 
            for _ in range(initial_population)]
    sell_trees = reusable_pool.starmap(create_sell_tree,args)

    return [{"buy":b,"sell":s,"state":"BUY","fitness":0.0} 
            for b,s in zip(buy_trees,sell_trees)]



def main(config):
    df = prepare_raw_data(
        data_path=config["data_file_path"],
        config_data=config)

    population = initialize_buysell_trees(config_data=config,data=df)
    for p in population:
        print(p)

    
    return

    vars_set = create_initialization_variables(df)
    tree1 = create_buy_tree(variables=vars_set,depth=4)
    print("TreeA")
    pprint_tree(tree1)
    
    tree2 = create_buy_tree(variables=vars_set,depth=4)
    print("TreeB")
    pprint_tree(tree2)

    child1,child2 = crossover_reproduction(treeA=tree1,treeB=tree2,probability=1)

    print("child1")
    pprint_tree(child1)
    print("child2")
    pprint_tree(child2)
    


    # vars_set = create_initial_variables(data=df)
    # vars_set.update({"bought_price":20000})

    # for i in range(100):
    #     point_mutate(tree,variables=vars_set,terminals=["SELL","HOLD"])
    # pprint_tree(tree)

    # serial = serialize_tree(tree)
    # pprint_tree(deserialize_tree(serial))
    # step_thresholds(tree,1,0.01)
    # print(stringify_tree(tree))
    # print(Node(var_name="percent_b",df=df))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train AutoTrading Genetic Algorithm')
    parser.add_argument('--config',type=str,dest="config",
                        help='Genetic Algorithm Configuration file',
                        required=True)
    config_path = parser.parse_args().config
    config = parse_configuration(os.path.abspath(config_path))
    print("Processes: ",start_pool(config))
    main(config)