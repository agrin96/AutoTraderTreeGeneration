from __future__ import annotations
from typing import List,Union,Dict,Tuple
import pandas as pd
import numpy as np
import uuid

from DataPrepare import (
    prepare_raw_data,
    continuos_train_test_split)
from Common import (
    random_choice,
    parse_configuration,
    check_configuration)
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

from Speciation import speciate_by_kmeans

from Selection import (
    tournament_selection,
    match_hanging_trees)

from TreeIO import serialize_tree,deserialize_tree
from TreeEvaluation import make_pop_decisions,score_decisions
from TreeMutation import mutate
from Crossover import single_crossover_reproduction,repopulate

import argparse
import os
from multiprocessing import Pool
from itertools import repeat
reusable_pool = None


def start_process_pool(config_data:Dict)->int:
    """Initializes a process pool with the number of processes specified in the
    config.
    Returns the number or processes initialized"""
    global reusable_pool
    procs = config_data["process_pool_size"]\
            if "process_pool_size" in config_data\
            else 1

    reusable_pool = Pool(processes=procs)
    return procs


def main(config):
    df = prepare_raw_data(
        data_path=config["data_file_path"],
        config_data=config)

    global reusable_pool
    if not reusable_pool:
        raise RuntimeError("The process pool was not initialized properly.")


    # Variable initialization
    starting_funds = config["initial_funds"]
    trading_fee = config["trading_fee_percent"]
    train_split = config["train_percent_split"]
    mutation_types = config["mutation_types"]
    mutation_rate = config["mutation_rate"]
    unique_tree_variables = config["unique_tree_variables"]
    generations = config["generations"]

    initial_population = config["initial_population"]
    buy_depth = config["initial_buy_node_depth"]
    sell_depth = config["initial_sell_node_depth"]
    search_modifier = config["search_distance_modifier"]
    crossover = config["crossover_rate"]
    max_population = config["max_population"]


    train_df,test_df = continuos_train_test_split(df,train_split)
    variables = create_initialization_variables(train_df)

    args = [[variables.copy(),idx,buy_depth] 
            for idx in range(initial_population)]
    buy_trees = reusable_pool.starmap(create_buy_tree,args)

    args = [[variables.copy(),idx,sell_depth] 
            for idx in range(initial_population)]
    sell_trees = reusable_pool.starmap(create_sell_tree,args)

    
    for i in range(generations):
        print(F"Generation {i+1}")

        # evaluate
        print("Evaluating")
        buy_trees = sorted(buy_trees,key=lambda k: k["popid"])
        sell_trees = sorted(sell_trees,key=lambda k: k["popid"])
        for b in buy_trees:
            print(b)
        print("")
        for b in sell_trees:
            print(b)

        args = zip(buy_trees,sell_trees,repeat(train_df,len(buy_trees)))
        decisions = reusable_pool.starmap(make_pop_decisions,args)
        
        print("Scoring")
        args = [[starting_funds,
                 trading_fee,
                 dset,
                 train_df["best_bid"].values,
                 train_df["best_ask"].values] for dset in decisions]
        balances = reusable_pool.starmap(score_decisions,args)
        
        for balance,buy,sell in zip(balances,buy_trees,sell_trees):
            buy["fitness"] = balance
            sell["fitness"] = balance

        print("Speciation")
        buy_trees = speciate_by_kmeans(buy_trees,search_modifier,reusable_pool)
        sell_trees = speciate_by_kmeans(sell_trees,search_modifier,reusable_pool)
        for b in buy_trees:
            print(b)
        print("")
        for b in sell_trees:
            print(b)


        print("Selection")
        buy_trees = tournament_selection(buy_trees,3,0.5)
        sell_trees = tournament_selection(sell_trees,3,0.5)
        buy_trees,sell_trees = match_hanging_trees(buy_trees,sell_trees)
        for b in buy_trees:
            print(b)
        print("")
        for b in sell_trees:
            print(b)

        print("Reproduction")
        buy_trees = repopulate(buy_trees,max_population,crossover)
        sell_trees = repopulate(sell_trees,max_population,crossover)

        print("Mutation")
        args = [
            [pop,
            variables.copy(),
            ["BUY","HOLD"],
            unique_tree_variables,
            mutation_rate,
            mutation_types.copy()] for pop in buy_trees]
        buy_trees = reusable_pool.starmap(mutate,args)
        args = [
            [pop,
            variables.copy(),
            ["SELL","HOLD"],
            unique_tree_variables,
            mutation_rate,
            mutation_types.copy()] for pop in sell_trees]
        sell_trees = reusable_pool.starmap(mutate,args)

        # Reset clusters
        print("Cluster reset")
        for i in range(len(buy_trees)):
            buy_trees[i]["cluster"] = None
            sell_trees[i]["cluster"] = None

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train AutoTrading Genetic Algorithm')
    parser.add_argument('--config',type=str,dest="config",
                        help='Genetic Algorithm Configuration file',
                        required=True)
    config_path = parser.parse_args().config
    config = parse_configuration(os.path.abspath(config_path))
    check_configuration(config)

    print("Processes: ",start_process_pool(config))
    main(config)