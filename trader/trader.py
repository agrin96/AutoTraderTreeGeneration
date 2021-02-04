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

from Speciation import speciate_by_coordinate,speciate_by_structure

from Selection import (
    tournament_selection,
    match_hanging_trees)

from TreeIO import serialize_tree,deserialize_tree
from TreeEvaluation import (
    make_pop_decisions,
    score_decisions,
    calculate_fitness,
    number_of_valid_trades)

from TreeMutation import point_mutate
from Crossover import single_crossover_reproduction,repopulate

import argparse
import os
from multiprocessing import Pool
from itertools import repeat
from time import time
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
    max_buy_depth = config["max_buy_node_depth"]
    max_sell_depth = config["max_sell_node_depth"]
    initial_step_size = config["threshold_step_percent"]


    train_df,test_df = continuos_train_test_split(df,train_split)
    variables = create_initialization_variables(train_df)

    args = [[variables.copy(),idx,buy_depth] 
            for idx in range(initial_population)]
    buy_trees = reusable_pool.starmap(create_buy_tree,args)

    args = [[variables.copy(),idx,sell_depth] 
            for idx in range(initial_population)]
    sell_trees = reusable_pool.starmap(create_sell_tree,args)

    evaluation_time = []
    scoring_time = []
    speciation_time = []
    selection_time = []
    reproduction_time = []
    mutation_time = []

    modulated_step_size = initial_step_size
    for i in range(1,generations+1):
        print("-----------------------------------")
        print(F"\tGeneration {i}")

        # evaluate
        print("Evaluating")
        start = time()
        buy_trees = sorted(buy_trees,key=lambda k: k["popid"])
        sell_trees = sorted(sell_trees,key=lambda k: k["popid"])

        args = zip(buy_trees,sell_trees,repeat(train_df,len(buy_trees)))
        decisions = reusable_pool.starmap(make_pop_decisions,args)

        evaluation_time.append(time()-start)
        
        print("Scoring")
        start = time()
        args = [[starting_funds,
                 trading_fee,
                 dset,
                 train_df["best_bid"].values,
                 train_df["best_ask"].values] for dset in decisions]
        scores = reusable_pool.starmap(score_decisions,args)
        
        for score,buy,sell in zip(scores,buy_trees,sell_trees):
            balance,trades = score
            buy["balance"] = balance
            buy["trades"] = trades

            sell["balance"] = balance
            sell["trades"] = trades

        args = [[pop,max_buy_depth,starting_funds] for pop in buy_trees]
        buy_trees = reusable_pool.starmap(calculate_fitness,args)

        args = [[pop,max_sell_depth,starting_funds] for pop in sell_trees]
        sell_trees = reusable_pool.starmap(calculate_fitness,args)
        scoring_time.append(time()-start)
        
        for b in buy_trees:
            print("\t" + str(b))
        print("--------")
        for b in sell_trees:
            print("\t" + str(b))

        print("Speciation")
        start = time()
        buy_trees = speciate_by_structure(buy_trees)
        sell_trees = speciate_by_structure(sell_trees)
        # buy_trees = speciate_by_coordinate(buy_trees,search_modifier,reusable_pool)
        # sell_trees = speciate_by_coordinate(sell_trees,search_modifier,reusable_pool)
        speciation_time.append(time()-start)

        print("Selection")
        start = time()
        buy_trees = tournament_selection(buy_trees,10,0.25)
        sell_trees = tournament_selection(sell_trees,10,0.25)
        selection_time.append(time()-start)

        print("-----------------------------------")
        print("\tCurrent Best")
        best_buy = max(buy_trees,key=lambda k: k["fitness"])
        best_sell = max(sell_trees,key=lambda k: k["fitness"])
        print(best_buy)
        print(best_sell)
        print("-----------------------------------")
        print("\tAverage Balance and Fitness of Buys")
        mean_fitness = np.mean(list(map(lambda k: k["fitness"],buy_trees)))
        mean_balance = np.mean(list(map(lambda k: k["balance"],buy_trees)))
        print(F"\tMean Fitness: {mean_fitness}")
        print(F"\tMean Balance: {mean_balance}")
        print("\tAverage Balance and Fitness of Sells")
        mean_fitness = np.mean(list(map(lambda k: k["fitness"],sell_trees)))
        mean_balance = np.mean(list(map(lambda k: k["balance"],sell_trees)))
        print(F"\tMean Fitness: {mean_fitness}")
        print(F"\tMean Balance: {mean_balance}")

        print("Reproduction")
        start = time()
        buy_trees = repopulate(buy_trees,max_population,crossover)
        sell_trees = repopulate(sell_trees,max_population,crossover)
        buy_trees,sell_trees = match_hanging_trees(buy_trees,sell_trees)
        reproduction_time.append(time()-start)

        print("Mutation")
        # As the generations go on our step size decreases
        if i % 10 == 0:
            modulated_step_size = initial_step_size * (1 / i)

        buy_trees = sorted(buy_trees,key=lambda k: k["fitness"])
        sell_trees = sorted(sell_trees,key=lambda k: k["fitness"])

        # save the best members for next run
        holding_buy_trees = []
        holding_sell_trees = []
        for _ in range(5):
            temp_buy = max(buy_trees,key=lambda k: k["fitness"])
            temp_sell = max(sell_trees,key=lambda k: k["fitness"])
            holding_buy_trees.append(temp_buy)
            holding_sell_trees.append(temp_sell)
            buy_trees.remove(temp_buy)
            sell_trees.remove(temp_sell)

        start = time()
        args = [
            [pop,
            variables.copy(),
            ["BUY","HOLD"],
            unique_tree_variables,
            mutation_rate,
            modulated_step_size,
            mutation_types.copy()] for pop in buy_trees]
        mutated_buys = reusable_pool.starmap(point_mutate,args)
        buy_trees = holding_buy_trees + mutated_buys

        args = [
            [pop,
            variables.copy(),
            ["SELL","HOLD"],
            unique_tree_variables,
            mutation_rate,
            modulated_step_size,
            mutation_types.copy()] for pop in sell_trees]
        mutated_sells = reusable_pool.starmap(point_mutate,args)
        sell_trees = holding_sell_trees + mutated_sells
        mutation_time.append(time()-start)


    print("Scoring TEST")
    # evaluate
    buy_trees = sorted(buy_trees,key=lambda k: k["popid"])
    sell_trees = sorted(sell_trees,key=lambda k: k["popid"])

    args = zip(buy_trees,sell_trees,repeat(test_df,len(buy_trees)))
    decisions = reusable_pool.starmap(make_pop_decisions,args)
    
    print("Scoring")
    args = [[starting_funds,
            trading_fee,
            dset,
            test_df["best_bid"].values,
            test_df["best_ask"].values] for dset in decisions]
    scores = reusable_pool.starmap(score_decisions,args)
        
    for score,buy,sell in zip(scores,buy_trees,sell_trees):
        balance,trades = score
        buy["balance"] = balance
        buy["trades"] = trades

        sell["balance"] = balance
        sell["trades"] = trades
    
    args = [[pop,max_buy_depth,starting_funds] for pop in buy_trees]
    buy_trees = reusable_pool.starmap(calculate_fitness,args)

    args = [[pop,max_sell_depth,starting_funds] for pop in sell_trees]
    sell_trees = reusable_pool.starmap(calculate_fitness,args)

    # Get the most fit members of each
    best_buy = max(buy_trees,key=lambda k: k["fitness"])
    best_sell = max(sell_trees,key=lambda k: k["fitness"])

    print("-----------------------------------")
    print("\tBest Trees")
    print(best_buy)
    print(best_sell)
    pprint_tree(best_buy["tree"])
    print("")
    print("\tBest Sell tree")
    pprint_tree(best_sell["tree"])

    print("-----------------------------------")
    print("\tRuntime Statistics")
    print(F"Mean Evaluation: {np.mean(evaluation_time)} seconds")
    print(F"Mean Scoring: {np.mean(scoring_time)} seconds")
    print(F"Mean Speciation: {np.mean(speciation_time)} seconds")
    print(F"Mean Selection: {np.mean(selection_time)} seconds")
    print(F"Mean Reproduction: {np.mean(reproduction_time)} seconds")
    print(F"Mean Mutation: {np.mean(mutation_time)} seconds")
    print("-----------------------------------")

    
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