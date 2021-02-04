from __future__ import annotations
from typing import List,Union,Dict,Tuple
import pandas as pd
import numpy as np
import uuid
from itertools import repeat
from time import time
import argparse
import os

from DataStructures.Node import Node
from DataStructures.Terminal import Terminal

from Common import (
    random_choice,
    parse_configuration,
    pprint_generation_statistics)

from DataPrepare import (
    prepare_raw_data,
    continuos_train_test_split,
    create_data_subset)

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

from Speciation import (
    speciate_by_coordinate,
    speciate_by_structure)

from Selection import (
    tournament_selection,
    match_hanging_trees)

from TreeIO import (
    serialize_tree,
    deserialize_tree)

from TreeEvaluation import (
    make_pop_decisions,
    score_decisions,
    calculate_fitness,
    number_of_valid_trades)

from TreeMutation import point_mutate

from Crossover import (
    single_crossover_reproduction,
    repopulate)

# ----------------------- GLOBALS ----------------------- #
from multiprocessing import Pool
reusable_pool = None
# --------------------- END GLOBALS --------------------- #

def start_process_pool(config_data:Dict)->int:
    """Initializes a process pool with the number of processes specified in the
    config. If the pool is size 1 then we dont initialize the pool and all
    functions will default to iteration.
    Returns the number or processes initialized"""
    global reusable_pool
    procs = config_data["process_pool_size"]\
            if "process_pool_size" in config_data\
            else 1
    
    if procs == 1:
        print("Running on current single process.")
        return procs

    reusable_pool = Pool(processes=procs)
    return procs


def population_initialization(config:Dict,variables:Dict)->Tuple:
    """Generate a population of buy and sell trees using the configuration
    options and variables provided.
    Returns a tuple of (buy_trees,sell_trees)"""
    global reusable_pool
    
    buy_depth = config["population"]["initial_buy_node_depth"]
    sell_depth = config["population"]["initial_sell_node_depth"]
    initial_population = config["population"]["initial_population"]

    args = [[variables.copy(),idx,buy_depth] 
            for idx in range(initial_population)]

    buy_trees = reusable_pool.starmap(create_buy_tree,args)\
                if reusable_pool\
                else [create_buy_tree(*a) for a in args]

    args = [[variables.copy(),idx,sell_depth] 
            for idx in range(initial_population)]

    sell_trees = reusable_pool.starmap(create_sell_tree,args)\
                 if reusable_pool\
                 else [create_sell_tree(*a) for a in args]

    return buy_trees,sell_trees


def population_evaluation(buy_trees:List[Dict],
                          sell_trees:List[Dict],
                          data:pd.DataFrame)->List:
    """Evaluate the buy and sell trees in popid pair order and return a set of
    decisions."""
    global reusable_pool

    buy_trees = sorted(buy_trees,key=lambda k: k["popid"])
    sell_trees = sorted(sell_trees,key=lambda k: k["popid"])

    args = zip(buy_trees,sell_trees,repeat(data,len(buy_trees)))
    decisions = reusable_pool.starmap(make_pop_decisions,args)\
                if reusable_pool\
                else [make_pop_decisions(*a) for a in args]

    return decisions


def population_fitness(config:Dict,
                       buy_trees:List[Dict],
                       sell_trees:List[Dict],
                       decisions:List[str],
                       data:pd.DataFrame)->Tuple:
    """Uses the previously evaluated decisions to generate scores for each
    buy/sell pair. These are stores as balance and trades. Then calculates the
    fitness values using the score information.
    Returns tuple of updated (buy_trees,sell_trees)"""
    global reusable_pool

    starting_funds = config["evaluation"]["initial_funds"]
    trading_fee = config["evaluation"]["trading_fee_percent"]
    max_buy_depth = config["population"]["max_buy_node_depth"]
    max_sell_depth = config["population"]["max_sell_node_depth"]

    args = [[starting_funds,
            trading_fee,
            dset,
            data["best_bid"].values,
            data["best_ask"].values] for dset in decisions]
    scores = reusable_pool.starmap(score_decisions,args)\
             if reusable_pool\
             else [score_decisions(*a) for a in args]
    
    for score,buy,sell in zip(scores,buy_trees,sell_trees):
        balance,trades = score
        buy["balance"] = balance
        buy["trades"] = trades

        sell["balance"] = balance
        sell["trades"] = trades

    args = [[pop,max_buy_depth,starting_funds] for pop in buy_trees]
    buy_trees = reusable_pool.starmap(calculate_fitness,args)\
                if reusable_pool\
                else [calculate_fitness(*a) for a in args]

    args = [[pop,max_sell_depth,starting_funds] for pop in sell_trees]
    sell_trees = reusable_pool.starmap(calculate_fitness,args)\
                 if reusable_pool\
                 else [calculate_fitness(*a) for a in args]

    return buy_trees,sell_trees


def population_mutation(config:Dict,
                        buy_trees:List[Dict],
                        sell_trees:List[Dict],
                        variables:Dict,
                        step_size:float)->Tuple:
    """Execute mutation on the population of buy and sell trees. Returns the
    mutated set of buy and sell trees including any non affected trees."""
    global reusable_pool
    
    mutation_types = config["mutation"]["mutation_types"]
    mutation_rate = config["mutation"]["mutation_rate"]
    unique_tree_variables = config["population"]["unique_tree_variables"]
    alphas = config["mutation"]["alphas"]

    buy_trees = sorted(buy_trees,key=lambda k: k["fitness"])
    sell_trees = sorted(sell_trees,key=lambda k: k["fitness"])

    # save the best members for next run
    holding_buy_trees = []
    holding_sell_trees = []
    for _ in range(alphas):
        temp_buy = max(buy_trees,key=lambda k: k["fitness"])
        temp_sell = max(sell_trees,key=lambda k: k["fitness"])
        holding_buy_trees.append(temp_buy)
        holding_sell_trees.append(temp_sell)
        buy_trees.remove(temp_buy)
        sell_trees.remove(temp_sell)

    args = [
        [pop,
        variables.copy(),
        ["BUY","HOLD"],
        unique_tree_variables,
        mutation_rate,
        step_size,
        mutation_types.copy()] for pop in buy_trees]

    mutated_buys = reusable_pool.starmap(point_mutate,args)\
                   if reusable_pool\
                   else [point_mutate(*a) for a in args]
    buy_trees = holding_buy_trees + mutated_buys

    args = [
        [pop,
        variables.copy(),
        ["SELL","HOLD"],
        unique_tree_variables,
        mutation_rate,
        step_size,
        mutation_types.copy()] for pop in sell_trees]

    mutated_sells = reusable_pool.starmap(point_mutate,args)\
                    if reusable_pool\
                    else [point_mutate(*a) for a in args]
    sell_trees = holding_sell_trees + mutated_sells

    return buy_trees,sell_trees 


def population_selection(config:Dict,
                         buy_trees:List[Dict],
                         sell_trees:List[Dict])->Tuple:
    """Execute tournament selection on the population using the parameters
    set in the config. Returns the buy and sell trees tuple."""
    tourn_size = config["selection"]["tournament_size"]
    survivor_percent = config["selection"]["survivors_percent"]

    buy_trees = tournament_selection(buy_trees,tourn_size,survivor_percent)
    sell_trees = tournament_selection(sell_trees,tourn_size,survivor_percent)

    return buy_trees,sell_trees


def population_reproduction(config:Dict,
                            buy_trees:List[Dict],
                            sell_trees:List[Dict])->Tuple:
    """Run crossover reproduction on the population with config options as is.
    and return the tuple of the buy/sell population."""
    crossover = config["crossover"]["crossover_rate"]
    max_population = config["population"]["max_population"]

    buy_trees = repopulate(buy_trees,max_population,crossover)
    sell_trees = repopulate(sell_trees,max_population,crossover)
    buy_trees,sell_trees = match_hanging_trees(buy_trees,sell_trees)

    return buy_trees,sell_trees


def train_trader(config):
    global reusable_pool
    if not reusable_pool:
        raise RuntimeError("The process pool was not initialized properly.")
    
    runtime_start = time()
    df = prepare_raw_data(
        data_path=config["data_file_path"],
        config_data=config)


    # Variable initialization
    train_split = config["train_percent_split"]
    generations = config["generations"]
    initial_step_size = config["mutation"]["threshold_step_percent"]
    threshold_step_interval = config["mutation"]["threshold_step_interval"]

    data_subset_split = config["training_sampling_split"]\
                        if "training_sampling_split" in config\
                        else None
    subset_interval = config["training_sampling_interval"] 

    total_train_df,test_df = continuos_train_test_split(df,train_split)
    train_df = total_train_df
    if data_subset_split:
        train_df = create_data_subset(total_train_df,data_subset_split)

    variables = create_initialization_variables(train_df)
    buy_trees,sell_trees = population_initialization(config,variables)

    evaluation_time = []
    scoring_time = []
    speciation_time = []
    selection_time = []
    reproduction_time = []
    mutation_time = []

    modulated_step_size = initial_step_size
    for generation in range(1,generations+1):
        print("|-------------------------------------------------------------|")
        print(F"|-Executing Generation {generation}")

        print("|-Evaluating")
        start = time()
        decisions = population_evaluation(buy_trees,sell_trees,train_df)
        evaluation_time.append(time()-start)
        
        print("|-Calculating Fitness")
        start = time()
        buy_trees,sell_trees = population_fitness(config,
                                                  buy_trees,
                                                  sell_trees,
                                                  decisions,
                                                  train_df)
        scoring_time.append(time()-start)

        print("|-Clustering Speciation")
        start = time()
        buy_trees = speciate_by_structure(buy_trees)
        sell_trees = speciate_by_structure(sell_trees)
        speciation_time.append(time()-start)

        print("|-Selection")
        start = time()
        buy_trees,sell_trees = population_selection(config,buy_trees,sell_trees)
        selection_time.append(time()-start)

        pprint_generation_statistics(buy_trees,sell_trees)

        print("|-Repopulating via Reproduction")
        start = time()
        buy_trees,sell_trees = population_reproduction(config,
                                                       buy_trees,
                                                       sell_trees)
        reproduction_time.append(time()-start)

        print("|-Mutating Population")
        # As the generations go on our step size decreases
        if generation % 10 == 0:
            modulated_step_size = initial_step_size * (1 / generation)

        buy_trees,sell_trees = population_mutation(config,
                                                   buy_trees,
                                                   sell_trees,
                                                   variables,
                                                   modulated_step_size)
        mutation_time.append(time()-start)

        if data_subset_split:
            if generation % subset_interval == 0:
                print("|-Sampling new training data subset")
                train_df = create_data_subset(total_train_df,data_subset_split)
        print("|-------------------------------------------------------------|")


    print("Evaluating Best Members on Test Dataset")
    # evaluate
    decisions = population_evaluation(buy_trees,sell_trees,test_df)
    buy_trees,sell_trees = population_fitness(config,
                                              buy_trees,
                                              sell_trees,
                                              decisions,
                                              test_df)

    # Get the most fit members of each
    best_buy = max(buy_trees,key=lambda k: k["fitness"])
    best_sell = max(sell_trees,key=lambda k: k["fitness"])

    print(best_buy)
    print(best_sell)
    pprint_tree(best_buy["tree"])
    print("")
    pprint_tree(best_sell["tree"])

    print("\n|-------------------------------------------------------------|")
    print("Runtime Statistics")
    print(F"\tMean Evaluation: {np.mean(evaluation_time)} seconds")
    print(F"\tMean Scoring: {np.mean(scoring_time)} seconds")
    print(F"\tMean Speciation: {np.mean(speciation_time)} seconds")
    print(F"\tMean Selection: {np.mean(selection_time)} seconds")
    print(F"\tMean Reproduction: {np.mean(reproduction_time)} seconds")
    print(F"\tMean Mutation: {np.mean(mutation_time)} seconds")
    print("|-------------------------------------------------------------|\n")
    
    print(F"\n\tTotal Runtime: {(time() - runtime_start)/360} hours\t")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train AutoTrading Genetic Algorithm')
    parser.add_argument('--config',type=str,dest="config",
                        help='Genetic Algorithm Configuration file',
                        required=True)
    config_path = parser.parse_args().config
    config = parse_configuration(os.path.abspath(config_path))
    procs = start_process_pool(config)
    print(F"Starting Trader Training with {procs} processes.")
    train_trader(config)