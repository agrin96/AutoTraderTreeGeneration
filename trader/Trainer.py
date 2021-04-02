from __future__ import annotations
from typing import List,Union,Dict,Tuple
import pandas as pd
import numpy as np
import uuid
from itertools import repeat
from time import time
import argparse
import os

from Common import (
    random_choice,
    parse_configuration,
    store_serialized_pop)

from DataPrepare import (
    prepare_raw_data,
    continuos_train_test_split,
    create_data_subset,
    create_data_samples,
    convert_ticker_to_candles)

from CreateTree import create_indicator_tree

from TreeActions import pprint_tree

from Speciation import (
    speciate_by_coordinate,
    speciate_by_structure)

from TreeIO import (
    serialize_tree,
    deserialize_tree)

from TreeEvaluation import natural_price_increase
from Reporting import (
    pprint_generation_statistics,
    plot_decisions)

from Population import (
    population_initialization,
    population_evaluation,
    population_fitness,
    population_mutation,
    population_reproduction,
    population_selection)

from Indicators.IndicatorVariables import indicator_variables

# ----------------------- GLOBALS ----------------------- #
from multiprocessing import Pool
reusable_pool = None
evaluation_memo = {}
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


def main(config:Dict):
    global reusable_pool,evaluation_memo
    runtime_start = time()
    candle_period = config["candle_period"]
    split = config["training"]["traintest_split"]

    ticker = prepare_raw_data(data_path="./Data/BTCUSDT_ticker.csv")
    candles = convert_ticker_to_candles(ticker=ticker,period=candle_period)
    all_train,test = continuos_train_test_split(candles,split=split)
    train = all_train

    # Decide if we will do any sampling of the training set.
    rotate_data = config["training"]["rotate_training_data"]
    rotation_interval = config["training"]["rotation_interval"]

    fixed_training_sets = []
    current_train_set = 0

    # If our training is sampled, decide whether its a fixed N samples or 
    # continous sample every K generations.
    if rotate_data:
        print("\tCreating Rotating Training Sets.")
        sets = config["training"]["training_sets"]
        set_split = config["training"]["train_sampling_split"]

        fixed_training_sets = create_data_samples(all_train,sets,set_split)
        train = fixed_training_sets[current_train_set]

    pops = population_initialization(config,indicator_variables,reusable_pool)

    evaluation_time = []
    scoring_time = []
    speciation_time = []
    selection_time = []
    reproduction_time = []
    mutation_time = []
    balances = []

    for generation in range(1,config["generations"]+1):
        print("|-------------------------------------------------------------|")
        print(F"|-Executing Generation {generation}")
        print("|-Long Position Baseline growth")
        print("\tbalance: ",natural_price_increase(config,train))
        print("|-Evaluating")
        start = time()
        decisions = population_evaluation(pops,train,evaluation_memo,reusable_pool)
        evaluation_time.append(time()-start)

        print("|-Calculating Fitness")
        start = time()
        pops = population_fitness(config,pops,decisions,train,reusable_pool)
        scoring_time.append(time()-start)

        print("|-Clustering Speciation")
        start = time()
        pops = speciate_by_structure(pops)
        speciation_time.append(time()-start)

        print("|-Selection")
        start = time()
        pops = population_selection(config,pops)
        selection_time.append(time()-start)

        pprint_generation_statistics(pops,balances)

        print("|-Repopulating via Reproduction")
        start = time()
        pops = population_reproduction(config,pops)
        reproduction_time.append(time()-start)


        print("|-Mutating Population")
        start = time()
        pops = population_mutation(config,pops,indicator_variables,reusable_pool)
        mutation_time.append(time()-start)

        if rotate_data:
            if generation % rotation_interval == 0:
                print("|-Rotating training dataset.")
                current_train_set += 1
                if current_train_set >= len(fixed_training_sets):
                    current_train_set = 0
                train = fixed_training_sets[current_train_set]
        print("|-------------------------------------------------------------|")
    
    # Reset memo dict since values are no longer valid
    evaluation_memo = {}
    print("\n\nEvaluating Best Members on Test Dataset")
    print("|-Long position baseline.")
    print(F"\tbalance: {natural_price_increase(config,test)}\n")

    decisions = population_evaluation(pops,test,evaluation_memo,reusable_pool)
    pops = population_fitness(config,pops,decisions,test,reusable_pool)

    # Get the most fit members of each
    best = max(pops,key=lambda k: k["fitness"])
    
    if config["save_best"]:
        print("|-Saving best members serialized.")
        store_serialized_pop(serial_pop=serialize_tree(best["tree"]))
    
    output = F"POPID: {best['popid']} Fitness: {best['fitness']}"
    output += F" Balance: {best['balance']} Gain Trades: {best['gtrades']}"
    output += F" Lose Trades: {best['ltrades']}"
    output += F" Invalid Trades: {best['itrades']}"
    print("\t"+str(output))
    pprint_tree(best["tree"])

    print("\n|-------------------------------------------------------------|")
    print("Runtime Statistics")
    print(F"\tMean Evaluation: {np.mean(evaluation_time)} seconds")
    print(F"\tMean Scoring: {np.mean(scoring_time)} seconds")
    print(F"\tMean Speciation: {np.mean(speciation_time)} seconds")
    print(F"\tMean Selection: {np.mean(selection_time)} seconds")
    print(F"\tMean Reproduction: {np.mean(reproduction_time)} seconds")
    print(F"\tMean Mutation: {np.mean(mutation_time)} seconds")
    print("|-------------------------------------------------------------|\n")

    print(F"\n\tTotal Runtime: {(time() - runtime_start)/3600} hours\n")

    plot_decisions(best,candles,candle_period,display_range=(0,-1))



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
    main(config)