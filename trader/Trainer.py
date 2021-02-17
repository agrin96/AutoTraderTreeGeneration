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
from Reporting import pprint_generation_statistics

from Population import (
    population_initialization,
    population_evaluation,
    population_fitness,
    population_mutation,
    population_reproduction,
    population_selection)

from Indicators.IndicatorVariables import indicator_variables,create_memo_hash

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

    ticker = prepare_raw_data(data_path="./Data/BTCUSDT_ticker.csv")
    candles = convert_ticker_to_candles(ticker=ticker,period=30)
    test,train = continuos_train_test_split(candles,split=0.5)

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
        print("MEMO: ",evaluation_memo)
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
        print("|-------------------------------------------------------------|")
        
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
    
    print(best)
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