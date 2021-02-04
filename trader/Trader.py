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
    pprint_generation_statistics,
    store_serialized_pop)

from DataPrepare import (
    prepare_raw_data,
    continuos_train_test_split,
    create_data_subset,
    create_data_samples)

from CreateTree import create_initialization_variables

from TreeActions import pprint_tree

from Speciation import (
    speciate_by_coordinate,
    speciate_by_structure)

from TreeIO import (
    serialize_tree,
    deserialize_tree)

from TreeEvaluation import natural_price_increase

from Population import (
    population_initialization,
    population_evaluation,
    population_fitness,
    population_mutation,
    population_reproduction,
    population_selection)


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


def train_trader(config):
    global reusable_pool
    if not reusable_pool:
        raise RuntimeError("The process pool was not initialized properly.")
    
    runtime_start = time()
    df = prepare_raw_data(
        data_path=config["data_file_path"],
        config_data=config)


    # Initialize the train and test datasets
    train_split = config["training"]["traintest_split"]
    total_train_df,test_df = continuos_train_test_split(df,train_split)
    train_df = total_train_df
    
    # Decide if we will do any sampling of the training set.
    rotate_flag = config["training"]["rotate_training_data"]
    rotation_interval = config["training"]["rotation_interval"]
    
    fixed_training_sets = None
    current_set = 0

    # If our training is sampled, decide whether its a fixed N samples or 
    # continous sample every K generations.
    if rotate_flag:
        sets = config["training"]["training_sets"]
        set_split = config["training"]["train_sampling_split"]
        if sets == -1:
            train_df = create_data_subset(total_train_df,set_split)
        else:
            fixed_training_sets = create_data_samples(total_train_df,sets,set_split)
            train_df = fixed_training_sets[0]

    variables = create_initialization_variables(train_df)
    buy_trees,sell_trees = population_initialization(config,
                                                     variables,
                                                     reusable_pool)

    evaluation_time = []
    scoring_time = []
    speciation_time = []
    selection_time = []
    reproduction_time = []
    mutation_time = []

    initial_step_size = config["mutation"]["threshold_step_percent"]
    modulated_step_size = initial_step_size
    
    for generation in range(1,config["generations"]+1):
        print("|-------------------------------------------------------------|")
        print(F"|-Executing Generation {generation}")
        print("|-Long Position Baseline growth")
        print("\tbalance: ",natural_price_increase(config,train_df))
        print("|-Evaluating")
        start = time()
        decisions = population_evaluation(buy_trees,
                                          sell_trees,
                                          train_df,
                                          reusable_pool)
        evaluation_time.append(time()-start)
        
        print("|-Calculating Fitness")
        start = time()
        buy_trees,sell_trees = population_fitness(config,
                                                  buy_trees,
                                                  sell_trees,
                                                  decisions,
                                                  train_df,
                                                  reusable_pool)
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
                                                   modulated_step_size,
                                                   reusable_pool)
        mutation_time.append(time()-start)
        
        if rotate_flag:
            if fixed_training_sets:
                if generation % rotation_interval == 0:
                    print("|-Rotating training dataset.")
                    current_set += 1
                    if current_set >= len(fixed_training_sets):
                        current_set = 0
                    train_df = fixed_training_sets[current_set]
            else:
                if generation % rotation_interval == 0:
                    print("|-Sampling new training dataset.")
                    set_split = config["training"]["train_sampling_split"]
                    train_df = create_data_subset(total_train_df,set_split)
            
            # Update the variables list with training subsets.
            variables = create_initialization_variables(train_df)
        print("|-------------------------------------------------------------|")


    print("\n\nEvaluating Best Members on Test Dataset")
    print("|-Long position baseline.")
    print(F"\tbalance: {natural_price_increase(config,test_df)}\n")

    decisions = population_evaluation(buy_trees,sell_trees,test_df,reusable_pool)
    buy_trees,sell_trees = population_fitness(config,
                                              buy_trees,
                                              sell_trees,
                                              decisions,
                                              test_df,
                                              reusable_pool)

    # Get the most fit members of each
    best_buy = max(buy_trees,key=lambda k: k["fitness"])
    best_sell = max(sell_trees,key=lambda k: k["fitness"])
    
    if config["save_best"]:
        print("|-Saving best members serialized.")
        store_serialized_pop(serial_buy=serialize_tree(best_buy["tree"]),
                            serial_sell=serialize_tree(best_sell["tree"]))
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

    print(F"\n\tTotal Runtime: {(time() - runtime_start)/3600} hours\t")

    
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