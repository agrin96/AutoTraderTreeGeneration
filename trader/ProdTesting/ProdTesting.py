from __future__ import annotations
import pandas as pd
import argparse
import os
from time import time

import sys
sys.path.append("../")
from TreeIO import deserialize_tree
from TreeEvaluation import natural_price_increase
from Common import parse_configuration
from Population import population_evaluation,population_fitness

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

    candles = pd.read_csv("../Data/prod_test_candles.csv")
    pops = []
    with open("./ProdTrees/popfile-0.json") as model:
        tree = deserialize_tree(model.read())
        pop = {"popid":0,"balance":None,"fitness":None,"Trades":0,"tree":tree}
        pops.append(pop)

    print("\n\nEvaluating Model on Prod Dataset")
    print("|-Long position baseline.")
    print(F"\tbalance: {natural_price_increase(config,candles)}\n")

    decisions = population_evaluation(pops,candles,evaluation_memo,reusable_pool)
    pops = population_fitness(config,pops,decisions,candles,reusable_pool)
    
    best = pops[0]
    output = F"POPID: {best['popid']} Fitness: {best['fitness']}"
    output += F" Balance: {best['balance']} Trades: {best['trades']}"
    print("\t"+str(output))
    print(F"\n\tTotal Runtime: {(time() - runtime_start)/3600} hours\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test AutoTrading Genetic Algorithm')
    parser.add_argument('--config',type=str,dest="config",
                        help='Genetic Algorithm Configuration file',
                        required=True)
    config_path = parser.parse_args().config
    config = parse_configuration(os.path.abspath(config_path))
    
    procs = start_process_pool({})
    print(F"Starting Trader Training with {procs} processes.")
    main(config)