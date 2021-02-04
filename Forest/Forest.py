import argparse
import sys
from typing import List,Dict,Tuple
import os
from time import time

sys.path.append("../trader/")
from Common import parse_configuration
from TreeIO import deserialize_tree
from DataPrepare import prepare_raw_data
from TreeEvaluation import natural_price_increase,score_decisions
from Population import population_evaluation

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


def import_models(path:str)->Tuple:
    """Read all serialized model pairs at the specified path and return the
    tuple of buy_trees and sell_tress. The models popids are set based on the
    file names."""
    files = os.listdir(path)
    model_files = [f for f in files if "popfile" in f]
    
    buy_files = [f for f in model_files if "buy" in f]
    sell_files = [f for f in model_files if "sell" in f]

    buy_trees = []
    for buy in buy_files:
        popid = int((buy.split(".")[0]).split("-")[-1])

        with open(os.path.join(path,buy),"r") as data:
            buy_trees.append({
                "popid": popid,
                "tree": deserialize_tree(data.read())
            })

    sell_trees = []
    for sell in sell_files:
        popid = int((sell.split(".")[0]).split("-")[-1])

        with open(os.path.join(path,sell),"r") as data:
            sell_trees.append({
                "popid": popid,
                "tree": deserialize_tree(data.read())
            })

    buy_trees = sorted(buy_trees,key=lambda k: k["popid"])
    sell_trees = sorted(sell_trees,key=lambda k: k["popid"])

    return buy_trees,sell_trees


def forest_voting(decisions:List[List[str]])->List[str]:
    """Uses a simply majority voting system to decide what the decision should
    be at every point. Returns 1D list of decisions as if a single pop had been
    evaluated."""
    final_decisions = []

    column = 0
    while column < len(decisions[0]):
        count_buy = ["BUY",0]
        count_sell = ["SELL",0]
        count_hold = ["HOLD",0]

        for row in decisions:
            if row[column] == count_buy[0]:
                count_buy[1] += 1
            elif row[column] == count_sell[0]:
                count_sell[1] += 1
            else:
                count_hold[1] += 1
        # print("DECISION: ",[count_buy,count_sell,count_hold])
        vote = max(count_buy,count_sell,count_hold,key=lambda k:k[1])

        final_decisions.append(vote[0])
        column +=1

    return final_decisions
    

def RandomForest(config:Dict):
    runtime_start = time()
    global reusable_pool
    if not reusable_pool:
        raise RuntimeError("The process pool was not initialized properly.")

    # In this case we are only testing so no train/test split necessary
    data = prepare_raw_data(data_path=config["data_file_path"],
                            config_data=config)
    print("|-----------------------------------------------------------------|")
    print("|-Evaluating Forest on full data.")
    print("|-Long Position reference balance:")

    long_balance = natural_price_increase(config,data)
    print(F"\tbalance: {long_balance}")

    print("|-Importing Models into Forest")
    buy_trees,sell_trees = import_models(config["serialized_models_path"])
    print(F"\tModels in Forest: {len(buy_trees)}")

    print("|-Evaluating Forest")
    decisions = population_evaluation(buy_trees,sell_trees,data,reusable_pool)
    final_decisions = forest_voting(decisions)
    
    print("|-Scoring Decisions")
    starting_funds = config["evaluation"]["initial_funds"]
    trading_fee = config["evaluation"]["trading_fee_percent"]

    balance,trades = score_decisions(starting_funds,
                                     trading_fee,
                                     final_decisions,
                                     data["best_bid"].values,
                                     data["best_ask"].values)
    print("\nForest Performance")
    print(F"\tFinal Balance: {balance}")
    print(F"\tTrades Made: {trades}")
    print(F"\tPerformance against Long: {balance/long_balance}")
    print("|-----------------------------------------------------------------|")

    print(F"\n\tTotal Runtime: {(time() - runtime_start)/60} minutes\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Evaluate Random Forest')
    parser.add_argument('--config',type=str,dest="config",
                        help='Random Forest Configuration file',
                        required=True)
    config_path = parser.parse_args().config
    config = parse_configuration(os.path.abspath(config_path))
    start_process_pool(config)
    RandomForest(config)