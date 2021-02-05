from __future__ import annotations
from typing import List,Union,Dict,Tuple
import pandas as pd
import numpy as np
import uuid
from itertools import repeat
from time import time
import argparse
import os
from multiprocessing import Pool

from Selection import (
    tournament_selection,
    match_hanging_trees)

from TreeEvaluation import (
    make_pop_decisions,
    score_decisions,
    calculate_fitness,
    natural_price_increase)

from Crossover import (
    repopulate)

from CreateTree import (
    create_buy_tree,
    create_sell_tree)

from TreeMutation import point_mutate


def population_initialization(config:Dict,variables:Dict,pool:Pool=None)->Tuple:
    """Generate a population of buy and sell trees using the configuration
    options and variables provided.
    Returns a tuple of (buy_trees,sell_trees)"""
    buy_depth = config["population"]["initial_buy_node_depth"]
    sell_depth = config["population"]["initial_sell_node_depth"]
    initial_population = config["population"]["initial_population"]

    args = [[variables.copy(),idx,buy_depth] 
            for idx in range(initial_population)]

    buy_trees = pool.starmap(create_buy_tree,args)\
                if pool\
                else [create_buy_tree(*a) for a in args]

    args = [[variables.copy(),idx,sell_depth] 
            for idx in range(initial_population)]

    sell_trees = pool.starmap(create_sell_tree,args)\
                 if pool\
                 else [create_sell_tree(*a) for a in args]

    return buy_trees,sell_trees


def population_evaluation(buy_trees:List[Dict],
                          sell_trees:List[Dict],
                          data:pd.DataFrame,
                          pool:Pool=None)->List:
    """Evaluate the buy and sell trees in popid pair order and return a set of
    decisions."""
    buy_trees = sorted(buy_trees,key=lambda k: k["popid"])
    sell_trees = sorted(sell_trees,key=lambda k: k["popid"])
    
    args = zip(buy_trees,sell_trees,repeat(data,len(buy_trees)))
    decisions = pool.starmap(make_pop_decisions,args)\
                if pool\
                else [make_pop_decisions(*a) for a in args]

    return decisions


def population_fitness(config:Dict,
                       buy_trees:List[Dict],
                       sell_trees:List[Dict],
                       decisions:List[str],
                       data:pd.DataFrame,
                       pool:Pool=None)->Tuple:
    """Uses the previously evaluated decisions to generate scores for each
    buy/sell pair. These are stores as balance and trades. Then calculates the
    fitness values using the score information.
    Returns tuple of updated (buy_trees,sell_trees)"""
    starting_funds = config["evaluation"]["initial_funds"]
    trading_fee = config["evaluation"]["trading_fee_percent"]
    max_buy_depth = config["population"]["max_buy_node_depth"]
    max_sell_depth = config["population"]["max_sell_node_depth"]

    args = [[starting_funds,
            trading_fee,
            dset,
            data["best_bid"].values,
            data["best_ask"].values] for dset in decisions]
    scores = pool.starmap(score_decisions,args)\
             if pool\
             else [score_decisions(*a) for a in args]
    
    for score,buy,sell in zip(scores,buy_trees,sell_trees):
        balance,trades = score
        buy["balance"] = balance
        buy["trades"] = trades

        sell["balance"] = balance
        sell["trades"] = trades

    long_balance = natural_price_increase(config,data)

    args = [[pop,max_buy_depth,starting_funds,long_balance] 
            for pop in buy_trees]
    buy_trees = pool.starmap(calculate_fitness,args)\
                if pool\
                else [calculate_fitness(*a) for a in args]

    args = [[pop,max_sell_depth,starting_funds,long_balance]
            for pop in sell_trees]
    sell_trees = pool.starmap(calculate_fitness,args)\
                 if pool\
                 else [calculate_fitness(*a) for a in args]

    return buy_trees,sell_trees


def population_mutation(config:Dict,
                        buy_trees:List[Dict],
                        sell_trees:List[Dict],
                        variables:Dict,
                        step_size:float,
                        pool:Pool=None)->Tuple:
    """Execute mutation on the population of buy and sell trees. Returns the
    mutated set of buy and sell trees including any non affected trees."""
    mutation_types = config["mutation"]["mutation_types"]
    mutation_rate = config["mutation"]["mutation_rate"]
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
        mutation_rate,
        mutation_types.copy()] for pop in buy_trees]

    mutated_buys = pool.starmap(point_mutate,args)\
                   if pool\
                   else [point_mutate(*a) for a in args]
    buy_trees = holding_buy_trees + mutated_buys

    args = [
        [pop,
        variables.copy(),
        ["SELL","HOLD"],
        mutation_rate,
        mutation_types.copy()] for pop in sell_trees]

    mutated_sells = pool.starmap(point_mutate,args)\
                    if pool\
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