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
from copy import deepcopy

from Selection import tournament_selection
from TreeEvaluation import (
    make_pop_decisions,
    score_decisions,
    calculate_fitness,
    natural_price_increase)

from Crossover import repopulate
from CreateTree import create_indicator_tree
from TreeMutation import point_mutate

from TreeActions import pprint_tree


def population_initialization(config:Dict,
                              variables:List[Dict],
                              pool:Pool=None)->Tuple:
    """Generate a population of buy and sell trees using the configuration
    options and variables provided.
    Returns a tuple of trees"""
    depth = config["population"]["initial_tree_depth"]
    initial_population = config["population"]["initial_population"]
    terminals = config["terminals"]

    args = [[deepcopy(variables),terminals.copy(),idx,depth] 
            for idx in range(initial_population)]

    trees = pool.starmap(create_indicator_tree,args)\
                if pool\
                else [create_indicator_tree(*a) for a in args]

    return trees


def population_evaluation(pops:List[Dict],
                          data:pd.DataFrame,
                          decision_memo:Dict,
                          pool:Pool=None)->List:
    """Evaluate the buy and sell trees in popid pair order and return a set of
    decisions."""
    args = zip(pops,repeat(data,len(pops)),repeat(decision_memo,len(pops)))
    response = pool.starmap(make_pop_decisions,args)\
                if pool\
                else [make_pop_decisions(*a) for a in args]

    decisions = []
    for r in response:
        decision_memo.update(r[1])
        decisions.append(r[0])

    return decisions


def population_fitness(config:Dict,
                       pops:List[Dict],
                       decisions:List[List[str]],
                       data:pd.DataFrame,
                       pool:Pool=None,
                       final:bool=False)->Tuple:
    """Uses the previously evaluated decisions to generate scores for each
    buy/sell pair. These are stores as balance and trades. Then calculates the
    fitness values using the score information.
    Returns tuple of updated (buy_trees,sell_trees)"""
    starting_funds = config["evaluation"]["initial_funds"]
    trading_fee = config["evaluation"]["trading_fee_percent"]
    max_depth = config["population"]["max_tree_depth"]

    args = [[starting_funds,
            trading_fee,
            dset,
            data["close"].values] for dset in decisions]
    scores = pool.starmap(score_decisions,args)\
             if pool\
             else [score_decisions(*a) for a in args]
    
    for score,pop in zip(scores,pops):
        balance,gain_trades,lose_trades,invalid_trades = score
        pop["balance"] = balance
        pop["gtrades"] = gain_trades
        pop["ltrades"] = lose_trades
        pop["itrades"] = invalid_trades

    long_balance = natural_price_increase(config,data)

    args = [[pop,max_depth,starting_funds,long_balance] 
            for pop in pops]
    pops = pool.starmap(calculate_fitness,args)\
           if pool\
           else [calculate_fitness(*a) for a in args]

    return pops


def population_mutation(config:Dict,
                        pops:List[Dict],
                        variables:Dict,
                        pool:Pool=None)->Tuple:
    """Execute mutation on the population of buy and sell trees. Returns the
    mutated set of buy and sell trees including any non affected trees."""
    mutation_types = config["mutation"]["mutation_types"]
    mutation_rate = config["mutation"]["mutation_rate"]
    alphas = config["mutation"]["alphas"]

    # save the best members for next run
    held = []
    for _ in range(alphas):
        temp = max(pops,key=lambda k: k["fitness"])
        held.append(temp)
        pops.remove(temp)

    args = [
        [pop,
        variables.copy(),
        ["BUY","HOLD","SELL"],
        mutation_rate,
        mutation_types.copy()] for pop in pops]

    mutated = pool.starmap(point_mutate,args)\
                   if pool\
                   else [point_mutate(*a) for a in args]
    pops = held + mutated
    
    return pops


def population_selection(config:Dict,
                         pops:List[Dict])->Tuple:
    """Execute tournament selection on the population using the parameters
    set in the config. Returns the buy and sell trees tuple."""
    tourn_size = config["selection"]["tournament_size"]
    survivor_percent = config["selection"]["survivors_percent"]
    alphas = config["mutation"]["alphas"]

    # save the best members for next run
    held = []
    for _ in range(alphas):
        temp = max(pops,key=lambda k: k["fitness"])
        held.append(temp)
        pops.remove(temp)

    pops = held + tournament_selection(pops,tourn_size,survivor_percent)

    return pops


def population_reproduction(config:Dict,
                            pops:List[Dict])->Tuple:
    """Run crossover reproduction on the population with config options as is.
    and return the tuple of the buy/sell population."""
    crossover = config["crossover"]["crossover_rate"]
    max_population = config["population"]["max_population"]

    return repopulate(pops,max_population,crossover)