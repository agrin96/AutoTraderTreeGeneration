from typing import Dict,List,Tuple
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from TreeActions import count_nodes,tree_depth
from TreeEvaluation import make_pop_decisions


def pprint_generation_statistics(pops:List[Dict],
                                 rolling_balances:List[float]):
    """Prints key statistics of the current generation. Shows the best buy
    and sell tree as well as the mean fitness and mean balance of the 
    generation population. This step occurs before mutation and selection."""
    print("\n\tCurrent Best Tree")
    best = max(pops,key=lambda k: k["fitness"])
    
    output = F"POPID: {best['popid']} Fitness: {best['fitness']}"
    output += F" Balance: {best['balance']} Gain Trades: {best['gtrades']}"
    output += F" Lose Trades: {best['ltrades']}"
    output += F" Invalid Trades: {best['itrades']}"
    print("\t"+str(output))
    
    print("\n\tPopulation Statistics")
    mean_fitness = np.mean(list(map(lambda k: k["fitness"],pops)))
    mean_balance = np.mean(list(map(lambda k: k["balance"],pops)))
    print(F"\t\tMean Fitness: {mean_fitness}")
    print(F"\t\tMean Balance: {mean_balance}")
    lowestf = min(pops,key=lambda k: k["fitness"])["fitness"]
    lowestb = min(pops,key=lambda k: k["balance"])["balance"]
    print(F"\t\tLowest Fitness: {lowestf}")
    print(F"\t\tLowest Balance: {lowestb}")
    f_var = np.var(list(map(lambda k: k["fitness"],pops)))
    b_var = np.var(list(map(lambda k: k["balance"],pops)))
    print(F"\t\tFitness Variance: {f_var}")
    print(F"\t\tBalance Variance: {b_var}")

    # print("\n\tCluster Information")
    # num_clusters = max(pops,key=lambda k: k["cluster"])["cluster"]
    # for i in range(num_clusters):
    #     size = np.sum([1 for p in pops if p["cluster"] == i])
    #     print(F"\t\tCluster [{i}] size: {size}")

    print("\n\tAverage Tree Depth")
    depth = np.mean(list(map(lambda t: tree_depth(t["tree"]),pops)))
    print(F"\t\tMean Depth: {depth}")

    print("\n\tAverage Node Counts")
    count = np.mean(list(map(lambda t: count_nodes(t["tree"]),pops)))
    print(F"\t\tMean Node Count: {count}")

    print("\n\tRolling Mean of Best Member Balance")
    rolling_balances.append(best["balance"])
    rolling_mean = np.mean(rolling_balances)
    print(F"\t\tCurrent rolling mean balance: {rolling_mean}\n")


def plot_decisions(pop:Dict,
                   candles:pd.DataFrame,
                   candle_period:int=30,
                   display_range:Tuple=(250,-1),
                   axs=None):
    """Allows us to plot a decision list against price data to view them more
    clearly.
    Parameters:
        decisions (List[str]): A list of BUY/SELL/HOLD.
        candles (pd.DataFrame): Dataframe of candles with open close price.
        display_range (Tuple): Which candles to plot for.
        axs: A matplotlib axes object which can be plotted on. If not passed
            then it will create a new object."""

    decisions,_ = make_pop_decisions(pop,candles,{})
    view = candles[display_range[0]:display_range[1]]
    decisions = decisions[display_range[0]:display_range[1]]

    show = False
    if not axs:
        show = True
        fig,axs = plt.subplots(1)
    
    plot_candles(candles,candle_period,display_range,axs)
    axs2 = axs.twinx()
    
    buy_mask = np.where(np.array(decisions)=="BUY",True,False)
    sell_mask = np.where(np.array(decisions)=="SELL",True,False)

    buy_indexes = view.loc[buy_mask,"index"]
    sell_indexes = view.loc[sell_mask,"index"]

    buy_closes = view.loc[buy_mask,"close"]
    sell_closes = view.loc[sell_mask,"close"]

    axs2.scatter(sell_indexes,sell_closes,color="orange",marker="v")
    axs2.scatter(buy_indexes,buy_closes,color="black",marker="^")

    buy_patch = mpatches.Patch(color='black', label='BUY')
    sell_patch = mpatches.Patch(color='orange', label='SELL')
    axs2.legend(handles=[buy_patch,sell_patch])

    if show:
        plt.show()


def plot_candles(df:pd.DataFrame,
                 candle_period:int=30,
                 display_range:Tuple=(250,-1),
                 axs=None):
    """Generate the candle plots for the candle dataframe. It is expected
    to contain high low close and open columns. Axs are provided the plot will 
    not be immediately shown but instead will simply add its plots.
    Parameters:
        display_range (Tuple): Decides from which point in the data we will be
            displaying since most indicators have a ramp up we can safely ignore
            the first 250 periods.
        axs (Object): The marplotlib axes object."""
    view = df[display_range[0]:display_range[1]]
    show = False
    if not axs:
        show = True
        fig,axs = plt.subplots(1)

    axs.bar(view["index"],
            np.subtract(view["close"],view["open"]),
            bottom=view["open"],
            width=candle_period,
            color=np.where(
                np.subtract(view["close"],view["open"])>0,"green","red"))

    axs.bar(view["index"],
            np.subtract(view["high"],view["low"]),
            bottom=view["low"],
            width=candle_period/10,
            color=np.where(
                np.subtract(view["close"],view["open"])>0,"green","red"))

    if show:
        plt.show()
