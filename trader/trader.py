from __future__ import annotations
from typing import List,Union,Dict
import pandas as pd
import numpy as np
import uuid

from Common import random_choice
from DataStructures.Node import Node
from DataStructures.Terminal import Terminal
from CreateTree import (
    create_tree,
    create_buy_tree,
    create_sell_tree,
    create_initial_variables,
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

from TreeIO import serialize_tree,deserialize_tree
from TreeEvaluation import evaluate_tree
from TreeMutation import point_mutate
from TreeCrossover import crossover_reproduction


def prepare_data(data_path:str):
    df = pd.read_csv(data_path)
    df.drop("Unnamed: 0",inplace=True,axis=1)
    return df


def main():
    df = prepare_data(data_path="../test_indicator_data.csv")
    # tree = create_tree(terminals=["BUY","HOLD"],data=df,depth=2,unique_nodes=True)
    # tree1 = create_sell_tree(data=df)
    # pprint_tree(tree1)

    tree1 = create_buy_tree(data=df,depth=3)
    print("TreeA")
    pprint_tree(tree1)
    
    tree2 = create_buy_tree(data=df,depth=3)
    print("TreeB")
    pprint_tree(tree2)

    child1,child2 = crossover_reproduction(treeA=tree1,treeB=tree2,probability=1)

    print("child1")
    pprint_tree(child1)
    print("child2")
    pprint_tree(child2)
    


    # vars_set = create_initial_variables(data=df)
    # vars_set.update({"bought_price":20000})

    # for i in range(100):
    #     point_mutate(tree,variables=vars_set,terminals=["SELL","HOLD"])
    # pprint_tree(tree)

    # serial = serialize_tree(tree)
    # pprint_tree(deserialize_tree(serial))
    # step_thresholds(tree,1,0.01)
    # print(stringify_tree(tree))
    # print(Node(var_name="percent_b",df=df))

if __name__ == "__main__":
    main()