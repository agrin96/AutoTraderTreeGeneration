from __future__ import annotations
from typing import List,Union,Dict,Tuple
import json
import numpy as np
from multiprocessing import Pool

from Common import random_choice
from DataStructures.Node import Node
from DataStructures.Terminal import Terminal
from TreeActions import (
    count_nodes,
    count_terminals,
    tree_depth,
    list_tree_variables,
    pprint_tree)

def tournament_selection(population:List)->List:
    completed_clusters = []
    for pop in population:
        if pop["buy_cluster"] in completed_clusters:
            continue