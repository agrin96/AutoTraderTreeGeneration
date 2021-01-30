import numpy as np
from typing import Dict,Any,List
import json
import os


def random_choice(prob_true:float=0.5)->bool:
    return np.random.choice([True,False],p=[prob_true,1-prob_true])


def parse_configuration(path:str)->Dict:
    """Read in a json configuration file and return it as a dictionary."""
    if not os.path.exists(path):
        raise RuntimeError(
        "The configuration path you specified doesn't exist.")  
    
    with open(path,"r") as file:
        return json.loads(file.read())


def check_configuration(config:Dict):
    """Check the configuration passed in and apply default values to missing
    parameters."""
    required = [
        "process_pool_size",
        "generations",
        "initial_population",
        "max_population",
        "mutation_rate",
        "unique_tree_variables",
        "mutation_types",
        "initial_buy_node_depth", 
        "max_buy_node_depth", 
        "initial_sell_node_depth", 
        "max_sell_node_depth", 
        "crossover_rate",
        "initial_funds", 
        "trading_fee_percent", 
        "threshold_step_percent", 
        "train_percent_split", 
        "split_type",
        "data_file_path",
        "variables"]
    for req in required:
        if req not in config:
            raise RuntimeError(
            F"Required parameter `{req}` missing from config.") 