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


def store_serialized_pop(serial_buy:str,
                         serial_sell:str,
                         output_folder:str="SerializedTrees/"):
    """Write the serialized population members to the root directory to store
    them. Does not retain information in the pop dictionary just the trees."""
    current_dir = os.getcwd()
    for idx,folder in enumerate(current_dir.split("/")):
        if folder == "AutoTrader":
            current_dir = "/".join(current_dir.split("/")[:idx+1])
            break
    
    output_path = os.path.join(current_dir,output_folder)
    popid = 0
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        # Get the current largest file id and iterate by one
        contents = os.listdir(output_path)
        files = [c for c in contents if "popfile" in c]
        files = [c.split('.')[0] for c in contents]
        fileids = [int(c.split("-")[-1]) for c in files]
        popid = max(fileids)+1
    
    filename = F"buy-popfile-{popid}.json"
    with open(os.path.join(output_path,filename),"w+") as out:
        out.write(serial_buy)

    filename = F"sell-popfile-{popid}.json"
    with open(os.path.join(output_path,filename),"w+") as out:
        out.write(serial_sell)
    