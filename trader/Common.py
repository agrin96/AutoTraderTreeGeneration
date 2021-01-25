import numpy as np
from typing import Dict
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
