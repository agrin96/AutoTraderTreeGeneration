import numpy as np
from typing import Dict,Any,List
import json
import os

def random_choice(prob_true:float=0.5)->bool:
    return np.random.choice([True,False],p=[prob_true,1-prob_true])


def arange_with_endpoint(data:np.array,step:int)->np.array:
    """Convenience function to get around the fact that the np.arange function
    doesnt consider the endpoint properly. They reccomend using linspace to
    solve this, but we need arange, so we manually consider the endpoint.
    Returns the arange results."""
    aranged = np.arange(0,data.shape[0],step=step)
    
    # Adjust for how np.arange fails to consider the endpoint
    if aranged[-1] < data.shape[0] and (aranged[-1]+step) < data.shape[0]:
        aranged =np.append(aranged,aranged[-1]+period)
    return aranged


def parse_configuration(path:str)->Dict:
    """Read in a json configuration file and return it as a dictionary."""
    if not os.path.exists(path):
        raise RuntimeError(
        "The configuration path you specified doesn't exist.")  
    
    with open(path,"r") as file:
        return json.loads(file.read())


def store_serialized_pop(serial_pop:str,
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
        popid = max(fileids)+1 if len(fileids)>0 else 0
    
    filename = F"popfile-{popid}.json"
    with open(os.path.join(output_path,filename),"w+") as out:
        out.write(serial_pop)
