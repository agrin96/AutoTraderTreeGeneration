import numpy as np

def random_choice(prob_true:float=0.5)->bool:
    return np.random.choice([True,False],p=[prob_true,1-prob_true])