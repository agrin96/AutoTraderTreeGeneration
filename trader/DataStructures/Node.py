from __future__ import annotations
from typing import List,Union,Dict
import pandas as pd
import numpy as np
import uuid

from Common import random_choice
from .BaseNode import BaseNode
from .Terminal import Terminal

class Node(BaseNode):
    def __init__(self,var_name:str,initial_threshold:float,is_fixed:bool=False):
        super().__init__()
        self._variable = var_name
        self._threshold = initial_threshold
        self._children = []
        self._isfixed = is_fixed

    @staticmethod
    def node_from_dict(data:Dict,fixed:bool=False)->Node:
        if data["type"] == "NODE":
            return Node(
                var_name=data["variable"],
                initial_threshold=data["threshold"],
                is_fixed=data["fixed"])


    def evaluate(self,data:Dict)->Node:
        value = data[self._variable]
        if value > self._threshold:
            return self._children[0]
        else:
            return self._children[-1]

    
    def node_as_dict(self)->Dict:
        return {
            "type": "NODE",
            "variable": self._variable,
            "threshold": self._threshold,
            "fixed":self._isfixed}


    def add_child(self,node:Union[Node,Terminal],index:int=-1):
        if index == -1:
            self._children.append(node)
        else:
            self._children.insert(index,node)
        node.set_parent(self)


    def remove_child(self,node:Union[Node,Terminal])->int:
        removed = 0
        for idx,child in enumerate(self._children):
            if child.node_id() == node.node_id():
                child.set_parent(None)
                removed = idx
                break

        self._children = [ch for ch in self._children 
                          if ch.node_id() != node.node_id()]
        return removed


    def step_threshold(self,base_step:float,generation:int):
        step = self._threshold * (base_step / generation)

        if random_choice(prob_true=0.5):
            self._threshold = self._threshold + step
        else:
            self._threshold = self._threshold - step


    def set_threshold(self,threshold:float):
        self._threshold = threshold


    def get_threshold(self)->float:
        return self._threshold


    def children(self)->List:
        return self._children


    def __repr__(self):
        return F"{self._variable} > {self._threshold}"


    def __str__(self):
        return self.__repr__()