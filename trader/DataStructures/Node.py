from __future__ import annotations
from typing import List,Union,Dict
import numpy as np
import sys
from copy import deepcopy

sys.path.append("../Indicators")

from Indicators.IndicatorVariables import indicator_variables
from .BaseNode import BaseNode
from .Terminal import Terminal

class Node(BaseNode):
    def __init__(self,variable:Dict):
        super().__init__()
        self._variable = variable
        self._children = []


    @staticmethod
    def node_from_dict(data:Dict)->Node:
        if data["type"] == "NODE":
            variable = [I for I in indicator_variables 
                        if I['name'] == data["variable"]["name"]][0]
            variable["variables"] = data["variable"]["variables"]

            return Node(variable=deepcopy(variable))


    def evaluate(self,data:Dict)->Node:
        "Decides based on whether it is a buy, sell or hold."
        value = data[self._variable]

        if value == "BUY":
            return self._children[0]
        elif value == "HOLD":
            return self.children[1]
        else:
            return self._children[2]

    
    def get_decisions(self)->np.array:
        """Return an array mask of which decisions to make."""
        return np.array(self._variable["decisions"])

    
    def node_as_dict(self)->Dict:
        written_variable = {
            "name":self._variable["name"],
            "variables":self._variable["variables"]}

        return deepcopy({"type": "NODE","variable": written_variable})


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


    def children(self)->List:
        return self._children


    def __repr__(self):
        variables = self._variable["variables"]
        # values = (str(variables[k]["value"]) for k in variables.keys())
        values = (F'{k}={variables[k]["value"]}' for k in variables.keys())
        values = " ".join(values)
        return F"{self._variable['name']} of {values}"


    def __str__(self):
        return self.__repr__()