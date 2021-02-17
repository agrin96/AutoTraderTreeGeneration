import pandas as pd
from typing import Dict

from .GenerateDecisions import (
    generate_bop_decisions,
    generate_chaikin_decisions,
    generate_macd_decisions,
    generate_mfi_decisions,
    generate_rsi_decisions)    


def create_memo_hash(indicator:Dict)->str:
    """Create an integar hash for the variable using the Tuple of variable
    values as the object."""
    values = (indicator["variables"][k]["value"] 
            for k in indicator["variables"].keys())
    return hash(values)


indicator_variables = [
    {
        "name": "bop",
        "memo":None,
        "decisions": [],
        "generator":generate_bop_decisions,
        "variables": {
            "buy_threshold": {
                "value":-0.1,
                "range":{
                    "upper":0,
                    "lower":-1,
                }
            },
            "sell_threshold": {
                "value":0.3,
                "range":{
                    "upper":1,
                    "lower":0,
                }
            },
            "period": {
                "value":12,
                "range":{
                    "upper":100,
                    "lower":2,
                }
            }
        }
    },
    {
        "name": "rsi",
        "memo":None,
        "decisions": [],
        "generator":generate_rsi_decisions,
        "variables": {
            "buy_threshold": {
                "value":30,
                "range":{
                    "upper":100,
                    "lower":50,
                }
            },
            "sell_threshold": {
                "value":70,
                "range":{
                    "upper":50,
                    "lower":0,
                }
            },
            "period": {
                "value":12,
                "range":{
                    "upper":100,
                    "lower":2,
                }
            }
        }
    },
    {
        "name":"mfi",
        "memo":None,
        "decisions": [],
        "generator":generate_mfi_decisions,
        "variables": {
            "buy_threshold": {
                "value":30,
                "range":{
                    "upper":100,
                    "lower":50,
                }
            },
            "sell_threshold": {
                "value":70,
                "range":{
                    "upper":50,
                    "lower":0,
                }
            },
            "period": {
                "value":12,
                "range":{
                    "upper":100,
                    "lower":2,
                }
            }
        }
    },
    {
        "name":"chaikin",
        "memo":None,
        "decisions": [],
        "generator":generate_chaikin_decisions,
        "variables": {
            "fast_period": {
                "value":3,
                "range":{
                    "upper":100,
                    "lower":2,
                }
            },
            "slow_period": {
                "value":18,
                "range":{
                    "upper":100,
                    "lower":2,
                }
            },
            "signal_period": {
                "value":9,
                "range":{
                    "upper":100,
                    "lower":2,
                }
            }
        }
    },
    {
        "name":"macd",
        "memo":None,
        "decisions": [],
        "generator":generate_macd_decisions,
        "variables": {
            "fast_period": {
                "value":3,
                "range":{
                    "upper":100,
                    "lower":2,
                }
            },
            "slow_period": {
                "value":18,
                "range":{
                    "upper":100,
                    "lower":2,
                }
            },
            "signal_period": {
                "value":9,
                "range":{
                    "upper":100,
                    "lower":2,
                }
            }
        }
    }
]