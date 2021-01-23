from typing import Dict,List,Any
import asyncio
import websockets
import json
from datetime import datetime,timezone
import logging
import os

tickers = [
    "btcusdt",
    "ethusdt",
    "xlmusdt",
    "linkusdt",
    "adausdt",
    "dotusdt",
    "filusdt",
    "wavesusdt",
    "omgusdt",
    "ltcusdt"
]

stream_subscribe = {
    "method": "SUBSCRIBE",
    "params": [t+"@ticker" for t in tickers],
    "id": 1
}
stream_unsubscribe = {
    "method": "UNSUBSCRIBE",
    "params": [t+"@ticker" for t in tickers],
    "id": 2
}

ticker_mapping = {
    "E":"event_time",
    "p":"price_change",
    "P":"price_change_pct",
    "w":"weighted_average_price",
    "c":"last_price",
    "Q":"last_quantity",
    "b":"best_bid",
    "B":"best_bid_qty",
    "a":"best_ask",
    "A":"best_ask_qty",
    "o":"open_price",
    "h":"high_price",
    "l":"low_price",
    "v":"total_traded_asset",
    "q":"total_traded_usdt",
    "n":"total_num_trades"
}

raw_stream_uri = "wss://stream.binance.com:9443/ws"
combined_stream_uri = "wss://stream.binance.com:9443/stream"


def safe_append_to_file(path:str,row:List,header:List[str]):
    if os.path.exists(path):
        with open(path,"a") as file:
            file.write(','.join([str(v) for v in row]) + '\n')
    else:
        with open(path,"w+") as file:
            file.write(','.join(header) + '\n')
            file.write(','.join([str(v) for v in row]) + '\n')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DATA COLLECTOR")


def get_timestamp()->int:
    return int(datetime.now(tz=timezone.utc).timestamp()*1000)    


def parse_ticker_data_row(data:Dict)->Dict:
    """Expand the returned ticker names and use them to get data we want."""
    return {v:data[k] for k,v in ticker_mapping.items()}


def parse_depth_data_row(data:Dict)->List:
    """Read in the dictionary of depth data and get a list of its contents"""
    output = []
    for bid in data["bids"]:
        output.extend(bid)
    for ask in data["asks"]:
        output.extend(ask)
    return output


def ticker_data_header()->List:
    return ["timestamp","triptime",*ticker_mapping.values()]


def depth_data_header()->List:
    bids = [val for pair in zip(
            [F'bid_p{v+1}' for v in range(20)],
            [F'bid_q{v+1}' for v in range(20)]) 
            for val in pair]
    asks = [val for pair in zip(
            [F'ask_p{v+1}' for v in range(20)],
            [F'ask_q{v+1}' for v in range(20)]) 
            for val in pair]

    return ["timestamp","triptime",*bids,*asks]


def parse_model_from_file(path:str,only_one:bool=True):
    model_file = ""
    with open(path,"r") as file:
        model_file = json.loads(file.read())
    
    variable_set = model_file["variable_set"]
    models = model_file["solutions"]

    if only_one:
        model = ComplexProgram(contained_subtrees=0,max_subtree_depth=0)
        model.generate_program_from_prefix(models[0]["equation"])
        return variable_set,model
    else:
        programs = []
        for pop in models:
            model = ComplexProgram(contained_subtrees=0,max_subtree_depth=0)
            model.generate_program_from_prefix(pop["equation"])
            programs.append(model)
        return variable_set,programs


async def keep_alive(ws):
    """Binance expects a pong sent every 3 minutes to keep alive."""
    logger.info("Starting keep alive")
    while True:
        await asyncio.sleep(120)
        logger.info("Sending Pong")
        await ws.pong()


async def collect_data_combined(ws):
    logger.info("Starting Data Collection...")
    while True:
        # Because ticker and depth happen at 1s intervals we can get away
        # with this
        start_t = get_timestamp()
        data = json.loads(await ws.recv())
        end_t = get_timestamp()

        logger.info(F'Collecting {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}')
        file_ts = get_timestamp() 

        ticker_data = None
        for ticker in tickers:
            if data["stream"] == ticker+"@ticker":
                logger.info(
                    F'Writing ticker {ticker.upper()}'\
                    F' for {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}')

                ticker_data = parse_ticker_data_row(data["data"])
                time_delta = end_t-start_t
                output = [file_ts,time_delta,*ticker_data.values()]
                safe_append_to_file(
                    path=F"./{ticker.upper()}_ticker.csv",
                    row=output,
                    header=ticker_data_header())
                break


async def stream_connection():
    while True:
        try:
            async with websockets.connect(combined_stream_uri,ping_interval=None) as ws:
                await ws.send(json.dumps(stream_subscribe))
                
                res = json.loads(await ws.recv())
                if res != {"result": None,"id": 1}:
                    logging.error(F"Invalid response on subscribe {res}")
                    raise RuntimeError

                await asyncio.gather(keep_alive(ws),collect_data_combined(ws))

                res = await ws.send(json.dumps(stream_unsubscribe))
        except Exception as e:
            logger.error(F"Encountered error: {e}")
            logger.info("Restarting socket connection...")
            continue

asyncio.run(stream_connection())