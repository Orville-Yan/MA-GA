from datetime import datetime
import logging
import pandas as pd
import numpy as np

def init_logger() -> logging.Logger:
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    fh = logging.FileHandler('ReshapeData.log')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def generate_trading_time() -> list:
    morning_start = datetime.strptime("0930", "%H%M")
    morning_end = datetime.strptime("1130", "%H%M")
    afternoon_start = datetime.strptime("1300", "%H%M")
    afternoon_end = datetime.strptime("1500", "%H%M")

    morning_times = pd.date_range(morning_start, morning_end, freq='T').strftime('%H%M').tolist()
    afternoon_times = pd.date_range(afternoon_start, afternoon_end, freq='T').strftime('%H%M').tolist()

    all_trading_times = morning_times + afternoon_times
    return all_trading_times

def read_stock_exchange(path: str="./docs/StocksExchange.xlsx") -> pd.DataFrame:
    """
    prepare stocks' information, including exchange and on-list date
    """
    all_stocks = pd.read_excel(path)
    all_stocks.columns = ["code", "name", "exchange", "status", "on_date"]
    all_stocks["exchange"] = all_stocks["code"].str.split(".", expand=True)[1].str.lower()
    all_stocks["code"] = all_stocks["code"].str.split(".", expand=True)[0]
    all_stocks["status"] = all_stocks["status"].replace("--", np.nan)

    # on_date: 90 days after the stock's on-list date
    all_stocks["on_date"] = (pd.to_datetime(all_stocks["on_date"].replace("--", np.nan)) + pd.Timedelta(days=90)).dt.strftime("%Y%m%d").astype(float)
    all_stocks = all_stocks.set_index("code")
    return all_stocks