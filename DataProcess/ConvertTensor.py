import pandas as pd
import numpy as np
from tqdm import tqdm
import os

import torch

DATA_PATH = "./processed_data"
COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
YERA_ID = range(2016, 2025)

ALL_STOCKS = list(map(lambda x: x.split(".")[0].split("_")[1], os.listdir("./data")))

RESULT_PATH = "./results"
os.makedirs(RESULT_PATH, exist_ok=True)
for col in COLUMNS:
    os.makedirs(os.path.join(RESULT_PATH, col), exist_ok=True)

def generate_trading_time(date):
    """
    The minutes trading time of a day
    """
    date = pd.to_datetime(date)
    morning_times = pd.date_range(date + pd.Timedelta("09:30:00"), date + pd.Timedelta("11:30:00"), freq="T")
    afternoon_times = pd.date_range(date + pd.Timedelta("13:00:00"), date + pd.Timedelta("15:00:00"), freq="T")
    return morning_times.union(afternoon_times).sort_values()

def fill_full_day(day_data: pd.DataFrame):
    """
    Fill the missing minutes of a day with NaN
    """
    date = day_data.index[0].date()
    trading_time = generate_trading_time(date)
    day_data = day_data.reindex(trading_time)
    return day_data

def dataframe_to_tensor(df: pd.DataFrame, minute_len=242):
    """
    Convert the DataFrame (one year's data) to a tensor
    """
    # DateTimeIndex
    df.index = pd.to_datetime(df.index)
    # Sort by date and stock code
    df = df.sort_index(axis=0).sort_index(axis=1)

    # Group by date
    grouped = df.groupby(df.index.date)
    day_data = []
    for _, group in grouped:
        # Fill the missing minutes of a day with NaN
        if len(group) != minute_len:
            day_data = fill_full_day(group)
        
        day_data.append(group.values)
    
    # exchange dimensions to (num_stock, day_len, minute_len), from (day_len, minute_len, num_stock)
    tensor_data = torch.tensor(np.array(day_data)).permute(2, 0, 1)  
    return tensor_data

def read_data(col: str, year: int) -> pd.DataFrame:
    """
    Read all batch data of a year by col
    """
    files = os.listdir(f"{DATA_PATH}/{col}")
    files = [f for f in files if f"{year}" in f]
    data = [pd.read_parquet(f"{DATA_PATH}/{col}/{f}") for f in files]
    data = pd.concat(data, axis=1)
    return data

if __name__ == "__main__":
    for col, year in tqdm([(col, year) for col in COLUMNS for year in YERA_ID]):
        data = read_data(col, year)
        
        # if the number of stocks is not equal to all stocks, fill the missing stocks with NaN
        if data.shape[1] != len(ALL_STOCKS):
            cur_miss_stocks = list(set(ALL_STOCKS) - set(data.columns))
            data = pd.concat([data, pd.DataFrame(columns=cur_miss_stocks)], axis=1)
        
        # convert to tensor
        data = data.astype(np.float64)
        # data.to_parquet(f"{DATA_PATH}/{col}_{year}.parquet") # save the data to parquet
        data = dataframe_to_tensor(data)
        assert data.shape[0] == len(ALL_STOCKS) and data.shape[2] == 242, f"{col}_{year} has missing stocks"
        torch.save(data, f"{RESULT_PATH}/{col}/{col}_{year}.pt")