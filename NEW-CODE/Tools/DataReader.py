# import mmap
import pandas as pd
import numpy as np
import torch
import os
import time


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


class DataReader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.cols = ["open", "high", "low", "close", "volume"] # "amount"
    
    def dataframe_to_tensor(self, df: pd.DataFrame, minute_len=242):
        """
        Convert the DataFrame (one year's data) to a tensor
        """
        df = df.astype(np.float32)

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
        tensor_data = torch.tensor(np.array(day_data, dtype=np.float32), dtype=torch.float32).permute(2, 0, 1)
        return tensor_data

    def read_data_by_col(self, col: str, year_lst: list[int]) -> torch.Tensor:
        """
        load data by column and year list with mmap
        """
        data = []
        for year in year_lst:
            file_path = f"{self.data_path}/{col}_{year}.parquet"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            # load data with mmap
            data.append(self.dataframe_to_tensor(pd.read_parquet(file_path)))
        # concatenate date in the day dimension
        return torch.cat(data, dim=1)
    
    def read_data(self, year_lst: list[int]) -> list[torch.Tensor]:
        """
        Read all columns data by year list
        """
        return [self.read_data_by_col(col, year_lst) for col in self.cols]

def get_barra(year_list):
    barra=torch.load('barra.pt')
    dict=torch.load('dict.pt')
    s=[]
    for year in year_list:
        mask=pd.to_datetime(dict['index']).year==year
        s.append(mask)

    w=s[0]
    for t in s:
        w=w|t

    return barra[w]

if __name__ == "__main__":
    bgn_time = time.time()
    data_path = "./processed_data"
    data_reader = DataReader(data_path)
    open_data = data_reader.read_data_by_col("open", [2016, 2017])
    print("Open [2016, 2017] shape: ", open_data.shape)
    print("Time cost: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - bgn_time)))

    bgn_time = time.time()
    open_data, high_data, low_data, close_data, volume_data = data_reader.read_data([2016, 2017])
    print("Open, High, Low, Close, Volume [2016, 2017] shape: ", open_data.shape, high_data.shape, low_data.shape, close_data.shape, volume_data.shape)
    print("Time cost: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - bgn_time)))
