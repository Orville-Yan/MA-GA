import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from OP import *
from scipy.io import loadmat
import json
import re
from GA.Config import Data_Config as Config
class Interface():
    @staticmethod
    def df2tensor(df):
        df = df.astype(np.float32)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(axis=0).sort_index(axis=1)
        grouped = df.groupby(df.index.date)
        day_data = []
        for _, group in grouped:
            if len(group) != 242: 
                group = Interface.fill_full_day(group)
            day_data.append(group.values)
        tensor = torch.tensor(np.array(day_data, dtype=np.float32), dtype=torch.float32)
        return tensor.permute(0,2,1)
    
    @staticmethod
    def generate_trading_time(date:str)->pd.DatetimeIndex:
        date = pd.to_datetime(date)
        morning_times = pd.date_range(date + pd.Timedelta("09:30:00"), date + pd.Timedelta("11:30:00"), freq="min")
        afternoon_times = pd.date_range(date + pd.Timedelta("13:00:00"), date + pd.Timedelta("15:00:00"), freq="min")
        return morning_times.union(afternoon_times).sort_values()
    
    @staticmethod
    def fill_full_day(day_data:pd.DataFrame)->pd.DataFrame:
        date = day_data.index[0].date()
        trading_time = Interface.generate_trading_time(date)
        day_data = day_data.reindex(trading_time)
        return day_data

    @staticmethod
    def get_all_files(folder_path, pattern:re.Pattern):
        entries = os.listdir(folder_path)
        files = [file for file in entries if os.path.isfile(os.path.join(folder_path, file))]
        filtered_files = [file for file in files if pattern.search(file)]
        return sorted(filtered_files)
    
    @staticmethod
    def get_pct_change(close:torch.tensor)-> torch.tensor:
        p = close / close.shift(1)
        return p

    @staticmethod
    def get_labels(d_o,d_c, freq=5):
        open = OP_AF2A.D_ts_delay(d_o, -1)
        close = OP_AF2A.D_ts_delay(d_c, -freq)
        return close / open - 1



class BasicReader():
    def __init__(self):
        self.MutualStockCodes = pd.read_parquet(Config.MUTUAL_STOCK_CODES_PATH)["Mutual"].values
        self.TradingDate = self._TradingDate()
        self.StockCodes = self._StockCodes()

    def _TradingDate(self):
        trading_date = loadmat(os.path.join(Config.PARQUET_Daily_PATH, 'TradingDate_Daily.mat'))['TradingDate_Daily']
        return pd.Series([datetime.strptime(str(d[0]), '%Y%m%d') for d in trading_date])

    def _StockCodes(self):
        code = pd.read_parquet(Config.MUTUAL_STOCK_CODES_PATH)
        return code["StockCodes"].loc[code["Mutual"]].values
    
    def _clean_data(self) -> torch.tensor:
        ListedDate = loadmat(os.path.join(Config.PARQUET_Daily_PATH, 'AllStock_DailyListedDate.mat'))['AllStock_DailyListedDate'][:, self.MutualStockCodes]
        Status = loadmat(os.path.join(Config.PARQUET_Daily_PATH, 'AllStock_DailyStatus.mat'))['AllStock_DailyStatus_use'][:, self.MutualStockCodes]
        ST = loadmat(os.path.join(Config.PARQUET_Daily_PATH, 'AllStock_DailyST.mat'))['AllStock_DailyST'][:, self.MutualStockCodes]
        status20 = OP_AF2A.D_ts_mean(torch.from_numpy((1 - ST) * (Status)), 20) > 0.5
        listed = torch.from_numpy(ListedDate >= 60)
        clean = ~(listed * status20)
        return clean

    def get_daylist(self,year_lst:list[int]) -> list[pd.Timestamp]:
        trading_dates = self._TradingDate()
        return list(trading_dates[trading_dates.dt.year.isin(year_lst)])
    
    def get_index(self, year_lst:list[int])->pd.DatetimeIndex:
        trading_dates = self._TradingDate()
        return (trading_dates.index[trading_dates.dt.year.isin(year_lst)])

    def tensor2df(self, tensor, year_list:list[int])->pd.DataFrame:
        day_list = self.get_daylist(year_list)
        stock_codes = self._StockCodes()
        idx = self.get_index(year_list)
        tensor = tensor[idx]
        df = pd.DataFrame(np.asarray(tensor), index=day_list, columns=stock_codes)
        return df   

    
    def adjust(self,data,year_lst,device='cpu')-> torch.Tensor:   
        index = self.get_index(year_lst)
        clean = self._clean_data()[index]
        tensor = torch.tensor(data, dtype=torch.float32, device=device)[index]
        tensor = torch.where(clean | (tensor < 1e-5), float('nan'), tensor)
        return tensor


    
class ParquetReader(BasicReader):
    def __init__(self,DailyDataPath=Config.PARQUET_Daily_PATH, MinuteDataPath=Config.PARQUET_Minute_PATH, BarraPath=Config.PARQUET_BARRA_PATH, DictPath=Config.PARQUET_DICT_PATH, device='cpu'):
        super().__init__()
        self.parquet_DailyDataPath = DailyDataPath
        self.parquet_MinuteDataPath = MinuteDataPath
        self.parquet_BarraPath = BarraPath
        self.parquet_DictPath = DictPath
        self.device = device
        self.M_name = ['open', 'high', 'low', 'close', 'volume']


    def GetOHLC(self):
        open = loadmat(os.path.join(self.parquet_DailyDataPath, 'AllStock_DailyOpen_dividend.mat'))['AllStock_DailyOpen_dividend'][:, self.MutualStockCodes]
        high = loadmat(os.path.join(self.parquet_DailyDataPath, 'AllStock_DailyHigh_dividend.mat'))['AllStock_DailyHigh_dividend'][:, self.MutualStockCodes]
        low = loadmat(os.path.join(self.parquet_DailyDataPath, 'AllStock_DailyLow_dividend.mat'))['AllStock_DailyLow_dividend'][:, self.MutualStockCodes]
        close = loadmat(os.path.join(self.parquet_DailyDataPath, 'AllStock_DailyClose_dividend.mat'))['AllStock_DailyClose_dividend'][:, self.MutualStockCodes]
        return [open, high, low, close]
    
    def GetVolume(self):
        return loadmat(os.path.join(self.parquet_DailyDataPath, 'AllStock_DailyVolume.mat'))['AllStock_DailyVolume'][:, self.MutualStockCodes]
    
    def get_Day_data(self,year_lst: list[int])-> list[torch.Tensor]:
        D_O, D_H, D_L, D_C = self.GetOHLC()
        D_V = self.GetVolume()
        return [self.adjust(D_O,year_lst), self.adjust(D_H, year_lst), self.adjust(D_L, year_lst), self.adjust(D_C, year_lst), self.adjust(D_V, year_lst)]

    def read_data_by_col(self, col: str, year_lst: list[int]) -> torch.Tensor:
        data = []
        for year in year_lst:
            file_path = f"{self.parquet_MinuteDataPath}/{col}_{year}.parquet"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            tensor = Interface.df2tensor(pd.read_parquet(file_path, columns=self.StockCodes))
            data.append(tensor)
        return torch.cat(data, dim=0)
    
    def get_Minute_data(self, year_lst: list[int]) -> list[torch.Tensor]:
        return [self.read_data_by_col(col, year_lst) for col in self.M_name]
    
    
    def get_Barra(self, year_lst):
        barra = torch.load(os.path.join(Config.DATA_PATH,'barra.pt'), weights_only=True)
        dict = torch.load(os.path.join(Config.DATA_PATH,'dict.pt'), weights_only=False)
        w = pd.to_datetime(dict['index']).year.isin(year_lst)
        return barra[w][:, self.MutualStockCodes]
    
class MmapReader(BasicReader):
    def __init__(self,  download = False, DailyDataPath: str = Config.MMAP_Daily_PATH, MinuteDataPath: str = Config.MMAP_Minute_PATH, BarraPath: str = Config.MMAP_BARRA_PATH, DictPath: str = Config.PARQUET_DICT_PATH, device: str = 'cpu'):
        if download:
            self.parquetreader = ParquetReader()
        super().__init__()
        self.mmap_DailyDataPath = DailyDataPath
        self.mmap_MinuteDataPath = MinuteDataPath
        self.mmap_BarraPath = BarraPath
        self.mmap_DictPath = DictPath
        self.device = device
        self.daily_data_info = {}
        self.D_name = ['D_O', 'D_H', 'D_L', 'D_C', 'D_V']
        self.M_name = ['M_O', 'M_H', 'M_L', 'M_C', 'M_V']
        with open(os.path.join(Config.DATA_PATH, "Mmap/data_shape.json"), "r", encoding="utf-8") as file:
            self.data_shape = json.load(file)

    def get_Day_data(self, year_lst: list[int]):
        length = 0
        width = len(self.StockCodes)

        for year in year_lst:
            length += self.data_shape[str(year)][0]

        start = 0

        D_O, D_H, D_L, D_C, D_V = [torch.full((length, width), float('nan')) for _ in range(5)]
        for i, year in enumerate(year_lst):
            for j, data in enumerate([D_O, D_H, D_L, D_C, D_V]):
                file_path = os.path.join(self.mmap_DailyDataPath, f'{self.D_name[j]}_{year}.mmap')
                shape = self.data_shape[str(year)]
                mmap_read = np.memmap(file_path, dtype=np.float32, mode='r', shape=tuple(shape))
                tensor_read = torch.from_numpy(mmap_read.copy())
                data[start:start + tensor_read.shape[0]] = tensor_read

            start += tensor_read.shape[0]

        return D_O, D_H, D_L, D_C, D_V

    def get_Minute_data(self, year_lst: list[int]):
        day_len = 0
        minute_num = 242
        stock_num = len(self.StockCodes)

        for year in year_lst:
            day_len += self.data_shape[str(year)][0]

        year_pattern = re.compile(r'_({})_'.format('|'.join(map(str, year_lst))))

        pos = 0
        M_O, M_H, M_L, M_C, M_V = [torch.full((day_len, stock_num, minute_num), float('nan')) for _ in range(5)]
        for j, data in enumerate([M_O, M_H, M_L, M_C, M_V]):
            folder_path = os.path.join(self.mmap_MinuteDataPath, self.M_name[j])
            files = Interface.get_all_files(folder_path, year_pattern)

            for file in files:
                file_path = os.path.join(folder_path, file)
                mmap_read = np.memmap(file_path, dtype=np.float32, mode='r', shape=(stock_num,minute_num))
                tensor_read = torch.from_numpy(mmap_read.copy())
                data[pos] = tensor_read
                pos += 1

            pos = 0

        return M_O, M_H, M_L, M_C, M_V

    def get_Minute_data_daily(self, day:str):
        day = pd.to_datetime(day)
        data = []
        minute_num = 242
        stock_num = len(self.StockCodes)
        for name in self.M_name:
            file_path = os.path.join(self.mmap_MinuteDataPath, f"{name}", f'{name}_{day.strftime("%Y_%m_%d")}.mmap')
            mmap_read = np.memmap(file_path, dtype=np.float32, mode='r', shape=(stock_num,minute_num))
            data.append(torch.from_numpy(mmap_read.copy()).unsqueeze(0))

        return data

    def get_Barra(self, year_lst):
        num_stock = len(self.StockCodes)
        day_list = self.get_daylist(year_lst)
        barra = torch.full((len(day_list), num_stock, 41), float('nan'))
        for i, day in enumerate(day_list):
            barra[i] = self.get_Barra_daily(day)
        return barra

    def get_Barra_daily(self, day:str):
        day = pd.to_datetime(day)
        num_stocks = len(self.StockCodes)
        file_path = os.path.join(self.mmap_BarraPath, f'barra_{day.strftime("%Y_%m_%d")}.mmap')
        mmap_read = np.memmap(file_path, dtype=np.float32, mode='r', shape=(num_stocks, 41))
        return torch.from_numpy(mmap_read.copy())

    
    def save_daily_data(self):
        for year in range(2016, 2024):
            data = self.parquetreader.get_Day_data([year])
            for i, name in enumerate(self.D_name):
                col_data = data[i]
                self.daily_data_info[year] = col_data.shape
                mmap = np.memmap(os.path.join(self.mmap_DailyDataPath, f'{name}_{year}.mmap'), dtype=np.float32, mode='w+',
                                    shape=col_data.shape)
                mmap[:] = col_data

    def save_minute_data(self):
        for year in range(2016,2024):
            day_list = self.get_daylist([year])
            M_O, M_H, M_L, M_C, M_V = self.parquetreader.get_Minute_data([year])
            for i, data in enumerate([M_O, M_H, M_L, M_C, M_V]):
                for j, day in enumerate(day_list):
                    day  = day.date().strftime("%Y_%m_%d")
                    curr_data = data[j]
                    mmap = np.memmap(
                        os.path.join(self.mmap_MinuteDataPath, f"{self.M_name[i]}",f"{self.M_name[i]}_{day}.mmap"),
                        dtype=np.float32, mode='w+', shape=curr_data.shape)
                    mmap[:] = curr_data

    def save_barra_data(self):
        year_list = list(range(2016, 2024))
        barra = self.parquetreader.get_Barra(year_list)
        day_list = self.get_daylist(year_list)

        for i, day in enumerate(day_list):
            curr_data = barra[i]
            mmap = np.memmap(os.path.join(self.mmap_BarraPath, f'barra_{day.strftime("%Y_%m_%d")}.mmap'), dtype=np.float32,
                             mode='w+',
                             shape=curr_data.shape)
            mmap[:] = curr_data
            
if __name__ == '__main__':
    # 测试df2tensor
    # import time
    # dates = pd.date_range(start="2023-01-01 09:30", periods=242, freq="T")
    # data = np.random.rand(242, 5)  
    # df = pd.DataFrame(data, index=dates, columns=[f"feature_{i}" for i in range(5)])
    # df = df.sample(200)
    # start = time.time()
    # tensor = Interface.df2tensor(df)
    # end = time.time()
    # print(df.shape)
    # print("Tensor shape:", tensor.shape)
    # print("Tensor data:", tensor)   
    # print(f"Time taken: {end - start} seconds")
    # 测试
    reader1 = MmapReader()
    reader2 = ParquetReader()
    print(reader1.data_shape)
    # M_O1, M_H1, M_L1, M_C1, M_V1 = reader1.get_Minute_data([2020])
    # M_O2, M_H2, M_L2, M_C2, M_V2 = reader2.get_Minute_data([2020])

    # print((torch.nan_to_num(M_O1) == torch.nan_to_num(M_O2)).all())
    # print((torch.nan_to_num(M_H1) == torch.nan_to_num(M_H2)).all())   
    # print((torch.nan_to_num(M_L1) == torch.nan_to_num(M_L2)).all())
    # print((torch.nan_to_num(M_C1) == torch.nan_to_num(M_C2)).all())
    # print((torch.nan_to_num(M_V1) == torch.nan_to_num(M_V2)).all())

    # D_O1, D_H1, D_L1, D_C1, D_V1 = reader1.get_Day_data([2020])
    # D_O2, D_H2, D_L2, D_C2, D_V2 = reader2.get_Day_data([2020])

    # print((torch.nan_to_num(D_O1) == torch.nan_to_num(D_O2)).all())
    # print((torch.nan_to_num(D_H1) == torch.nan_to_num(D_H2)).all())
    # print((torch.nan_to_num(D_L1) == torch.nan_to_num(D_L2)).all())
    # print((torch.nan_to_num(D_C1) == torch.nan_to_num(D_C2)).all())
    # print((torch.nan_to_num(D_V1) == torch.nan_to_num(D_V2)).all())


    # B1 = reader1.get_Barra([2020])
    # B2 = reader2.get_Barra([2020])

    # print((torch.nan_to_num(B1) == torch.nan_to_num(B2)).all())

