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
    def get_daylist(trading_dates,year_lst:list[int]):
        return list(trading_dates[trading_dates.dt.year.isin(year_lst)])
    
    @staticmethod
    def get_index(trading_dates, year_lst)->list:
        return [(trading_dates.index[trading_dates.dt.year.isin(year_lst)])]
    
    @staticmethod
    def tensor2df(tensor, trading_dates,year_list, stocks):
        day_list = Interface.get_daylist(trading_dates
                                         ,year_list)
        idx = Interface.get_index(trading_dates,year_list)
        tensor = tensor[idx]
        df = pd.DataFrame(np.asarray(tensor), index=day_list, columns=stocks)
        return df
    
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
    def generate_trading_time(date):
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
    def adjust(array,index,clean,device='cpu'):
        tensor = torch.tensor(array, dtype=torch.float32, device=device)[index]
        tensor = torch.where(clean | (tensor < 1e-5), float('nan'), tensor)
        return tensor
    
    @staticmethod
    def get_pct_change(close:torch.tensor)-> torch.tensor:
        p = close / close.shift(1)
        return Interface.adjust(p)

    @staticmethod
    def get_labels(d_o,d_c,index,clean, freq=5):
        clean = clean[index]
        d_o = Interface.adjust(d_o,index,clean)
        d_c = Interface.adjust(d_c,index,clean)
        open = OP_AF2A.D_ts_delay(d_o, -1)
        close = OP_AF2A.D_ts_delay(d_c, -freq)
        return close / open - 1



class BasicReader():
    def __init__(self):
        self.daily_data_path = Config.PARQUET_Daily_PATH
        self.MutualStockCodes = pd.read_parquet(Config.MUTUAL_STOCK_CODES_PATH)["Mutual"].values
        self.TradingDate = self._TradingDate()
        self.StockCodes = self._StockCodes()
        self.ListedDate = self._ListedDate()
        self.Status = self._Status()
        self.ST = self._ST()
        self.clean = self._clean_data()

    def _TradingDate(self):
        trading_date = loadmat(os.path.join(self.daily_data_path, 'TradingDate_Daily.mat'))['TradingDate_Daily']
        return pd.Series([datetime.strptime(str(d[0]), '%Y%m%d') for d in trading_date])

    def _StockCodes(self):
        code = loadmat(os.path.join(self.daily_data_path, 'AllStockCode.mat'))['AllStockCode']
        return pd.Series([code[0][i].tolist()[0] for i in range(len(code[0]))]).loc[self.MutualStockCodes].reset_index(drop=True)
    
    def _ListedDate(self):
        return loadmat(os.path.join(self.daily_data_path, 'AllStock_DailyListedDate.mat'))['AllStock_DailyListedDate'][:, self.MutualStockCodes]
    
    def _Status(self):
        return loadmat(os.path.join(self.daily_data_path, 'AllStock_DailyStatus.mat'))['AllStock_DailyStatus_use'][:, self.MutualStockCodes]

    def _ST(self):
        return loadmat(os.path.join(self.daily_data_path, 'AllStock_DailyST.mat'))['AllStock_DailyST'][:, self.MutualStockCodes]

    def _clean_data(self) -> pd.DataFrame:
        status20 = OP_AF2A.D_ts_mean(torch.from_numpy((1 - self.ST) * (self.Status)), 20) > 0.5
        listed = torch.from_numpy(self.ListedDate >= 60)
        clean = ~(listed * status20)
        return clean


    
class ParquetReader(BasicReader):
    def __init__(self,DailyDataPath=Config.PARQUET_Daily_PATH, MinuteDataPath=Config.PARQUET_Minute_PATH, BarraPath=Config.PARQUET_BARRA_PATH, DictPath=Config.PARQUET_DICT_PATH, device='cpu'):
        super().__init__()
        self.parquet_DailyDataPath = DailyDataPath
        self.parquet_MinuteDataPath = MinuteDataPath
        self.parquet_BarraPath = BarraPath
        self.parquet_DictPath = DictPath
        self.device = device
        self.M_name = ['open', 'high', 'low', 'close', 'volume']
        self.Mutualstk = pd.read_parquet(Config.MUTUAL_STOCK_CODES_PATH)
        self.Mutualstk = list(self.Mutualstk["StockCodes"].loc[self.Mutualstk["Mutual"]].values)

    def GetOHLC(self):
        open = loadmat(os.path.join(self.parquet_DailyDataPath, 'AllStock_DailyOpen_dividend.mat'))['AllStock_DailyOpen_dividend'][:, self.MutualStockCodes]
        high = loadmat(os.path.join(self.parquet_DailyDataPath, 'AllStock_DailyHigh_dividend.mat'))['AllStock_DailyHigh_dividend'][:, self.MutualStockCodes]
        low = loadmat(os.path.join(self.parquet_DailyDataPath, 'AllStock_DailyLow_dividend.mat'))['AllStock_DailyLow_dividend'][:, self.MutualStockCodes]
        close = loadmat(os.path.join(self.parquet_DailyDataPath, 'AllStock_DailyClose_dividend.mat'))['AllStock_DailyClose_dividend'][:, self.MutualStockCodes]
        return [open, high, low, close]
    
    def GetVolume(self):
        return loadmat(os.path.join(self.parquet_DailyDataPath, 'AllStock_DailyVolume.mat'))['AllStock_DailyVolume'][:, self.MutualStockCodes]
    
    def get_Day_data(self,year_lst: list[int])-> list[torch.Tensor]:
        index = Interface.get_index(self.TradingDate,year_lst)
        clean = self.clean[index]
        D_O, D_H, D_L, D_C = self.GetOHLC()
        D_V = self.GetVolume()
        return [Interface.adjust(D_O,index,clean), Interface.adjust(D_H,index,clean), Interface.adjust(D_L,index,clean), Interface.adjust(D_C,index,clean), Interface.adjust(D_V,index,clean )]

    def read_data_by_col(self, col: str, year_lst: list[int]) -> torch.Tensor:
        data = []
        for year in year_lst:
            file_path = f"{self.parquet_MinuteDataPath}/{col}_{year}.parquet"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            tensor = Interface.df2tensor(pd.read_parquet(file_path, columns=self.Mutualstk))
            data.append(tensor)
        return torch.cat(data, dim=0)
    
    def get_Minute_data(self, year_lst: list[int]) -> list[torch.Tensor]:
        return [self.read_data_by_col(col, year_lst) for col in self.M_name]
    
    
    def get_barra(self, year_lst):
        barra = torch.load(os.path.join(Config.DATA_PATH,'barra.pt'), weights_only=True)
        dict = torch.load(os.path.join(Config.DATA_PATH,'dict.pt'), weights_only=False)
        s = [pd.to_datetime(dict['index']).year.isin(year_lst)]
        w = s[0]
        return barra[w][:, self.MutualStockCodes]
    
class MmapReader(BasicReader):
    def __init__(self,  download = False, DailyDataPath: str = Config.MMAP_Daily_PATH, MinuteDataPath: str = Config.MMAP_Minute_PATH, BarraPath: str = Config.MMAP_BARRA_PATH, DictPath: str = Config.PARQUET_DICT_PATH, device: str = 'cpu'):
        if download:
            self.parquetreader = ParquetReader()
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
        width = self.data_shape[str(year_lst[0])][1]

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
        def get_all_files(folder_path, year_lst):
            year_pattern = re.compile(r'_({})_'.format('|'.join(map(str, year_lst))))
            entries = os.listdir(folder_path)
            files = [file for file in entries if os.path.isfile(os.path.join(folder_path, file))]
            filtered_files = [file for file in files if year_pattern.search(file)]
            return sorted(filtered_files)

        day_len = 0
        minute_num = 242
        stock_num = self.data_shape[str(year_lst[0])][1]

        for year in year_lst:
            day_len += self.data_shape[str(year)][0]

        pos = 0
        M_O, M_H, M_L, M_C, M_V = [torch.full((day_len, stock_num, minute_num), float('nan')) for _ in range(5)]
        for j, data in enumerate([M_O, M_H, M_L, M_C, M_V]):
            folder_path = os.path.join(self.mmap_MinuteDataPath, self.M_name[j])
            files = get_all_files(folder_path, year_lst)

            for file in files:
                file_path = os.path.join(folder_path, file)
                mmap_read = np.memmap(file_path, dtype=np.float32, mode='r', shape=(stock_num,minute_num))
                tensor_read = torch.from_numpy(mmap_read.copy())
                data[pos] = tensor_read
                pos += 1

            pos = 0

        return M_O, M_H, M_L, M_C, M_V

    def get_Minute_data_daily(self, day):
        data = []
        minute_num = 242
        stock_num = 5483
        for name in self.M_name:
            file_path = os.path.join(self.mmap_MinuteDataPath, f"{name}", f'{name}_{day.strftime("%Y_%m_%d")}.mmap')
            mmap_read = np.memmap(file_path, dtype=np.float32, mode='r', shape=(stock_num,minute_num))
            data.append(torch.from_numpy(mmap_read.copy()).unsqueeze(0))

        return data

    def get_Barra(self, year_lst):
        num_stock = self.data_shape[str(year_lst[0])][1]
        day_list = Interface.get_daylist(self._TradingDate,year_lst)
        barra = torch.full((len(day_list), num_stock, 41), float('nan'))
        for i, day in enumerate(day_list):
            barra[i] = self.get_Barra_daily(day)
        return barra

    def get_Barra_daily(self, day):
        num_stocks = self.data_shape['2016'][1]
        file_path = os.path.join(self.mmap_BarraPath, f'barra_{day.strftime("%Y_%m_%d")}.mmap')
        mmap_read = np.memmap(file_path, dtype=np.float32, mode='r', shape=(num_stocks, 41))
        return torch.from_numpy(mmap_read.copy())

    
    def save_daily_data(self):
        for year in range(2016, 2025):
            data = self.parquetreader.get_Day_data([year])
            for i, name in enumerate(self.D_name):
                col_data = data[i]
                self.daily_data_info[year] = col_data.shape
                mmap = np.memmap(os.path.join(self.mmap_DailyDataPath, f'{name}_{year}.mmap'), dtype=np.float32, mode='w+',
                                    shape=col_data.shape)
                mmap[:] = data[0] 

    def save_minute_data(self):
        trading_dates = self._TradingDate
        for year in range(2016, 2025):
            day_list = Interface.get_daylist(trading_dates, [year])
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
        barra = self.parquetreader.get_barra(year_list)
        day_list = Interface.get_daylist(self._TradingDate,year_list)

        for i, day in enumerate(day_list):
            curr_data = barra[i]
            mmap = np.memmap(os.path.join(self.mmap_BarraPath, f'barra_{day.strftime("%Y_%m_%d")}.mmap'), dtype=np.float32,
                             mode='w+',
                             shape=curr_data.shape)
            mmap[:] = curr_data
            
if __name__ == '__main__':
    reader = MmapReader()
    M_O, M_H, M_L, M_C, M_V = reader.get_Minute_data([2016])
    print(M_H)