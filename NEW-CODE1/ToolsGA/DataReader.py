import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from OP import *
from ToolsGA.Data_tools import DailyDataReader, MinuteDataReader
import pandas as pd
import torch
import os
import re
import numpy as np
import tqdm
import json

class ParquetReader:
    def __init__(
        self, 
        DailyDataPath: str = "../Data/Daily",
        MinuteDataPath: str = "../Data/Minute",
        BarraPath: str = "../Data/barra.pt",
        DictPath: str = "../Data/dict.pt",
        device: str = 'cpu'
    ):
        self.DailyDataReader = DailyDataReader(DailyDataPath)
        self.MinuteDataReader = MinuteDataReader(MinuteDataPath, device)
        self.BarraPath = BarraPath
        self.DictPath = DictPath
        self.device = device

    def get_Minute_data(self, year_lst: list[int]) -> list[torch.Tensor]:
        """
        Read all columns data by year list
        """
        return self.MinuteDataReader.get_Minute_data(year_lst)

    def get_Day_data(self,year_lst: list[int])-> list[torch.Tensor]:
        index = self.DailyDataReader.get_index(year_lst)
        clean = self.DailyDataReader.get_clean()[index]
        
        D_O, D_H, D_L, D_C = self.DailyDataReader.GetOHLC()
        D_V = self.DailyDataReader.GetVolume().values

        def adjust(tensor):
            tensor=torch.tensor(tensor, dtype=torch.float32,device=self.device)[index]
            tensor = torch.where(clean | (tensor < 1e-5), float('nan'), tensor)
            return tensor

        return [adjust(D_O), adjust(D_H), adjust(D_L), adjust(D_C), adjust(D_V)]

    def get_barra(self, year_lst: list[int]) -> torch.Tensor:
        barra = torch.load(self.BarraPath, weights_only=True)
        dict = torch.load(self.DictPath, weights_only=False)
        s = []
        for year in year_lst:
            mask = pd.to_datetime(dict['index']).year == year
            s.append(mask)

        w = s[0]
        for t in s:
            w = w | t

        return barra[w][:, self.DailyDataReader.MutualStockCodes]

    def get_labels(self, year_lst: list[int], freq: int=5) -> torch.Tensor:
        index = self.DailyDataReader.get_index(year_lst)
        clean = self.DailyDataReader.get_clean()[index]
        D_O = torch.tensor(self.DailyDataReader._get_open(), dtype=torch.float32, device=self.device)[index]
        D_C = torch.tensor(self.DailyDataReader._get_close(), dtype=torch.float32, device=self.device)[index]
        D_O, D_C=[
            torch.where(clean | (tensor < 1e-5), float('nan'), tensor) for tensor in [D_O, D_C]
        ]
        
        open = OP_AF2A.D_ts_delay(D_O, -1)
        close = OP_AF2A.D_ts_delay(D_C, -freq)
        interval_return = close / open - 1
        return interval_return

    def get_barra_by_daylist(self,day_list):
        barra = torch.load(self.BarraPath, weights_only=True)
        dict = torch.load(self.DictPath, weights_only=False)['index']
        index=[dict.get_loc(day)  for day in day_list]
        return barra[index][:, self.DailyDataReader.MutualStockCodes]

    def get_daylist(self,year_list):
        trading_dates=self.DailyDataReader.TradingDate
        return trading_dates[trading_dates.dt.year.isin(year_list)]

class Interface:
    def __init__(self):
        pass
    def get_daylist(self,year):
        if not hasattr(self, 'DailyDataReader'):
            self.DailyDataReader = DailyDataReader()
        trading_dates = self.DailyDataReader.TradingDate
        return list(trading_dates[trading_dates.dt.year == year])

    def tensor2df(self,tensor,year_list):
        day_list=self.get_daylist(year_list)
        stocks=self.DailyDataReader.StockCodes
        df=pd.DataFrame(np.asarray(tensor),index=day_list,columns=stocks)
        return df


class MmapReader(Interface):
    def __init__(self,
                 DailyDataPath: str = "../Data/Daily",
                 MinuteDataPath: str = "../Data/Minute",
                 MmapPath: str = "../Data/Mmap",
                 device='cpu',download = False):
        if download:
            self.MinuteDataReader = MinuteDataReader(MinuteDataPath, device)
            self.DailyDataReader = DailyDataReader(DailyDataPath)
        self.MmapPath = MmapPath
        self.daily_data_info = {}

        self.D_name = ['D_O', 'D_H', 'D_L', 'D_C', 'D_V']
        self.M_name = ['M_O', 'M_H', 'M_L', 'M_C', 'M_V']
        with open("../Data/Mmap/data_shape.json", "r", encoding="utf-8") as file:
            self.data_shape = json.load(file)

    def save_daily_data(self):
        D_O, D_H, D_L, D_C = self.DailyDataReader.GetOHLC()
        D_O, D_H, D_L, D_C, D_V = self.DailyDataReader.get_df_ohlcv(D_O, D_H, D_L, D_C)
        root_path = self.MmapPath + '/Daily'

        for i, data in enumerate([D_O, D_H, D_L, D_C, D_V]):
            for year in range(2016, 2024):
                curr_data = data.loc[str(year)]
                self.daily_data_info[year] = curr_data.shape
                mmap = np.memmap(os.path.join(root_path, f'{self.D_name[i]}_{year}.mmap'), dtype=np.float32, mode='w+',
                                 shape=curr_data.shape)
                mmap[:] = curr_data

    def save_minute_data(self):
        for year in range(2016, 2024):
            day_list = self.get_daylist(year)
            M_O, M_H, M_L, M_C, M_V = self.MinuteDataReader.get_Minute_data([year])
            for i, data in enumerate([M_O, M_H, M_L, M_C, M_V]):
                for j, day in enumerate(day_list):
                    day  = day.date().strftime("%Y_%m_%d")
                    curr_data = data[j]
                    mmap = np.memmap(
                        os.path.join(f"../Data/Mmap/Minute/{self.M_name[i]}", f'{self.M_name[i]}_{day}.mmap'),
                        dtype=np.float32, mode='w+', shape=curr_data.shape)
                    mmap[:] = curr_data

    def save_barra_data(self):
        root_path = self.MmapPath + '/Barra'
        year_list = list(range(2016, 2024))

        reader = ParquetReader()
        barra = reader.get_barra(year_list)
        day_list = self.get_daylist(year_list)

        for i, day in enumerate(day_list):
            curr_data = barra[i]
            mmap = np.memmap(os.path.join(root_path, f'barra_{day.strftime("%Y_%m_%d")}.mmap'), dtype=np.float32,
                             mode='w+',
                             shape=curr_data.shape)
            mmap[:] = curr_data

    def get_Day_data(self, year_lst: list[int]):
        length = 0
        width = self.data_shape[str(year_lst[0])][1]

        for year in year_lst:
            length += self.data_shape[str(year)][0]

        start = 0

        D_O, D_H, D_L, D_C, D_V = [torch.full((length, width), float('nan')) for _ in range(5)]
        for i, year in enumerate(year_lst):

            for j, data in enumerate([D_O, D_H, D_L, D_C, D_V]):
                file_path = os.path.join(self.MmapPath + '/Daily', f'{self.D_name[j]}_{year}.mmap')
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
            folder_path = os.path.join(self.MmapPath + '/Minute', self.M_name[j])
            files = get_all_files(folder_path, year_lst)

            for file in files:
                file_path = os.path.join(folder_path, file)
                mmap_read = np.memmap(file_path, dtype=np.float32, mode='r', shape=(minute_num, stock_num))
                tensor_read = torch.from_numpy(mmap_read.copy())
                data[pos] = tensor_read.permute(1, 0)
                pos += 1

            pos = 0

        return M_O, M_H, M_L, M_C, M_V

    def get_Minute_data_daily(self, day):
        data = []
        for name in self.M_name:
            file_path = os.path.join(self.MmapPath + f'/Minute/{name}', f'{name}_{day.strftime("%Y_%m_%d")}.mmap')
            mmap_read = np.memmap(file_path, dtype=np.float32, mode='r', shape=(242, 5483))
            data.append(torch.from_numpy(mmap_read.copy()).permute(1, 0).unsqueeze(0))

        return data

    def get_Barra(self, year_lst):
        num_stock = self.data_shape[str(year_lst[0])][1]
        day_list = self.get_daylist(year_lst)
        barra = torch.full((len(day_list), num_stock, 41), float('nan'))
        for i, day in enumerate(day_list):
            barra[i] = self.get_Barra_daily(day)
        return barra

    def get_Barra_daily(self, day):
        num_stocks = self.data_shape['2016'][1]
        file_path = os.path.join(self.MmapPath + '/Barra', f'barra_{day.strftime("%Y_%m_%d")}.mmap')
        mmap_read = np.memmap(file_path, dtype=np.float32, mode='r', shape=(num_stocks, 41))
        return torch.from_numpy(mmap_read.copy())

    def get_labels(self, year_lst, freq):
        if not hasattr(self, 'DailyDataReader'):
            self.DailyDataReader = DailyDataReader()

        index = self.DailyDataReader.get_index(year_lst)
        clean = self.DailyDataReader.get_clean()[index]


        D_O,_,_, D_C,_ = self.get_Day_data(year_lst)
        D_O, D_C = [torch.where(clean | (tensor < 1e-5), float('nan'), tensor) for tensor in [D_O, D_C]]

        open = OP_AF2A.D_ts_delay(D_O, -1)
        close = OP_AF2A.D_ts_delay(D_C, -freq)
        interval_return = close / open

        return interval_return



if __name__ == "__main__":
    # bgn_time = time.time()
    # data_path = "./processed_data"
    # data_reader = DataReader(data_path)
    # open_data = data_reader.read_data_by_col("open", [2016, 2017])
    # print("Open [2016, 2017] shape: ", open_data.shape)
    # print("Time cost: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - bgn_time)))
    #
    # bgn_time = time.time()
    # open_data, high_data, low_data, close_data, volume_data = data_reader.read_data([2016, 2017])
    # print("Open, High, Low, Close, Volume [2016, 2017] shape: ", open_data.shape, high_data.shape, low_data.shape,
    #       close_data.shape, volume_data.shape)
    # print("Time cost: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - bgn_time)))
    # data_reader1 = ParquetReader()
    # MO1 , MH1, ML1, MC1, MV1 = data_reader1.get_Minute_data([2016])
    data_reader2 = MmapReader(download=False)
    MO2 , MH2, ML2, MC2, MV2 = data_reader2.get_Minute_data([2016])
    # print(MO1.shape == MO2.shape, MH1.shape == MH2.shape, ML1.shape == ML2.shape, MC1.shape == MC2.shape, MV1.shape == MV2.shape)



    # barra = data_reader.get_barra([2016, 2017])
    # print(barra.shape)

    # interval_rtn = data_reader.get_labels([2016, 2017])
    # print(interval_rtn.shape)