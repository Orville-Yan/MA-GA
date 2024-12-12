import sys
sys.path.append('..')

from OP.ToA import OP_AF2A
from ToolsGA.Data_tools import DailyDataReader, MinuteDataReader
import pandas as pd
import numpy as np
import torch
import os
import time


class DataReader:
    def __init__(
        self, 
        DailyDataPath: str = "../../Data/DailyData", 
        MinuteDataPath: str = "../../Data/MinuteData", 
        BarraPath: str = "../../Data/barra.pt", 
        DictPath: str = "../../Data/dict.pt", 
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
        D_O=torch.tensor(D_O, dtype=torch.float32,device=self.device)[index]
        D_C = torch.tensor(D_C, dtype=torch.float32, device=self.device)[index]
        D_H = torch.tensor(D_H, dtype=torch.float32, device=self.device)[index]
        D_L = torch.tensor(D_L, dtype=torch.float32, device=self.device)[index]
        D_V = torch.tensor(self.DailyDataReader.GetVolume().values, dtype=torch.float32, device=self.device)[index]
        for tensor in [D_O, D_H, D_L, D_C, D_V]:
            tensor = torch.where(clean | (tensor < 1e-5), float('nan'), tensor)
        return D_O, D_H, D_L, D_C, D_V

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

    data_reader = DataReader()

    DO, DH, DL, DC, DV = data_reader.get_Day_data([2016, 2017])
    print(DO.shape, DH.shape, DL.shape, DC.shape, DV.shape)

    MO, MH, ML, MC, MV = data_reader.get_Minute_data([2016, 2017])
    print(MO.shape, MH.shape, ML.shape, MC.shape, MV.shape)

    barra = data_reader.get_barra([2016, 2017])
    print(barra.shape)

    interval_rtn = data_reader.get_labels([2016, 2017])
    print(interval_rtn.shape)