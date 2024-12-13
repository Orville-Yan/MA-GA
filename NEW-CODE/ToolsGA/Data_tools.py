import sys
sys.path.append('..')
from OP.ToA import OP_AF2A

import pandas as pd
import numpy as np
from scipy.io import loadmat
import os
from datetime import datetime
import torch

def get_first_day(df: pd.DataFrame) -> pd.DatetimeIndex:
    p = df.resample('ME').last().index
    groups = df.groupby(df.index.map(lambda x: sum(x > p)))
    a = groups.apply(lambda x: pd.concat([x.iloc[0], pd.Series(x.iloc[0].name)]))
    a = a.set_index(a.columns[-1])
    return a.index

def get_last_day(df: pd.DataFrame) -> pd.DatetimeIndex:
    p = df.resample('ME').last().index
    groups = df.groupby(df.index.map(lambda x: sum(x > p)))
    a = groups.apply(lambda x: pd.concat([x.iloc[-1], pd.Series(x.iloc[-1].name)]))
    a = a.set_index(a.columns[-1])
    return a.index

def generate_trading_time(date: str) -> pd.DatetimeIndex:
    """
    The minutes trading time of a day
    """
    date = pd.to_datetime(date)
    morning_times = pd.date_range(date + pd.Timedelta("09:30:00"), date + pd.Timedelta("11:30:00"), freq="T")
    afternoon_times = pd.date_range(date + pd.Timedelta("13:00:00"), date + pd.Timedelta("15:00:00"), freq="T")
    return morning_times.union(afternoon_times).sort_values()

def fill_full_day(day_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill the missing minutes of a day with NaN
    """
    date = day_data.index[0].date()
    trading_time = generate_trading_time(date)
    day_data = day_data.reindex(trading_time)
    return day_data


class DailyDataReader:
    def __init__(self, daily_data_path: str="../../Data/DailyData"):
        self.daily_data_path = daily_data_path
        self.MutualStockCodes = pd.read_parquet("../../Data/MutualStockCodes.parquet")["Mutual"].values

        self.ListedDate = self._ListedDate()
        self.TradingDate = self._TradingDate()
        self.StockCodes = self._StockCodes()
        self.Status = self._Status()
        self.ST = self._ST()

    def _ListedDate(self) -> np.ndarray:
        """
        截至昨日收盘的上市交易日数
        """
        return loadmat(
            os.path.join(self.daily_data_path, 'AllStock_DailyListedDate.mat')
        )['AllStock_DailyListedDate'][:, self.MutualStockCodes]

    def _TradingDate(self) -> pd.Series:
        """
        交易日
        """
        TradingDate = loadmat(
            os.path.join(self.daily_data_path, 'TradingDate_Daily.mat')
        )['TradingDate_Daily']

        date = []
        for i in range(len(TradingDate)):
            # date.append(pd.to_datetime(TradingDate[i][0],format='%Y%m%d'))
            # print(TradingDate[i][0])
            time =  datetime.strptime(str(TradingDate[i][0]), '%Y%m%d')
            date.append(time)
            # print(TradingDate[i][0])
        # print(date)
        # print(pd.Series(date))
        return pd.Series(date)

    def _StockCodes(self) -> pd.Series:
        """
        股票代码
        """
        code = loadmat(
            os.path.join(self.daily_data_path, 'AllStockCode.mat')
        )['AllStockCode']
        code1 = []
        for i in range(len(code[0])):
            code1.append(code[0][i].tolist()[0])
        return pd.Series(code1).loc[self.MutualStockCodes].reset_index(drop=True)

    def _Status(self) -> np.ndarray:
        """
        交易状态, 已经把文本处理成数字, 1 代表正常交易, 0 不是
        """
        return loadmat(
            os.path.join(self.daily_data_path, 'AllStock_DailyStatus.mat')
        )['AllStock_DailyStatus_use'][:, self.MutualStockCodes]

    def _ST(self) -> np.ndarray:
        """
        1 代表是 st, 0 代表不是
        """
        return loadmat(
            os.path.join(self.daily_data_path, 'AllStock_DailyST.mat')
        )['AllStock_DailyST'][:, self.MutualStockCodes]
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        status20 = (
            pd.DataFrame(
                (1 - self.ST) * (self.Status), 
                index=self.TradingDate, 
                columns=self.StockCodes
            ).rolling(20).sum() >= 10
        ).astype(int).replace(0, np.nan)

        listed = pd.DataFrame(
            (self.ListedDate >= 60).astype(int), index=self.TradingDate, columns=self.StockCodes
        )
        good_bad = listed * status20
        df1 = df[good_bad.replace(0, np.nan).notna()]
        return df1
    
    def _get_open(self) -> np.ndarray:
        return loadmat(
            os.path.join(self.daily_data_path, 'AllStock_DailyOpen_dividend.mat')
        )['AllStock_DailyOpen_dividend'][:, self.MutualStockCodes]
    
    def _get_close(self) -> np.ndarray:
        return loadmat(
            os.path.join(self.daily_data_path, 'AllStock_DailyClose_dividend.mat')
        )['AllStock_DailyClose_dividend'][:, self.MutualStockCodes]
    
    def _get_high(self) -> np.ndarray:
        return loadmat(
            os.path.join(self.daily_data_path, 'AllStock_DailyHigh_dividend.mat')
        )['AllStock_DailyHigh_dividend'][:, self.MutualStockCodes]
    
    def _get_low(self) -> np.ndarray:
        return loadmat(
            os.path.join(self.daily_data_path, 'AllStock_DailyLow_dividend.mat')
        )['AllStock_DailyLow_dividend'][:, self.MutualStockCodes]
    
    def GetOHLC(self) -> list[np.ndarray]:
        """
        OHLC 数据
        """
        return [self._get_open(), self._get_high(), self._get_low(), self._get_close()]
    
    def GetVolume(self) -> pd.DataFrame:
        volume = loadmat(
            os.path.join(self.daily_data_path, 'AllStock_DailyVolume.mat')
        )['AllStock_DailyVolume'][:, self.MutualStockCodes]
        volume = pd.DataFrame(
            volume, index=self.TradingDate, columns=self.StockCodes
        ).replace(0, np.nan)
        return self._clean_data(volume)

    def get_df_ohlcv(
            self, 
            open: np.ndarray, high: np.ndarray, 
            low: np.ndarray, close: np.ndarray
    ) -> list[pd.DataFrame]:
        def to_df(data: np.ndarray):
            return pd.DataFrame(data, index=self.TradingDate, columns=self.StockCodes).replace(0, np.nan)
        return [to_df(open), to_df(high), to_df(low), to_df(close), self.get_volume()]

    def get_pv(self) -> pd.DataFrame:
        s1 = loadmat(
            self.daily_data_path + '/AllStock_DailyAShareNum.mat'
        )['AllStock_DailyAShareNum'][:, self.MutualStockCodes]
        s2 = loadmat(
            self.daily_data_path + '/AllStock_DailyClose.mat'
        )['AllStock_DailyClose'][:, self.MutualStockCodes]
        pv = pd.DataFrame(s1*s2, index=self.TradingDate, columns=self.StockCodes)
        return pv

    def get_TR(self) -> pd.DataFrame:
        TR = loadmat(
            self.daily_data_path + '/AllStock_DailyTR.mat'
        )['AllStock_DailyTR'][:, self.MutualStockCodes]
        TR = pd.DataFrame(TR, index=self.TradingDate, columns=self.StockCodes).replace(0, np.nan)
        return self._clean_data(TR)

    def get_index(self, year_lst: list[int]):
        index=[]
        for year in year_lst:
            index.extend(self.TradingDate.index[self.TradingDate.dt.year == year])
        return index

    def get_clean(self) -> np.ndarray:
        status20 = OP_AF2A.D_ts_mean(torch.from_numpy((1 - self.ST) * (self.Status)), 20) > 0.5
        listed = torch.from_numpy(self.ListedDate >= 60)
        clean = ~(listed * status20)
        return clean

    def get_pct_change(self, close: np.ndarray):
        p = pd.DataFrame(close, index=self.TradingDate, columns=self.StockCodes)
        pct_change = p / p.shift(1)
        return self._clean_data(pct_change)


class MinuteDataReader:
    def __init__(self, minute_data_path: str="../../Data/MinuteData", device: str='cpu'):
        self.data_path = minute_data_path  # M_tensor's Path
        self.device = device
        self.cols = ["open", "high", "low", "close", "volume"]  # "amount"
        self.MutualStockCodes = pd.read_parquet("../../Data/MutualStockCodes.parquet")
        self.MutualStockCodes = self.MutualStockCodes["StockCodes"].loc[self.MutualStockCodes["Mutual"]].values

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
        tensor_data = torch.tensor(np.array(day_data, dtype=np.float32), dtype=torch.float32, device=self.device).permute(2, 0, 1)
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
            data.append(self.dataframe_to_tensor(pd.read_parquet(file_path, columns=self.MutualStockCodes)))
        # concatenate date in the day dimension
        return torch.cat(data, dim=1)

    def get_Minute_data(self, year_lst: list[int]) -> list[torch.Tensor]:
        """
        Read all columns data by year list
        """
        return [self.read_data_by_col(col, year_lst) for col in self.cols]






