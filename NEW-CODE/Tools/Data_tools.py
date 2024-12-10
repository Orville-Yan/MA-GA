import sys
sys.path.append('..')
from OP.ToA import OP_AF2A

import pandas as pd
import numpy as np
from scipy.io import loadmat
import os
from datetime import datetime
import torch

root_path = "../../Data/DailyData"
minute_data_path="../../Data/Minute_data"
barra_path="../../Data/barra.pt"
dict_path="../../Data/dict.pt"

status_path = os.path.join(root_path, 'AllStock_DailyStatus.mat')
ST_path = os.path.join(root_path, 'AllStock_DailyST.mat')
ListedDate_path = os.path.join(root_path, 'AllStock_DailyListedDate.mat')
TradingDate_path = os.path.join(root_path, 'TradingDate_Daily.mat')
code_path = os.path.join(root_path, 'AllStockCode.mat')
open_path = os.path.join(root_path, 'AllStock_DailyOpen_dividend.mat')
close_path = os.path.join(root_path, 'AllStock_DailyClose_dividend.mat')
high_path = os.path.join(root_path, 'AllStock_DailyHigh_dividend.mat')
low_path = os.path.join(root_path, 'AllStock_DailyLow_dividend.mat')

status=loadmat(status_path)['AllStock_DailyStatus_use']
ST=loadmat(ST_path)['AllStock_DailyST']
ListedDate=loadmat(ListedDate_path)['AllStock_DailyListedDate']
close = loadmat(close_path)['AllStock_DailyClose_dividend']
open=loadmat(open_path)['AllStock_DailyOpen_dividend']
high=loadmat(high_path)['AllStock_DailyHigh_dividend']
low=loadmat(low_path)['AllStock_DailyLow_dividend']

TradingDate=loadmat(TradingDate_path)['TradingDate_Daily']
code=loadmat(code_path)['AllStockCode']

def process_code():
    code1 = []
    for i in range(len(code[0])):
        code1.append(code[0][i].tolist()[0])
    return pd.Series(code1)

def process_TradingDate():
    date=[]
    for i in range(len(TradingDate)):
        # date.append(pd.to_datetime(TradingDate[i][0],format='%Y%m%d'))
        # print(TradingDate[i][0])
        time =  datetime.strptime(str(TradingDate[i][0]), '%Y%m%d')
        date.append(time)
        # print(TradingDate[i][0])
    # print(date)
    # print(pd.Series(date))
    return pd.Series(date)
TradingDate=process_TradingDate()
code=process_code()

def get_trading_date():
    return TradingDate

def get_code():
    return code

def get_pv():
    s1=loadmat(root_path+'/AllStock_DailyAShareNum.mat')['AllStock_DailyAShareNum']
    s2=loadmat(root_path+'/AllStock_DailyClose.mat')['AllStock_DailyClose']
    pv=pd.DataFrame(s1*s2,index=TradingDate,columns=code)
    return pv

def get_open_and_close():
    s1=pd.DataFrame(open,index=TradingDate,columns=code).replace(0,np.nan)
    s2=pd.DataFrame(close,index=TradingDate,columns=code).replace(0,np.nan)
    return s1,s2

def get_high_and_low():
    s1 = pd.DataFrame(high, index=TradingDate, columns=code).replace(0, np.nan)
    s2 = pd.DataFrame(low, index=TradingDate, columns=code).replace(0, np.nan)
    return s1, s2

def get_TR():
    TR_path = root_path + '/AllStock_DailyTR.mat'
    TR = loadmat(TR_path)['AllStock_DailyTR']
    TR=pd.DataFrame(TR,index=TradingDate,columns=code).replace(0,np.nan)
    return clean_data(TR)

def get_volume():
    volume_path = root_path + '/AllStock_DailyVolume.mat'
    volume = loadmat(volume_path)['AllStock_DailyVolume']
    volume = pd.DataFrame(volume, index=TradingDate, columns=code).replace(0, np.nan)
    return clean_data(volume)

def clean_data(df):
    status20=(pd.DataFrame((1-ST)*(status),index=TradingDate,columns=code).rolling(20).sum()>=10).astype(int).replace(0,np.nan)
    listed=pd.DataFrame((ListedDate>=60).astype(int),index=TradingDate,columns=code)
    good_bad=listed*status20
    df1=df[good_bad.replace(0,np.nan).notna()]
    return df1

def get_index(year_lst):
    date_series = pd.Series(TradingDate)
    index=[]
    for year in year_lst:
        index.extend(date_series.index[date_series.dt.year == year])
    return index

def get_pct_change():
    p = pd.DataFrame(close, index=TradingDate, columns=code)
    pct_change = p / p.shift(1)
    return clean_data(pct_change)

def get_first_day(df):
    p = df.resample('ME').last().index
    groups = df.groupby(df.index.map(lambda x: sum(x > p)))
    a = groups.apply(lambda x: pd.concat([x.iloc[0], pd.Series(x.iloc[0].name)]))
    a = a.set_index(a.columns[-1])
    return a.index

def get_last_day(df):
    p = df.resample('ME').last().index
    groups = df.groupby(df.index.map(lambda x: sum(x > p)))
    a = groups.apply(lambda x: pd.concat([x.iloc[-1], pd.Series(x.iloc[-1].name)]))
    a = a.set_index(a.columns[-1])
    return a.index
def get_clean():
    status20 = OP_AF2A.D_ts_mean(torch.from_numpy((1 - ST) * (status)), 20) > 0.5
    listed = torch.from_numpy(ListedDate >= 60)
    clean = ~(listed * status20)
    return clean
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

class CustomClass:
    def __init__(self):
        self.factor_list = []

    def get_from_list(self,df_list):
        self.factor_list+=['feature_{}'.format(i) for i in range(1, len(df_list) + 1)]
        for i in range(1, len(df_list) + 1):
            setattr(self, 'feature_{}'.format(i), df_list[i - 1])

    def extend(self,factor_class):
        ini_length=len(self.factor_list)
        for i,factor_name in enumerate(factor_class.factor_list):
            factor=getattr(factor_class,factor_name)
            self.factor_list.append(factor_name)
            setattr(self, factor_name, factor)





