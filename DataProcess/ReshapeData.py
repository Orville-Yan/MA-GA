import pandas as pd
import numpy as np
import akshare as ak
import os
from scipy.io import loadmat
from tqdm import tqdm

from utils import generate_trading_time, read_stock_exchange, init_logger

DATA_PATH = "./data"
HFQ_FACTOR_PATH = "./hfq_factors"
PROCESSED_DATA_PATH = "./processed_data"
os.makedirs(HFQ_FACTOR_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(f"{PROCESSED_DATA_PATH}/Original", exist_ok=True)


COLUMNS_NAMES = ["time", "open", "high", "low", "close", "volume", "amount"]
for col in COLUMNS_NAMES[1:]:
    os.makedirs(os.path.join(PROCESSED_DATA_PATH, col), exist_ok=True)

# add logger
Logger = init_logger()
# generate trading time (str), 242 minutes per day, to label the data
ALL_TRADING_TIMES = generate_trading_time()
# stock - exchange - on list date, akshare hfq factor api need the exchange of the stock
STOCKS_EXCHANGE = read_stock_exchange()

# for memory usage, reshape the data by batch
ALL_STOCK_FILES = os.listdir(DATA_PATH)
BATCH_NUMBER = 20   # you need to adjust the batch number according to the number of stocks, around 100 stocks per batch is recommended
BATCH_SIZE = len(ALL_STOCK_FILES) // BATCH_NUMBER
BATCHES = [ALL_STOCK_FILES[i: i + BATCH_SIZE] for i in range(0, len(ALL_STOCK_FILES), BATCH_SIZE)]
Logger.info(f"Total {len(ALL_STOCK_FILES)} files, {len(BATCHES)} batches, {BATCH_SIZE} files per batch")


def minute_timestamp(data: pd.DataFrame, timestamps: list=ALL_TRADING_TIMES) -> pd.DataFrame:
    """
    Just label the minute info of the data
    """
    data["time"] += pd.Series(timestamps * data["time"].nunique())
    return data

def read_data(path: str, file_name: str, on_date=None) -> pd.DataFrame:
    """
    on_date: 90 days after the stock's on-list date. on_date is None for those stocks that are not in the STOCKS_EXCHANGE (StocksExchange.xlsx)
    """
    data = pd.DataFrame(loadmat(path)[file_name], columns=COLUMNS_NAMES)

    # we only need the data after 2016, and we need to drop the data before the stock on list date (here we use 90 days after the on list date)
    cond = (data["time"] >= 20160101) & (data["time"] >= on_date) if on_date else (data["time"] >= 20160101)
    data = data.loc[cond].reset_index(drop=True)

    # stock name
    data.insert(1, "stock_name", file_name.split("_")[1])

    # convert the time to str, add the minute info, and convert to datetime
    data["time"] = data["time"].astype(int).astype(str)
    data = minute_timestamp(data)
    data["time"] = pd.to_datetime(data["time"], format="%Y%m%d%H%M")
    return data

def read_hfq_factor(stock_name: str, exchange=None) -> pd.DataFrame | None:
    """
    Get the hfq factor of the stock
    If we have the hfq factor file, read it directly
    If not, use akshare to get the hfq factor

    akshare api: stock_zh_a_daily(symbol=exchange + stock_name, adjust="hfq-factor")
    if we don't have the exchange info, we need to try all the exchanges
    if akshare api doesn't have the hfq factor, return None
    """
    try:
        hfq_factor_df = pd.read_csv(f"{HFQ_FACTOR_PATH}/{stock_name}_hfq.csv")
        return hfq_factor_df
    except:
        pass

    if exchange is None:
        try:
            hfq_factor_df = ak.stock_zh_a_daily(symbol="sz" + stock_name, adjust="hfq-factor")
        except:
            try:
                hfq_factor_df = ak.stock_zh_a_daily(symbol="sh" + stock_name, adjust="hfq-factor")
            except:
                try:
                    hfq_factor_df = ak.stock_zh_a_daily(symbol="bj" + stock_name, adjust="hfq-factor")
                except:
                    hfq_factor_df = None
                    Logger.error(f"{stock_name}'s hfq factor not found!")
    else:
        symbol = exchange + stock_name
        hfq_factor_df = ak.stock_zh_a_daily(symbol=symbol, adjust="hfq-factor")
    
    # save the hfq factor file for the next time
    if hfq_factor_df is not None:
        hfq_factor_df.to_csv(f"{HFQ_FACTOR_PATH}/{stock_name}_hfq.csv", index=False)
    return hfq_factor_df

def hfq_process(stock_name: str, data: pd.DataFrame, exchange: str=None) -> pd.DataFrame:
    """
    process the data with the hfq factor
    if we don't have the hfq factor, the stock's data will not be processed
    """
    # if the data is empty, fill the data with the first row (some stocks don't have data after 2016)
    if data.empty:
        data.loc[0] = ["201601040930", stock_name] + [np.nan] * 6
        return data
    
    # get the hfq factor
    hfq_factor_df = read_hfq_factor(stock_name, exchange)

    # if we don't have the hfq factor, fill 1 (multiply 1 is the same)
    if hfq_factor_df is None:
        data["hfq_factor"] = 1
    else:
        # pre process the hfq factor data
        hfq_factor_df["date"] = pd.to_datetime(hfq_factor_df["date"]) + pd.Timedelta(minutes=30, hours=9)
        hfq_factor_df["hfq_factor"] = hfq_factor_df["hfq_factor"].astype(float)
        hfq_factor_df = hfq_factor_df.sort_values("date").set_index("date")

        # for merge the hfq factor to the data, we need to match the time
        if data["time"][0] not in hfq_factor_df.index:
            hfq_factor_df.loc[data["time"][0]] = np.nan
        hfq_factor_df = hfq_factor_df.ffill().fillna(1)
        data = pd.merge(data, hfq_factor_df, left_on="time", right_on="date", how="left")
        data["hfq_factor"] = data["hfq_factor"].fillna(method="ffill")
        assert data["hfq_factor"].isnull().sum() == 0, f"{stock_name} hfq factor missing!"
    
    # multiply the data with the hfq factor
    data[["open", "high", "low", "close"]] *= data["hfq_factor"].values.reshape(-1, 1)
    data["time"] = data["time"].dt.strftime("%Y%m%d%H%M")
    data = data.loc[data["time"] >= "201601040930"].reset_index(drop=True).drop(columns=["hfq_factor"])
    
    del hfq_factor_df
    return data

def process_data(file_name: str, stock_exchange: pd.DataFrame=STOCKS_EXCHANGE) -> pd.DataFrame:
    """
    read the data, process the data with the hfq factor, and return the data
    """
    stock_name = file_name.split(".")[0].split("_")[1]
    stock_on_date = stock_exchange.loc[stock_name, "on_date"] if stock_name in stock_exchange.index else None
    stock_exchange = stock_exchange.loc[stock_name, "exchange"] if stock_name in stock_exchange.index else None
    data = read_data(path=os.path.join(DATA_PATH, file_name), file_name=file_name.split(".")[0], on_date=stock_on_date)
    hfq_data = hfq_process(stock_name, data, exchange=stock_exchange)
    del data
    return hfq_data

def save_data_by_column(datas: pd.DataFrame, batch_id: int, num_stocks: int) -> bool:
    """
    pivot the data by column, and save the data by year
    here we check if the data is complete, if not, return True
    """
    flag = False
    for col in COLUMNS_NAMES[1:]:
        cur_df = datas.pivot(index="time", columns="stock_name", values=col)
        if cur_df.shape[1] != num_stocks:
            flag = True
        years = datas["time"].str[:4].unique()
        for year in years:
            cur_df.loc[cur_df.index.str.startswith(year)].to_parquet(f"{PROCESSED_DATA_PATH}/{col}/{col}_{year}_{batch_id}.parquet")
    return flag

def process_by_batch(batch: list, batch_id: int) -> None:
    """
    process the data by batch
    here we check if the data is complete, if not, log the error
    """
    # check if the data is already processed
    try:
        datas = pd.read_parquet(f"{PROCESSED_DATA_PATH}/Original/Original_{batch_id}.parquet")
    except:
        datas = [process_data(file_name) for file_name in tqdm(batch)]
        datas = pd.concat(datas, axis=0).sort_values(by=["time", "stock_name"]).reset_index(drop=True)
        datas.to_parquet(f"{PROCESSED_DATA_PATH}/Original/Original_{batch_id}.parquet")

    num_stocks = len(batch)
    aflag = False
    if datas["stock_name"].nunique() != num_stocks:
        aflag = True
    
    bflag = save_data_by_column(datas, batch_id, num_stocks)

    if aflag or bflag:
        Logger.error(f"Batch {batch_id} has missing data!")
        Logger.error(f"aflag: {aflag}, bflag: {bflag}")

if __name__ == "__main__":
    Logger.info("Start processing data...")
    for i, batch in enumerate(BATCHES):
        # # akshare api may be blocked, you can skip some batches
        # if i <= 9:
        #     continue
        Logger.info("\n" + "+"*50 + f"\nProcessing batch {i}\n" + "+"*50)
        process_by_batch(batch, i)