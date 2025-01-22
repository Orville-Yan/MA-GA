import sys
sys.path.append('..')

from OP import *
from ToolsGA.Data_tools import DailyDataReader, MinuteDataReader
import pandas as pd
import torch

class ParquetReader:
    def __init__(
        self, 
        DailyDataPath: str = "C:/个股日频数据 20240329",
        MinuteDataPath: str = "D:/MA-GA-Data/Minute_Data",
        BarraPath: str = "C:/Users/74989/Desktop/GP/GP/barra.pt",
        DictPath: str = "C:/Users/74989/Desktop/GP/GP/dict.pt",
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

class MmapReader:
    def __init__(self):
        self.origin_daily='../../Data/DailyData'
        self.origin_minute='../../Data/MinuteData'
        self.years=range(2017,2018)#设定将哪些年的数据修改为mmap格式，最好不要很多年一起修改
        self.output_daily='..'#mmap文件存放的路径
        self.output_minute='..'

    def save_daily_to_mmap(self,data, dtype=np.float32):
        num_rows, num_cols = data.shape
        with open(self.origin_daily, 'w+b') as f:
            mmap = np.memmap(f, dtype=dtype, mode='w+', shape=(num_rows, num_cols))
            mmap[:] = data

    def save_minute_to_mmap(self,data, dtype=np.float32):
        num_rows, num_cols , num_days= data.shape
        with open(self.origin_minute, 'w+b') as f:
            mmap = np.memmap(f, dtype=dtype, mode='w+', shape=(num_rows, num_cols,num_days))
            mmap[:] = data

    def save_daily(self):
        data_reader = ParquetReader()
        years = self.years
        os.makedirs(self.output_daily, exist_ok=True)

    # 定义数据的名称和对应的索引
        data_names = ['DO', 'DH', 'DL', 'DC', 'DV']

        for year in years:
            data = data_reader.get_Day_data([year])
            for i, name in enumerate(data_names):
                tensor = data[i]  # 获取当前Tensor
                file_path = os.path.join(self.output_daily, f'{name}{year}.mmap')  # 构造文件路径
                self.save_daily_to_mmap(file_path, tensor)  # 保存为Memory Map文件
                print(f"Saved {name} data for year {year} to {file_path}")

    def save_minute(self):
        data_reader = ParquetReader()
        years = self.years
        os.makedirs(self.output_minute, exist_ok=True)
        data_names = ['MO', 'MH', 'ML', 'MC', 'MV']

        for year in years:
            data = data_reader.get_Minute_data([year])
            for i, name in enumerate(data_names):
                tensor = data[i]  
                file_path = os.path.join(self.output_minute, f'{name}{year}.mmap')  
                self.save_minute_to_mmap(file_path, tensor)  
                print(f"Saved {name} data for year {year} to {file_path}")
    
    def mmap_readDaily(self,file_path):
        # 定义数据的形状和数据类型
        num_rows = 244
        num_cols = 5601
        dtype = np.float32

        # 读取Memory Map文件
        mmap_read = np.memmap(file_path, dtype=dtype, mode='r', shape=(num_rows, num_cols))

        tensor_read = torch.from_numpy(mmap_read)
        print(f"Tensor的形状: {tensor_read.shape}")
        return tensor_read

    def mmap_readMinute(self,file_path):
        num_rows = 5528
        num_cols = 244
        num_days=242
        dtype = np.float32
        mmap_read = np.memmap(file_path, dtype=dtype, mode='r', shape=(num_rows, num_cols,num_days))
        tensor_read = torch.from_numpy(mmap_read)
        return tensor_read


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

    data_reader = ParquetReader()

    DO, DH, DL, DC, DV = data_reader.get_Day_data([2016, 2017])
    print(DO.shape, DH.shape, DL.shape, DC.shape, DV.shape)

    MO, MH, ML, MC, MV = data_reader.get_Minute_data([2016, 2017])
    print(MO.shape, MH.shape, ML.shape, MC.shape, MV.shape)

    barra = data_reader.get_barra([2016, 2017])
    print(barra.shape)

    interval_rtn = data_reader.get_labels([2016, 2017])
    print(interval_rtn.shape)
