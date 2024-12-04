import mmap
import torch
import os

class DataReader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.cols = ["open", "high", "low", "close", "volume", "amount"]

    def _load_with_mmap(self, file_path: str) -> torch.Tensor:
        """
        load .pt file with mmap
        """
        with open(file_path, "rb") as f:
            # read data with mmap
            mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # load tensor data
            tensor_data = torch.load(mmapped_file)
            mmapped_file.close()
        return tensor_data

    def read_data_by_col(self, col: str, year_lst: list[int]) -> torch.Tensor:
        """
        load data by column and year list with mmap
        """
        data = []
        for year in year_lst:
            file_path = f"{self.data_path}/{col}/{col}_{year}.pt"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            # load data with mmap
            data.append(self._load_with_mmap(file_path))
        # concatenate date in the day dimension
        return torch.cat(data, dim=1)

if __name__ == "__main__":
    data_path = "./results"
    data_reader = DataReader(data_path)
    open_data = data_reader.read_data_by_col("open", [2016, 2017])
    print(open_data.shape)