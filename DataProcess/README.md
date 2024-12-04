# Minute-Freq Data Processing
## 运行方式
1. 修改 `ReshapeData.py` 中的 `BATCH_NUMBER`，决定每个批次处理多少只股票，我这边大概 `100/batch`。
2. `python ReshapeData.py`
3. `python ConvertTensor.py`

## 文件结构说明
1. 个股数据存放到 `./data` 文件夹下，个股交易所、上市日期数据存放在 `./docs/StocksExchange.xlsx`，从同花顺下载的，部分股票缺失。
   1. 没找到原因，但缺失的股票全部已退市，这个部分只涉及后复权因子下载和去除上市不满三个月的数据。
   2. 后复权因子如果下载不到，就不做后复权。
   3. 上市日期如果缺失，就不去除，取 `2016` 年之后的数据
2. `python ReshapeData.py` 生成的文件位于 `./hfq_factors`（后复权因子）和 `./processed_data/{col}`，每个 `{col}` 子文件夹下存放每年、每个 `batch` 的数据。
3. `python ConvertTensor.py` 生成最终的处理结果。如果你需要存 `.parquet` 文件，取消第 `81` 行代码的注释，文件会存放到 `./processed_data` 下。 `.pt` 文件会存放到 `./results` 文件夹下。
4. `DataReader.py` 使用 `mmap` 读取 `.pt` 文件，需要指定 `.pt` 文件存放目录，如 `./results`。

一些后复权因子文件位于 [nju box](https://box.nju.edu.cn/f/5616d23ad9f14154aecf/).