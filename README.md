# 备注
- 本项目只需要看NEWCODE1这个文件
- 项目推进见飞书文档 https://x1rzpzd8gc3.feishu.cn/wiki/Flozw2hg9iXrzKkf1JLc3YQ1nKc
- OLDCODE里面加了G3GP这个文件，是遗传算法的主要框架，大家可以从这里面看如何添加Terminal和Primitives。


- nan值采用float('nan')

# 变量类型
| Type   | Name           | Tensor_Shape                             |
|--------|----------------|-------------------------------------------|
| TypeA  | day_OHLCV      | (day_len, num_stock)                      |
| TypeB  | minute_OHLCV   | (day_len, num_stock, minute_len=240)      |
| TypeC  | day_mask       | (day_len, num_stock, rolling_day)         |
| TypeD  | minute_mask    | (day_len, num_stock, minute_len=240)      |
| TypeE  | industry       | (day_len, num_stock, industry_num=31)     |
| TypeF  | time_int       | int                                       |
| TypeG  | threshold      | float                                     |





# RPN包
备注：生成方式为half-and-half
需要手动设置的参数，如lookback，fun_lst都在代码上方，不用到class里面改

## RPN_Producer
>通过 producer.run()  执行完整生成流程

| 阶段     | 方法              | 输出特征                                                 |
|----------|-------------------|----------------------------------------------------------|
| Seed     | generate_seed()    | 在保证不改变量纲的情况下完成数据集的参数扩展             |
| Root     | generate_root()    | 去除量纲，形成基础的代理变量                             |
| Branch   | generate_branch()  | 利用量价数据构建基本的代表性子集Mask                     |
| Trunk    | generate_trunk()   | 因子主体，最主要的不可解释的部分——复杂代理变量         |
| Subtree  | generate_subtree() | 选取代表性子集                                           |
| Tree     | generate_tree()    | 构建统计量形成因子                                       |

## RPN_Parser
### 核心功能：实现字符串与AcyclicTree结构的互相转化：
- 初始化传入rpn字符串，可获得：
  - 每部分的缩写与Acyclic_Tree：直接查询对应部分的属性可获得相应的字典
  - 每部分的缩写：调用tree2dict_abbr()
  - 每部分的全称：调用tree2dict_full()
  - 打印树结构：plot.tree()
- 初始化传入Acyclic_Tree:

### 具体函数
- `get_tree_structure(self)`：将rpn转换成Acyclic_Tree
- `get_tree_depth(tree)`：获取Acyclic_Tree的深度
- `get_abbrnsub(ac_tree, substr, flag=0, count=0)`：递归获取树的缩写，形如“['D_ts_std(subtree_ARG0, 5)']”，递归的前提是树的深度已知
- `argnorm(seed_str)`：将seed的D_O替换为ARG

- `tree2str`：获取深度后逐级获得树结构的缩写，
- `tree2lst`：对传入的lst依元素获得缩写
- `plot_tree`：根据self.tree_structure打印树的结构
- `parse_tree()`：根据self.tree_structure生成各级类属性
- `tree2dict_abbr`获取所有级属性的缩写
- `tree2dict_full`获取所有级属性的全称

## RPN_Compiler
### 核心功能：依据数据年份编译输入的算子
- 初始化compiler = RPN_Compiler(year_lst)后直接使用compiler.compile(tree)编译算子数

### 具体方法
-  `extract_op(self, expression)`：提取表达式中的op

- `add_op_class(self, op)`：根据算子的名称返回op的类路径。

- `replace_primities(self, rpn)`：替换rpn中的算子为类路径

- `replace_D_tensor(self, rpn)`：替换rpn中的 `D_tensor`

- `compile_module1(self, rpn, D_tensor: [torch.Tensor])`：日期数据的编译模块

- `compile_module2(self, rpn, D_tensor: [torch.Tensor])`：分钟数据的编译模块

-  `adjust_memorizer(self, deap_primitive, string_memorizer)`

- `compile(self, rpn)`：编译算子
  - 根据算子类型调整`string_memorizer`。
  - 根据 `D_tensor` 调用 `compile_module1` 或 `compile_module2` 执行计算。
- 返回最终的计算结果。




# ToolsGA包
## Acyclic_Tree
### 核心功能：
- 由rpn字符串创建Acyclic_Tree对象, self.node记载嵌套的Acyclic_Tree,self.root_node记载根部的primitive，self.abbreviation记载该树的缩写
### 具体方法
- `parse_deap_str(deap_str, pset)`：将DEAP字符串转换为Acyclic_Tree结构，并构建树的节点信息。
- `build_tree(primitive_tree)`：递归构建Acyclic_Tree树
- `extract_string(s)`：提取最外层括号内的字符串

## change_name方法:为population中的ARG terminal更名为前一树结构/日频分钟频数据

## OrganAbstractClass模块
- Organ类为所有操作符的基类，定义了操作符的输入输出类型，以及操作符的参数
- Organ类中定义了add_primitive_byfunclist方法，用于添加操作符
- Organ类中定义了add_constant_terminal方法，用于添加常量
- Organ类中定义了add_float_terminal方法，用于添加浮点数
- Organ类中定义了generate_toolbox方法，用于生成工具箱

>具体的模块用于加入特异参数与primitive集
## Seed模块
- MV_Seed, MP_Seed: 添加OP_BF2B
- DV_Seed, DP_Seed: 添加OP_AF2A
## Root模块
- MV_Root, MP_Root: 添加OP_B2B，OP_BB2B
- DV_Root, DP_Root: 添加OP_AF2A，OP_AA2A
## Branch模块
- M_Branch_MP2D, M_Branch_MV2D：添加OP_BF2D， OP_B2D
- M_Branch_MVDV2D, M_Branch_MPDP2D：添加OP_BA2D
## Subtree模块
- SubtreeWithMask：添加OP_BD2A，OP_BBD2A
- SubtreeNoMask：添加OP_B2A，OP_BB2A
## Tree模块
- 添加OP_AF2A
# ToolsGA包


## DataReader模块
### 基本介绍：
  - 核心功能实现——数据读取：初始化任意一个reader，用`get_Day_data()`,`get_Minute_data()`,`get_Barra()`,即可获取数据；Mmap还内置了`get_Minute_data_daily()`,`get_Minute_data_daily()`方法；

  - mmap文件保存：按下图所示结构建立对应的文件夹，初始化`MmapReader(download=True)`，调用`save_daily_data()`,`save_minute_data()`,`save_barra_data()`即可

  - 数据的形状与筛选：大部分数据的初始形状为(`self.TradingDate`,`self.StockCodes`)，即(4917,5601)，读取时已按`MutualStockCodes`进行筛选，在后面仅仅需要用`yearlst`从`self.TradingDate`中筛选对应年份的数据即可

  - 日期的筛选方式：日期筛选的方式是用`BasicReader`中的`get_index`方法返回`self.TradingDate`中对应年份的index，处理tensor时只要tensor的长度与`self.TradingDate`一致便可直接`tensor[index]`筛选；
    - 反例：`Barra`的日期范围与`self.TradingDate`不一致，因此不可用index直接筛选；操作中用配套的`dict`筛选与传入的`year_lst`对应的index
  
  - 数据明细
    - 时间范围：原始数据基本与`self.TradingDate`的时间范围相同，但与`Barra`数据不相同，本模块仅处理2016-2023年的数据
    - `Barra`共有41个风格因子

  - 数据清洗要求：在`ParquetReader`读取时，从`mat`中读取的初始数据还要根据能否交易，用`adjust`方法进行筛选；但在`MmapReader`中所有读取数据均已清洗完毕，不需要考虑清洗问题；

  - 两种不同接口方法的调用：`Interface`类和`BasicReader`类储存了一系列数据处理方法：`Interface`类中均为静态方法，不用实例化直接调用即可；`BasicReader`中的方法需要实例化，一般除输入数据外还需要调用交易日期与股票代码，请自行判断调用。
    - 例如：分钟数据的df与tensor格式的转换中,df到tensor仅仅需要调整和补充df的格式，因此不需要其他信息可直接调用，可在`Interface`中找到，而tensor到df则需要用到交易日期与股票代码，因此在`BasicReader`中,  

### 数据结构
```
Data
├─ Daily
├─ Minute
├─ Mmap
│  ├─ Barra
│  ├─ Daily
│  ├─ data_shape.json
│  └─ Minute
│     ├─ M_O
│     ├─ M_H
│     ├─ M_L
│     ├─ M_C
├─ barra.pt
├─ dict.pt
├─ MutualStockCodes.parquet
└─ README.md
```
### 具体介绍
#### Interface类
- `df2tensor(df)`：
  - 输入`(date_minute,stock)`的dataframe，补充`minute_len`，输出形状为`(day_num,stock_num,minute_len)`的tensor

- `generate_trading_time(date:str)->pd.DatetimeIndex:`：
  - 输入`%Y-%m-%d`格式的str，生成对应`pd.DatetimeIndex`格式的交易事件，长度为242

- `fill_full_day(day_data:pd.DataFrame)->pd.DataFrame:`：
  - 将某日的`pd.DataFrame`的index补充为242个交易时间

- `get_all_files(folder_path, pattern:re.Pattern)`：
  - 输入文件夹路径与给定的命名模式，返回符合命名模式的命名文件

- `get_pct_change(close)：计算close的百分比变化 
- `get_labels(d_o,d_c, freq=5)`：计算给定freq的close与前日open比

#### BasicReader类
- 初始化：
  - `self.Mutual StockCodes`：筛选arr/tensor股票代码的mask，用`[:,self.Mutual StockCodes]`直接筛选，形状为`(5601,)`
  - `self.TradingDate`：范围内的所有交易日期，形状为`(4917,)`
  - `self.StockCodes`：筛选出来的股票代码，用作df的columns和Mmap保存时的数据宽度，形状为`(5483,)`

- `get_daylist(self,year_lst:list[int]) -> list[pd.Timestamp]`
  - 返回`self.TradingDate`中对应年份的值；

- `get_index(self, year_lst:list[int])->pd.DatetimeIndex`：
  - 返回`self.TradingDate`中对应年份的index；

- `tensor2df(self, tensor, year_list:list[int])->pd.DataFrame:`：
  - 将形状为`(self.TradingDate,self.StockCodes)`的tensor转换为对应年份的dataframe

- `adjust(self,data,year_lst,device='cpu')-> torch.Tensor`：
  - 传入np.array或torch.tensor的data, index筛选data并根据clean清洗；需要传入year_lst数据指示该数据的年份以便clean与之对应
  - 考虑到data可能是np.array,无device属性，函数中设有device参数

    

#### ParquetReader类：...
#### MmapReader类：...
  - 保存日期，分钟，barra的事件约为两分钟，七分钟，五秒
  - `self.data_shape`字典以str格式的年份为键存储mmap读取的年份数据格式，主要是第0维的日期有所差别






## Backtest模块
### FactorTest
#### 初始化
| Parameter    | Type       | Description                          |
|--------------|------------|--------------------------------------|
| factor       | torch.Tensor | Target factor matrix (T x N)         |
| yearlist     | list[int]  | Analysis years (e.g., [2019,2020])   |
| bins_num     | int        | Number of stratification groups      |
| period_num   | int        | Annual trading periods (default 252) |
| factor_name  | str        | Factor identifier (optional)         |


#### 方法一览

- `get_stratified_return()`: 分层收益计算。
- `get_rank_IC()`: 因子Rank IC序列。
- `get_turnover()`: 组合换手率分析。
- `get_long_short_sharpe()`: 多空夏普。
- `get_mean_over_short_sharpe()`: 均值-空头夏普。
- `get_rank_ICIR()`: IC信息比率。
- `get_turnover_punishment()`: 换手惩罚后夏普。

- `barra_test()`: Barra风格因子暴露。
- `pv_neutra()`: 市值中性化。
- `industry_neutra()`: 行业中性化。

- `plot(output_path=None)`: 生成完整分析报告。
- `plot_stratified_rtn()`: 分层收益曲线。
- `plot_long_short()`: 多空组合对比。

### class: GroupTest

## GA_tools模块
### CustomError
- 自定义异常类，用于抛出工具中的特定错误。
  
### 类型定义

- **注意**：`TypeF` 和 `TypeG` 重载了 `__new__` 方法，强制将数据类型设置为 `torch.int` 和 `torch.float32`，确保数据类型一致性。

### chaotic_map

- 该类包含多个静态方法，用于执行不同的混沌映射算法。

| 方法名                        | 数学表达式                                                 | 参数范围           | 注意事项                           |
|-------------------------------|-----------------------------------------------------------|--------------------|------------------------------------|
| **chebyshev_map(x, a=4)**      | \( \cos(a \cdot \arccos(x)) \)                            | \([-1, 1]\)         | 过于接近0/1时不会更新               |
| **circle_map(x, a=0.5, b=2.2)**| \( (x + a - \frac{b}{2\pi} \sin(2\pi x)) \mod 1 \)       | \([0, 1]\)          | 无                                |
| **iterative_map(x, a=0.7)**    | \( \sin\left(a \cdot \frac{\pi}{x}\right) \)             | \([-1, 1]\)         | 无                                |
| **logistic_map(x, a=4)**       | \( a \cdot x \cdot (1 - x) \)                             | \([0, 1]\)          | 无                                |
| **piecewise_map(x, d=0.3)**    | 分段函数                                                 | \([0, 1]\)          | 过于接近0/1时不会更新               |
| **sine_map(x, a=4)**           | \( \frac{a}{4} \sin(\pi x) \)                             | \([0, 1]\)          | 可能数值溢出，过于接近0/1时不会更新 |
| **singer_map(x, a=1.07)**      | \( a \cdot (7.86x - 23.31x^2 + 28.75x^3 - 13.302875x^4) \) | \([0, 1]\)          | 过于接近1时可能会数值溢出         |
| **tent_map(x, a=0.4)**         | \( \text{if } x < a \text{ then } \frac{x}{a} \text{ else } \frac{1-x}{1-a} \) | \([0, 1]\)          | 无                                |
| **spm_map(x, eta=0.4, mu=0.3)**| 分段函数                                                 | \([0, 1]\)          | 无                                |
| **tent_logistic_cosine_map(x, r=0.7)** | 混合函数                                             | \([0, 1]\)          | 可能数值溢出                       |
| **sine_tent_cosine_map(x, r=0.7)**  | 混合函数                                             | \([0, 1]\)          | 可能数值溢出                       |
| **logistic_sine_cosine_map(x, r=0.7)** | 混合函数                                             | \([0, 1]\)          | 可能数值溢出                       |
| **cubic_map(x, a=2.595)**      | \( a \cdot x \cdot (1 - x^2) \)                           | \([0, 1]\)          | 过于接近0/1时可能不会更新          |
| **logistic_tent_map(x, r=0.3)**| 混合函数                                                 | \([0, 1]\)          | 过于接近0/1时可能不会更新          |
| **bernoulli_map(x, a=0.4)**    | 分段函数                                                 | \([0, 1]\)          | 过于接近1时不会更新               |

- **注意**：多个映射函数会在输入值接近0或1时不更新，或可能出现数值溢出的情况。请根据具体应用场景谨慎使用。

### change_name 
- **作用**：该函数用于重命名遗传算法公式中的参数。
- **输入**：
  - `formula_list`：待修改的公式列表，每个公式包含多个节点。
  - `substitute_list`：包含新参数名的列表，用于替换公式中的旧参数名。
- **输出**：
  - `renamed_individual_code`：包含重命名后的公式代码。
  - `renamed_individual_str`：包含重命名后的公式字符串。

# GA
## FIS `FactorIntoStorage` 
- `__init__(self, new_factor: list)`：初始化方法，接收一个新的因子列表 `new_factor`，并设定存储路径。

- `get_exist_factor(self)`：从指定的存储路径读取已存在的因子，返回一个因子列表。

- `greedy_algo(self)`：实现贪心算法从已有因子中筛选与新因子相似度小于0.6的因子，并将其添加到 `greedy_factor` 列表中。

- `calculate_similarity(factor1: gp.PrimitiveTree, factor2: gp.PrimitiveTree)`：计算两个因子（基因树）的相似度，返回一个介于0到1之间的相似度值。

- `build_factor_graph(self, factors_lst: list)`：根据因子列表构建因子图，图的边表示因子之间相似度小于0.6时的关系。

- `bk_algo(self)`：使用Bron-Kerbosch算法（BK算法）从因子图中找到最大团，并返回在新因子列表中存在的因子。

- `factor_evaluating(self)`：对选择的因子进行评估并绘制评估结果图。

- `store_factors(self)`：将选择的因子存储到指定路径的Excel文件中。

# OP包
注意OP更改后要去RPN里面看一看对应的func_lst需不需要添加
## ToA
`OP_A2A`
- `D_at_abs(x)`：取绝对值。
- `D_cs_rank(x)`：计算截面分位数。
- `D_cs_scale(x)`：标准化截面最大最小值。
- `D_cs_zscore(x)`：z-score标准化。
- `D_cs_harmonic_mean(x)`：计算调和平均数。
- `D_cs_demean(x)`：去除均值。
- `D_cs_winsor(x, limit=[0.05, 0.95])`：尾部磨平操作。

`OP_AE2A`
- `D_cs_demean_industry(day_OHLCV, industry)`：行业均值计算。
- `D_cs_industry_neutra(day_OHLCV, industry)`：行业中性化。

`OP_AA2A`
- `D_cs_norm_spread(x, y)`：计算截面规范化差异。
- `D_cs_cut(x, y)`：根据符号处理数据。
- `D_cs_regress_res(x, y)`：截面回归取残差。
- `D_at_add(x, y)`：加法操作。
- `D_at_sub(x, y)`：减法操作。
- `D_at_div(x, y)`：除法操作。
- `D_at_prod(x, y)`：乘法操作。
- `D_at_mean(x, y)`：均值计算。

`OP_AG2A`
- `D_cs_edge_flip(x, thresh)`：对数据进行边缘翻转，调整特定阈值。

`OP_AAF2A`
- `D_ts_corr(x, y, d)`：计算时间序列的相关性。
- `D_ts_rankcorr(x, y, d)`：计算时间序列的秩相关性。
- `D_ts_regress_res(x, y, lookback)`：计算时间序列回归残差。
- `D_ts_weight_mean(x, y, lookback)`：时间序列加权平均。

`OP_AF2A`
- `D_ts_max(x, lookback)`：计算时间序列最大值。
- `D_ts_min(x, lookback)`：计算时间序列最小值。
- `D_ts_delay(x, d)`：延迟操作。
- `D_ts_delta(x, d)`：计算时间序列的差值。
- `D_ts_pctchg(x, lookback)`：计算百分比变化。
- `D_ts_mean(x, lookback)`：计算时间序列均值。
- `D_ts_harmonic_mean(x, lookback)`：计算时间序列的调和平均。
- `D_ts_std(x, lookback)`：计算时间序列的标准差。
- `D_ts_max_to_min(x, lookback)`：计算最大值与最小值之比。
## ToB模块
### `OP_B2B`  
- `M_ignore_wobble(M_tensor, window_size=5)`：将开盘前五分钟和收盘前五分钟的数据变为 `NaN`。  
- `M_cs_zscore(x)`：计算日内数据的 z-score 标准化。  
- `M_cs_rank(x)`：对日内数据进行排名并标准化为 [0, 1] 范围内。  
- `M_cs_scale(x)`：进行日内数据的最大值-最小值标准化。  
- `M_cs_demean(x)`：计算数据与均值的距离。  
- `M_cs_winsor(x, limit=[0.05, 0.95])`：对数据进行 Winsorization，限制上下百分位。  
- `M_at_abs(M_tensor)`：返回张量的绝对值。  

### `OP_BB2B`  
- `M_at_add(x, y)`：返回两个张量元素逐项相加的结果。  
- `M_at_sub(x, y)`：返回两个张量元素逐项相减的结果。  
- `M_at_div(x, y)`：返回两个张量元素逐项相除的结果，对除数为零的情况返回 `NaN`。  
- `M_at_sign(x)`：返回张量每个元素的符号。  
- `M_cs_cut(x, y)`：根据均值的符号调整数据。  
- `M_cs_umr(x, y)`：计算数据与均值的乘积。  
- `M_at_prod(d_tensor_x, d_tensor_y)`：返回两个张量元素逐项相乘的结果。  
- `M_cs_norm_spread(x, y)`：计算两个张量的标准化差异。  

### `OP_BA2B`  
- `M_toD_standard(M_tensor, D_tensor)`：按标准化公式将 B 维度数据除以 D 维度数据。  

### `OP_BG2B`  
- `M_cs_edge_flip(x, thresh)`：根据排名与给定阈值反转数据的边缘部分。  

### `OP_BF2B`  
- `M_ts_delay(x, d)`：延迟 `x` 张量中的数据，向前或向后移动。  
- `M_ts_pctchg(x, lookback)`：计算数据的百分比变化。  
- `M_ts_delta(x, lookback)`：计算数据与延迟数据的差异。  
- `M_ts_mean_xx_neighbor(m_tensor, neighbor_range, orit=-1)`：计算指定邻域范围内的均值。  
- `M_ts_std_xx_neighbor(m_tensor, neighbor_range, orit=-1)`：计算指定邻域范围内的标准差。  
- `M_ts_prod_xx_neighbor(m_tensor, neighbor_range, orit=-1)`：计算指定邻域范围内的乘积。  
## ToC模块
### `OP_AF2C`
- `Dmask_min(x, lookback)`：获取过去`lookback`天中最小的1/4天的数据。
- `Dmask_max(x, lookback)`：获取过去`lookback`天中最大的1/4天的数据。
- `Dmask_middle(x, lookback)`：获取过去`lookback`天中间1/2天的数据。
- `Dmask_mean_plus_std(x, lookback)`：对过去`lookback`天数据进行标准化，取大于均值+标准差部分的数据。
- `Dmask_mean_sub_std(x, lookback)`：对过去`lookback`天数据进行标准化，取小于均值-标准差部分的数据。
## ToD模块
### `OP_B2D`
- `Mmask_min(m_tensor)`：返回日内最小1/4部分的掩码。
- `Mmask_max(m_tensor)`：返回日内最大1/4部分的掩码。
- `Mmask_middle(m_tensor)`：返回日内中间1/2部分的掩码。
- `Mmask_min_to_max(m_tensor)`：返回日内最大值和最小值中间的部分的掩码。
- `Mmask_mean_plus_std(m_tensor)`：生成大于均值加1倍标准差的掩码。
- `Mmask_mean_sub_std(m_tensor)`：生成小于均值减1倍标准差的掩码。
- `Mmask_1h_after_open(m_tensor)`：取开盘后的第一个小时的掩码。
- `Mmask_1h_before_close(m_tensor)`：取收盘前的第一个小时的掩码。
- `Mmask_2h_middle(m_tensor)`：取中间的两个小时的掩码。
- `Mmask_morning(m_tensor)`：取早上的两个小时的掩码。
- `Mmask_afternoon(m_tensor)`：取下午的两个小时的掩码。

### `OP_BA2D`
- `Mmask_day_plus(m_tensor, d_tensor)`：返回大于日频数据的部分掩码。
- `Mmask_day_sub(m_tensor, d_tensor)`：返回小于日频数据的部分掩码。

### `OP_BF2D`
- `Mmask_rolling_plus(m_tensor, lookback)`：返回大于lookback期内最大日均值的部分掩码。
- `Mmask_rolling_sub(m_tensor, lookback)`：返回小于lookback期内最小日均值的部分掩码。

### `OP_DD2D`
- `Mmask_and(m_mask_x, m_mask_y)`：掩码的并操作
- `Mmask_or(m_mask_x, m_mask_y)`：掩码的或操作

## Others模块
### `OP_Closure`
- `id_industry(industry)`：返回行业数据。
- `id_int(int)`：返回整数数据。
- `id_float(thresh)`：返回浮动阈值。
- `id_tensor(tensor)`：返回张量数据。

### `OP_Basic`
- `nanmean(tensor, dim=-1)`：计算给定维度上的均值，忽略NaN值。
- `nanstd(x, dim=-1)`：计算给定维度上的标准差，忽略NaN值。
- `corrwith(tensor1, tensor2, dim=-1)`：计算两个张量在指定维度上的相关性。
- `rank_corrwith(tensor1, tensor2, dim=-1)`：计算两个张量在指定维度上的秩相关性。
- `multi_regress(y, x_s, dim=-1)`：进行多元线性回归，计算回归系数、截距及残差。
- `regress(y, x_s, dim=-1)`：根据维度进行回归分析。
