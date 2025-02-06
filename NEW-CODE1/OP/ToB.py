import sys

sys.path.append('..')

from OP.Others import OP_Basic
import torch

OPclass_name_2B = ['OP_B2B', 'OP_BB2B', 'OP_BA2B', 'OP_BG2B',
                   'OP_BF2B']


class OP_B2B:
    def __init__(self):
        self.func_list = ['M_ignore_wobble', 'M_cs_rank', 'M_cs_scale', 'M_cs_zscore', 'M_at_abs', 'M_cs_demean',
                          'M_cs_winsor',
                          ]

    @staticmethod
    def M_ignore_wobble(M_tensor, window_size=5):
        """
        将开盘前五分钟和收盘前五分钟的数据变成nan。

        参数:
        M_tensor (torch.Tensor): 输入的张量，假设其形状为 (num_stock, day_len, minute_len)。
        window_size (int): 忽略的窗口大小，单位为分钟。

        返回:
        torch.Tensor: 处理后的张量。
        """
        num_stock, day_len, minute_len = M_tensor.shape

        # 创建一个与M_tensor形状相同的掩码，初始值为1
        mask = torch.ones_like(M_tensor, dtype=torch.bool)

        # 开盘前五分钟为nan
        mask[:, :, :window_size] = False
        # 收盘前五分钟为nan
        mask[:, :, -window_size:] = False

        # 将掩码为False的位置设置为nan
        return torch.where(mask, M_tensor, torch.tensor(float('nan'), device=M_tensor.device))

    @staticmethod
    def M_cs_zscore(x):
        # 计算每个股票每天的均值和标准差
        x_mean = OP_Basic.nanmean(x, dim=1).unsqueeze(1)
        x_std = OP_Basic.nanstd(x, dim=1).unsqueeze(1)
        zscore = (x - x_mean) / x_std
        s = zscore
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)


    @staticmethod
    def M_cs_rank(x):
        """
        将日内按照排序[0,1]标准化。

        参数:
        M_tensor (torch.Tensor): 输入的张量，假设其形状为 (num_stock, day_len, minute_len)。

        返回:
        torch.Tensor: 排名标准化后的张量。
        """
        # 计算每个元素在每个交易日中的排名
        mask = ~torch.isnan(x)
        data_no_nan = torch.where(mask, x, torch.full_like(x, float('inf')))
        ranks = torch.argsort(torch.argsort(data_no_nan, dim=1), dim=1).float()  # 首先排序，然后取序数
        quantiles = ranks / torch.sum(mask, 1).unsqueeze(1)  # 计算分位数
        s = torch.where(mask, quantiles, torch.tensor(float('nan')))

        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)


    @staticmethod
    def M_cs_scale(x):
        """
        日内最大值最小值标准化。

        参数:
        M_tensor (torch.Tensor): 输入的张量，假设其形状为 (num_stock, day_len, minute_len)。

        返回:
        torch.Tensor: 最大值最小值标准化后的张量。
        """
        mask = ~torch.isnan(x)
        data_no_nan = torch.where(mask, x, torch.full_like(x, 0))
        max = torch.max(torch.where(mask, x, torch.full_like(x, float('-inf'))), dim=1)[0].unsqueeze(dim=1)
        min = torch.min(torch.where(mask, x, torch.full_like(x, float('inf'))), dim=1)[0].unsqueeze(dim=1)
        sacled_data_no_nan = (data_no_nan - min) / (max - min)  # 公式核心
        scaled_data = torch.where(mask, sacled_data_no_nan, torch.tensor(float('nan')))
        s = scaled_data + 1
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)



    @staticmethod
    def M_cs_demean(x):
        """
        计算日内到均值的距离。
        """
        return OP_B2B.M_at_abs(x - OP_Basic.nanmean(x, dim=1).unsqueeze(1))

    @staticmethod
    def M_cs_winsor(x, limit=[0.05, 0.95]):
        rank = OP_B2B.D_cs_rank(x)
        min_limit = torch.where(rank >= limit[0], rank, float('nan'))
        max_limit = torch.where(rank <= limit[1], rank, float('nan'))
        mask = (~torch.isnan(min_limit)) & (~torch.isnan(max_limit))
        max = torch.max(torch.where(mask, x, torch.full_like(x, float('-inf'))), dim=1)[0].unsqueeze(dim=1)
        min = torch.min(torch.where(mask, x, torch.full_like(x, float('inf'))), dim=1)[0].unsqueeze(dim=1)
        winsored_min = torch.where(rank <= limit[0], min, x)  # 最小值变化
        winsored_max = torch.where(rank >= limit[1], max, winsored_min)
        x_with_nan = torch.where(~torch.isnan(x), winsored_max, float('nan'))
        return x_with_nan

    @staticmethod
    def M_at_abs(M_tensor):
        return torch.abs(M_tensor)


class OP_BB2B:
    def __init__(self):
        self.func_list = ['M_at_add', 'M_at_sub', 'M_at_div', 'M_at_sign', 'M_cs_cut', 'M_cs_umr', 'M_at_prod',
                          'M_cs_norm_spread']

    @staticmethod
    def M_at_add(x, y):
        return torch.add(x, y)

    @staticmethod
    def M_at_sub(x, y):
        return torch.sub(x, y)

    @staticmethod
    def M_at_div(x, y):
        zero_mask = (y == 0)
        result = torch.div(x, y)
        result[zero_mask] = torch.nan
        return result

    @staticmethod
    def M_at_sign(x):
        mask = ~torch.isnan(x)
        x_no_nan = torch.where(mask, x, 0)
        sign = torch.sign(x_no_nan)
        return torch.where(mask, sign, float('nan'))

    @staticmethod
    def M_cs_cut(x, y):
        sign = OP_BB2B.at_sign(x - OP_Basic.nanmean(x, dim=1).unsqueeze(1))
        return sign * y

    @staticmethod
    def M_cs_umr(x, y):
        s = torch.multiply(x - OP_Basic.nanmean(y, dim=1).unsqueeze(1), y)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def M_at_prod(d_tensor_x, d_tensor_y):
        return torch.mul(d_tensor_x, d_tensor_y)

    @staticmethod
    def M_cs_norm_spread(x, y):
        s = (x - y) / (torch.abs(x) + torch.abs(y))
        inf_mask = torch.isinf(s)
        return torch.where(inf_mask, torch.tensor(float('nan'), device=s.device), s)


class OP_BA2B:  # B*A-B
    def __init__(self):
        self.func_list = ['M_toD_standard']  # B*A-B

    @staticmethod
    def M_toD_standard(M_tensor, D_tensor):
        D_tensor_adjusted = D_tensor.unsqueeze(-1) 
        return M_tensor / D_tensor_adjusted  


class OP_BG2B:  # B*G-B
    def __init__(self):
        self.func_list = ['M_cs_edge_flip']  # B*G-B

    @staticmethod
    def M_cs_edge_flip(x, thresh):
        rank = OP_B2B.D_cs_rank(x)
        if thresh < 0.3:
            edge_fliped = torch.where((rank < thresh) | (rank > 1 - thresh), -x, x)
        elif thresh > 0.7:
            edge_fliped = torch.where((rank < 1 - thresh) | (rank > thresh), -x, x)
        else:
            edge_fliped = torch.where((rank < thresh) | (rank > 1 - thresh), x, -x)

        return edge_fliped

class OP_BF2B:  # B*F-B
    def __init__(self):
        self.func_list = [
            'M_ts_delta',
            'M_ts_mean_left_neighbor',
            'M_ts_mean_mid_neighbor',
            'M_ts_mean_right_neighbor',
            'M_ts_std_left_neighbor',
            'M_ts_std_mid_neighbor',
            'M_ts_std_right_neighbor',
            'M_ts_product_left_neighbor',
            'M_ts_product_mid_neighbor',
            'M_ts_product_right_neighbor',
            'M_ts_pctchg',
            'M_ts_delay',
        ]  # B*F-B

    @staticmethod
    def M_ts_delay(x, d):
        if d > 0:
            new_tensor = torch.full(x.shape, float('nan'))
            new_tensor[d:, :] = x[:-d, :]
            return new_tensor
        elif d == 0:
            return x
        else:
            new_tensor = torch.full(x.shape, float('nan'))
            new_tensor[:d, :] = x[-d:, :]
            return new_tensor

    @staticmethod
    def M_ts_pctchg(x, lookback):
        s = (x - OP_BF2B.M_ts_delay(x, lookback)) / OP_BF2B.M_ts_delay(x, lookback)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def M_ts_delta(x, lookback):
        return x - OP_BF2B.D_ts_delay(x, lookback)

    @staticmethod
    def M_ts_mean_right_neighbor(m_tensor, neighbor_range):
        window_size = neighbor_range
        unfolded = m_tensor.unfold(dimension=-1, size=window_size, step=1)
        window_mean = OP_Basic.nanmean(unfolded)
        result_tensor = torch.full_like(m_tensor, float('nan'))
        valid_length = m_tensor.size(-1) - window_size
        result_tensor[..., :valid_length] = window_mean[..., 1 : 1 + valid_length]
        return result_tensor
    
    @staticmethod
    def M_ts_mean_left_neighbor(m_tensor, neighbor_range):
        window_size = neighbor_range
        unfolded = m_tensor.unfold(dimension=-1, size=window_size, step=1)
        window_mean = OP_Basic.nanmean(unfolded)
        result_tensor = torch.full_like(m_tensor, float('nan'))
        valid_length = m_tensor.size(-1) - window_size
        result_tensor[..., window_size:] = window_mean[..., :valid_length]
        return result_tensor

    @staticmethod
    def M_ts_mean_mid_neighbor(m_tensor, neighbor_range):
        window_size = neighbor_range
        offset = (window_size - 1) // 2  
        unfolded = m_tensor.unfold(dimension=-1, size=window_size, step=1)
        window_mean = OP_Basic.nanmean(unfolded)
        result_tensor = torch.full_like(m_tensor, float('nan'))
        result_tensor[..., offset : offset + unfolded.size(-1)] = window_mean
        return result_tensor
    
    @staticmethod
    def M_ts_std_right_neighbor(m_tensor, neighbor_range):
        window_size = neighbor_range
        unfolded = m_tensor.unfold(dimension=-1, size=window_size, step=1)
        window_std = OP_Basic.nanstd(unfolded)
        result_tensor = torch.full_like(m_tensor, float('nan'))
        valid_length = m_tensor.size(-1) - window_size
        result_tensor[..., :valid_length] = window_std[..., 1 : 1 + valid_length]
        return result_tensor
    
    @staticmethod
    def M_ts_std_left_neighbor(m_tensor, neighbor_range):
        window_size = neighbor_range
        unfolded = m_tensor.unfold(dimension=-1, size=window_size, step=1)
        window_std = OP_Basic.nanstd(unfolded)
        result_tensor = torch.full_like(m_tensor, float('nan'))
        valid_length = m_tensor.size(-1) - window_size
        result_tensor[..., window_size:] = window_std[..., :valid_length]
        return result_tensor

    @staticmethod
    def M_ts_std_mid_neighbor(m_tensor, neighbor_range):
        window_size = neighbor_range
        offset = (window_size - 1) // 2  
        unfolded = m_tensor.unfold(dimension=-1, size=window_size, step=1)
        window_std = OP_Basic.nanstd(unfolded)
        result_tensor = torch.full_like(m_tensor, float('nan'))
        result_tensor[..., offset : offset + unfolded.size(-1)] = window_std
        return result_tensor

    @staticmethod
    def M_ts_product_right_neighbor(m_tensor, neighbor_range):
        window_size = neighbor_range
        unfolded = m_tensor.unfold(dimension=-1, size=window_size, step=1)
        window_prod = torch.prod(unfolded, dim=-1)
        result_tensor = torch.full_like(m_tensor, float('nan'))
        valid_length = m_tensor.size(-1) - window_size
        result_tensor[..., :valid_length] = window_prod[..., 1 : 1 + valid_length]
        return result_tensor
    
    @staticmethod
    def M_ts_product_left_neighbor(m_tensor, neighbor_range):
        window_size = neighbor_range
        unfolded = m_tensor.unfold(dimension=-1, size=window_size, step=1)
        window_prod = torch.prod(unfolded, dim=-1)
        result_tensor = torch.full_like(m_tensor, float('nan'))
        valid_length = m_tensor.size(-1) - window_size
        result_tensor[..., window_size:] = window_prod[..., :valid_length]
        return result_tensor

    @staticmethod
    def M_ts_product_mid_neighbor(m_tensor, neighbor_range):
        window_size = neighbor_range
        offset = (window_size - 1) // 2  
        unfolded = m_tensor.unfold(dimension=-1, size=window_size, step=1)
        window_prod = torch.prod(unfolded, dim=-1)
        result_tensor = torch.full_like(m_tensor, float('nan'))
        result_tensor[..., offset : offset + unfolded.size(-1)] = window_prod
        return result_tensor
