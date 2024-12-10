import sys
sys.path.append('..')

from OP.Others import OP_Basic
import torch

OPclass_name_2B=['OP_B2B','OP_BB2B','OP_BA2B','OP_BG2B',
                 'OP_BF2B']

class OP_B2B:
    def __init__(self):
        self.func_list = ['M_ignore_wobble', 'M_cs_rank', 'M_cs_scale', 'M_cs_zscore', 'M_at_abs', 'M_cs_demean', 'M_cs_winsor',
                          'M_ts_pctchg']

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
    def M_cs_zscore(M_tensor):
        num_stock, day_len, minute_len = M_tensor.shape
        # 计算每个股票每天的均值和标准差
        mean = torch.mean(M_tensor, dim=2, keepdim=True)
        std = torch.std(M_tensor, dim=2, keepdim=True)

        # 避免除以零的情况
        std[std == 0] = 1

        # 进行标准化
        zscore_tensor = (M_tensor - mean) / std

        return zscore_tensor

    @staticmethod
    def M_cs_rank(M_tensor):
        """
        将日内按照排序[0,1]标准化。

        参数:
        M_tensor (torch.Tensor): 输入的张量，假设其形状为 (num_stock, day_len, minute_len)。

        返回:
        torch.Tensor: 排名标准化后的张量。
        """
        # 计算每个元素在每个交易日中的排名
        sorted_indices = torch.argsort(torch.argsort(M_tensor, dim=-1), dim=-1)
        ranks = torch.arange(M_tensor.shape[-1], device=M_tensor.device)[None, None, :]
        ranks = ranks.expand_as(sorted_indices)

        # 使用排名索引来获取每个元素的排名
        rank_tensor = ranks.gather(-1, sorted_indices)

        # 将排名标准化到[0, 1]区间
        rank_tensor = (rank_tensor - 1) / (M_tensor.shape[-1] - 1)

        return rank_tensor

    @staticmethod
    def M_cs_scale(M_tensor):
        """
        日内最大值最小值标准化。

        参数:
        M_tensor (torch.Tensor): 输入的张量，假设其形状为 (num_stock, day_len, minute_len)。

        返回:
        torch.Tensor: 最大值最小值标准化后的张量。
        """
        num_stock, day_len, minute_len = M_tensor.shape

        # 初始化最大值最小值标准化后的张量
        scaled_tensor = torch.zeros_like(M_tensor)

        # 计算每个股票每天的最大值和最小值
        max_values = torch.max(M_tensor, dim=2)[0].unsqueeze(2)  # shape: (num_stock, day_len, 1)
        min_values = torch.min(M_tensor, dim=2)[0].unsqueeze(2)  # shape: (num_stock, day_len, 1)

        # 计算缩放因子
        range_values = max_values - min_values

        # 避免除以零的情况
        range_values[range_values == 0] = 1

        # 进行最大值最小值标准化
        scaled_tensor = (M_tensor - min_values) / range_values

        return scaled_tensor

    @staticmethod
    def M_cs_demean(M_tensor):
        """
        计算日内到均值的距离。

        参数:
        M_tensor (torch.Tensor): 输入的张量，假设其形状为 (num_stock, day_len, minute_len)。

        返回:
        torch.Tensor: 减去日内均值后的张量。
        """
        num_stock, day_len, minute_len = M_tensor.shape

        # 初始化减去均值后的张量
        demeaned_tensor = torch.zeros_like(M_tensor)

        # 计算每个股票每天的均值
        daily_mean = torch.nanmean(M_tensor, dim=2, keepdim=True)  # shape: (num_stock, day_len, 1)

        # 减去均值
        demeaned_tensor = M_tensor - daily_mean

        return demeaned_tensor

    @staticmethod
    def M_cs_winsor(M_tensor, lower_percentile=0.05, upper_percentile=0.95):
        num_stock, day_len, minute_len = M_tensor.shape

        # 计算每个股票每天的上下百分位数
        lower_quantiles = torch.quantile(M_tensor, lower_percentile, dim=2, keepdim=True)
        upper_quantiles = torch.quantile(M_tensor, upper_percentile, dim=2, keepdim=True)

        # 将低于下百分位数的值设置为下百分位数的值
        M_tensor = torch.where(M_tensor < lower_quantiles, lower_quantiles, M_tensor)
        # 将高于上百分位数的值设置为上百分位数的值
        M_tensor = torch.where(M_tensor > upper_quantiles, upper_quantiles, M_tensor)

        return M_tensor

    @staticmethod
    def M_at_abs(M_tensor):

        return torch.abs(M_tensor)

    @staticmethod
    def M_ts_delay(x, d):
        if d > 0:
            new_tensor = torch.full(x.shape, float('nan'), device=x.device)
            new_tensor[d:, :] = x[:-d, :]
            return new_tensor
        elif d == 0:
            return x
        else:
            new_tensor = torch.full(x.shape, float('nan'), device=x.device)
            new_tensor[:d, :] = x[-d:, :]
            return new_tensor

    @staticmethod
    def M_ts_pctchg(x, lookback):
        s = (x - OP_B2B.M_ts_delay(x, lookback)) / OP_B2B.M_ts_delay(x, lookback)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

class OP_BB2B:
    def __init__(self):
        self.func_list = ['at_add', 'at_sub', 'at_div', 'at_sign', 'cs_cut', 'cs_umr', 'at_prod', 'cs_norm_spread']

    @staticmethod
    def M_at_add(x, y):
        return torch.add(x, y)

    @staticmethod
    def M_at_sub(x, y):
        return torch.sub(x, y)

    @staticmethod
    def M_at_div(x, y):
        zero_mask = y == 0
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
        mask = ~((d_tensor_y == 0) | torch.isnan(d_tensor_y))
        result = torch.full_like(d_tensor_x, float('nan'))
        result[mask] = torch.div(d_tensor_x[mask], d_tensor_y[mask])

        return result

    @staticmethod
    def M_cs_norm_spread(x, y):
        s = (x - y) / (torch.abs(x) + torch.abs(y))
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)


class OP_BA2B:  # B*A-B
    def __init__(self):
        self.func_list = ['M_toD_standard']  # B*A-B

    @staticmethod
    def M_toD_standard(M_tensor, D_tensor):
        D_tensor_adjusted = D_tensor.transpose(0, 1).unsqueeze(2)
        return M_tensor / D_tensor_adjusted


class OP_BG2B:  # B*G-B
    def __init__(self):
        self.func_list = ['M_cs_edge_flip']  # B*G-B

    @staticmethod
    def M_cs_edge_flip(M_tensor, thresh):
        flipped = torch.where((M_tensor > thresh) & (M_tensor < 1 - thresh), -M_tensor, M_tensor)
        return torch.where((M_tensor <= 0.3) | (M_tensor >= 0.7), M_tensor.flip(dims=[-1]), flipped)


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

        ]  # B*F-B

    @staticmethod
    def M_ts_delta(m_tensor, lookback):
        return m_tensor - m_tensor.roll(lookback, dims=-1)

    @staticmethod
    def M_ts_mean_left_neighbor(m_tensor, neighbor_range):
        rolled = m_tensor.roll(-1 * neighbor_range, dims=-1)
        return rolled.mean(dim=-1)

    @staticmethod
    def M_ts_mean_mid_neighbor(m_tensor, neighbor_range):
        window_size = neighbor_range * 2 + 1
        unfolded = m_tensor.unfold(dimension=2, size=window_size, step=1)  
        window_mean = unfolded.mean(dim=-1)  
        result_tensor = torch.full_like(m_tensor, float('nan'))
        result_tensor[:, :, neighbor_range:-neighbor_range] = window_mean
        return result_tensor
    



    @staticmethod
    def M_ts_mean_right_neighbor(m_tensor, neighbor_range):
        rolled = m_tensor.roll(neighbor_range, dims=-1)
        return rolled.mean(dim=-1)

    @staticmethod
    def M_ts_std_left_neighbor(m_tensor, neighbor_range):
        rolled = m_tensor.roll(-1 * neighbor_range, dims=-1)
        return rolled.std(dim=-1)

    @staticmethod
    def M_ts_std_mid_neighbor(m_tensor, neighbor_range):
        window_size = neighbor_range * 2 + 1
        unfolded = m_tensor.unfold(dimension=2, size=window_size, step=1)  
        window_std = unfolded.std(dim=-1)  
        result_tensor = torch.full_like(m_tensor, float('nan'))
        result_tensor[:, :, neighbor_range:-neighbor_range] = window_std
        return result_tensor
    
    @staticmethod
    def M_ts_std_right_neighbor(m_tensor, neighbor_range):
        rolled = m_tensor.roll(neighbor_range, dims=-1)
        return rolled.std(dim=-1)

    @staticmethod
    def M_ts_product_left_neighbor(m_tensor, neighbor_range):
        rolled = m_tensor.roll(-1 * neighbor_range, dims=-1)
        return torch.prod(rolled, dim=-1)

    @staticmethod
    def M_ts_product_mid_neighbor(m_tensor, neighbor_range):
        window_size = neighbor_range * 2 + 1
        unfolded = m_tensor.unfold(dimension=2, size=window_size, step=1)  
        window_prod= unfolded.prod(dim=-1)  
        result_tensor = torch.full_like(m_tensor, float('nan'))
        result_tensor[:, :, neighbor_range:-neighbor_range] = window_prod
        return result_tensor



    @staticmethod
    def M_ts_product_right_neighbor(m_tensor, neighbor_range):
        rolled = m_tensor.roll(neighbor_range, dims=-1)
        return torch.prod(rolled, dim=-1)
