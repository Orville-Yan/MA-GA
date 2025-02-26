import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from OP.Others import OP_Basic
import torch

OPclass_name_2A = ['OP_A2A', 'OP_AE2A', 'OP_AA2A', 'OP_AG2A',
                   'OP_AAF2A', 'OP_AF2A', 'OP_AC2A', 'OP_BD2A',
                   'OP_BBD2A', 'OP_BB2A', 'OP_B2A', 'OP_D2A']

device = torch.device("cuda:0")


class OP_A2A:
    def __init__(self):
        self.func_list = ['D_at_abs', 'D_cs_rank', 'D_cs_scale', 'D_cs_zscore', 'D_cs_harmonic_mean', 'D_cs_demean',
                          'D_cs_winsor']

    @staticmethod
    def D_at_abs(x):  # 取绝对值
        return torch.abs(x)

    @staticmethod
    def D_cs_rank(x):  # 截面分位数
        mask = ~torch.isnan(x)
        data_no_nan = torch.where(mask, x, float('inf'))
        ranks = torch.argsort(torch.argsort(data_no_nan, dim=1), dim=1).float()  # 首先排序，然后取序数
        quantiles = ranks / torch.sum(mask, 1).unsqueeze(1)  # 计算分位数
        s = torch.where(mask, quantiles, torch.tensor(float('nan')))

        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_cs_scale(x):  # 标准化截面最大最小值
        mask = ~torch.isnan(x)
        data_no_nan = torch.where(mask, x, 0)
        max = torch.max(torch.where(mask, x, float('-inf')), dim=1)[0].unsqueeze(dim=1)
        min = torch.min(torch.where(mask, x, float('inf')), dim=1)[0].unsqueeze(dim=1)
        sacled_data_no_nan = (data_no_nan - min) / (max - min)  # 公式核心
        scaled_data = torch.where(mask, sacled_data_no_nan, float('nan'))
        s = scaled_data + 1
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_cs_zscore(x):  # z-score标准化
        x_mean = OP_Basic.nanmean(x, dim=1).unsqueeze(1)
        x_std = OP_Basic.nanstd(x, dim=1).unsqueeze(1)
        zscore = (x - x_mean) / x_std
        s = zscore
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_cs_harmonic_mean(x):  # 调和平均
        mask = (~torch.isnan(x)) & (x != 0)
        data_no_nan = 1 / torch.where(mask, x, 1)
        harmonic_mean = torch.sum(mask, dim=1) / torch.nansum(
            torch.where(mask, data_no_nan, float('nan')), dim=1)
        result = torch.full_like(x, float('nan'),device=x.device)
        result[mask] = harmonic_mean.unsqueeze(dim=1).expand_as(x)[mask]
        s = result
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_cs_demean(x):
        return OP_A2A.D_at_abs(x - OP_Basic.nanmean(x, dim=1).unsqueeze(1))

    @staticmethod
    def D_cs_winsor(x, limit=[0.05, 0.95]):  # 尾部磨平，将分位数小于0.05或大于0.95的部分全部改为0.05和0.95处的值
        rank = OP_A2A.D_cs_rank(x)
        min_limit = torch.where(rank >= limit[0], rank, float('nan'))
        max_limit = torch.where(rank <= limit[1], rank, float('nan'))
        mask = (~torch.isnan(min_limit)) & (~torch.isnan(max_limit))
        max = torch.max(torch.where(mask, x, float('-inf')), dim=1)[0].unsqueeze(dim=1)
        min = torch.min(torch.where(mask, x, float('inf')), dim=1)[0].unsqueeze(dim=1)
        winsored_min = torch.where(rank <= limit[0], min, x)  # 最小值变化
        winsored_max = torch.where(rank >= limit[1], max, winsored_min)
        x_with_nan = torch.where(~torch.isnan(x), winsored_max, float('nan'))
        return x_with_nan


class OP_AE2A:
    def __init__(self):
        self.func_list = ['D_cs_demean_industry', 'D_cs_industry_neutra']

    @staticmethod
    def D_cs_demean_industry(x, y):  # 行业均值计算
        day_len, num_stock = x.shape
        _, _, industry_num = y.shape
        y = y.float()   
        industry_sums = torch.bmm(y.permute(0, 2, 1), x.unsqueeze(-1))
        industry_counts = y.sum(dim=1).unsqueeze(-1)
        industry_means = industry_sums / industry_counts
        weighted_industry_means = torch.bmm(y, industry_means)
        num_industries_per_stock = y.sum(dim=2).unsqueeze(-1)
        valid_mask = (num_industries_per_stock > 0)
        industry_means_final = torch.where(valid_mask, weighted_industry_means / num_industries_per_stock,
                                           torch.tensor(0.0, device=x.device))
        demeaned_abs = torch.abs(x - industry_means_final.squeeze(-1))

        return demeaned_abs

    @staticmethod
    def D_cs_industry_neutra(x, y):  # 行业中性化
        return OP_AE2A.D_cs_demean_industry(x, y)


class OP_AA2A:
    def __init__(self):
        self.func_list = ['D_cs_norm_spread', 'D_cs_cut', 'D_cs_regress_res', 'D_at_add', 'D_at_sub', 'D_at_div',
                          'D_at_prod', 'D_at_mean']

    @staticmethod
    def D_cs_norm_spread(x, y):
        s = (x - y) / (torch.abs(x) + torch.abs(y))
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_cs_cut(x, y):
        def at_sign(x):
            mask = ~torch.isnan(x)
            x_no_nan = torch.where(mask, x, 0)
            sign = torch.sign(x_no_nan)
            return torch.where(mask, sign, float('nan'))

        sign = at_sign(x - OP_Basic.nanmean(x, dim=1).unsqueeze(1))
        return sign * y

    @staticmethod
    def D_cs_regress_res(x, y):  # 截面回归取残差
        res = OP_Basic.multi_regress(x, y)[-1]
        return res

    @staticmethod
    def D_at_add(x, y):
        return torch.add(x, y)

    @staticmethod
    def D_at_div(x, y):  # 除
        zero_mask = y == 0
        result = torch.div(x, y)
        result[zero_mask] = torch.nan
        return result

    @staticmethod
    def D_at_sub(x, y):  # 减
        return torch.sub(x, y)

    @staticmethod
    def D_at_prod(x, y):  # 乘法

        result = torch.prod(x, y)

        return result

    @staticmethod
    def D_at_mean(x, y):  # 均值
        return OP_AA2A.D_at_add(x, y) / 2


class OP_AG2A:
    def __init__(self):
        self.func_list = ['D_cs_edge_flip']

    @staticmethod
    def D_cs_edge_flip(x, thresh):
        rank = OP_A2A.D_cs_rank(x)
        if thresh < 0.3:
            edge_fliped = torch.where((rank < thresh) | (rank > 1 - thresh), -x, x)
        elif thresh > 0.7:
            edge_fliped = torch.where((rank < 1 - thresh) | (rank > thresh), -x, x)
        else:
            edge_fliped = torch.where((rank < thresh) | (rank > 1 - thresh), x, -x)

        return edge_fliped


class OP_AAF2A:
    def __init__(self):
        self.func_list = ['D_ts_corr', 'D_ts_rankcorr', 'D_ts_regress_res', 'D_ts_weight_mean', 'D_ts_regress']

    @staticmethod
    def D_ts_corr(x, y, d):  # d天内的相关性
        nan_fill = torch.full((d - 1, x.shape[1]), float('nan'),device=x.device)
        x = x.unfold(0, d, 1)
        y = y.unfold(0, d, 1)
        correlation = OP_Basic.corrwith(x, y, dim=-1)
        s = torch.cat([nan_fill, correlation], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_rankcorr(x, y, lookback):  # 回溯lookback天，x 和 y的秩相关性
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'),device=x.device)
        x = x.unfold(0, lookback, 1)
        y = y.unfold(0, lookback, 1)
        correlation = OP_Basic.rank_corrwith(x, y, dim=-1)
        s = torch.cat([nan_fill, correlation], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_regress(x, y, lookback):  # 用于下面计算残差
        epsilon = 1e-10
        # x = x.type(torch.float64)
        # y = y.type(torch.float64)
        nan_fill = torch.full((x[:lookback - 1].shape), float('nan'),device=x.device)

        x_unfold = x.unfold(0, lookback, 1)
        y_unfold = y.unfold(0, lookback, 1)

        mask = torch.isnan(x_unfold) | torch.isnan(y_unfold)
        nan_all = (torch.prod(mask, dim=-1) == 1)
        x_unfold = torch.where(mask, 0, x_unfold).masked_fill(mask, 0)
        y_unfold = torch.where(mask, 0, y_unfold).masked_fill(mask, 0)
        constant = torch.ones((x_unfold.unsqueeze(-1).shape)).masked_fill(mask.unsqueeze(-1), 0)
        x_const = torch.cat([x_unfold.unsqueeze(-1), constant], dim=-1)
        x_T = x_const.permute(0, 1, -1, 2)
        w = torch.matmul(x_T, x_const).masked_fill(nan_all.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2),
                                                   float('nan'))
        singularity = (torch.det(w) < epsilon).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2)
        theta = (torch.inverse(torch.where(singularity, float('nan'), w)) @ x_T @ y_unfold.unsqueeze(-1)).squeeze(-1)

        k = torch.cat([nan_fill, torch.where(nan_all, float('nan'), theta[:, :, 0])], dim=0)
        b = torch.cat([nan_fill, torch.where(nan_all, float('nan'), theta[:, :, 1])], dim=0)
        res = torch.where(torch.cat([~torch.isnan(nan_fill), nan_all], dim=0), float('nan'), y - (k * x + b))

        k = torch.where(abs(k) < epsilon, 0, k)
        b = torch.where(abs(b) < epsilon, 0, b)
        res = torch.where(abs(res) < epsilon, 0, res)
        return k, b, res

    @staticmethod
    def D_ts_regress_res(x, y, lookback):  # 回归取残差
        return OP_AAF2A.D_ts_regress(x, y, lookback)[2]

    @staticmethod
    def D_ts_weight_mean(x, y, lookback):  # 回溯lookback天，以d_tensor_y为权重，计算d_tensor_x 的加权平均
        if lookback == 1:
            return x
        else:
            x = x.unfold(0, lookback, 1)
            y = y.unfold(0, lookback, 1)
            mask = torch.isnan(x) | torch.isnan(y)
            x = torch.where(mask, float('nan'), x)
            y = torch.where(mask, float('nan'), y)

            nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'),device=x.device)
            p = torch.nansum(x * y, dim=-1) / torch.nansum(y, dim=-1)
            p = torch.cat([nan_fill, p], dim=0)
            return torch.where((p == torch.inf) | (p == -torch.inf), float('nan'), p)


class OP_AF2A:
    def __init__(self):
        self.func_list = ['D_ts_max', 'D_ts_min', 'D_ts_delay', 'D_ts_delta', 'D_ts_pctchg',
                          'D_ts_mean', 'D_ts_harmonic_mean', 'D_ts_std', 'D_ts_to_max',
                          'D_ts_to_min', 'D_ts_to_mean', 'D_ts_max_to_min', 'D_ts_maxmin_norm',
                          'D_ts_norm', 'D_ts_detrend'
                          ]

    @staticmethod
    def D_ts_max(x, lookback):  # lookback天内最大值
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'),device=x.device)
        x_3d = x.unfold(0, lookback, 1)
        max_tensor = torch.max(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, max_tensor], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_min(x, lookback):  # 最小值
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'),device=x.device)
        x_3d = x.unfold(0, lookback, 1)
        max_tensor = torch.min(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, max_tensor], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_delay(x, d):  # 用于计算delta
        if d > 0:
            new_tensor = torch.full(x.shape, float('nan'),device=x.device)
            new_tensor[d:, :] = x[:-d, :]
            return new_tensor
        elif d == 0:
            return x
        else:
            new_tensor = torch.full(x.shape, float('nan'),device=x.device)
            new_tensor[:d, :] = x[-d:, :]
            return new_tensor

    @staticmethod
    def D_ts_delta(x, d):
        return x - OP_AF2A.D_ts_delay(x, d)

    @staticmethod
    def D_ts_pctchg(x, lookback):
        s = (x - OP_AF2A.D_ts_delay(x, lookback)) / OP_AF2A.D_ts_delay(x, lookback)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_mean(x, lookback):  # 均值
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'),device=x.device)
        x_3d = x.unfold(0, lookback, 1)
        x_mean = OP_Basic.nanmean(x_3d)
        s = torch.cat([nan_fill, x_mean], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_harmonic_mean(x, lookback):  # 调和平均
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'),device=x.device)
        x_3d = x.unfold(0, lookback, 1)
        mask = (x_3d == 0) | torch.isnan(x_3d)
        dominator = 1 / x_3d
        dominator = torch.where(mask, 0, dominator)
        dominator = torch.sum(dominator, dim=-1)
        numerator = torch.sum(~mask, dim=-1)
        s = numerator / dominator
        s = torch.where(dominator == 0, float('nan'), s)
        s = torch.cat([nan_fill, s], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_std(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'),device=x.device)
        x_3d = x.unfold(0, lookback, 1)
        x_std = OP_Basic.nanstd(x_3d)
        s = torch.cat([nan_fill, x_std], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_to_max(x, lookback):  # d_tensor/D_ts_max(d_tensor, lookback)
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'),device=x.device)
        x_3d = x.unfold(0, lookback, 1)
        max_tensor = torch.max(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, max_tensor], dim=0)
        s = x / s
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_to_min(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'),device=x.device)
        x_3d = x.unfold(0, lookback, 1)
        min_tensor = torch.min(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, min_tensor], dim=0)
        s = x / s
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_to_mean(x, lookback):  # d_tensor/D_ts_mean(d_tensor, lookback)
        mean_tensor = OP_AF2A.D_ts_mean(x, lookback)
        s = x / mean_tensor
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_max_to_min(x, lookback):  # d_tensor/(D_ts_max(d_tensor, lookback)-D_ts_min(d_tensor, lookback))
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'),device=x.device)
        x_3d = x.unfold(0, lookback, 1)
        min_tensor = torch.min(x_3d, dim=-1)[0]
        max_tensor = torch.max(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, max_tensor - min_tensor], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_maxmin_norm(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'),device=x.device)
        x_3d = x.unfold(0, lookback, 1)
        min_tensor = torch.min(x_3d, dim=-1)[0]
        max_tensor = torch.max(x_3d, dim=-1)[0]
        min_tensor = torch.cat([nan_fill, min_tensor], dim=0)
        max_tensor = torch.cat([nan_fill, max_tensor], dim=0)
        s = (x - min_tensor) / (max_tensor - min_tensor)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_norm(x, lookback):  # 时序标准化
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'),device=x.device)
        x_3d = x.unfold(0, lookback, 1)
        x_mean = OP_Basic.nanmean(x_3d)
        mean = torch.cat([nan_fill, x_mean], dim=0)
        x_std = OP_Basic.nanstd(x_3d)
        std = torch.cat([nan_fill, x_std], dim=0)
        s = (x - mean) / std
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_detrend(x, lookback):  # ts_regress，去除lookback天内的趋势
        x = x.float()
        time_idx = torch.arange(x.shape[0], dtype=torch.float32).unsqueeze(-1)
        time_idx_expanded = time_idx.repeat(1, x.shape[1])
        k, b, _ = OP_AAF2A.D_ts_regress(time_idx_expanded, x, lookback)
        trend = (k * time_idx_expanded) + b
        s = x - trend
        s[:lookback - 1, :] = float('nan')
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)


class OP_AC2A:
    def __init__(self):
        self.func_list = ['D_ts_mask_mean', 'D_ts_mask_std', 'D_ts_mask_sum', 'D_ts_mask_prod']

    @staticmethod
    def D_ts_mask_mean(x, mask):
        nan_fill = torch.full((mask.shape[2] - 1, x.shape[1], mask.shape[2]), float('nan'),device=x.device)
        x_3d = x.unfold(0, mask.shape[2], 1)
        x_3d = torch.cat([nan_fill, x_3d], dim=0)
        x_3d = torch.where(mask, float('nan'), x_3d)
        s = OP_Basic.nanmean(x_3d)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_mask_std(x, mask):
        nan_fill = torch.full((mask.shape[2] - 1, x.shape[1], mask.shape[2]), float('nan'),device=x.device)
        x_3d = x.unfold(0, mask.shape[2], 1)
        x_3d = torch.cat([nan_fill, x_3d], dim=0)
        x_3d = torch.where(mask, x_3d, float('nan'))
        s = OP_Basic.nanstd(x_3d)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_mask_sum(x, mask):
        nan_fill = torch.full((mask.shape[2] - 1, x.shape[1], mask.shape[2]), float('nan'),device=x.device)
        x_3d = x.unfold(0, mask.shape[2], 1)
        x_3d = torch.cat([nan_fill, x_3d], dim=0)
        x_3d = torch.where(mask, x_3d, float('nan'))
        all_nan = torch.all(torch.isnan(x_3d), dim=2)
        s = torch.nansum(x_3d, dim=-1)
        s[all_nan] = float('nan')
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_ts_mask_prod(x, mask):
        nan_fill = torch.full((mask.shape[2] - 1, x.shape[1], mask.shape[2]), float('nan'),device=x.device)
        x_3d = x.unfold(0, mask.shape[2], 1)
        x_3d = torch.cat([nan_fill, x_3d], dim=0)
        x_3d = torch.where(mask, x_3d, float('nan'))
        all_nan = torch.all(torch.isnan(x_3d), dim=2)
        x_3d = torch.where(torch.isnan(x_3d), torch.ones_like(x_3d), x_3d)
        s = torch.prod(x_3d, dim=-1)
        s[all_nan] = float('nan')
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)


class OP_BD2A:
    def __init__(self):
        self.func_list = ['D_Minute_area_mean', 'D_Minute_area_std', 'D_Minute_area_sum', 'D_Minute_area_prod']

    @staticmethod
    def D_Minute_area_mean(x, mask):
        x = torch.where(mask, x, float('nan'))
        s = OP_Basic.nanmean(x)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_Minute_area_std(x, mask):
        x = torch.where(mask, x, float('nan'))
        s = OP_Basic.nanstd(x)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_Minute_area_sum(x, mask):
        x = torch.where(mask, x, float('nan'))
        all_nan = torch.all(torch.isnan(x), dim=2)
        s = torch.nansum(x, dim=-1)
        s[all_nan] = float('nan')
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_Minute_area_prod(x, mask):
        x = torch.where(mask, x, 1)
        s = torch.prod(x, dim=-1)
        s[torch.all(mask, dim=-1)] = float('nan')
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)


class OP_B2A:
    def __init__(self):
        self.func_list = ['D_Minute_std', 'D_Minute_mean', 'D_Minute_trend']

    @staticmethod
    def D_Minute_std(x):
        # 计算日内标准差。
        return OP_Basic.nanstd(x, dim=-1)

    @staticmethod
    def D_Minute_mean(x):
        # 计算日内均值。
        return OP_Basic.nanmean(x, dim=-1)

    @staticmethod
    def D_Minute_trend(x):
        # 计算日内数据的变化趋势。
        time_index = torch.arange(x.shape[-1], dtype=torch.float32, device=x.device)
        time_index = time_index.unsqueeze(0).expand_as(x)
        slopes, _, _ = OP_Basic.regress(x, time_index, dim=-1)
        return slopes.squeeze(-1)


class OP_BBD2A:
    def __init__(self):
        self.func_list = ['D_Minute_area_weight_mean', 'D_Minute_area_corr', 'D_Minute_area_rankcorr',
                          'D_Minute_area_bifurcate_mean', 'D_Minute_area_bifurcate_std']

    @staticmethod
    def D_Minute_area_weight_mean(x, weight, mask):
        x = torch.where(mask, x, float('nan'))
        x_ = x * weight
        s = OP_Basic.nanmean(x_)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def D_Minute_area_corr(x, y, mask):
        x = torch.where(mask, x, float('nan'))
        corr = OP_Basic.corrwith(x, y)
        return torch.where((corr == torch.inf) | (corr == -torch.inf), float('nan'), corr)

    @staticmethod
    def D_Minute_area_rankcorr(x, y, mask):
        x = torch.where(mask, x, float('nan'))
        corr = OP_Basic.rank_corrwith(x, y, )
        return torch.where((corr == torch.inf) | (corr == -torch.inf), float('nan'), corr)

    @staticmethod
    def D_Minute_area_bifurcate_mean(x, y, mask):
        day_expanded = OP_BD2A.D_Minute_area_mean(y, mask).unsqueeze(-1).repeat(1, 1, 242)  # (day_len, num_stock, minute_len)
        maskplus = day_expanded < y
        masksub = day_expanded > y
        return OP_AA2A.D_at_sub(OP_BD2A.D_Minute_area_mean(x, maskplus),
                                OP_BD2A.D_Minute_area_mean(x, masksub))

    @staticmethod
    def D_Minute_area_bifurcate_std(x, y, mask):
        day_expanded = OP_BD2A.D_Minute_area_mean(y, mask).unsqueeze(-1).repeat(1, 1, 242)  # (day_len, num_stock, minute_len)
        maskplus = day_expanded < y
        masksub = day_expanded > y
        return OP_AA2A.D_at_sub(OP_BD2A.D_Minute_area_std(x, maskplus),
                                OP_BD2A.D_Minute_area_std(x, masksub))


class OP_BB2A:
    def __init__(self):
        self.func_list = ['D_Minute_corr', 'D_Minute_weight_mean']

    @staticmethod
    def D_Minute_corr(x, y):
        corr = OP_Basic.corrwith(x, y)
        return torch.where((corr == torch.inf) | (corr == -torch.inf), float('nan'), corr)

    @staticmethod
    def D_Minute_weight_mean(x, weight=1):
        x_ = x * weight
        s = OP_Basic.nanmean(x_)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)


class OP_D2A:
    def __init__(self):
        self.func_list = ['D_Minute_abnormal_point_count']

    @staticmethod
    def D_Minute_abnormal_point_count(mask):
        s = torch.nansum(mask, dim=-1)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)


if __name__ == '__main__':
    import time
    TypeA_shape = (10, 100) 
    TypeC_shape = (10,100,2)
    TypeB_shape = TypeD_shape = (10, 100, 242)
    TypeE_shape = (10, 100, 31)

    # 创建随机数据
    A = torch.randn(TypeA_shape)
    A2 = torch.randn(TypeA_shape)
    C = torch.randint(0, 2, TypeC_shape).bool()
    B = torch.randn(TypeB_shape)
    B2 = torch.randn(TypeB_shape)
    D = torch.randint(0, 2, TypeD_shape).bool()
    F = 5
    G = 0.5
    E = torch.randint(0, 2, TypeE_shape)  # 假设有10个行业


    # 测试函数
    def test_functions(class_instance, data, *args):
        results = {}
        for func_name in class_instance.func_list:
            func = getattr(class_instance, func_name)
            start_time = time.time()
            try:
                result = func(*data, *args)
                results[func_name] = time.time() - start_time
                shape_result = (result.shape == TypeA_shape)
                if not shape_result:
                    print(func_name)
                    print('shape fault')
            except Exception as e:
                results[func_name] = str(e)
                print(func_name)
                print(e)
        
        



    # 测试每个类
    def test_class(class_type, *args):
        instance = class_type()
        if class_type in [OP_A2A]:
            return test_functions(instance, (A,))
        elif class_type in [OP_AE2A]:
            return test_functions(instance, (A, E,))
        elif class_type in [OP_AA2A]:
            return test_functions(instance, (A, A2,))
        elif class_type in [OP_AF2A]:
            return test_functions(instance, (A, F,))
        elif class_type in [OP_AG2A]:
            return test_functions(instance, (A, G,))
        elif class_type in [OP_AAF2A]:
            return test_functions(instance, (A, A2, F,))
        elif class_type in [OP_AC2A]:
            return test_functions(instance, (A, C,))
        elif class_type in [OP_BD2A]:
            return test_functions(instance, (B, D,))
        elif class_type in [OP_BBD2A]:
            return test_functions(instance, (B, B2, D))
        elif class_type in [OP_BB2A]:
            return test_functions(instance, (B, B2,))
        elif class_type in [OP_B2A]:
            return test_functions(instance, (B,))
        elif class_type in [OP_D2A]:
            return test_functions(instance, (D,))


    # # 打印结果
    # def print_results(results, class_name):
    #     print(f"Results for {class_name}:")
    #     for func_name, duration in results.items():
    #         if isinstance(duration, float):
    #             print(f"  {func_name}: {duration:.6f} seconds")
    #         else:
    #             print(f"  {func_name}: {duration}")


    # 运行测试
    classes = [OP_A2A, OP_AE2A, OP_AA2A, OP_AG2A, OP_AAF2A, OP_AF2A, OP_AC2A, OP_BD2A, OP_B2A, OP_BBD2A, OP_BB2A,
               OP_D2A]
    classes = [OP_AA2A]
    for class_type in classes:
        results = test_class(class_type)
        # print_results(results, class_type.__name__)

