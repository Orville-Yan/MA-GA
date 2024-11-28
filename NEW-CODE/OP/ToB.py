import torch
import numpy as np

'''
先把示例代码copy，有些可以直接用
'''


def nanmean(tensor, dim=-1, weights=None):
    if weights is None:
        mean = torch.nansum(tensor, dim=dim) / torch.sum(~torch.isnan(tensor), dim=dim)
    else:
        nan_mask = torch.isnan(tensor) | torch.isnan(weights)
        tensor = tensor.masked_fill(nan_mask, float('nan'))
        weights = weights.masked_fill(nan_mask, float('nan'))
        mean = torch.nansum(tensor * weights, dim=dim) / torch.nansum(weights, dim=dim)
    return mean


def nanstd(x, unbiased=False, weights=None, dim=-1):
    if weights is None:
        mean = nanmean(x, dim=dim)
        centered_tensor = (x - mean.unsqueeze(dim))
        centered_tensor = centered_tensor.masked_fill(torch.isnan(centered_tensor), 0)
        var = torch.nansum(centered_tensor.pow(2), dim=dim)
        n = torch.sum(~torch.isnan(x), dim=dim)
        denom = n - 1 if unbiased else n
        std = torch.sqrt(var / denom)
    else:
        nan_mask = torch.isnan(x) | torch.isnan(weights)
        x = x.masked_fill(nan_mask, float('nan'))
        weights = weights.masked_fill(nan_mask, float('nan'))

        mean = nanmean(x, weights=weights, dim=dim).unsqueeze(dim)
        var = nanmean((x - mean) ** 2, weights=weights, dim=dim)
        std = torch.sqrt(var)

    return std


def corrwith(tensor1, tensor2, dim=-1):
    mask = ~(torch.isnan(tensor1) | torch.isnan(tensor2))
    tensor1 = torch.where(mask, tensor1, float('nan'))
    tensor2 = torch.where(mask, tensor2, float('nan'))

    tensor1_mean = nanmean(tensor1, dim=dim)
    tensor2_mean = nanmean(tensor2, dim=dim)
    tensor1_std = nanstd(tensor1, dim=dim, unbiased=False)
    tensor2_std = nanstd(tensor2, dim=dim, unbiased=False)
    centered_tensor1 = tensor1 - tensor1_mean.unsqueeze(dim)
    centered_tensor2 = tensor2 - tensor2_mean.unsqueeze(dim)
    covariance = nanmean(centered_tensor1 * centered_tensor2, dim=dim)
    correlation = covariance / (tensor1_std * tensor2_std)
    return correlation


def rank_corrwith(tensor1, tensor2, dim=-1):
    nan_mask = torch.isnan(tensor1) | torch.isnan(tensor2)
    tensor1 = tensor1.masked_fill(nan_mask, float('nan'))
    tensor1 = torch.argsort(torch.argsort(tensor1, dim=dim))
    tensor1 = torch.where(~nan_mask, tensor1, float('nan'))
    tensor2 = tensor2.masked_fill(nan_mask, float('nan'))
    tensor2 = torch.argsort(torch.argsort(tensor2, dim=dim))
    tensor2 = torch.where(~nan_mask, tensor2, float('nan'))

    tensor1_mean = nanmean(tensor1, dim=dim)
    tensor2_mean = nanmean(tensor2, dim=dim)
    tensor1_std = nanstd(tensor1, dim=dim, unbiased=False)
    tensor2_std = nanstd(tensor2, dim=dim, unbiased=False)
    centered_tensor1 = tensor1 - tensor1_mean.unsqueeze(dim)
    centered_tensor2 = tensor2 - tensor2_mean.unsqueeze(dim)
    covariance = nanmean(centered_tensor1 * centered_tensor2, dim=dim)
    correlation = covariance / (tensor1_std * tensor2_std)
    return correlation


def multi_regress(y, x_s):
    down_epsilon = 1e-10
    up_epsilon = 1e20
    # y = y.type(torch.float64)
    x = x_s  # .type(torch.float64)

    mask_x = torch.sum(torch.isnan(x), dim=-1) > 0
    mask_y = torch.isnan(y)
    mask = ~(mask_x | mask_y)
    y = torch.where(mask.unsqueeze(-1), y.unsqueeze(-1), 0)

    const = torch.ones(x[:, :, 0].shape).unsqueeze(-1)
    x_const = torch.cat([x, const], dim=-1)
    x_const = torch.where(mask.unsqueeze(-1).expand(-1, -1, x_const.shape[-1]), x_const, 0)
    x_T = x_const.permute(0, -1, 1)
    w = torch.matmul(x_T, x_const)
    det = torch.det(w)
    singularity = ((det < down_epsilon) | (abs(det) >= up_epsilon)).unsqueeze(-1).unsqueeze(-1).expand(-1, x_s.shape[
        -1] + 1,
                                                                                                       x_s.shape[
                                                                                                           -1] + 1)
    theta = (torch.inverse(torch.where(singularity, float('nan'), w)) @ x_T @ y).squeeze(-1)

    k = theta[:, :-1]
    b = theta[:, -1]
    predict = torch.sum(k.unsqueeze(1) * x_s, dim=-1) + b.unsqueeze(-1)
    res = torch.where(mask, y.squeeze(-1), float('nan')) - predict

    k = torch.where(abs(k) < down_epsilon, 0, k)
    b = torch.where(abs(b) < down_epsilon, 0, b)
    res = torch.where(abs(res) < down_epsilon, 0, res)

    return k, b, res


def ts_regress(x, y, lookback):
    epsilon = 1e-10
    # x = x.type(torch.float64)
    # y = y.type(torch.float64)
    nan_fill = torch.full((x[:lookback - 1].shape), float('nan'))

    x_unfold = x.unfold(0, lookback, 1)
    y_unfold = y.unfold(0, lookback, 1)

    mask = torch.isnan(x_unfold) | torch.isnan(y_unfold)
    nan_all = (torch.prod(mask, dim=-1) == 1)
    x_unfold = torch.where(mask, 0, x_unfold).masked_fill(mask, 0)
    y_unfold = torch.where(mask, 0, y_unfold).masked_fill(mask, 0)
    constant = torch.ones((x_unfold.unsqueeze(-1).shape)).masked_fill(mask.unsqueeze(-1), 0)
    x_const = torch.cat([x_unfold.unsqueeze(-1), constant], dim=-1)
    x_T = x_const.permute(0, 1, -1, 2)
    w = torch.matmul(x_T, x_const).masked_fill(nan_all.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2), float('nan'))
    singularity = (torch.det(w) < epsilon).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2)
    theta = (torch.inverse(torch.where(singularity, float('nan'), w)) @ x_T @ y_unfold.unsqueeze(-1)).squeeze(-1)

    k = torch.cat([nan_fill, torch.where(nan_all, float('nan'), theta[:, :, 0])], dim=0)
    b = torch.cat([nan_fill, torch.where(nan_all, float('nan'), theta[:, :, 1])], dim=0)
    res = torch.where(torch.cat([~torch.isnan(nan_fill), nan_all], dim=0), float('nan'), y - (k * x + b))

    k = torch.where(abs(k) < epsilon, 0, k)
    b = torch.where(abs(b) < epsilon, 0, b)
    res = torch.where(abs(res) < epsilon, 0, res)
    return k, b, res


class op:
    def __init__(self):
        self.variable1_func_list = ['at_log', 'at_sign', 'at_signsqrt', 'at_sigmoid', 'at_neg', 'cs_rank'
            , 'cs_scale', 'cs_zscore', 'cs_harmonic_mean', 'cs_edge_flip', 'cs_winsor', 'cs_demean',
                                    'cs_distance2mean', ]

        self.on_parameter_func_list = ['at_signpower', 'ts_harmonic_mean', 'ts_pctchg_abs', 'ts_pctchg', 'ts_delay',
                                       'ts_delta', 'ts_mean', 'ts_std', 'ts_to_max',
                                       'ts_to_min', 'ts_skewness', 'ts_kurtosis', 'ts_ir',
                                       'ts_to_mean', 'ts_max_to_min', 'ts_to_maxmin_norm', 'ts_mon', 'ts_median',
                                       'ts_product', 'ts_max', 'ts_min', 'ts_middle_mean']

        self.variable2_func_list = ['at_add', 'at_div', 'at_mul', 'at_sub', 'cs_norm_spread', 'cs_umr', 'cs_cut',
                                    'cs_regress_res']
        self.variable3_func_list = ['ts_regress_res', 'ts_regress_k', 'ts_regress_b', 'ts_rankcorr', 'ts_correlation',
                                    'ts_weight_mean']

    @staticmethod
    def create_constant_function(value):
        def constant_function(df):
            return int(value)

        return constant_function

    @staticmethod
    def at_add(x, y):
        return torch.add(x, y)

    @staticmethod
    def at_div(x, y):
        zero_mask = y == 0
        result = torch.div(x, y)
        result[zero_mask] = torch.nan
        return result

    @staticmethod
    def at_log(x):
        zero_mask = (x == 0) | (x == torch.inf) | (x == -torch.inf)
        result = torch.log(torch.abs(x))
        result[zero_mask] = torch.nan
        return result

    @staticmethod
    def at_mul(x, y):
        result = torch.multiply(x, y)
        return result

    @staticmethod
    def at_neg(x):
        return -x

    @staticmethod
    def at_sign(x):
        mask = ~torch.isnan(x)
        x_no_nan = torch.where(mask, x, 0)
        sign = torch.sign(x_no_nan)
        return torch.where(mask, sign, float('nan'))

    @staticmethod
    def at_signsqrt(x):
        return torch.sqrt(torch.abs(x))

    @staticmethod
    def at_sub(x, y):
        return torch.sub(x, y)

    @staticmethod
    def at_signpower(x, d):
        return torch.pow(torch.abs(x), d)

    @staticmethod
    def at_sigmoid(x):
        return torch.div(1, torch.add(1, torch.exp(-x)))

    @staticmethod
    def cs_rank(x):
        mask = ~torch.isnan(x)
        data_no_nan = torch.where(mask, x, torch.full_like(x, float('inf')))
        ranks = torch.argsort(torch.argsort(data_no_nan, dim=1), dim=1).float()
        quantiles = ranks / torch.sum(mask, 1).unsqueeze(1)
        s = torch.where(mask, quantiles, torch.tensor(float('nan')))

        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def cs_scale(x):
        mask = ~torch.isnan(x)
        data_no_nan = torch.where(mask, x, torch.full_like(x, 0))
        max = torch.max(torch.where(mask, x, torch.full_like(x, float('-inf'))), dim=1)[0].unsqueeze(dim=1)
        min = torch.min(torch.where(mask, x, torch.full_like(x, float('inf'))), dim=1)[0].unsqueeze(dim=1)
        sacled_data_no_nan = (data_no_nan - min) / (max - min)
        scaled_data = torch.where(mask, sacled_data_no_nan, torch.tensor(float('nan')))
        s = scaled_data + 1
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def cs_zscore(x):
        x_std = nanstd(x, dim=1).unsqueeze(1)
        x_mean = nanmean(x, dim=1).unsqueeze(1)
        zscore_x = (x - x_mean) / x_std
        s = zscore_x
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def cs_harmonic_mean(x):
        mask = (~torch.isnan(x)) & (x != 0)
        data_no_nan = 1 / torch.where(mask, x, torch.full_like(x, 1))
        harmonic_mean = torch.sum(mask, dim=1) / torch.nansum(
            torch.where(mask, data_no_nan, torch.tensor(float('nan'))), dim=1)
        result = torch.full_like(x, float('nan'))
        result[mask] = harmonic_mean.unsqueeze(dim=1).expand_as(x)[mask]
        s = result
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def cs_norm_spread(x, y):
        s = (x - y) / (torch.abs(x) + torch.abs(y))
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def cs_edge_flip(x, percent=0.3):
        rank = op.cs_rank(x)
        edge_fliped = torch.where(torch.abs(rank - 0.5) > percent / 2, x, -x)
        return edge_fliped

    @staticmethod
    def cs_winsor(x, limit=[0.05, 0.95]):
        rank = op.cs_rank(x)
        min_limit = torch.where(rank >= limit[0], rank, float('nan'))
        max_limit = torch.where(rank <= limit[1], rank, float('nan'))
        mask = (~torch.isnan(min_limit)) & (~torch.isnan(max_limit))
        max = torch.max(torch.where(mask, x, torch.full_like(x, float('-inf'))), dim=1)[0].unsqueeze(dim=1)
        min = torch.min(torch.where(mask, x, torch.full_like(x, float('inf'))), dim=1)[0].unsqueeze(dim=1)
        winsored_min = torch.where(rank <= limit[0], min, x)
        winsored_max = torch.where(rank >= limit[1], max, winsored_min)
        x_with_nan = torch.where(~torch.isnan(x), winsored_max, float('nan'))
        return x_with_nan

    @staticmethod
    def cs_demean(x):
        return x - nanmean(x, dim=1).unsqueeze(1)

    @staticmethod
    def cs_distance2mean(x):
        return abs(x - nanmean(x, dim=1).unsqueeze(1))

    @staticmethod
    def cs_cut(x, y):
        sign = op.at_sign(x - nanmean(x, dim=1).unsqueeze(1))
        return sign * y

    @staticmethod
    def cs_umr(x, y):
        s = torch.multiply(x - nanmean(y, dim=1).unsqueeze(1), y)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def cs_ind_neut(x, ind):
        res = multi_regress(x, ind)[-1]
        return res

    @staticmethod
    def cs_barra_neut(x, barra_factor):
        res = multi_regress(x, barra_factor.unsqueeze(-1))[-1]
        return res

    @staticmethod
    def cs_regress_res(x, y):
        res = multi_regress(x, y.unsqueeze(-1))[-1]
        return res

    @staticmethod
    def ts_correlation(x, y, d):
        nan_fill = torch.full((d - 1, x.shape[1]), float('nan'))
        x = x.unfold(0, d, 1)
        y = y.unfold(0, d, 1)
        correlation = corrwith(x, y, dim=-1)
        s = torch.cat([nan_fill, correlation], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_rankcorr(x, y, d):
        nan_fill = torch.full((d - 1, x.shape[1]), float('nan'))
        x = x.unfold(0, d, 1)
        y = y.unfold(0, d, 1)
        correlation = rank_corrwith(x, y, dim=-1)
        s = torch.cat([nan_fill, correlation], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_delay(x, d):
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
    def ts_delta(x, d):
        return x - op.ts_delay(x, d)

    @staticmethod
    def ts_ir(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        s = nanmean(x_3d, -1) / nanstd(x_3d, dim=-1)
        s = torch.cat([nan_fill, s], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_kurtosis(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        x_mean = nanmean(x_3d).unsqueeze(-1)
        x_4mean = nanmean(torch.pow(x_3d - x_mean, 4))
        x_4std = torch.pow(nanstd(x_3d), 4)
        s = x_4mean / x_4std
        s = torch.cat([nan_fill, s], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_skewness(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        x_mean = nanmean(x_3d).unsqueeze(-1)
        x_3mean = nanmean(torch.pow(x_3d - x_mean, 3))
        x_3std = torch.pow(nanstd(x_3d), 3)
        s = x_3mean / x_3std
        s = torch.cat([nan_fill, s], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_mean(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        x_mean = nanmean(x_3d)
        s = torch.cat([nan_fill, x_mean], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_harmonic_mean(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
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
    def ts_std(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        x_std = nanstd(x_3d)
        s = torch.cat([nan_fill, x_std], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_median(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        median = torch.median(x_3d, -1)[0]
        s = torch.cat([nan_fill, median], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_to_max(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        max_tensor = torch.max(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, max_tensor], dim=0)
        s = x / s
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_to_min(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        min_tensor = torch.min(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, min_tensor], dim=0)
        s = x / s
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_to_mean(x, lookback):
        mean_tensor = op.ts_mean(x, lookback)
        s = x / mean_tensor
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_max_to_min(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        min_tensor = torch.min(x_3d, dim=-1)[0]
        max_tensor = torch.max(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, max_tensor - min_tensor], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_to_maxmin_norm(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        min_tensor = torch.min(x_3d, dim=-1)[0]
        max_tensor = torch.max(x_3d, dim=-1)[0]
        min_tensor = torch.cat([nan_fill, min_tensor], dim=0)
        max_tensor = torch.cat([nan_fill, max_tensor], dim=0)
        s = (x - min_tensor) / (max_tensor - min_tensor)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_mon(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        summed_tensor = torch.sum(x_3d, dim=-1)
        abs_tensor = torch.sum(torch.abs(x_3d), dim=-1)
        s = torch.cat([nan_fill, summed_tensor / abs_tensor], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_product(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        prod_tensor = torch.prod(x_3d, dim=-1)
        s = torch.cat([nan_fill, prod_tensor], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_max(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        max_tensor = torch.max(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, max_tensor], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_min(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        max_tensor = torch.min(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, max_tensor], dim=0)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_pctchg_abs(x, lookback):
        s = (x - op.ts_delay(x, lookback)) / abs(op.ts_delay(x, lookback))
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_pctchg(x, lookback):
        s = (x - op.ts_delay(x, lookback)) / op.ts_delay(x, lookback)
        return torch.where((s == torch.inf) | (s == -torch.inf), float('nan'), s)

    @staticmethod
    def ts_regress_k(x, y, lookback):
        return ts_regress(x, y, lookback)[0]

    @staticmethod
    def ts_regress_b(x, y, lookback):
        return ts_regress(x, y, lookback)[1]

    @staticmethod
    def ts_regress_res(x, y, lookback):
        return ts_regress(x, y, lookback)[2]

    @staticmethod
    def ts_mask(x, y, lookback, part, method):
        if (lookback < 21) & (lookback > 9):
            x = x.unfold(0, lookback, 1)
            y = y.unfold(0, lookback, 1)

            q1 = torch.nanquantile(y, 0.25, dim=-1).unsqueeze(-1)
            q2 = torch.nanquantile(y, 0.50, dim=-1).unsqueeze(-1)
            q3 = torch.nanquantile(y, 0.75, dim=-1).unsqueeze(-1)
            mask0 = (y < q1)
            mask1 = (y > q1) & (y <= q2)
            mask2 = (y > q2) & (y <= q3)
            mask3 = (y > q3)

            mask = [mask0, mask1, mask2, mask3][part]
            nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
            if method == 'std':
                p = nanstd(torch.where(mask, x, float('nan')), dim=-1)
                p = torch.cat([nan_fill, p], dim=0)
                return torch.where((p == torch.inf) | (p == -torch.inf), float('nan'), p)

            if method == 'mean':
                p = nanmean(torch.where(mask, x, float('nan')), dim=-1)
                p = torch.cat([nan_fill, p], dim=0)
                return torch.where((p == torch.inf) | (p == -torch.inf), float('nan'), p)

            if method == 'prod':
                p = torch.prod(torch.where(mask, x, 1), dim=-1)
                p = torch.cat([nan_fill, p], dim=0)
                return torch.where((p == torch.inf) | (p == -torch.inf), float('nan'), p)

            if method == 'weight_mean':
                mask = torch.isnan(x) | torch.isnan(y)
                x = torch.where(mask, float('nan'), x)
                y = torch.where(mask, float('nan'), y)
                fill = x * y
                p = torch.nansum(torch.where(mask, float('nan'), fill), dim=-1) / torch.nansum(y, dim=-1)
                p = torch.cat([nan_fill, p], dim=0)
                return torch.where((p == torch.inf) | (p == -torch.inf), float('nan'), p)

        else:
            return x

    @staticmethod
    def ts_middle_mean(x, lookback, limit=[0.2, 0.8]):
        if lookback < 5:
            return x
        else:
            x = x.unfold(0, lookback, 1)

            q1 = torch.nanquantile(x, limit[0], dim=-1).unsqueeze(-1)
            q2 = torch.nanquantile(x, limit[1], dim=-1).unsqueeze(-1)
            mask = (x > q1) & (x <= q2)
            nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
            p = nanmean(torch.where(mask, x, float('nan')), dim=-1)
            p = torch.cat([nan_fill, p])
            return torch.where((p == torch.inf) | (p == -torch.inf), float('nan'), p)

    @staticmethod
    def ts_weight_mean(x, y, lookback):
        if lookback == 1:
            return x
        else:
            x = x.unfold(0, lookback, 1)
            y = y.unfold(0, lookback, 1)
            mask = torch.isnan(x) | torch.isnan(y)
            x = torch.where(mask, float('nan'), x)
            y = torch.where(mask, float('nan'), y)

            nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
            p = torch.nansum(x * y, dim=-1) / torch.nansum(y, dim=-1)
            p = torch.cat([nan_fill, p], dim=0)
            return torch.where((p == torch.inf) | (p == -torch.inf), float('nan'), p)


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
    daily_mean = torch.mean(M_tensor, dim=2, keepdim=True)  # shape: (num_stock, day_len, 1)

    # 减去均值
    demeaned_tensor = M_tensor - daily_mean

    return demeaned_tensor


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


def M_at_abs(M_tensor):
    return torch.abs(M_tensor)


def M_at_add(M_tensor1, M_tensor2):
    """
    将两个张量相加。
    """
    return op.at_add(M_tensor1, M_tensor2)


def M_ts_pctchg(M_tensor, lookback=1):
    """
    计算(M_tensor - M_ts_delta(M_tensor, lookback)) / M_ts_delta(M_tensor, lookback)。

    """
    return op.ts_pctchg(M_tensor, lookback)


def M_at_sub(M_tensor1, M_tensor2):
    return op.at_sub(M_tensor1, M_tensor2)


def M_at_div(M_tensor1, M_tensor2):
    return op.at_div(M_tensor1, M_tensor2)


def M_at_prod(M_tensor1, M_tensor2):
    return op.at_mul(M_tensor1, M_tensor2)


def M_cs_umr(M_tensor1, M_tensor2):
    return op.cs_umr(M_tensor1, M_tensor2)


def M_cs_cut(M_tensor1, M_tensor2):
    return op.cs_cut(M_tensor1, M_tensor2)


def M_cs_norm_spread(M_tensor1, M_tensor2):
    return op.cs_norm_spread(M_tensor1, M_tensor2)


def M_toD_standard(M_tensor, D_tensor):
    D_tensor_adjusted = D_tensor.transpose(0, 1).unsqueeze(2)
    return M_tensor / D_tensor_adjusted


def M_cs_edge_flip(M_tensor, thresh):
    flipped = torch.where((M_tensor > thresh) & (M_tensor < 1 - thresh), -M_tensor, M_tensor)
    return torch.where((M_tensor <= 0.3) | (M_tensor >= 0.7), M_tensor.flip(dims=[-1]), flipped)


def M_ts_delta(m_tensor, lookback):
    return m_tensor - m_tensor.roll(lookback, dims=-1)


def M_ts_mean_xx_neighbor(m_tensor, neighbor_range, orit):
    if orit == 0:
        return m_tensor.mean(dim=-1)
    else:
        rolled = m_tensor.roll(orit * neighbor_range, dims=-1)
        return rolled.mean(dim=-1)


def M_ts_std_xx_neighbor(m_tensor, neighbor_range, orit):
    if orit == 0:
        return m_tensor.std(dim=-1)
    else:
        rolled = m_tensor.roll(orit * neighbor_range, dims=-1)
        return rolled.std(dim=-1)


def M_ts_product_xx_neighbor(m_tensor, neighbor_range, orit):
    if orit == 0:
        return torch.prod(m_tensor, dim=-1)
    else:
        rolled = m_tensor.roll(orit * neighbor_range, dims=-1)
        return torch.prod(rolled, dim=-1)
