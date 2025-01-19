import sys
sys.path.append('..')

from .GA_tools import *
from OP import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


class GroupTest(RPN_Compiler):
    def __init__(self,factor_list:[str]):
        super().__init__()
        self.factors=factor_list
        self.in_sample_time=range(2016,2019)
        self.out_sample_time=range(2019,2022)

    def in_sample_response_rate(self):
        pass

    def out_sample_response_rate(self):
        pass


class FactorTest:
    def __init__(self, factor_tensor, factor_target, bins_num, factor_name='factor'):
        self.factor = factor_tensor
        self.factor_target = factor_target
        self.bins_num = bins_num
        self.factor_name = factor_name

        self.bins_record = None
        self.every_interval_rate = None
        self.long_turn_over = None
        self.IC_series = None

    def get_stratified_return(self):
        tensor = self.factor
        target = self.factor_target
        bins_num = self.bins_num

        quantiles = torch.linspace(0, 1, bins_num + 1, device=tensor.device)
        q = quantiles[1:-1]

        boundaries = torch.nanquantile(tensor, q, dim=1).T
        boundaries_all_nan = torch.isnan(boundaries).all(dim=1)

        comparison = tensor.unsqueeze(-1) > boundaries.unsqueeze(1)
        bins = comparison.sum(dim=-1).float()
        bins[boundaries_all_nan, :] = float('nan')
        mask = torch.isnan(tensor)
        bins[mask] = float('nan')

        # 计算每个桶的平均收益
        bin_indices = torch.arange(bins_num, device=tensor.device).view(1, 1, bins_num).float()
        mask_b = (bins.unsqueeze(-1) == bin_indices)

        masked_target = torch.where(
            mask_b,
            target.unsqueeze(-1),
            torch.full_like(target.unsqueeze(-1), float('nan'))
        )
        group_ret = torch.nanmean(masked_target, dim=1)
        all_mean_ret = torch.nanmean(target, dim=1, keepdim=True)
        bins_return = torch.cat([group_ret, all_mean_ret], dim=1)
        self.every_interval_rate = torch.nan_to_num(bins_return, nan=0.0)
        self.bins_record = bins

    def get_mean_over_short_sharpe(self,period_num):
        # 计算多均与空均Sharpe。
        # 其中 bins_num+1列中的最后一列为ret_mean平均收益，其余为各组收益
        nav = (self.every_interval_rate[:, :-1] - 1).mean(dim=0)
        long_idx = torch.argmax(nav).item()
        short_idx = torch.argmin(nav).item()

        mean_ret = self.every_interval_rate[:, -1]
        long_ret = self.every_interval_rate[:, long_idx]
        short_ret = self.every_interval_rate[:, short_idx]

        # 多均
        long_mean_ret = (long_ret - mean_ret + 1)
        # 空均
        mean_short_ret = (-short_ret + mean_ret + 1)

        # Sharpe
        long_mean_sharp = (long_mean_ret - 1).mean() / ((long_mean_ret - 1).std() + 1e-12) * np.sqrt(period_num)
        mean_short_sharp = (mean_short_ret - 1).mean() / ((mean_short_ret - 1).std() + 1e-12) * np.sqrt(period_num)

        return long_mean_sharp.item(), mean_short_sharp.item()

    def get_long_over_mean_sharpe(self,period_num):
        nav = (self.every_interval_rate[:, :-1] - 1).mean(dim=0)
        long_idx = torch.argmax(nav).item()
        short_idx = torch.argmin(nav).item()

        mean_ret = self.every_interval_rate[:, -1]
        long_ret = self.every_interval_rate[:, long_idx]
        short_ret = self.every_interval_rate[:, short_idx]

        long_mean_ret = (long_ret - mean_ret + 1)
        long_mean_sharp = (long_mean_ret - 1).mean() / ((long_mean_ret - 1).std() + 1e-12) * np.sqrt(period_num)
        # mean_sharp
        mean_sharp = (mean_ret - 1).mean() / ((mean_ret - 1).std() + 1e-12) * np.sqrt(period_num)

        return long_mean_sharp.item(), mean_sharp.item()

    def get_rank_IC(self):
        pass

    def get_rank_ICIR(self):
        T, N = self.factor.shape
        mask = (~torch.isnan(self.factor)) & (~torch.isnan(self.factor_target))
        valid_counts = mask.sum(dim=1)
        IC_list = torch.full((T,), float('nan'), dtype=torch.float)
        valid_indices = valid_counts >= 5
        if valid_indices.any():
            valid_factors = self.factor[valid_indices]
            valid_targets = self.factor_target[valid_indices]
            valid_mask = mask[valid_indices]
            IC_values = torch.tensor([
                OP_Basic.rank_corrwith(f[valid], r[valid]).item()
                for f, r, valid in zip(valid_factors, valid_targets, valid_mask)
            ])
            IC_list[valid_indices] = IC_values

        IC_series = pd.Series(IC_list)
        self.IC_series = IC_series
        IC_mean = IC_series.mean()
        IC_std = IC_series.std()
        ICIR = IC_mean / (IC_std + 1e-12) * np.sqrt(self.period_num)
        return ICIR

    def get_turnover(self):
        if self.bins_record is None:
            raise ValueError("Please run get_stratified_return first")

        bins = self.bins_record
        curr_top = bins[1:] == 0
        prev_top = bins[:-1] == 0
        a1_tensor = (curr_top & prev_top).sum(dim=1)
        a2_tensor = prev_top.sum(dim=1)

        valid_indices = a2_tensor > 0
        a_list = a1_tensor[valid_indices].tolist()
        a2_list = a2_tensor[valid_indices].tolist()

        if len(a2_list) == 0:
            turnover = float('nan')
        else:
            turnover = np.nanmean(np.array(a_list) / np.array(a2_list))
        self.long_turn_over = turnover
        return turnover

    def get_long_equity(self)->TypeC:
        pass

    def get_short_equity(self)->TypeC:
        pass

    def get_turnover_punishment(self):
        pass

    def get_short_addition(self):
        pass

    def pv_neutra(self,pv):
        pass

    def industry_neutra(self):
        pass

    def barra_test(self):
        pass

    def plot(self):
        pass

