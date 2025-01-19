import sys
import os
import tempfile

from IPython.core.pylabtools import figsize
from matplotlib.backends.backend_pdf import PdfPages
from sympy import shape
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from io import BytesIO
sys.path.append('..')
import DataReader

from OP import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from RPN.RPNbuilder import RPN_Compiler

class GroupTest(RPN_Compiler):
    def __init__(self,factor_list:[str]):
        super().__init__()
        self.factors=factor_list
        self.in_sample_time=range(2010,2020)
        self.out_sample_time=range(2019,2022)

    def in_sample_response_rate(self):
        ft = FactorTest(factor_tensor, factor_target, self.in_sample_time, bins_num=5, factor_name='factor')
        ft.plot(output_pdf='in_sample_output.pdf')
        
    def out_sample_response_rate(self):
        ft = FactorTest(factor_tensor, factor_target, self.out_sample_time, bins_num=5, factor_name='factor')
        ft.plot(output_pdf='out_sample_output.pdf')


class FactorTest:
    def __init__(self, factor_tensor, factor_target,yearlist, bins_num,period_num=12, factor_name='factor'):
        self.data_reader = DataReader.DataReader()
        self.factor = factor_tensor
        self.factor_target = factor_target
        self.yearlist=yearlist
        self.bins_num = bins_num
        self.factor_name = factor_name
        self.period_num = period_num

        self.bins_record = None
        self.every_interval_rate = None
        self.long_turn_over = None
        self.IC_series = None
        self.industry_factor=torch.empty_like(factor_tensor)
        self.pv_factor=torch.empty_like(factor_tensor)
        self.barra_factor=torch.empty_like(factor_tensor)

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
        nav = (self.every_interval_rate[:, :-1] - 1).mean(dim=0)
        short_idx = torch.argmin(nav).item()
        mean_ret = self.every_interval_rate[:, -1]
        short_ret = self.every_interval_rate[:, short_idx]
        # 空均
        mean_short_ret = (-short_ret + mean_ret + 1)
        # Sharpe
        mean_short_sharp = (mean_short_ret - 1).mean() / ((mean_short_ret - 1).std() + 1e-12) * np.sqrt(period_num)
        return mean_short_sharp.item()

    def get_long_over_mean_sharpe(self,period_num):
        nav = (self.every_interval_rate[:, :-1] - 1).mean(dim=0)
        long_idx = torch.argmax(nav).item()
        mean_ret = self.every_interval_rate[:, -1]
        long_ret = self.every_interval_rate[:, long_idx]

        long_mean_ret = (long_ret - mean_ret + 1)
        long_mean_sharp = (long_mean_ret - 1).mean() / ((long_mean_ret - 1).std() + 1e-12) * np.sqrt(period_num)
        return long_mean_sharp.item()
    def get_mean_sharpe(self,period_num):
        mean_ret = self.every_interval_rate[:, -1]
        mean_sharp = (mean_ret - 1).mean() / ((mean_ret - 1).std() + 1e-12) * np.sqrt(period_num)
        return mean_sharp.item()

    def get_rank_IC(self):
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
        return IC_series


    def get_rank_ICIR(self,period_num):
        IC_series= self.get_rank_IC()
        IC_mean = IC_series.mean()
        IC_std = IC_series.std()
        ICIR = IC_mean / (IC_std + 1e-12) * np.sqrt(period_num)
        return ICIR

    def get_turnover(self):
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

    def get_long_equity(self):
        nav = (self.every_interval_rate[:, :-1] - 1).mean(dim=0)
        long_idx = torch.argmax(nav).item()
        daily_ret = self.every_interval_rate[:, long_idx]
        equity_curve = np.cumprod(daily_ret+1)
        print(equity_curve)
        return equity_curve

    def get_short_equity(self):
        if self.bins_record is None or self.every_interval_rate is None:
            raise ValueError("Please run get_stratified_return first.")
        nav = (self.every_interval_rate[:, :-1] - 1).mean(dim=0)
        short_idx = torch.argmin(nav).item()
        daily_ret = self.every_interval_rate[:, short_idx]
        equity_curve = np.cumprod(daily_ret+1)
        print(equity_curve)
        return equity_curve

    def get_turnover_punishment(self):
        turnover = self.get_turnover()
        mean_sharp= self.get_mean_sharpe(self.period_num)
        # 换手率惩罚千分之三
        turnover_penalty = turnover * 0.003
        penalized_sharpe = mean_sharp - turnover_penalty
        return penalized_sharpe

    def get_short_addition(self):
        long_curve = self.get_long_equity()
        short_curve = self.get_short_equity()
        long_daily = (long_curve / long_curve.roll(1, dims=0)) - 1.0
        short_daily = (short_curve / short_curve.roll(1, dims=0)) - 1.0

        # 空头加成千分之二
        short_daily_with_addition = short_daily + 0.002
        combined_daily = 0.5 * long_daily + 0.5 * short_daily_with_addition
        combined_equity = np.cumprod(combined_daily + 1.0)
        return combined_equity

    def pv_neutra(self, pv):
        T, N = self.factor.shape
        industries = self.data_reader.get_labels()
        valid_mask = ~torch.isnan(self.factor)
        valid_factors = self.factor[valid_mask]
        valid_industries = industries[valid_mask]
        unique_industries = torch.unique(valid_industries)
        industry_means = torch.zeros_like(unique_industries, dtype=torch.float)

        for i, ind in enumerate(unique_industries):
            industry_means[i] = valid_factors[valid_industries == ind].mean()
        industry_mean_map = industry_means[torch.searchsorted(unique_industries, valid_industries)]
        self.pv_factor = self.factor - industry_mean_map.unsqueeze(0)

    def industry_neutra(self):
        T, N = self.factor.shape
        industries = self.data_reader.get_barra(self.yearlist)[10:]
        valid_mask = ~torch.isnan(self.factor)
        valid_factors = self.factor[valid_mask]
        valid_industries = industries[valid_mask]
        unique_industries = torch.unique(valid_industries)
        industry_means = torch.zeros_like(unique_industries, dtype=torch.float)
        for i, ind in enumerate(unique_industries):
            industry_means[i] = valid_factors[valid_industries == ind].mean()
        industry_mean_map = industry_means[torch.searchsorted(unique_industries, valid_industries)]
        self.industry_factor = self.factor - industry_mean_map.unsqueeze(0)

    def barra_test(self):
        F = 10
        index = ['size', 'beta', 'momentum', 'volatility', 'nlsize', 'value',
                 'liquidity', 'earnings_yield', 'growth', 'leverage']
        barra_data = self.data_reader.get_barra(self.yearlist)[:10]

        results = []
        for i in range(F):
            factor = self.factor[:, i]
            target = self.factor_target

            X = barra_data[:, i].view(-1, 1)
            Y = target.view(-1, 1)

            X_transpose = X.T
            beta = torch.linalg.inv(X_transpose @ X) @ (X_transpose @ Y)
            results.append({
                'Factor': index[i],
                'Beta': beta.item(),
                'R^2': self.calculate_r2(X, Y, beta)
            })

        results_df = pd.DataFrame(results)
        return results_df
    def calculate_r2(self, X, Y, beta):
        Y_pred = X @ beta
        ss_total = torch.sum((Y - torch.mean(Y)) ** 2)
        ss_residual = torch.sum((Y - Y_pred) ** 2)
        return 1 - ss_residual / ss_total


    def plot(self, output_pdf='output.pdf'):
        self.get_stratified_return()
        monthly_returns = self.every_interval_rate[:, 4] - self.every_interval_rate[:, 0]
        if isinstance(monthly_returns, torch.Tensor):
            monthly_returns = monthly_returns.numpy()
        mean_monthly_return = np.mean(monthly_returns)  # 计算月度收益率的平均值
        annualized_return = (1 + mean_monthly_return) ** self.period_num - 1  # 计算年化收益率
        monthly_returns_std = np.std(monthly_returns)  # 计算月度收益率的标准差
        annualized_volatility = monthly_returns_std * np.sqrt(self.period_num)  # 计算年化波动率
        print(annualized_return)
        print(annualized_volatility)
        mean_sharp=self.get_mean_sharpe(self.period_num)
        long_mean_sharp = self.get_long_over_mean_sharpe(self.period_num)
        mean_short_sharp=self.get_mean_over_short_sharpe(self.period_num)
        rankic=self.get_rank_IC()
        rank_icir = self.get_rank_ICIR(self.period_num)
        turnover = self.get_turnover()
        penalized_sharpe=self.get_turnover_punishment()

        c = canvas.Canvas(output_pdf, pagesize=letter)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        bins_num = self.bins_num
        for i in range(bins_num):
            axs[0, 0].plot(np.cumprod(self.every_interval_rate[:, i] + 1), label=f'Group {i + 1}')
        axs[0, 0].plot(np.cumprod(self.every_interval_rate[:, 4]-self.every_interval_rate[:, 0] + 1), color='black', label='5-1')
        axs[0, 0].set_title('Stratified Return by Factor Bins', fontsize=14)
        axs[0, 0].set_xlabel('Time', fontsize=12)
        axs[0, 0].set_ylabel('Return', fontsize=12)
        axs[0, 0].legend()

        if self.every_interval_rate is not None:
            axs[0, 1].plot(self.get_long_equity(), label='Long Equity', color='red')
            axs[0, 1].plot(self.get_short_equity(), label='Short Equity', color='green')
            axs[0, 1].set_title(f'Long v.s. Short Equity')
            axs[0, 1].set_ylabel('Return', fontsize=12)
            axs[0, 1].legend()

        if self.IC_series is not None:
            axs[1, 0].plot(rankic, label='IC Series', color='blue')
            axs[1, 0].axhline(rank_icir, color='red', linestyle='--', label=f'ICIR: {rank_icir:.2f}')
            axs[1, 0].set_title('IC Series and ICIR')
            axs[1, 0].set_xlabel('Time')
            axs[1, 0].set_ylabel('Value')
            axs[1, 0].legend()

        if self.every_interval_rate is not None:
            axs[1, 1].plot(self.get_short_addition(), label='combined_equity', color='blue')
            axs[1, 1].set_title('combined_equity')
            axs[1, 1].set_xlabel('Time')
            axs[1, 1].set_ylabel('Return')
            axs[1, 1].legend()

        plt.tight_layout()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            img_path = tmpfile.name
            plt.savefig(img_path)
            plt.close()
            c.drawImage(img_path, 50, 400, width=500, height=400)
            c.showPage()

        fig, axs = plt.subplots(1, 2, figsize=(12, 10))
        pv_ft=FactorTest(self.pv_factor,self.factor_target,self.bins_num,self.period_num)
        for i in range(bins_num):
            axs[0, 0].plot(np.cumprod(pv_ft.every_interval_rate[:, i] + 1), label=f'Group {i + 1}')
        axs[0, 0].plot(np.cumprod(pv_ft.every_interval_rate[:, 4]-pv_ft.every_interval_rate[:, 0] + 1), color='black', label='5-1')
        axs[0, 0].set_title('Stratified Return by pv_factor Bins', fontsize=14)
        axs[0, 0].set_xlabel('Time', fontsize=12)
        axs[0, 0].set_ylabel('Return', fontsize=12)
        axs[0, 0].legend()

        industry_ft=FactorTest(self.industry_factor,self.factor_target,self.bins_num,self.period_num)
        for i in range(bins_num):
            axs[0, 0].plot(np.cumprod(industry_ft.every_interval_rate[:, i] + 1), label=f'Group {i + 1}')
        axs[0, 1].plot(np.cumprod(industry_ft.every_interval_rate[:, 4]-industry_ft.every_interval_rate[:, 0] + 1), color='black', label='5-1')
        axs[0, 1].set_title('Stratified Return by pv_factor Bins', fontsize=14)
        axs[0, 1].set_xlabel('Time', fontsize=12)
        axs[0, 1].set_ylabel('Return', fontsize=12)
        axs[0, 1].legend()

        plt.tight_layout()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            img_path = tmpfile.name
            plt.savefig(img_path)
            plt.close()
            c.drawImage(img_path, 50, 400, width=500, height=400)
            c.showPage()

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        results_df=self.barra_test()
        ax[0].bar(results_df['Factor'], results_df['Beta'], color='skyblue')
        ax[0].set_title('Barra Factor Betas', fontsize=14)
        ax[0].set_xlabel('Factor', fontsize=12)
        ax[0].set_ylabel('Beta', fontsize=12)
        ax[1].bar(results_df['Factor'], results_df['R^2'], color='lightgreen')
        ax[1].set_title('Barra R^2 Values', fontsize=14)
        ax[1].set_xlabel('Factor', fontsize=12)
        ax[1].set_ylabel('R^2', fontsize=12)
        plt.tight_layout()
        plt.show()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            img_path = tmpfile.name
            plt.savefig(img_path)
            plt.close()
            c.drawImage(img_path, 50, 400, width=500, height=400)
            c.showPage()
        c.setFont("Helvetica", 10)
        y_position = 350
        table_data = {
            'Metric': ['Annualized Return', 'Annualized Volatility', 'Mean Sharpe', 'Long Mean Sharpe',
                       'Mean Short Sharpe', 'Rank IC', 'Rank ICIR', 'Turnover','Penalized Sharpe'],
            'Value': [annualized_return, annualized_volatility, mean_sharp, long_mean_sharp,
                      mean_short_sharp, rankic.mean(), rank_icir, turnover,penalized_sharpe]
        }

        # 绘制表格
        for i in range(len(table_data['Metric'])):
            if y_position < 50:
                c.showPage()
                y_position = 750

            c.drawString(200, y_position, table_data['Metric'][i])
            data=table_data['Value'][i]
            c.drawString(400, y_position, f'{data:.2f}')
            y_position -= 20
        c.save()
        os.remove(img_path)
        print(f"PDF file saved to: {output_pdf}")


if __name__ == '__main__':
    data_reader = DataReader.DataReader()
    yearlist=[i for i in range(2010,2021,1)]
    DO, DH, DL, DC, DV = data_reader.get_Day_data(yearlist)
    # 每月价格
    MO=DO[41:-120:20,:]
    MC = DC[41:-120:20, :]
    print('shape')
    print(MO.shape)
    T, N = MC.shape
    ret20 = (MC[1:T] / MC[0:T - 1]) - 1
    print(ret20.shape)
    factor_tensor = ret20[:-1,:]
    print(factor_tensor[1:])
    ret = (MC[1:T]/ MO[0:T - 1]) - 1
    ret[torch.isinf(ret)] = torch.nan  # 替换 Inf 和 -Inf
    factor_target = ret[1:,:]
    ft = FactorTest(factor_tensor, factor_target, yearlist,bins_num=5, factor_name='factor')
    ft.plot(output_pdf='demo_output.pdf')
    print("Plot saved to demo_output.pdf")
