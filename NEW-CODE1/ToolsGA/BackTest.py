import os.path
import sys

sys.path.append('..')
import DataReader

from OP import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import torch


class FactorTest:
    def __init__(self, factor, yearlist, bins_num, period_num=252, factor_name='ret20'):
        self.data_reader = DataReader.ParquetReader()
        DO, DH, DL, DC, DV = self.data_reader.get_Day_data(yearlist)
        returns = torch.full_like(DC, 0, dtype=torch.float32)
        returns[1:] = (DC[1:] - DC[:-1]) / DC[:-1]
        index = self.data_reader.DailyDataReader.get_index(yearlist)
        self.trading_dates = pd.to_datetime(self.data_reader.DailyDataReader._TradingDate()[index]).tolist()
        self.factor = factor
        self.factor_target = returns
        self.yearlist = yearlist
        self.bins_num = bins_num
        self.factor_name = factor_name
        self.period_num = period_num

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
        return

    def factor_stratified_return(self, factor):
        tensor = factor
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
        return torch.nan_to_num(bins_return, nan=0.0)

    def get_mean_over_short_sharpe(self):
        mean_ret = self.every_interval_rate[:, -1]
        short_ret = self.every_interval_rate[:, -2]
        # 均空
        mean_short_ret = -short_ret + mean_ret
        # Sharpe
        mean_short_sharp = mean_short_ret.mean() / (mean_short_ret.std() + 1e-12) * np.sqrt(self.period_num)
        return mean_short_sharp

    def get_long_over_mean_sharpe(self):
        mean_ret = self.every_interval_rate[:, -1]
        long_ret = self.every_interval_rate[:, 0]
        # 多均
        long_mean_ret = (long_ret - mean_ret)
        # Sharpe
        long_mean_sharp = long_mean_ret.mean() / (long_mean_ret.std() + 1e-12) * np.sqrt(self.period_num)
        return long_mean_sharp

    def get_long_short_sharpe(self):
        mean_ret = self.every_interval_rate[:, 0] - self.every_interval_rate[:, -2]
        mean_sharp = mean_ret.mean() / (mean_ret.std() + 1e-12) * np.sqrt(self.period_num)
        return mean_sharp

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

    def get_rank_ICIR(self):
        IC_series = self.get_rank_IC().dropna()
        IC_mean = IC_series.mean()
        IC_std = IC_series.std()
        ICIR = IC_mean / (IC_std + 1e-12) * np.sqrt(self.period_num)
        return ICIR

    def get_turnover(self):
        if self.every_interval_rate is None:
            self.get_stratified_return()
        bins = self.bins_record
        non_nan_mask = ~torch.isnan(bins)
        prev_state = bins[:-1]
        curr_state = bins[1:]
        valid_mask = ~(torch.isnan(curr_state) | torch.isnan(prev_state))
        state_change = torch.zeros_like(curr_state, dtype=torch.int)
        state_change[valid_mask] = (curr_state[valid_mask] != prev_state[valid_mask]).to(torch.int)

        a1_tensor = state_change.sum(axis=1)
        a2_tensor = non_nan_mask.sum(dim=1)[1:]

        a_list = a1_tensor.tolist()
        a2_list = a2_tensor.tolist()

        turnover = np.array(a_list) / np.array(a2_list)
        adjusted_turnover = np.zeros_like(bins[:, 0])
        adjusted_turnover[1:] = turnover
        return adjusted_turnover

    def get_long_equity(self, bins_num=0) -> pd.DataFrame:
        if self.every_interval_rate is None:
            self.get_stratified_return()
        bins = self.bins_record
        long_mask = bins == bins_num
        stock_codes = self.data_reader.DailyDataReader._StockCodes().tolist()
        index = self.data_reader.DailyDataReader.get_index(yearlist)
        trading_dates = pd.to_datetime(self.data_reader.DailyDataReader._TradingDate()[index])

        stock_array = np.tile(stock_codes, (len(trading_dates), 1))
        result = np.where(long_mask, stock_array, None)
        df = pd.DataFrame(data=result, index=trading_dates, columns=stock_codes)
        return df

    def get_short_equity(self):
        return self.get_long_equity(-2)

    def get_turnover_punishment(self):
        turnover = self.get_turnover()
        turnover_penalty = turnover * 0.003
        adjusted_returns = self.every_interval_rate[:, 0] - self.every_interval_rate[:, -2] - turnover_penalty
        penalized_sharpe = adjusted_returns.mean() / (adjusted_returns.std() + 1e-12) * np.sqrt(self.period_num)
        return penalized_sharpe

    def get_short_addition(self):
        adjusted_returns = (self.every_interval_rate[:, -1] - self.every_interval_rate[:, -2]) * 1.3
        sharpe = adjusted_returns.mean() / (adjusted_returns.std() + 1e-12) * np.sqrt(self.period_num)
        return sharpe

    def pv_neutra(self):
        index = self.data_reader.DailyDataReader.get_index(yearlist)
        pv = self.data_reader.DailyDataReader.get_pv().iloc[index]
        pv_tensor = torch.tensor(pv.values, dtype=torch.float32)
        k, b, res = OP_Basic.regress(self.factor, pv_tensor)
        return res

    def industry_neutra(self):
        industries = self.data_reader.get_barra(self.yearlist)[..., 10:]
        k, b, res = OP_Basic.regress(self.factor, industries)
        return res

    def barra_test(self):
        barra_data = self.data_reader.get_barra(self.yearlist)[..., :10]
        k, b, res = OP_Basic.regress(self.factor, barra_data)
        return k, res

    def plot_stratified_rtn(self, every_interval_rate, factor_name, ax=None):
        if ax is None:
            ax = plt.gca()
        bins_num = self.bins_num
        for i in range(bins_num):
            cumulative_rtn = np.cumprod(every_interval_rate[:, i] + 1)
            ax.plot(self.trading_dates,cumulative_rtn, label=f'Group {i + 1}')
        ret=every_interval_rate[:, 0] - every_interval_rate[:, -2]
        long_short = np.cumprod(
            (every_interval_rate[:, 0] - every_interval_rate[:, -2]) + 1
        )
        sharp = ret.mean() / (ret.std() + 1e-12) * np.sqrt(self.period_num)
        ax.plot(self.trading_dates,long_short,
                color='black',
                linewidth=2,
                label=f'Long-Short, Sharp: {sharp:.2f}')

        ax.set_title(f'Stratified Returns: {factor_name}', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend(fontsize=6)
        return ax

    def plot_long_short(self, every_interval_rate, factor_name, ax=None):
        if ax is None:
            ax = plt.gca()
        long_mean=every_interval_rate[:, 0] - every_interval_rate[:, -1]
        mean_short=every_interval_rate[:, -1] - every_interval_rate[:, -2]
        long_short=every_interval_rate[:, 0] - every_interval_rate[:, -2]
        long_rtn = np.cumprod(every_interval_rate[:, 0] - every_interval_rate[:, -1] + 1)
        short_rtn = np.cumprod(every_interval_rate[:, -1] - every_interval_rate[:, -2] + 1)
        spread = np.cumprod((every_interval_rate[:, 0] - every_interval_rate[:, -2]) + 1)

        long_short_sharp = long_short.mean() / (long_short.std() + 1e-12) * np.sqrt(self.period_num)
        mean_short_sharp = mean_short.mean() / (mean_short.std() + 1e-12) * np.sqrt(self.period_num)
        long_mean_sharp = long_mean.mean() / (long_mean.std() + 1e-12) * np.sqrt(self.period_num)

        ax.plot(self.trading_dates,long_rtn, color='red', label=f'Long-Mean, Sharp: {long_mean_sharp:.2f}')
        ax.plot(self.trading_dates,short_rtn, color='green', label=f'Mean-Short, Sharp: {mean_short_sharp:.2f}')
        ax.plot(self.trading_dates,spread, color='black', linewidth=2, label=f'Long-Short, Sharp: {long_short_sharp:.2f}')
        ax.set_title(f'Long/Short Performance Comparison: {factor_name}', fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
        return ax

    def get_annual_metrics(self):
        index = self.data_reader.DailyDataReader.get_index(yearlist)
        dates = pd.to_datetime(self.data_reader.DailyDataReader._TradingDate()[index])
        annual_data = []
        for year in sorted(self.yearlist):
            year_mask = (dates.dt.year == year).to_numpy()
            long_ret = self.every_interval_rate[year_mask, 0]
            mean_ret = self.every_interval_rate[year_mask, -1]
            short_ret = self.every_interval_rate[year_mask, -2]

            annual_long = torch.prod(long_ret-mean_ret + 1) - 1
            annual_short = torch.prod(-short_ret+mean_ret + 1) - 1
            annual_spread = torch.prod((long_ret - short_ret) + 1) - 1

            spread_returns = (long_ret - short_ret)
            sharpe = spread_returns.mean() / (spread_returns.std() + 1e-12) * np.sqrt(len(spread_returns))
            long_mean_sharp=(long_ret - mean_ret).mean() / ((long_ret - mean_ret).std() + 1e-12) * np.sqrt(len(spread_returns))
            mean_short_sharp=(mean_ret - short_ret).mean() / ((mean_ret - short_ret).std() + 1e-12) * np.sqrt(len(spread_returns))

            year_ic = self.get_rank_IC()[year_mask]
            valid_ic = year_ic.dropna()
            ic_mean = valid_ic.mean()
            ic_ir = ic_mean / valid_ic.std() * np.sqrt(len(spread_returns))

            annual_data.append([
                str(year),
                f"{annual_long.item():.2%}",  # 多头组收益
                f"{annual_short.item():.2%}",  # 空头组收益
                f"{annual_spread.item():.2%}",  # 多空价差收益
                f"{long_mean_sharp.item():.2f}",  # 多空夏普
                f"{mean_short_sharp.item():.2f}",  # 多空夏普
                f"{sharpe.item():.2f}",  # 多空夏普
                f"{ic_mean:.2f}",  # Rank IC均值
                f"{ic_ir:.2f}"
            ])
        return annual_data

    def plot(self, output_path=None):
        # ----------------- 存储路径 -----------------
        # 输入的output_path指向结果文件夹
        if output_path is None:
            output_path = f'Backtest over Factor {self.factor_name}.png'
        else:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_path = os.path.join(output_path, f'Backtest over Factor {self.factor_name}.png')

        self.get_stratified_return()
        long_short_sharp = self.get_long_short_sharpe()
        mean_short_sharp = self.get_mean_over_short_sharpe()
        long_mean_sharp = self.get_long_over_mean_sharpe()

        # ----------------- 布局参数 -----------------
        fig = plt.figure(figsize=(10, 16), constrained_layout=True)
        gs = fig.add_gridspec(
            6, 2,
            height_ratios=[2.8, 2.8, 2.8, 2.8, 0.5, 2.8],
            hspace=0.05,
            wspace=0
        )

        # ----------------- 图表区域-----------------
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_stratified_rtn(self.every_interval_rate, self.factor_name, ax=ax1)

        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)
        self.plot_long_short(self.every_interval_rate, self.factor_name, ax=ax2)

        ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3.plot(self.trading_dates,np.cumsum(self.get_rank_IC()), label=f'Cumulative Rank IC, Mean: {self.get_rank_IC().mean():.2f}')
        ax3.plot([],[],label=f'Rank ICIR: {self.get_rank_ICIR():.2f}')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax3.xaxis.set_major_locator(mdates.YearLocator())
        ax3.tick_params(axis='x', labelrotation=45)
        ax3.legend()
        ax3.set_title('Rank IC Trend Analysis', fontsize=12)

        k, barrra_factor = self.barra_test()
        every_interval_rate = self.factor_stratified_return(barrra_factor)
        ax4 = fig.add_subplot(gs[1, 1], sharex=ax1)
        self.plot_long_short(every_interval_rate, 'Barra Neutralization', ax=ax4)
        ax4.legend()

        ax5 = fig.add_subplot(gs[2, 0])
        barra_labels = ['Size', 'Beta', 'Momentum', 'Vol', 'NonLinSize',
                        'Value', 'Liquidity', 'Earnings', 'Growth', 'Leverage']
        ax5.barh(barra_labels, k.mean(axis=0), color='steelblue')
        ax5.set_title('Average Barra Factor Exposure', fontsize=12)

        ax6 = fig.add_subplot(gs[2, 1], sharex=ax1)
        turnover = self.get_turnover()
        ax6.plot(self.trading_dates,turnover, label='Daily Turnover')
        ax6.axhline(np.mean(turnover), color='red', linestyle='--', label=f'Mean: {np.mean(turnover):.2f}')
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax6.xaxis.set_major_locator(mdates.YearLocator())
        ax6.tick_params(axis='x', labelrotation=45)
        ax6.legend()
        ax6.set_title('Turnover Analysis', fontsize=12)

        ax7 = fig.add_subplot(gs[3, 0])
        sharpe_values = [
            self.get_long_short_sharpe(),
            self.get_short_addition(),
            self.get_turnover_punishment()
        ]
        labels = ['Long-Short Sharpe', 'Short', 'Turnover']
        bars = ax7.bar(labels, sharpe_values, color='skyblue')
        for bar in bars:
            yval = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width() / 2,
                     yval,
                     f'{yval:.2f}',
                     ha='center', va='bottom',
                     fontsize=9)

        ax7.set_title('Sharpe Ratios', fontsize=10)
        ax7.grid(axis='y', alpha=0.3)

        # 分年度统计表
        ax_table1 = fig.add_subplot(gs[4, :])
        ax_table1.axis('off')
        annual_data = self.get_annual_metrics()
        annual_headers = ["Year", "Long-Mean", "Mean-Short", "Long-Short",
                           "Long-Mean Sharpe","Mean-Short", "Long-Short","Rank IC", "ICIR"]
        annual_table = ax_table1.table(
            cellText=annual_data,
            colLabels=annual_headers,
            loc='center',
            cellLoc='center',
            colColours=['#f0f0f0'] * len(annual_headers)
        )
        annual_table.auto_set_font_size(False)
        annual_table.set_fontsize(8)
        annual_table.scale(1, 1.8)

        # 汇总指标
        ax_table2 = fig.add_subplot(gs[5, :])
        ax_table2.axis('off')

        monthly_returns = self.every_interval_rate[:, 0] - self.every_interval_rate[:, -2]
        mean_monthly_return = monthly_returns.mean()
        annualized_return = (1 + mean_monthly_return) ** self.period_num - 1
        monthly_returns_std = monthly_returns.std()
        annualized_volatility = monthly_returns_std * np.sqrt(self.period_num)


        metrics = [
            ["Annualized Return", f"{annualized_return:.2%}"],
            ["Annualized Volatility", f"{annualized_volatility:.2%}"],
            ["Long-Short Sharpe", f"{long_short_sharp:.2f}"],
            ["Long-Mean Sharpe", f"{long_mean_sharp:.2f}"],
            ["Mean-Short Sharpe", f"{mean_short_sharp:.2f}"],
            ["Rank IC", f"{np.mean(self.get_rank_IC()):.2f}"],
            ["ICIR", f"{np.mean(self.get_rank_ICIR()):.2f}"],
            ["Turnover", f"{np.mean(self.get_turnover()):.2f}"]
        ]
        summary_table = ax_table2.table(
            cellText=metrics,
            colLabels=["Metric", "Value"],
            loc='center',
            cellLoc='center',
            colColours=['#f0f0f0', '#f8f8f8']
        )
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(10)
        summary_table.scale(1, 2)

        # 表格样式
        for table in [annual_table, summary_table]:
            for (i, j), cell in table.get_celld().items():
                if i == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#40466e')

        # ----------------- 保存参数调整 -----------------
        plt.savefig(
            output_path,
            dpi=250,
            bbox_inches='tight',
            pad_inches=0.08
        )
        plt.show()
        plt.close()
        print(f"results saved：{output_path}")


if __name__ == '__main__':
    yearlist = [i for i in range(2010, 2021, 1)]

    DO, DH, DL, DC, DV = DataReader.ParquetReader().get_Day_data(yearlist)
    returns = torch.full_like(DC, 0, dtype=torch.float32)
    returns[1:] = (DC[1:] - DC[:-1]) / DC[:-1]

    def compute_factor(return_tensor):
        days, stocks = return_tensor.shape
        factor = torch.zeros_like(return_tensor)
        start_indices = list(range(20, days + 1, 20))
        for start_idx in start_indices:
            window = return_tensor[start_idx - 20:start_idx, :]
            cum_returns = torch.prod(1 + window, dim=0) - 1
            fill_end = min(start_idx + 20, days)
            factor[start_idx:fill_end, :] = cum_returns.unsqueeze(0).expand(fill_end - start_idx, -1)
        return factor

    # invoke example
    factor = compute_factor(returns)
    ft = FactorTest(factor, yearlist, bins_num=5, factor_name='ret20')
    # saved under folder res
    ft.plot('res')

    #中性化后的因子值 e.g.pv
    # pv_factor=ft.pv_neutra()
    # every_interval_rate=ft.factor_stratified_return(pv_factor)
    # fig,ax = plt.subplots(figsize=(8, 6))
    # ft.plot_stratified_rtn(every_interval_rate,'factor_name',ax=ax)
    # plt.show()
