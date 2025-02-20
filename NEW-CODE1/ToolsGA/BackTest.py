import os.path
import sys

import pandas as pd

sys.path.append('..')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ToolsGA.DataReader import *


class FactorTest:
    def __init__(self, factor, factor_target, bins_num, freq=1, factor_name='ret20'):
        self.factor = factor.astype(np.float32)
        self.factor_target = factor_target.astype(np.float32)
        self.bins_num = bins_num
        self.factor_name = factor_name
        self.freq = freq
        self._init_data()

    def _init_data(self):
        self.factor[self.factor_target.isna()]=np.nan
        self.factor_target[self.factor.isna()] = np.nan

        self.trading_dates = self.factor.index
        self.stock_codes = self.factor.columns

        barra_reader = MmapReader()
        self.barra=torch.full((len(self.trading_dates),len(self.stock_codes),41),float('nan'))
        for i,day in enumerate(self.trading_dates):
            self.barra[i]=barra_reader.get_Barra_daily(day)


    @staticmethod
    def get_stratified_return(tensor: torch.Tensor, target: torch.Tensor, bins_num):
        quantiles = torch.linspace(0, 1, bins_num + 1, device=tensor.device)
        boundaries = torch.nanquantile(tensor, quantiles[1:-1], dim=1).T

        comparison = tensor.unsqueeze(-1) > boundaries.unsqueeze(1)
        bins = comparison.sum(dim=-1).float()
        mask = torch.isnan(tensor) | torch.isnan(target)
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
        all_mean_ret = torch.nanmean(torch.where(~torch.isnan(tensor), target, float('nan')), dim=1, keepdim=True)

        every_interval_rate = torch.cat([group_ret, all_mean_ret], dim=1)
        bins_record = bins

        return every_interval_rate, bins_record

    def get_rank_IC(self, factor: torch.Tensor, keep=False):
        target = torch.tensor(self.factor_target.values)
        IC_list = np.array(OP_Basic.rank_corrwith(factor, target))
        rank_IC = pd.Series(IC_list, index=self.factor_target.index, name='IC')
        if keep:
            self.direction = rank_IC.mean() > 0
            self.rank_IC = rank_IC
        return rank_IC

    def get_rank_ICIR(self, IC_series, keep=False):
        IC_mean = IC_series.mean()
        IC_std = IC_series.std()
        ICIR = IC_mean / IC_std * np.sqrt(252 / self.freq)
        if keep:
            self.ICIR = ICIR
        return ICIR

    def get_long_short_sharpe(self, every_interval_rate, direction=True):
        mean_ret = every_interval_rate[:, -1]
        if direction:
            short_ret = every_interval_rate[:, 0]
            long_ret = every_interval_rate[:, -2]
        else:
            short_ret = every_interval_rate[:, -2]
            long_ret = every_interval_rate[:, 0]

        mean_short_ret = (-short_ret + mean_ret)
        long_mean_ret = (long_ret - mean_ret)
        long_short_ret = (long_ret - short_ret)
        mean_short_sharp = (OP_Basic.nanmean(mean_short_ret)) / OP_Basic.nanstd(mean_short_ret) * np.sqrt(
            252 / self.freq)
        long_mean_sharp = (OP_Basic.nanmean(long_mean_ret)) / OP_Basic.nanstd(long_mean_ret) * np.sqrt(252 / self.freq)
        long_short_sharp = (OP_Basic.nanmean(long_short_ret)) / OP_Basic.nanstd(long_short_ret) * np.sqrt(
            252 / self.freq)

        sharp = [mean_short_sharp.item(), long_mean_sharp.item(), long_short_sharp.item()]
        long_short_decompose = [mean_short_ret, long_mean_ret, long_short_ret]

        return long_short_decompose, sharp

    def get_turnover(self, bins_record):
        non_nan_mask = ~torch.isnan(bins_record)
        prev_state = bins_record[:-1]
        curr_state = bins_record[1:]
        valid_mask = ~(torch.isnan(curr_state) | torch.isnan(prev_state))

        state_change = torch.full_like(curr_state, False, dtype=torch.bool)
        state_change[valid_mask] = (curr_state[valid_mask] != prev_state[valid_mask])

        changed_num = state_change.sum(dim=1)
        total_num = non_nan_mask.sum(dim=1)[1:]

        turnover = np.asarray(changed_num.tolist()) / np.asarray(total_num.tolist())

        return pd.Series(np.concatenate(([0], turnover)), index=self.trading_dates, name='turnover')

    def get_equity(self, bins, direction=True):
        if direction:
            long_mask = bins == (self.bins_num - 1)
            short_mask = bins == 0
        else:
            long_mask = bins == 0
            short_mask = bins == (self.bins_num - 1)


        return (pd.DataFrame(long_mask, index=self.trading_dates, columns=self.stock_codes),
                pd.DataFrame(short_mask, index=self.trading_dates, columns=self.stock_codes))

    def neutra(self, factor):
        pv_reader=DailyDataReader()
        pv = pv_reader.get_pv().loc[self.trading_dates]
        pv_tensor = torch.tensor(pv.values, dtype=torch.float32)
        k, b, res = OP_Basic.regress(factor, pv_tensor)
        k, b, res = OP_Basic.multi_regress(res.unsqueeze(-1), self.barra[:, :, 10:])
        return res.squeeze(-1)

    def barra_test(self, factor, keep=False):
        k, b, res = OP_Basic.multi_regress(factor.unsqueeze(-1), self.barra[:, :, :10])
        barra_corr = []
        for i in range(10):
            barra_corr.append(OP_Basic.nanmean(OP_Basic.rank_corrwith(factor, self.barra[:, :, i])).item())
        if keep:
            self.barra_corr = np.array(barra_corr)
            self.pure_factor = res.squeeze(-1)
        return barra_corr, res.squeeze(-1)

    def plot_stratified_rtn(self, every_interval_rate, factor_name, ax=None):
        if ax is None:
            ax = plt.gca()
        bins_num = self.bins_num
        every_interval_rate = torch.where(torch.isnan(every_interval_rate), 1, every_interval_rate)

        for i in range(bins_num):
            cumulative_rtn = np.cumprod(every_interval_rate[:, i])
            ax.plot(self.trading_dates, cumulative_rtn, label=f'Group {i + 1}')

        ax.set_title(f'Stratified Returns: {factor_name}', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend(fontsize=6)
        return ax

    def plot_long_short(self, long_short_decompose, sharp, factor_name, ax=None):
        if ax is None:
            ax = plt.gca()
        long_short_decompose = [torch.where(torch.isnan(ret), 1, ret + 1) for ret in long_short_decompose]

        mean_short_ret, long_mean_ret, long_short_ret = long_short_decompose
        mean_short_sharp, long_mean_sharp, long_short_sharp = sharp

        long_rtn = np.cumprod(long_mean_ret)
        short_rtn = np.cumprod(mean_short_ret)
        spread = np.cumprod(long_short_ret)

        ax.plot(self.trading_dates, long_rtn, color='red', label=f'Long-Mean, Sharp: {long_mean_sharp:.2f}')
        ax.plot(self.trading_dates, short_rtn, color='green', label=f'Mean-Short, Sharp: {mean_short_sharp:.2f}')
        ax.plot(self.trading_dates, spread, color='black', linewidth=2,
                label=f'Long-Short, Sharp: {long_short_sharp:.2f}')
        ax.set_title(f'Long/Short Performance Comparison: {factor_name}', fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
        return ax

    def plot_ICIR(self, ax):
        if ax is None:
            ax = plt.gca()

        Orig_IC_list = self.get_rank_IC(torch.tensor(self.factor.values), keep=True)
        Orig_IC_IR = self.get_rank_ICIR(Orig_IC_list, keep=True)

        Pure_IC_list = self.get_rank_IC(self.pure_factor)
        Pure_IC_IR = self.get_rank_ICIR(Pure_IC_list)

        ax.plot(self.trading_dates, np.cumsum(Orig_IC_list),
                label=f'Orig_CRIC, Mean: {Orig_IC_list.mean():.2f}')
        ax.plot([], [], label=f'Orig_CRICIR: {Orig_IC_IR:.2f}')

        ax.plot(self.trading_dates, np.cumsum(Pure_IC_list),
                label=f'Pure_CRIC, Mean: {Pure_IC_list.mean():.2f}')
        ax.plot([], [], label=f'Pure_CRICIR: {Pure_IC_IR:.2f}')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
        ax.set_title('Rank IC Trend Analysis', fontsize=12)

    def get_annual_metrics(self, long_short_decompose):
        mean_short_ret, long_mean_ret, spread_returns = [ret + 1 for ret in long_short_decompose]

        annual_data = []
        for year in self.trading_dates.year.unique():
            year_mask = np.array(self.trading_dates.year == year)
            mask = np.array(~torch.isnan(long_mean_ret)) & year_mask

            year_tensor = torch.tensor(self.factor[mask].values)
            year_target = torch.tensor(self.factor_target[mask].values)

            annual_long = torch.prod(long_mean_ret[mask]) ** (np.sum(mask) * self.freq / 252) - 1
            annual_short = torch.prod(mean_short_ret[mask]) ** (np.sum(mask) * self.freq / 252) - 1
            annual_spread = torch.prod(spread_returns[mask]) ** (np.sum(mask) * self.freq / 252) - 1

            sharpe = (spread_returns[mask].mean() - 1) / spread_returns[mask].std() * np.sqrt(np.sum(mask))
            long_mean_sharp = (long_mean_ret[mask].mean() - 1) / long_mean_ret[mask].std() * np.sqrt(
                np.sum(mask))
            mean_short_sharp = (mean_short_ret[mask].mean() - 1) / mean_short_ret[mask].std() * np.sqrt(
                np.sum(mask))

            year_ic = np.array(OP_Basic.rank_corrwith(year_tensor, year_target))
            ic_mean = year_ic.mean()
            ic_ir = ic_mean / year_ic.std() * np.sqrt(np.sum(mask))

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

        # ----------------- 布局参数 -----------------
        fig = plt.figure(figsize=(10, 16), constrained_layout=True)
        gs = fig.add_gridspec(
            5, 2,
            height_ratios=[2.8, 2.8, 2.8, 0.5, 2.8],
            hspace=0.05,
            wspace=0
        )

        # ----------------- 图表区域-----------------

        every_interval_rate, bins_record = self.get_stratified_return(
            torch.tensor(self.factor.values, dtype=torch.float32),
            torch.tensor(self.factor_target.values),
            self.bins_num
        )
        self.every_interval_rate, self.bins_record = every_interval_rate, bins_record
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_stratified_rtn(every_interval_rate, self.factor_name, ax=ax1)

        Orig_IC_list = self.get_rank_IC(torch.tensor(self.factor.values), keep=True)
        long_short_decompose, sharp = self.get_long_short_sharpe(every_interval_rate, self.direction)
        mean_short_sharp, long_mean_sharp, long_short_sharp = sharp
        ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
        self.plot_long_short(long_short_decompose, sharp, self.factor_name, ax=ax2)

        pv_ind_neutra_factor = self.neutra(torch.tensor(self.factor.values))
        barra_corr, pure_factor = self.barra_test(pv_ind_neutra_factor, keep=True)
        pure_every_interval_rate, pure_bins_record = self.get_stratified_return(
            pure_factor,
            torch.tensor(self.factor_target.values),
            self.bins_num)
        pure_long_short_decompose, pure_sharp = self.get_long_short_sharpe(pure_every_interval_rate, self.direction)
        ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
        self.plot_long_short(pure_long_short_decompose, pure_sharp, 'Barra Neutralization', ax=ax3)
        ax3.legend()

        ax4 = fig.add_subplot(gs[0, 1], sharex=ax1)
        self.plot_ICIR(ax=ax4)

        ax5 = fig.add_subplot(gs[2, 0])
        barra_labels = ['Size', 'Beta', 'Momentum', 'Vol', 'NonLinSize',
                        'Value', 'Liquidity', 'Earnings', 'Growth', 'Leverage']
        ax5.barh(barra_labels, barra_corr, color='steelblue')
        ax5.set_title('Average Barra Factor Exposure', fontsize=12)

        ax6 = fig.add_subplot(gs[2, 1], sharex=ax1)
        turnover = self.get_turnover(bins_record)
        self.turn_over = turnover
        ax6.bar(self.trading_dates, turnover, label='Daily Turnover', width=20)
        ax6.axhline(np.mean(turnover), color='red', linestyle='--', label=f'Mean: {np.mean(turnover):.2f}')
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax6.xaxis.set_major_locator(mdates.YearLocator())
        ax6.tick_params(axis='x', labelrotation=45)
        ax6.legend()
        ax6.set_title('Turnover Analysis', fontsize=12)

        # 分年度统计表
        ax_table1 = fig.add_subplot(gs[3, :])
        ax_table1.axis('off')
        annual_data = self.get_annual_metrics(long_short_decompose)
        annual_headers = ["Year", "Long-Mean", "Mean-Short", "Long-Short",
                          "Long-Mean Sharpe", "Mean-Short", "Long-Short", "Rank IC", "ICIR"]
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
        ax_table2 = fig.add_subplot(gs[4, :])
        ax_table2.axis('off')

        long_short_ret = np.array(long_short_decompose[-1])
        long_short_ret = long_short_ret[~np.isnan(long_short_ret)]

        annualized_return = np.prod(long_short_ret + 1) ** (252 / self.freq / len(long_short_ret)) - 1
        annualized_volatility = np.std(long_short_ret) * np.sqrt(252 / self.freq)

        metrics = [
            ["Annualized Return", f"{annualized_return:.2%}"],
            ["Annualized Volatility", f"{annualized_volatility:.2%}"],
            ["Long-Short Sharpe", f"{long_short_sharp:.2f}"],
            ["Long-Mean Sharpe", f"{long_mean_sharp:.2f}"],
            ["Mean-Short Sharpe", f"{mean_short_sharp:.2f}"],
            ["Rank IC", f"{self.rank_IC.mean():.2f}"],
            ["ICIR", f"{self.ICIR:.2f}"],
            ["Turnover", f"{np.mean(turnover):.2f}"]
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

class TensorTest:
    def __init__(self,factor,year_list):
        self.factor=factor
        self.year_list=year_list




if __name__ == '__main__':
    from ToolsGA.Data_tools import *

    # reader=DailyDataReader()
    # D_O,D_H,D_L,D_C=reader.GetOHLC()
    # D_O, D_H, D_L, D_C,D_V=reader.get_df_ohlcv(D_O,D_H,D_L,D_C)
    #
    # buy_price=D_O.loc['2010':'2020']
    # sell_price=D_C.loc['2010':'2020']
    #
    # buydays=get_first_day(buy_price)
    # selldays=get_last_day(sell_price)
    #
    # buy_price = buy_price.loc[buydays]
    # sell_price = sell_price.loc[selldays]
    #
    # factor=sell_price/sell_price.shift(1)
    # factor_target=pd.DataFrame(sell_price.values/buy_price.values,index=sell_price.index,columns=sell_price.columns).shift(-1)
    #
    #
    # barra_reader=ParquetReader()
    # barra=barra_reader.get_barra_by_daylist(factor.index)
    #
    # test=FactorTest(factor,factor_target,5,21,'Ret20')
    # test.barra=barra
    # test.plot()

    # # reader=DailyDataReader()
    # # D_O,D_H,D_L,D_C=reader.GetOHLC()
    # # D_O, D_H, D_L, D_C,D_V=reader.get_df_ohlcv(D_O,D_H,D_L,D_C)
    # #
    # # buy_price=D_O.loc['2010':'2020'].shift(-1)
    # # sell_price=D_C.loc['2010':'2020'].shift(-20)
    # #
    # # factor=D_C.loc['2010':'2020']/D_C.loc['2010':'2020'].shift(1)
    # # factor_target=sell_price/buy_price
    # #
    # #
    # # barra_reader=ParquetReader()
    # # barra=barra_reader.get_barra_by_daylist(factor.index[::20])
    # #
    # # test=FactorTest(factor[::20],factor_target[::20],5,20,'Ret20')
    # # test.barra=barra
    # # test.plot()

