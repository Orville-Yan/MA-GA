import pandas as pd
import data_tools as tools
import torch
from OP import *
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
"""
1. 参数
    factor_frame:因子值，要求是pd.DataFrame,index是datetime格式的日期
    factor_name:因子名称，方便回测图像存储
    buydays:买入日期，要求是datetime格式的list，且长度与factor_frame的index.unique()长度一致，buydays[0]要求大于factor_frame.index.unique().tolist().min()
    selldays:卖出日期，要求是datetime格式的list，且长度与factor_frame的index.unique()长度一致，selldays[0]要求大于等于buydays[0]
    period_num:回测周期。period_num = 252/调仓周期
    bins_num:分组数量，default=5
    plot:bool值传参，plot=True表示plt.show(),plot=False表示plt.savefig()
2. 函数调用
    step1:run()
    step2:
    plot_backtest:计算各类指标，画pnl图，需要传参root_path表示回测图像存储地址。
    plot_IC:计算Rank_IC和画累计IC图，需要传参root_path表示回测图像存储地址。
调用示例
假设因子是open
from back_test import stratified
import data_tools as tools
from datetime import datetime,timedelta
考虑到回测方便观察，我们要求回测期最早从2015年开始
open,close = tools.get_open_and_close()
factor_days = list(open[open.index>datetime(2015,1,1)].index)
buydays = factor_days[1:-1]
selldays = factor_days[2:]
open = open.loc[factor_days[:-2]]
调用模块
layer = stratified(factor_frame=open,factor_name='open',buydays=buydays,selldays=selldays,period_num=252,bins_num=10,plot=False)
layer.run()
layer.plot_backtest(root_path = '../test')
layer.plot_IC(root_path = '../test')
"""


class CustomError(Exception):
    def __init__(self, message="发生了自定义错误"):
        self.message = message
        super().__init__(self.message)

# class Cunstom
class stratified:
    def __init__(self,factor_frame,factor_name,buydays,selldays,period_num=12,bins_num=5,plot=True):
        self.factor_frame=tools.clean_data(factor_frame)
        self.factor_name = factor_name
        self.buydays=buydays
        self.selldays=selldays
        self.n= bins_num
        self.plot = plot
        self.period_num = period_num # 回测周期
        self.factor_dates = self.factor_frame.index.unique().tolist()
        if len(buydays)!=len(selldays):
            raise CustomError(f'Length of buydays is {len(buydays)} not equal to length of selldays {len(selldays)}')
        if len(self.factor_dates)!=len(selldays):
            raise CustomError(f'Length of factor unique dates is {len(self.factor_dates)} not equal to length of selldays {len(selldays)}')

    def get_interval_return(self):
        self.open,self.close=tools.get_open_and_close()
        self.open=self.open[(self.open/self.close.shift(1))<1.095]
        self.interval_return=pd.DataFrame(self.close.loc[self.selldays].values/self.open.loc[self.buydays].values,index=self.factor_dates,columns=tools.code)
        self.limit_down_mask=torch.from_numpy((self.close/self.close.shift(1)<0.905).loc[self.selldays].values)

    def get_every_interval_rate(self):
        interval_return=torch.from_numpy(self.interval_return.values)
        tensor=torch.tensor(self.factor_frame.values)
        s=[torch.nanquantile(tensor, i.item(),dim=1) for i in torch.linspace(0,1,self.n+1)[1:-1]]
        mask=torch.isnan(tensor)
        bins=torch.full_like(tensor,float('nan'))
        boundaries=torch.stack(s).permute(1,0)
        for i in range(tensor.shape[0]):
            bins[i] = torch.bucketize(tensor[i], boundaries[i], right=True)
        bins=torch.masked_fill(bins,mask,float('nan'))
        last_bins=op.ts_delay(bins,1)
        # print(bins.shape,self.limit_down_mask.shape,last_bins.shape)
        bins=torch.where(self.limit_down_mask,last_bins,bins)

        bins_return=torch.zeros((self.n+1,tensor.shape[0]))
        for i in range(self.n):
            bins_return[i]=nanmean(torch.where(bins==i,interval_return,float('nan')),dim=1)
        bins_return[self.n]=nanmean(interval_return,dim=1)
        self.every_interval_rate=pd.DataFrame(bins_return,index=list(range(self.n))+['ret_mean'],columns=self.buydays).T.dropna()
        self.orders=pd.DataFrame(bins,index=self.buydays,columns=tools.code)
        a=torch.where(bins==0,bins,float('nan'))
        b=torch.where(last_bins==0,last_bins,float('nan'))
        a1=torch.nansum(torch.where((a==0)&(b==0),1,float('nan')),dim=1)
        a2=torch.nansum(torch.where((b==0),1,float('nan')),dim=1)
        self.long_turn_over=nanmean(a1/a2)
        
    def run(self):
        self.get_interval_return()
        self.get_every_interval_rate()
        
    def plot_backtest(self,root_path=None):
        fig = plt.figure(figsize=(40, 15))
        ax1 = fig.add_subplot(221) # 分组pnl
        ax2 = fig.add_subplot(223) # 分组收益
        ax3 = fig.add_subplot(222) # 多均、均空及对应sharp
        ax4 = fig.add_subplot(224) # Statistical DataFrame
        
        # self.every_interval_rate = self.every_interval_rate[self.every_interval_rate.index>datetime(2015,1,1)]
        period_length=len(self.every_interval_rate)
        bins_return  = self.every_interval_rate.fillna(1).cumprod()
        nav = []
        for i in range(self.n):
            ret = bins_return[i]
            nav.append((self.every_interval_rate[i]-1).mean())
            ax1.plot(ret.index,ret,label=f'group {i+1}')
        ax1.set_ylabel(u"Net", fontsize=16)
        ax1.set_title(u"Group Cumulative Net", fontsize=16)
        ax1.legend(loc=0)

        ind = np.arange(self.n)
        ax2.bar(ind + 1.0 / self.n, nav, 0.3, color='r')
        ax2.set_xlim((0, ind[-1] + 1))
        ax2.set_xticks(ind + 0.35)
        ax2.set_xticklabels([f'group {i+1}' for i in ind])
        ax2.set_title(u"Group Average Return", fontsize=16)
        
        long_idx = np.argmax(nav)
        short_idx = np.argmin(nav)
        
        long_ret = self.every_interval_rate[long_idx]
        short_ret = self.every_interval_rate[short_idx]
        
        long_cum_ret = bins_return[long_idx]
        short_cum_ret = bins_return[short_idx]
        
        long_mean_ret = pd.Series(self.every_interval_rate[long_idx] - self.every_interval_rate['ret_mean'] + 1)
        mean_short_ret = pd.Series(- self.every_interval_rate[short_idx] + self.every_interval_rate['ret_mean'] + 1)
        mean_ret = self.every_interval_rate['ret_mean']
        
        long_mean_cum_ret = long_mean_ret.cumprod()
        mean_short_cum_ret = mean_short_ret.cumprod()
        mean_cum_ret = mean_ret.cumprod()
        
        long_annualized_ret = long_cum_ret.iloc[-1]**(self.period_num/period_length)
        short_annualized_ret = short_cum_ret.iloc[-1]**(self.period_num/period_length)
        long_mean_annualized_ret = long_mean_cum_ret.iloc[-1]**(self.period_num/period_length)-1
        mean_short_annualized_ret = mean_short_cum_ret.iloc[-1]**(self.period_num/period_length)-1
        mean_annualized_ret = mean_cum_ret.iloc[-1]**(self.period_num/period_length)
        
        long_annualized_vola = long_ret.std()*np.sqrt(self.period_num)
        short_annualized_vola = short_ret.std()*np.sqrt(self.period_num)
        long_mean_annualized_vola = long_mean_ret.std()*np.sqrt(self.period_num)
        mean_short_annualized_vola = mean_short_ret.std()*np.sqrt(self.period_num)
        mean_annualized_vola = mean_ret.std()*np.sqrt(self.period_num)
        
        long_sharp = (long_ret-1).mean()/(long_ret-1).std()*np.sqrt(self.period_num)
        short_sharp = (short_ret-1).mean()/(short_ret-1).std()*np.sqrt(self.period_num)
        long_mean_sharp = (long_mean_ret-1).mean()/(long_mean_ret-1).std()*np.sqrt(self.period_num)
        mean_short_sharp = (mean_short_ret-1).mean()/(mean_short_ret-1).std()*np.sqrt(self.period_num)
        mean_sharp = (mean_ret-1).mean()/(mean_ret-1).std()*np.sqrt(self.period_num)
        
        ax3.plot(short_cum_ret.index, short_cum_ret, label=f'short_ret:{short_sharp}')
        ax3.plot(mean_short_cum_ret.index, mean_short_cum_ret, label=f'mean_short_ret:{mean_short_sharp}')
        ax3.plot(long_cum_ret.index, long_cum_ret, label=f'long_ret:{long_sharp}')
        ax3.plot(long_mean_cum_ret.index, long_mean_cum_ret, label=f'long_mean_ret:{long_mean_sharp}')
        # d = bt_df[['tradeDate','mean_ret']].drop_duplicates()
        # print(d)
        ax3.plot(mean_cum_ret.index,mean_cum_ret, label=f'mean_ret:{mean_sharp}')
        # ax3.plot(pd.to_datetime())
        ax3.set_ylabel(u"Net", fontsize=16)
        ax3.set_title(u"Group Cumulative Net", fontsize=16)
        ax3.legend(loc=0) 
        
        stats = [
        # ['Long Sharp Ratio', f'{long_sharp:.4f}'],
        ['Long Mean Sharp Ratio', f'{long_mean_sharp:.4f}'],
        # ['Long Annualized Return',f'{long_annualized_ret:.4f}'],
        # ['Long Annualized Volatility',f'{long_annualized_vola:.4f}'],
        ['Long Mean Annualized Return',f'{long_mean_annualized_ret:.4f}'],
        ['Long Mean Annualized Volatility',f'{long_mean_annualized_vola:.4f}'],
        # ['Short Sharp Ratio', f'{short_sharp:.4f}'],
        ['Short Mean Sharp Ratio', f'{mean_short_sharp:.4f}'],
        # ['Short Annualized Return',f'{short_annualized_ret:.4f}'],
        # ['Short Annualized Volatility',f'{short_annualized_vola:.4f}'],
        ['Short Mean Annualized Return',f'{mean_short_annualized_ret:.4f}'],
        ['Short Mean Annualized Volatility',f'{mean_short_annualized_vola:.4f}'],
        ['Long Turnover',f'{self.long_turn_over:.4f}']
        # ['Mean Sharp Ratio',f'{mean_sharp:.4f}'],
        # ['Mean Annualized Return',f'{mean_annualized_ret:.4f}'],
        # ['Mean Annualized Volatility',f'{mean_annualized_vola}']
        ]
        
        table = ax4.table(cellText=stats, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        ax4.axis('off')
        ax4.set_title('Statistics', fontsize=16)
        if self.plot:
            plt.show()
        else:
            if not os.path.exists(os.path.join(root_path,f'{self.factor_name}')):
                os.mkdir(os.path.join(root_path,f'{self.factor_name}'))
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(root_path,f'{self.factor_name}',f'{self.factor_name}_{now}_pnl.png'))
        plt.close()
        
    def plot_IC(self,root_path):
        self.p1_ret = self.interval_return.T
        self.factor = self.factor_frame.T
        IC_p1 = self.factor.corrwith(self.p1_ret,method='spearman')
      
        fig = plt.figure(figsize=(18, 5))
        plt.plot(IC_p1.index,IC_p1.cumsum(),label=f'IC_IR:{IC_p1.mean()/IC_p1.std()*np.sqrt(self.period_num)}')
        plt.title('2015-2023 IC')
        plt.legend()
        if self.plot:
            plt.show()
        else:
            if not os.path.exists(os.path.join(root_path,f'{self.factor_name}')):
                os.mkdir(os.path.join(root_path,f'{self.factor_name}'))
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(root_path,f'{self.factor_name}',f'{self.factor_name}_{now}_IC.png'))
        plt.close()

    
    
