import pandas as pd
import data_tools as tools
import torch

class stratified:
    def __init__(self,factor_frame,buydays,selldays,bins_num=5):
        self.factor_frame=tools.clean_data(factor_frame)
        self.buydays=buydays
        self.selldays=selldays
        self.n= bins_num

    def get_interval_return(self):
        self.open,self.close=tools.get_open_and_close()
        self.open=self.open[(self.open/self.close.shift(1))<1.095]
        self.interval_return=pd.DataFrame(self.close.loc[self.selldays].values/self.open.loc[self.buydays].values,index=self.selldays,columns=tools.code)
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
        last_bins=stats_tools.ts_delay(bins,1)
        bins=torch.where(self.limit_down_mask,last_bins,bins)

        bins_return=torch.zeros((self.n,tensor.shape[0]))
        for i in range(self.n):
            bins_return[i]=stats_tools.nanmean(torch.where(bins==i,interval_return,float('nan')),dim=1)
        self.every_interval_rate=pd.DataFrame(bins_return,index=range(self.n),columns=self.selldays).T
        self.orders=pd.DataFrame(bins,index=self.buydays,columns=tools.code)

    def run(self):
        self.get_interval_return()
        self.get_every_interval_rate()
