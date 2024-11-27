import torch

class op_A2A:
    def __init__(self):
        self.func_list=['at_abs','cs_rank','cs_scale','cs_zscore','cs_harmonic_mean','cs_demean','cs_winsor']
    @staticmethod
    def at_abs(x):
        return torch.abs(x)
    
    @staticmethod
    def cs_rank(x):
        mask = ~torch.isnan(x)
        data_no_nan = torch.where(mask, x, torch.full_like(x, float('inf')))
        ranks = torch.argsort(torch.argsort(data_no_nan, dim=1), dim=1).float()
        quantiles = ranks / torch.sum(mask, 1).unsqueeze(1)
        s = torch.where(mask, quantiles, torch.tensor(float('nan')))

        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def cs_scale( x):
        mask = ~torch.isnan(x)
        data_no_nan = torch.where(mask, x, torch.full_like(x, 0))
        max = torch.max(torch.where(mask, x, torch.full_like(x, float('-inf'))), dim=1)[0].unsqueeze(dim=1)
        min = torch.min(torch.where(mask, x, torch.full_like(x, float('inf'))), dim=1)[0].unsqueeze(dim=1)
        sacled_data_no_nan = (data_no_nan - min) / (max - min)
        scaled_data = torch.where(mask, sacled_data_no_nan, torch.tensor(float('nan')))
        s=scaled_data+1
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)

    @staticmethod
    def cs_zscore( x):
        x_mean = nanmean(x,dim=1).unsqueeze(1)
        x_std = nanstd(x,dim=1).unsqueeze(1)
        zscore = (x-x_mean)/x_std
        s = zscore
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def cs_harmonic_mean(x):
        mask = (~torch.isnan(x)) & (x != 0)
        data_no_nan = 1 / torch.where(mask, x, torch.full_like(x, 1))
        harmonic_mean = torch.sum(mask, dim=1) / torch.nansum(
            torch.where(mask, data_no_nan, torch.tensor(float('nan'))), dim=1)
        result = torch.full_like(x, float('nan'))
        result[mask] = harmonic_mean.unsqueeze(dim=1).expand_as(x)[mask]
        s=result
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def cs_demean( x):
        return op_A2A.at_abs(x - nanmean(x, dim=1).unsqueeze(1))
    
    @staticmethod
    def cs_winsor( x, limit=[0.05, 0.95]):
        rank = op_A2A.cs_rank(x)
        min_limit = torch.where(rank >= limit[0], rank, float('nan'))
        max_limit = torch.where(rank <= limit[1], rank, float('nan'))
        mask = (~torch.isnan(min_limit)) & (~torch.isnan(max_limit))
        max = torch.max(torch.where(mask, x, torch.full_like(x, float('-inf'))), dim=1)[0].unsqueeze(dim=1)
        min = torch.min(torch.where(mask, x, torch.full_like(x, float('inf'))), dim=1)[0].unsqueeze(dim=1)
        winsored_min = torch.where(rank <= limit[0], min, x)
        winsored_max = torch.where(rank >= limit[1], max, winsored_min)
        x_with_nan = torch.where(~torch.isnan(x), winsored_max, float('nan'))
        return x_with_nan
    
class op_AE2A:
    def __init__(self):
        self.func_list=['cs_demean_industry','cs_industry_neutra']
    def cs_demean_industry(day_OHLCV, industry):

        day_len, num_stock = day_OHLCV.shape
        _, _, industry_num = industry.shape

        industry_sums = torch.bmm(industry.permute(0, 2, 1), day_OHLCV.unsqueeze(-1))  
        industry_counts = industry.sum(dim=1).unsqueeze(-1).expand(day_len, industry_num, 1)  
        industry_means = industry_sums / industry_counts  


        weighted_industry_means = torch.bmm(industry, industry_means)  
        num_industries_per_stock = industry.sum(dim=2).unsqueeze(-1).expand(day_len, num_stock, 1)  
        valid_mask = (num_industries_per_stock > 0)
        industry_means_final = torch.where(valid_mask, weighted_industry_means / num_industries_per_stock, torch.tensor(0.0, device=day_OHLCV.device))

        demeaned_abs = torch.abs(day_OHLCV.unsqueeze(-1) - industry_means_final.squeeze(-1))

        return demeaned_abs

class op_AA2A:
    def __init__(self):
        self.func_list=['cs_norm_spread','cs_cut','cs_regress_res','at_add','at_sub','at_div','at_prod','at_mean']
       
    @staticmethod
    def cs_norm_spread(x, y):
        s=(x - y) / (torch.abs(x) + torch.abs(y))
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def cs_cut(x, y):
        sign = at_sign(x - nanmean(x, dim=1).unsqueeze(1))
        return sign * y
    
    @staticmethod
    def cs_regress_res(x, y):
        res = multi_regress(x, y.unsqueeze(-1))[-1]
        return res
    
    @staticmethod
    def at_add( x, y):
        return torch.add(x, y)
    
    @staticmethod
    def at_div( x, y):
        zero_mask = y == 0
        result = torch.div(x, y)
        result[zero_mask] = torch.nan
        return result

    @staticmethod
    def at_sub( x, y):
        return torch.sub(x, y)
    
    @staticmethod
    def at_prod(d_tensor_x, d_tensor_y):

        mask = ~((d_tensor_y == 0) | torch.isnan(d_tensor_y))
        result = torch.full_like(d_tensor_x, float('nan'))
        result[mask] = torch.div(d_tensor_x[mask], d_tensor_y[mask])

        return result

    @staticmethod
    def at_mean(x,y):
        return op_AA2A.at_add(x,y)/2
    
class op_AG2A:
    def __init__(self):
        self.func_list=['cs_edge_flip']
    @staticmethod
    def cs_edge_flip( x, percent=0.3):
        rank = op_A2A.cs_rank(x)
        edge_fliped = torch.where(torch.abs(rank - 0.5) > percent / 2, x, -x)
        return edge_fliped

class AAF2A:
    def __init__(self):
        self.func_list=['ts_corr','ts_rankcorr','ts_regress_res','ts_weight_mean']
    @staticmethod
    def ts_corr(x,y,d):
        nan_fill = torch.full((d - 1, x.shape[1]), float('nan'))
        x=x.unfold(0,d,1)
        y=y.unfold(0,d,1)
        correlation=corrwith(x,y,dim=-1)
        s = torch.cat([nan_fill, correlation], dim=0)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def ts_rankcorr(x,y,d):
        nan_fill = torch.full((d - 1, x.shape[1]), float('nan'))
        x=x.unfold(0,d,1)
        y=y.unfold(0,d,1)
        correlation=rank_corrwith(x,y,dim=-1)
        s = torch.cat([nan_fill, correlation], dim=0)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def ts_regress_res(x,y,lookback):
        return ts_regress(x,y,lookback)[2]
    
    @staticmethod
    def ts_weight_mean(x,y, lookback):
        if lookback==1:
            return x
        else:
            x = x.unfold(0, lookback, 1)
            y=y.unfold(0,lookback,1)
            mask=torch.isnan(x)|torch.isnan(y)
            x=torch.where(mask,float('nan'),x)
            y = torch.where(mask, float('nan'), y)

            nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
            p=torch.nansum(x*y,dim=-1)/torch.nansum(y,dim=-1)
            p=torch.cat([nan_fill,p],dim=0)
            return torch.where((p==torch.inf)|(p==-torch.inf),float('nan'),p)

class AF2A:
    def __init__(self):
        self.func_list=['ts_max','ts_min','ts_delay','ts_delta','ts_pctchg',
                        'ts_mean','ts_harmonic_mean','ts_std','ts_to_max',
                        'ts_to_min','ts_to_mean','ts_max_to_min','ts_maxmin_norm',
                        'D_ts_norm','D_ts_detrend'
                        ]
    
    @staticmethod
    def ts_max( x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        max_tensor = torch.max(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, max_tensor], dim=0)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)

    @staticmethod
    def ts_min( x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        max_tensor = torch.min(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, max_tensor], dim=0)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def ts_delay( x, d):
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
    def ts_delta( x, d):
        return x - AF2A.ts_delay(x, d)
    
    @staticmethod
    def ts_pctchg( x, lookback):
        s=(x-AF2A.ts_delay(x,lookback))/op.ts_delay(x,lookback)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    @staticmethod
    def ts_mean( x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        x_mean = nanmean(x_3d)
        s = torch.cat([nan_fill, x_mean], dim=0)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)

    @staticmethod
    def ts_harmonic_mean( x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        mask=(x_3d==0)|torch.isnan(x_3d)
        dominator=1/x_3d
        dominator=torch.where(mask,0,dominator)
        dominator=torch.sum(dominator,dim=-1)
        numerator=torch.sum(~mask,dim=-1)
        s=numerator/dominator
        s=torch.where(dominator==0,float('nan'),s)
        s=torch.cat([nan_fill,s],dim=0)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)

    @staticmethod
    def ts_std( x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        x_std = nanstd(x_3d)
        s = torch.cat([nan_fill, x_std], dim=0)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)

    @staticmethod
    def ts_to_max( x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        max_tensor = torch.max(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, max_tensor], dim=0)
        s=x/s
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)

    @staticmethod
    def ts_to_min( x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        min_tensor = torch.min(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, min_tensor], dim=0)
        s=x/s
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)

    @staticmethod
    def ts_to_mean( x, lookback):
        mean_tensor = op.ts_mean(x, lookback)
        s=x/mean_tensor
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)

    @staticmethod
    def ts_max_to_min( x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        min_tensor = torch.min(x_3d, dim=-1)[0]
        max_tensor = torch.max(x_3d, dim=-1)[0]
        s = torch.cat([nan_fill, max_tensor - min_tensor], dim=0)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)

    @staticmethod
    def ts_to_maxmin_norm(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        min_tensor = torch.min(x_3d, dim=-1)[0]
        max_tensor = torch.max(x_3d, dim=-1)[0]
        min_tensor = torch.cat([nan_fill,min_tensor ], dim=0)
        max_tensor = torch.cat([nan_fill,max_tensor ], dim=0)
        s=(x-min_tensor)/(max_tensor-min_tensor)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def D_ts_norm(x, lookback):
        nan_fill = torch.full((lookback - 1, x.shape[1]), float('nan'))
        x_3d = x.unfold(0, lookback, 1)
        x_mean = nanmean(x_3d)
        mean = torch.cat([nan_fill, x_mean], dim=0)
        x_std = nanstd(x_3d)
        std = torch.cat([nan_fill, x_std], dim=0)
        s = (x - mean) / std
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def D_ts_detrend(x, lookback):#ts_regress
        x = x.float()
        time_idx = torch.arange(x.shape[0], dtype=torch.float32).unsqueeze(-1)
        time_idx_expanded = time_idx.repeat(1, x.shape[1])
        k, b, _ = ts_regress(time_idx_expanded, x, lookback)
        trend = (k * time_idx_expanded) + b
        s= x - trend
        s[:lookback-1, :] = float('nan')
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    class op_AC2A:
        def __init__(self):
            self.func_list = ['D_ts_mask_mean','D_ts_mask_std','D_ts_mask_sum','D_ts_mask_prod']
    
    @staticmethod
    def D_ts_mask_mean(x, mask):
        nan_fill = torch.full((mask.shape[2]-1, x.shape[1], mask.shape[2]), float('nan'))
        x_3d = x.unfold(0, mask.shape[2], 1)
        x_3d = torch.cat([nan_fill, x_3d], dim=0)
        x_3d = torch.where(mask, float('nan'), x_3d)
        s = nanmean(x_3d)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def D_ts_mask_std(x, mask):
        nan_fill = torch.full((mask.shape[2]-1, x.shape[1], mask.shape[2]), float('nan'))
        x_3d = x.unfold(0, mask.shape[2], 1)
        x_3d = torch.cat([nan_fill, x_3d], dim=0)
        x_3d = torch.where(mask, x_3d, float('nan'))
        s = nanstd(x_3d)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def D_ts_mask_sum(x, mask):
        nan_fill = torch.full((mask.shape[2]-1, x.shape[1], mask.shape[2]), float('nan'))
        x_3d = x.unfold(0, mask.shape[2], 1)
        x_3d = torch.cat([nan_fill, x_3d], dim=0)
        x_3d = torch.where(mask, x_3d, float('nan'))
        all_nan = torch.all(torch.isnan(x_3d),dim=(0,1))
        s = torch.nansum(x_3d, dim=-1)
        s[all_nan] = float('nan')
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def D_ts_mask_prod(x, mask):
        nan_fill = torch.full((mask.shape[2]-1, x.shape[1], mask.shape[2]), float('nan'))
        x_3d = x.unfold(0, mask.shape[2], 1)
        x_3d = torch.cat([nan_fill, x_3d], dim=0)
        x_3d = torch.where(mask, x_3d, float('nan'))
        all_nan = torch.all(torch.isnan(x_3d),dim=(0,1))
        x_3d = torch.where(torch.isnan(x_3d), torch.ones_like(x_3d), x_3d)
        s = torch.prod(x_3d,dim=-1)
        s[all_nan] = float('nan')
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
class op_BD2A:
    def __init__(self):
        self.func_list = ['D_Minute_area_mean','D_Minute_area_std','D_Minute_area_sum','D_Minute_area_prod']

    @staticmethod
    def D_Minute_area_mean(x, mask):
        x = torch.where(mask, x, float('nan'))
        s = nanmean(x)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def D_Minute_area_std(x, mask):
        x = torch.where(mask, x, float('nan'))
        s = nanstd(x)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def D_Minute_area_sum(x, mask):
        x = torch.where(mask, x, float('nan'))
        all_nan = torch.all(torch.isnan(x),dim=(0,1))
        s = torch.nansum(x, dim=-1)
        s[all_nan] = float('nan')
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def D_Minute_area_prod(x, mask):
        x = torch.where(mask, x, float('nan'))
        all_nan = torch.all(torch.isnan(x),dim=(0,1))
        x = torch.where(torch.isnan(x), torch.ones_like(x), x)
        s = torch.prod(x, dim=-1)
        s[all_nan] = float('nan')
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)

class op_BBD2A:
    def __init__(self):
        self.func_list = ['D_Minute_area_weight_mean','D_Minute_area_corr','D_Minute_area_rankcorr','D_Minute_area_bifurcate_mean','D_Minute_area_bifurcate_std']

    @staticmethod
    def D_Minute_area_weight_mean(x,weight,mask):
        x = torch.where(mask, x, float('nan'))
        x_ = x * weight
        s = nanmean(x_)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def D_Minute_area_corr(x,y,mask):#ts_corrwith
        x = torch.where(mask, x, float('nan'))
        corr = corrwith(x, y)
        return torch.where((corr==torch.inf)|(corr==-torch.inf),float('nan'),corr)
    
    @staticmethod
    def D_Minute_area_rankcorr(x,y,mask):#rank_corrwith
        x = torch.where(mask, x, float('nan'))
        corr = rank_corrwith(x, y,)
        return torch.where((corr==torch.inf)|(corr==-torch.inf),float('nan'),corr)

    @staticmethod
    def D_Minute_area_bifurcate_mean(m_tensor_x,m_tensor_y,mask):
        pass
        #D_at_sub(D_Minute_area_mean(m_tensor_x, Mmask_day_plus(m_tensor_y,D_Minute_area_mean(m_tensor_y, m_mask)))),D_Minute_area_mean(m_tensor_x, Mmask_day_sub(m_tensor_y,D_Minute_area_mean(m_tensor_y, m_mask)))))

    @staticmethod
    def D_Minute_area_bifurcate_std(m_tensor_x,m_tensor_y,mask):
        pass
        #D_at_sub(D_Minute_area_std(m_tensor_x, Mmask_day_plus(m_tensor_y,D_Minute_area_mean(m_tensor_y, m_mask)))),D_Minute_area_std(m_tensor_x, Mmask_day_sub(m_tensor_y,D_Minute_area_mean(m_tensor_y, m_mask)))))

class op_BB2A():
    def __init__(self):
        self.func_list = ['D_Minute_corr','D_Minute_weight_mean']

    @staticmethod
    def D_Minute_corr(x,y):
        corr = corrwith(x, y)
        return torch.where((corr==torch.inf)|(corr==-torch.inf),float('nan'),corr)

    @staticmethod
    def D_Minute_weight_mean(x,weight=1):
        x_ = x * weight
        s = nanmean(x_)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
class op_B2A():
    def __init__(self):
        self.func_list = ['D_Minute_std','D_Minute_mean','D_Minute_trend']    
    
    @staticmethod
    def minute_std(x):
        s = nanstd(x)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def minute_mean(x):
        s = nanmean(x)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
    
    @staticmethod
    def minute_trend(x):#regress
        x = x.float()
        time_idx = torch.arange(x.shape[0], dtype=torch.float32).unsqueeze(-1)
        time_idx_expanded = time_idx.repeat(1, x.shape[1])
        k, b, _ = regress(time_idx_expanded, x)
        return torch.where((k==torch.inf)|(k==-torch.inf),float('nan'),k)
    
class op_D2A():
    def __init__(self):
        self.func_list = ['D_Minute_abnormal_point_count']

    @staticmethod
    def D_Minute_abnormal_point_count(mask):
        s = torch.nansum(mask, dim=-1)
        return torch.where((s==torch.inf)|(s==-torch.inf),float('nan'),s)
