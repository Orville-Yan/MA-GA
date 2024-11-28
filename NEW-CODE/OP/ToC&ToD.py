'''

TypeC
day_mask
(day_len, num_stock, rolling_day)
TypeD
minute_mask
(num_stock, day_len, minute_len=240)

Dmask_min(d_tensor, rolling_day)
unfold过去rolling_day天，取最小的1/4天作为True，不足1/4的取1
Dmask_max(d_tensor, rolling_day)
unfold过去rolling_day天，取最大的1/4天作为True，不足1/4的取1
Dmask_middle(d_tensor, rolling_day)
unfold过去rolling_day天，取中间的1/2作为True，不足1/4的取1
Dmask_mean_plus_std(d_tensor, rolling_day)
unfold过去rolling_day天，先做标准化处理，取大于 均值+标准差 部分的数据
Dmask_mean_sub_std(d_tensor, rolling_day)
unfold过去rolling_day天，先做标准化处理，取小于 均值+标准差 部分的数据
'''
def Dmask_min(x, lookback):
    min_true_days = max(1, lookback // 4)
    unfolded = x.unfold(0,lookback,1)
    sorted_unfolded, _ = torch.sort(unfolded, dim=-1, descending=False)
    threshold = sorted_unfolded[..., min_true_days - 1]
    mask = unfolded <= threshold.unsqueeze(-1)  # 扩展维度以便比较
    return mask
def Dmask_max(x, lookback):
    max_true_days = max(1, lookback // 4)
    unfolded = x.unfold(0,lookback,1)
    sorted_unfolded, _ = torch.sort(unfolded, dim=-1, descending=True)
    threshold = sorted_unfolded[..., max_true_days - 1]
    mask = unfolded >= threshold.unsqueeze(-1)  # 扩展维度以便比较
    return mask
def Dmask_middle(x, lookback):
    mask1 = Dmask_max(x, lookback)
    mask2 = Dmask_min(x, lookback)
    mask3 = mask1 | mask2
    return ~mask3

def Dmask_mean_plus_std(x, lookback):
    unfolded = x.unfold(0,lookback,1)
    unfolded_mean = nanmean(unfolded,dim=-1).unsqueeze(-1)
    unfolded_std = nanstd(unfolded,dim=-1).unsqueeze(-1)
    unfolded_zscore = (unfolded - unfolded_mean)/unfolded_std
    mask = (unfolded_zscore)>1
    return mask
def Dmask_mean_sub_std(x, lookback):
    unfolded = x.unfold(0,lookback,1)
    unfolded_mean = nanmean(unfolded,dim=1).unsqueeze(1)
    unfolded_std = nanstd(unfolded,dim=1).unsqueeze(1)
    unfolded_zscore = (unfolded - unfolded_mean)/unfolded_std
    mask = (unfolded_zscore)<1
    return mask

def Mmask_min(x):
    q = torch.nanquantile(x,0.25,dim=-1,keepdim=True)
    mask = x < q
    return mask
def Mmask_max(x):
    q = torch.nanquantile(x,0.75,dim=-1,keepdim=True)
    mask = x > q
    return mask
def Mmask_middle(x):
    q1 = torch.nanquantile(x,0.25,dim=-1,keepdim=True)
    q2 = torch.nanquantile(x,0.75,dim=-1,keepdim=True)
    mask = (x > q1)&(x < q2)
    return mask
def Mmask_min_to_max(x):
    max_tensor = torch.max(x,dim = -1)
    min_tensor = torch.min(x,dim = -1)
    mask = (x > min_tensor)&(x < max_tensor)
    return mask
def Mmask_mean_plus_std(x):
    x_mean = nanmean(x,dim=-1).unsqeeze(-1)
    x_std = nanstd(x,dim=-1).unsqeeze(-1)
    x_zscore = (x - x_mean)/x_std
    mask = x_zscore > 1
    return mask
def Mmask_mean_sub_std(x):
    x_mean = nanmean(x,dim=-1).unsqeeze(-1)
    x_std = nanstd(x,dim=-1).unsqeeze(-1)
    x_zscore = (x - x_mean)/x_std
    mask = x_zscore < 1
    return mask
def Mmask_and(m_mask_x,m_mask_y):
    return m_mask_x&m_mask_y
def Mmask_or(m_mask_x,m_mask_y):
    return m_mask_x | m_mask_y
def Mmask_day_plus(m_tensor,d_tensor):
    day_expanded =d_tensor.unsqueeze(-1).repeat(1,1, minute_len = 240) # (day_len, num_stock, minute_len)
    day_expanded = day_expanded.permute(1,0,2)
    mask = day_expanded < m_tensor
    print(mask)
def Mmask_day_sub(m_tensor,d_tensor):
    day_expanded =d_tensor.unsqueeze(-1).repeat(1,1, minute_len = 240) # (day_len, num_stock, minute_len)
    day_expanded = day_expanded.permute(1,0,2)
    mask = day_expanded > m_tensor
    print(mask)    

def Mmask_rolling_plus(m_tensor,lookback):
    d_max_mean = D_Minute_area_mean(m_tensor,Mmask_max(m_tensor))
    result = Mmask_day_plus(m_tensor,D_ts_max(d_max_mean,lookback))
    return result
def Mmask_rolling_sub(m_tensor,lookback):
    d_min_mean = D_Minute_area_mean(m_tensor,Mmask_min(m_tensor))
    result = Mmask_day_sub(m_tensor,D_ts_min(D_Minute_area_mean(d_min_mean,lookback)))
    return result