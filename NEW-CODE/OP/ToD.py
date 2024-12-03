import torch
OPclass_name_2D=['OP_B2D','OP_BF2D','OP_BA2D','OP_DD2D']  
class OP_B2D:
    def __init__(self):
        self.func_list = [
        'Mmask_min', 
        'Mmask_max', 
        'Mmask_middle', 
        'Mmask_min_to_max', 
        'Mmask_mean_plus_std', 
        'Mmask_mean_sub_std',
        'Mmask_1h_after_open',
        'Mmask_1h_before_close',
        'Mmask_2h_middle',
        'Mmask_morning',
        'Mmask_afternoon',
        ]
    
    @staticmethod
    def Mmask_min(x):
        """
        功能简介: 返回日内的最小1/4部分
        """
        q = torch.nanquantile(x, 0.25, dim=-1, keepdim=True)
        mask = x < q
        return mask

    @staticmethod
    def Mmask_max(x):
        """
        功能简介: 返回日内最大的1/4部分
        数据类型: 
        """
        q = torch.nanquantile(x, 0.75, dim=-1, keepdim=True)
        mask = x > q
        return mask

    @staticmethod
    def Mmask_middle(x):
        """
        功能简介: 返回日内中间1/2部分
        """
        q1 = torch.nanquantile(x, 0.25, dim=-1, keepdim=True)
        q2 = torch.nanquantile(x, 0.75, dim=-1, keepdim=True)
        mask = (x > q1) & (x < q2)
        return mask

    @staticmethod
    def Mmask_min_to_max(x):
        """
        功能简介: 日内最大值和最小值中间的部分
        """
        max_tensor = torch.max(x, dim=-1)
        min_tensor = torch.min(x, dim=-1)
        mask = (x > min_tensor) & (x < max_tensor)
        return mask

    @staticmethod
    def Mmask_mean_plus_std(x):
        """
        功能简介: 日内标准化处理后大于均值+标准差的部分
        """
        x_mean = nanmean(x, dim=-1).unsqueeze(-1)
        x_std = nanstd(x, dim=-1).unsqueeze(-1)
        x_zscore = (x - x_mean) / x_std
        mask = x_zscore > 1
        return mask

    @staticmethod
    def Mmask_mean_sub_std(x):
        """
        功能简介: 日内标准化处理后小于均值+标准差的部分
        """
        x_mean = nanmean(x, dim=-1).unsqueeze(-1)
        x_std = nanstd(x, dim=-1).unsqueeze(-1)
        x_zscore = (x - x_mean) / x_std
        mask = x_zscore < 1
        return mask
    def Mmask_1h_after_open(x):
        """
        功能简介: 取开盘后的第1个小时
        """
        return x[...,:60]
    def Mmask_1h_before_close(x):
        """
        功能简介: 取收盘前的第一个小时
        """
        return x[...,180:]
    '''取收盘前的一个小时'''
    def Mmask_2h_middle(x):
        """
        功能简介: 取中间的两个小时
        """
        return x[...,60:180]
    def Mmask_morning(x):
        """
        功能简介: 取早上的两个小时
        """
        return x[...,:120]
    def Mmask_afternoon(x):
        """
        功能简介: 取下午的两个小时
        """
        return x[...,120:]
class OP_BA2D:
    def __init__(self):
        self.func_list = [
        'Mmask_day_plus', 
        'Mmask_day_sub'
    ]
    @staticmethod
    def Mmask_day_plus(m_tensor, d_tensor):
        """
        功能简介: 返回大于日频数据的部分
        """
        day_expanded = d_tensor.unsqueeze(-1).repeat(1, 1, 240)  # (day_len, num_stock, minute_len)
        day_expanded = day_expanded.permute(1, 0, 2)
        mask = day_expanded < m_tensor
        return mask

    @staticmethod
    def Mmask_day_sub(m_tensor, d_tensor):
        """
        功能简介: 返回小于日频数据的部分
        """
        day_expanded = d_tensor.unsqueeze(-1).repeat(1, 1, 240) 
        day_expanded = day_expanded.permute(1, 0, 2)
        mask = day_expanded > m_tensor
        return mask
class OP_BF2D:
    def __init__(self):
        self.func_list = [
        'Mmask_rolling_plus', 
        'Mmask_rolling_sub'
        ]
    @staticmethod
    def Mmask_rolling_plus(m_tensor, lookback):
        """
        功能简介: 以日内数据最大1/4部分的均值作为日较大值，返回大于lookback期内最大日较大值的部分。
        """
        d_max_mean = OP_BD2A.D_Minute_area_mean(m_tensor, OP_B2D.Mmask_max(m_tensor))
        result = OP_BA2D.Mmask_day_plus(m_tensor, OP_AF2A.ts_max(d_max_mean, lookback))
        return result

    @staticmethod
    def Mmask_rolling_sub(m_tensor, lookback):
        """
        功能简介: 以日内数据最小1/4部分的均值作为日较大值，返回大于lookback期内最小日较大值的部分。
        """
        d_min_mean = OP_BD2A.D_Minute_area_mean(m_tensor, OP_B2D.Mmask_min(m_tensor))
        result = OP_BA2D.Mmask_day_sub(m_tensor, OP_AF2A.ts_min(d_min_mean, lookback))
        return result
class OP_DD2D:
    def __init__(self):
        self.func_list = [
        'Mmask_and', 
        'Mmask_or'
        ]
    @staticmethod
    def Mmask_and(m_mask_x, m_mask_y):
        """
        功能简介: minute_mask内部的并运算
        """
        return m_mask_x & m_mask_y

    @staticmethod
    def Mmask_or(m_mask_x, m_mask_y):
        """
        功能简介: minute_mask内部的和运算
        """
        return m_mask_x | m_mask_y
