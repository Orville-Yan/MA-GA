import torch
import ToB
# 注：BXC-D中引用的算子还未修改成有效的导入形式，等ToB算子库更新后修改
class ToD:
    func_list = [
        'Mmask_min', 
        'Mmask_max', 
        'Mmask_middle', 
        'Mmask_min_to_max', 
        'Mmask_mean_plus_std', 
        'Mmask_mean_sub_std', 
        'Mmask_day_plus', 
        'Mmask_day_sub', 
        'Mmask_rolling_plus', 
        'Mmask_rolling_sub', 
        'Mmask_and', 
        'Mmask_or'
    ]
    
    @staticmethod
    def Mmask_min(x):
        """
        功能简介: 返回日内的最小1/4部分
            输入类型: TypeB (minute_OHLCV) (num_stock, day_len, minute_len=240)
            输出类型: TypeD (minute_mask) (num_stock, day_len, minute_len=240)
        """
        q = torch.nanquantile(x, 0.25, dim=-1, keepdim=True)
        mask = x < q
        return mask

    @staticmethod
    def Mmask_max(x):
        """
        功能简介: 返回日内最大的1/4部分
        数据类型: 
            输入类型: TypeB (minute_OHLCV) (num_stock, day_len, minute_len=240)
            输出类型: TypeD (minute_mask) (num_stock, day_len, minute_len=240)
        """
        q = torch.nanquantile(x, 0.75, dim=-1, keepdim=True)
        mask = x > q
        return mask

    @staticmethod
    def Mmask_middle(x):
        """
        功能简介: 返回日内中间1/2部分
        数据类型: 
            输入类型: TypeB (minute_OHLCV) (num_stock, day_len, minute_len=240)
            输出类型: TypeD (minute_mask) (num_stock, day_len, minute_len=240)
        """
        q1 = torch.nanquantile(x, 0.25, dim=-1, keepdim=True)
        q2 = torch.nanquantile(x, 0.75, dim=-1, keepdim=True)
        mask = (x > q1) & (x < q2)
        return mask

    @staticmethod
    def Mmask_min_to_max(x):
        """
        功能简介: 日内最大值和最小值中间的部分
        数据类型: 
            输入类型: TypeB (minute_OHLCV) (num_stock, day_len, minute_len=240)
            输出类型: TypeD (minute_mask) (num_stock, day_len, minute_len=240)
        """
        max_tensor = torch.max(x, dim=-1)
        min_tensor = torch.min(x, dim=-1)
        mask = (x > min_tensor) & (x < max_tensor)
        return mask

    @staticmethod
    def Mmask_mean_plus_std(x):
        """
        功能简介: 日内标准化处理后大于均值+标准差的部分
        数据类型: 
            输入类型: TypeB (minute_OHLCV) (num_stock, day_len, minute_len=240)
            输出类型: TypeD (minute_mask) (num_stock, day_len, minute_len=240)
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
        数据类型: 
            输入类型: TypeB (minute_OHLCV) (num_stock, day_len, minute_len=240)
            输出类型: TypeD (minute_mask) (num_stock, day_len, minute_len=240)
        """
        x_mean = nanmean(x, dim=-1).unsqueeze(-1)
        x_std = nanstd(x, dim=-1).unsqueeze(-1)
        x_zscore = (x - x_mean) / x_std
        mask = x_zscore < 1
        return mask

    @staticmethod
    def Mmask_day_plus(m_tensor, d_tensor):
        """
        功能简介: 返回大于日频数据的部分
        数据类型: 
            输入类型: TypeB (minute_OHLCV) (num_stock, day_len, minute_len=240), TypeC (day_mask) (day_len, num_stock, rolling_day)
            输出类型: TypeD (minute_mask) (num_stock, day_len, minute_len=240)
        """
        day_expanded = d_tensor.unsqueeze(-1).repeat(1, 1, 240)  # (day_len, num_stock, minute_len)
        day_expanded = day_expanded.permute(1, 0, 2)
        mask = day_expanded < m_tensor
        print(mask)

    @staticmethod
    def Mmask_day_sub(m_tensor, d_tensor):
        """
        功能简介: 返回小于日频数据的部分
        数据类型: 
            输入类型: TypeB (minute_OHLCV) (num_stock, day_len, minute_len=240), TypeC (day_mask) (day_len, num_stock, rolling_day)
            输出类型: TypeD (minute_mask) (num_stock, day_len, minute_len=240)
        """
        day_expanded = d_tensor.unsqueeze(-1).repeat(1, 1, 240) 
        day_expanded = day_expanded.permute(1, 0, 2)
        mask = day_expanded > m_tensor
        print(mask)

    @staticmethod
    def Mmask_rolling_plus(m_tensor, lookback):
        """
        功能简介: 以日内数据最大1/4部分的均值作为日较大值，返回大于lookback期内最大日较大值的部分。
        数据类型: 
            输入类型: TypeB (minute_OHLCV) (num_stock, day_len, minute_len=240), TypeC (day_mask) (day_len, num_stock, rolling_day)
            输出类型: TypeD (minute_mask) (num_stock, day_len, minute_len=240)
        """
        d_max_mean = ToB.D_Minute_area_mean(m_tensor, ToD.Mmask_max(m_tensor))
        result = ToD.Mmask_day_plus(m_tensor, ToB.D_ts_max(d_max_mean, lookback))
        return result

    @staticmethod
    def Mmask_rolling_sub(m_tensor, lookback):
        """
        功能简介: 以日内数据最小1/4部分的均值作为日较大值，返回大于lookback期内最小日较大值的部分。
        数据类型: 
            输入类型: TypeB (minute_OHLCV) (num_stock, day_len, minute_len=240), TypeC (day_mask) (day_len, num_stock, rolling_day)
            输出类型: TypeD (minute_mask) (num_stock, day_len, minute_len=240)
        """
        d_min_mean = ToB.D_Minute_area_mean(m_tensor, ToD.Mmask_min(m_tensor))
        result = ToD.Mmask_day_sub(m_tensor, ToB.D_ts_min(d_min_mean, lookback))
        return result

    @staticmethod
    def Mmask_and(m_mask_x, m_mask_y):
        """
        功能简介: minute_mask内部的并运算
        数据类型: 
            输入类型: TypeD (minute_mask) (num_stock, day_len, minute_len=240)
            输出类型: TypeD (minute_mask) (num_stock, day_len, minute_len=240)
        """
        return m_mask_x & m_mask_y

    @staticmethod
    def Mmask_or(m_mask_x, m_mask_y):
        """
        功能简介: minute_mask内部的和运算
        数据类型: 
            输入类型: TypeD (minute_mask) (num_stock, day_len, minute_len=240)
            输出类型: TypeD (minute_mask) (num_stock, day_len, minute_len=240)
        """
        return m_mask_x | m_mask_y
