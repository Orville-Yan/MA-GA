import sys
sys.path.append('..')

from OP.ToA import OP_BD2A, OP_AF2A
from OP.Others import OP_Basic
import torch

OPclass_name_2D = ['OP_B2D', 'OP_BF2D', 'OP_BA2D', 'OP_DD2D']
device_default = 'cpu'
# device_default = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    def Mmask_min(m_tensor, device=device_default):
        """
        description: 返回日内的最小1/4部分 

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=240)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        m_tensor= m_tensor.to(device)
        q1 = torch.nanmean(m_tensor, dim=-1, keepdim=True) - 0.675 * OP_Basic.nanstd(m_tensor, dim=-1).unsqueeze(-1)
        mask = m_tensor<= q1
        return mask

    @staticmethod
    def Mmask_max(m_tensor, device=device_default):
        """
        description: 返回日内的最大1/4部分 

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=240)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        m_tensor= m_tensor.to(device)
        q3 = torch.nanmean(m_tensor, dim=-1, keepdim=True) + 0.675 * OP_Basic.nanstd(m_tensor, dim=-1).unsqueeze(-1)
        mask = m_tensor>= q3
        return mask

    @staticmethod
    def Mmask_middle(m_tensor, device=device_default):
        """
        description: 返回日内的中间1/2部分 

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=240)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        m_tensor= m_tensor.to(device)
        q1 = torch.nanmean(m_tensor, dim=-1, keepdim=True) - 0.675 * OP_Basic.nanstd(m_tensor, dim=-1).unsqueeze(-1)
        q3 = torch.nanmean(m_tensor, dim=-1, keepdim=True) + 0.675 * OP_Basic.nanstd(m_tensor, dim=-1).unsqueeze(-1)
        mask = (m_tensor> q1) & (m_tensor< q3)
        return mask

    @staticmethod
    def Mmask_min_to_max(m_tensor, device=device_default):
        """
        description: 返回日内最大值和最小值中间的部分 

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=240)。

        Returns:
            minute_mask(torch.Tensor): 分钟频率分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        m_tensor= m_tensor.to(device)
        x_filled = m_tensor.nan_to_num(nan=0)
        min_tensor = torch.min(x_filled, dim=-1, keepdim=True).values
        max_tensor = torch.max(x_filled, dim=-1, keepdim=True).values
        mask = (m_tensor> min_tensor) & (m_tensor< max_tensor)
        return mask

    @staticmethod
    def Mmask_mean_plus_std(m_tensor, device=device_default):
        """
        description: 生成大于均值加1倍标准差的掩码。

        Args:
            m_tensor(torch.Tensor): 输入数据张量，形状为 (num_stock, day_len, minute_len)。

        Returns:
            torch.Tensor: 掩码张量，True 表示对应位置大于均值加1倍标准差。
        """
        m_tensor= m_tensor.to(device)
        x_mean = torch.nanmean(m_tensor, dim=-1, keepdim=True)
        x_std = OP_Basic.nanstd(m_tensor, dim=-1).unsqueeze(-1)
        mask = m_tensor> (x_mean + x_std)
        return mask

    @staticmethod
    def Mmask_mean_sub_std(m_tensor, device=device_default):
        """
        description: 生成小于均值减1倍标准差的掩码。

        Args:
            m_tensor(torch.Tensor): 输入数据张量，形状为 (num_stock, day_len, minute_len)。

        Returns:
            torch.Tensor: 掩码张量，True 表示对应位置小于均值减1倍标准差。
        """
        m_tensor= m_tensor.to(device)
        x_mean = torch.nanmean(m_tensor, dim=-1, keepdim=True)
        x_std = OP_Basic.nanstd(m_tensor, dim=-1).unsqueeze(-1)
        mask = m_tensor< (x_mean - x_std)
        return mask
    @staticmethod
    def Mmask_1h_after_open(m_tensor, device=device_default):
        """
        description: 取开盘后的第一个小时

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=240)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        m_tensor = m_tensor.to(device)
        mask = torch.zeros_like(m_tensor, dtype=torch.bool, device=device)
        mask[..., :60] = True
        return mask

    @staticmethod
    def Mmask_1h_before_close(m_tensor, device=device_default):
        """
        description: 功能简介: 取收盘前的第一个小时

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=240)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        m_tensor = m_tensor.to(device)
        mask = torch.zeros_like(m_tensor, dtype=torch.bool, device=device)
        mask[..., 181:] = True
        return mask

    @staticmethod
    def Mmask_2h_middle(m_tensor, device=device_default):
        """
        description: 功能简介: 取中间的两个小时，返回一个布尔掩码。

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=240)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        m_tensor = m_tensor.to(device)
        mask = torch.zeros_like(m_tensor, dtype=torch.bool, device=device)
        mask[..., 60:181] = True
        return mask

    @staticmethod
    def Mmask_morning(m_tensor, device=device_default):
        """
        description: 取早上的两个小时

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=240)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        m_tensor = m_tensor.to(device)
        mask = torch.zeros_like(m_tensor, dtype=torch.bool, device=device)
        mask[..., :121] = True
        return mask

    @staticmethod
    def Mmask_afternoon(m_tensor, device=device_default):
        """
        description: 取下午的两个小时

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=240)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        m_tensor = m_tensor.to(device)
        mask = torch.zeros_like(m_tensor, dtype=torch.bool, device=device)
        mask[..., 121:] = True
        return mask


class OP_BA2D:
    def __init__(self):
        self.func_list = [
            'Mmask_day_plus',
            'Mmask_day_sub'
        ]

    @staticmethod
    def Mmask_day_plus(m_tensor, d_tensor,device = device_default):
        """
        description: 返回大于日频数据的部分 

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=240)。
            d_tensor (torch.Tensor): 日频数据张量，形状为 (day_len, num_stock)

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        m_tensor = m_tensor.to(device)
        d_tensor = d_tensor.to(device)
        day_expanded = d_tensor.unsqueeze(-1).repeat(1, 1, 242)  # (day_len, num_stock, minute_len)
        day_expanded = day_expanded.permute(1, 0, 2)
        mask = day_expanded < m_tensor
        return mask

    @staticmethod
    def Mmask_day_sub(m_tensor, d_tensor,device = device_default):
        """
        description: 返回小于日频数据的部分 

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=240)。
            d_tensor (torch.Tensor): 日频数据张量，形状为 (day_len, num_stock)

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        m_tensor = m_tensor.to(device)
        d_tensor = d_tensor.to(device)
        day_expanded = d_tensor.unsqueeze(-1).repeat(1, 1, 242)
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
    def Mmask_rolling_plus(m_tensor, lookback,device = device_default):
        """
        description: 返回大于lookback期内最大的日平均较大值的部分 

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=240)。
            lookback (int): 滚动窗口大小。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        m_tensor = m_tensor.to(device)
        d_max_mean = OP_BD2A.D_Minute_area_mean(m_tensor, OP_B2D.Mmask_max(m_tensor))
        rolling_max = OP_AF2A.D_ts_max(d_max_mean, lookback)
        result = OP_BA2D.Mmask_day_plus(m_tensor, rolling_max)
        return result

    @staticmethod
    def Mmask_rolling_sub(m_tensor, lookback,device = device_default):
        """
        description: 返回小于lookback期内最小的日平均较小值的部分 

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=240)。
            lookback (int): 滚动窗口大小。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        m_tensor = m_tensor.to(device)
        d_min_mean = OP_BD2A.D_Minute_area_mean(m_tensor, OP_B2D.Mmask_min(m_tensor))
        rolling_min = OP_AF2A.D_ts_min(d_min_mean, lookback)
        result = OP_BA2D.Mmask_day_sub(m_tensor,rolling_min )
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
        description: minute_mask内部的并运算

        Args:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        return m_mask_x& m_mask_y

    @staticmethod
    def Mmask_or(m_mask_x, m_mask_y):
        """
        description: minute_mask内部的和运算

        Args:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=240)
        """
        return m_mask_x| m_mask_y
