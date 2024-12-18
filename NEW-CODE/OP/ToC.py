import sys
sys.path.append('..')

from OP.Others import OP_Basic
import torch

OPclass_name_2C=['OP_AF2C']

class OP_AF2C:
    def __init__(self):
        self.func_list = ["Dmask_min", "Dmask_max", "Dmask_middle", "Dmask_mean_plus_std", "Dmask_mean_sub_std"]

    @staticmethod
    def Dmask_min(x: torch.Tensor, lookback: int) -> torch.Tensor:
        """
        功能简介: unfold过去lookback天的数据，取最小的1/4天
        """
        min_true_days = max(1, lookback // 4)
        unfolded = x.unfold(0, lookback, 1)  # 展开过去lookback天的数据
        sorted_unfolded, _ = torch.sort(unfolded, dim=-1, descending=False)
        threshold = sorted_unfolded[..., min_true_days - 1]  # 取最小的1/4
        mask = unfolded <= threshold.unsqueeze(-1)
        return mask

    @staticmethod
    def Dmask_max(x: torch.Tensor, lookback: int) -> torch.Tensor:
        """
        功能简介: unfold过去lookback天的数据，取最大的1/4天。
        """
        max_true_days = max(1, lookback // 4)
        unfolded = x.unfold(0, lookback, 1)
        sorted_unfolded, _ = torch.sort(unfolded, dim=-1, descending=True)
        threshold = sorted_unfolded[..., max_true_days - 1]  # 取最大的1/4
        mask = unfolded >= threshold.unsqueeze(-1)
        return mask

    @staticmethod
    def Dmask_middle(x: torch.Tensor, lookback: int) -> torch.Tensor:
        """
        功能简介: unfold过去lookback天的数据，取中间的1/2天
        """
        mask1 = OP_AF2C.Dmask_max(x, lookback)  # 取最大1/4
        mask2 = OP_AF2C.Dmask_min(x, lookback)  # 取最小1/4
        mask3 = mask1 | mask2  # 合并最大最小部分
        return ~mask3  # 返回中间部分

    @staticmethod
    def Dmask_mean_plus_std(x: torch.Tensor, lookback: int) -> torch.Tensor:
        """
        功能简介: unfold过去lookback天的数据，进行标准化处理，取大于均值+标准差的部分
        """
        unfolded = x.unfold(0, lookback, 1)
        unfolded_mean = OP_Basic.nanmean(unfolded, dim=-1).unsqueeze(-1)
        unfolded_std = OP_Basic.nanstd(unfolded, dim=-1).unsqueeze(-1)
        unfolded_zscore = (unfolded - unfolded_mean) / unfolded_std
        mask = (unfolded_zscore) > 1  # 大于均值+标准差的部分
        return mask

    @staticmethod
    def Dmask_mean_sub_std(x: torch.Tensor, lookback: int) -> torch.Tensor:
        """
        功能简介: unfold过去lookback天的数据，进行标准化处理，取小于均值-标准差的部分
        """
        unfolded = x.unfold(0, lookback, 1)
        unfolded_mean = OP_Basic.nanmean(unfolded, dim=1).unsqueeze(1)
        unfolded_std = OP_Basic.nanstd(unfolded, dim=1).unsqueeze(1)
        unfolded_zscore = (unfolded - unfolded_mean) / unfolded_std
        mask = (unfolded_zscore) < 1  # 小于均值-标准差的部分
        return mask
