import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

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
        print(mask.shape)
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
if __name__ == '__main__':
    import time
    TypeA_shape = (10, 100) 
    TypeC_shape = (10,100,2)

    # 创建随机数据
    A = torch.randn(TypeA_shape)
    F = 2


    # 测试函数
    def test_functions(class_instance, data, *args):
        results = {}
        for func_name in class_instance.func_list:
            func = getattr(class_instance, func_name)
            start_time = time.time()
            try:
                result = func(*data, *args)
                results[func_name] = time.time() - start_time
                shape_result = (result.shape == TypeC_shape)
                if not shape_result:
                    print(func_name)
                    print('shape fault')
            except Exception as e:
                results[func_name] = str(e)
        



    # 测试每个类
    def test_class(class_type, *args):
        instance = class_type()
        if class_type in [OP_AF2C]               :
            return test_functions(instance, (A,F))

    # # 打印结果
    # def print_results(results, class_name):
    #     print(f"Results for {class_name}:")
    #     for func_name, duration in results.items():
    #         if isinstance(duration, float):
    #             print(f"  {func_name}: {duration:.6f} seconds")
    #         else:
    #             print(f"  {func_name}: {duration}")


    # 运行测试
    classes = [OP_AF2C]
    for class_type in classes:
        results = test_class(class_type)
        # print_results(results, class_type.__name__)