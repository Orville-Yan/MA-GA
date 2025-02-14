import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

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
        description: 返回每个交易日内最小四分之一数据的掩码。

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=242)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
        """
        m_tensor = m_tensor.to(device)
        q1 = torch.quantile(m_tensor, 0.25, dim=-1, keepdim=True)
        mask = m_tensor <= q1
        return mask

    @staticmethod
    def Mmask_max(m_tensor, device=device_default):
        """
        description: 返回每个交易日内最大四分之一数据的掩码。

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=242)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
        """
        m_tensor = m_tensor.to(device)
        q3 = torch.quantile(m_tensor, 0.75, dim=-1, keepdim=True)
        mask = m_tensor >= q3
        return mask

    @staticmethod
    def Mmask_middle(m_tensor, device=device_default):
        """
        description: 返回每个交易日内中间一半数据的掩码。

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=242)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
        """
        m_tensor = m_tensor.to(device)
        q1 = torch.quantile(m_tensor, 0.25, dim=-1, keepdim=True)
        q3 = torch.quantile(m_tensor, 0.75, dim=-1, keepdim=True)
        mask = (m_tensor > q1) & (m_tensor < q3)
        return mask

    @staticmethod
    def Mmask_min_to_max(m_tensor, device=device_default):
        """
        description: 返回每个交易日内介于最小值和最大值之间的数据的掩码。

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=242)。

        Returns:
            minute_mask(torch.Tensor): 分钟频率分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
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
        description: 返回大于每个交易日内均值加一倍标准差的数据的掩码。

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
        description: 返回小于每个交易日内均值减一倍标准差的数据的掩码。

        Args:
            m_tensor(torch.Tensor): 输入数据张量，形状为 (num_stock, day_len, minute_len)。

        Returns:
            torch.Tensor: 掩码张量，True 表示对应位置小于均值减1倍标准差。
        """
        m_tensor= m_tensor.to(device)
        x_mean = torch.nanmean(m_tensor, dim=-1, keepdim=True)
        x_std = OP_Basic.nanstd(m_tensor, dim=-1).unsqueeze(-1)
        mask = m_tensor < (x_mean - x_std)
        return mask
    @staticmethod
    def Mmask_1h_after_open(m_tensor, device=device_default):
        """
        description: 返回开盘后第一个小时的数据的掩码。

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=242)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
        """
        m_tensor = m_tensor.to(device)
        mask = torch.zeros_like(m_tensor, dtype=torch.bool, device=device)
        mask[..., :60] = True
        return mask

    @staticmethod
    def Mmask_1h_before_close(m_tensor, device=device_default):
        """
        description: 返回收盘前最后一个小时的数据的掩码。

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=242)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
        """
        m_tensor = m_tensor.to(device)
        mask = torch.zeros_like(m_tensor, dtype=torch.bool, device=device)
        mask[..., 181:] = True
        return mask

    @staticmethod
    def Mmask_2h_middle(m_tensor, device=device_default):
        """
        description: 返回交易日中间两个小时的数据的掩码。

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=242)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
        """
        m_tensor = m_tensor.to(device)
        mask = torch.zeros_like(m_tensor, dtype=torch.bool, device=device)
        mask[..., 60:181] = True
        return mask

    @staticmethod
    def Mmask_morning(m_tensor, device=device_default):
        """
        description: 返回交易日早上两个小时的数据的掩码。

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=242)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
        """
        m_tensor = m_tensor.to(device)
        mask = torch.zeros_like(m_tensor, dtype=torch.bool, device=device)
        mask[..., :121] = True
        return mask

    @staticmethod
    def Mmask_afternoon(m_tensor, device=device_default):
        """
        description: 返回交易日下午两个小时的数据的掩码。

        Args:
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=242)。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
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
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=242)。
            d_tensor (torch.Tensor): 日频数据张量，形状为 (day_len, num_stock)

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
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
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=242)。
            d_tensor (torch.Tensor): 日频数据张量，形状为 (day_len, num_stock)

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
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
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=242)。
            lookback (int): 滚动窗口大小。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
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
            m_tensor (torch.Tensor): 分钟数据张量，形状为 (num_stock, day_len, minute_len=242)。
            lookback (int): 滚动窗口大小。

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
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
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
        """
        return m_mask_x & m_mask_y

    @staticmethod
    def Mmask_or(m_mask_x, m_mask_y):
        """
        description: minute_mask内部的和运算

        Args:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)

        Returns:
            minute_mask(torch.Tensor): 分钟数据掩码，形状为(num_stock, day_len, minute_len=242)
        """
        return m_mask_x| m_mask_y
if __name__ == "__main__":
    # 创建一个小型测试数据集，形状为 (4, 2, 242)
    TypeD_data_shape =  TypeB_data_shape = (4, 5, 242) #(num_stock = 4, day_len =2, minute_len=242)
    TypeA_data_shape = (5,4) #(day_len =2  , num_stock =4)

    test_data = torch.randint(0, 11, TypeB_data_shape ,dtype=torch.float)  # 生成 0 到 10 的随机整数
    test_data = test_data.to(device_default)

    day_data = torch.randint(0, 11, TypeA_data_shape ,dtype=torch.float) 
    day_data = day_data.to(device_default)

    # 初始化所有算子
    op_b2d = OP_B2D()
    op_ba2d = OP_BA2D()
    op_bf2d = OP_BF2D()
    op_dd2d = OP_DD2D()

    # 存储所有测试结果
    shape_tests = {}
    
    # 获取所有算子类的实例
    operators = {
        'OP_B2D': op_b2d,
        'OP_BA2D': op_ba2d,
        'OP_BF2D': op_bf2d,
        'OP_DD2D': op_dd2d
    }
    
    # 遍历每个算子类
    for op_class_name, op_instance in operators.items():
        # 获取该类的所有方法
        methods = [method for method in dir(op_instance) if method.startswith('Mmask_')]
        
        # 遍历每个方法并测试
        for method_name in methods:
            method = getattr(op_instance, method_name)
            try:
                if op_class_name == 'OP_BA2D':
                    # BA2D类需要额外的day_data参数
                    result = method(test_data, day_data)
                elif op_class_name == 'OP_BF2D':
                    # BF2D类需要额外的lookback参数
                    result = method(test_data, lookback=1)
                elif op_class_name == 'OP_DD2D':
                    # DD2D类需要两个掩码作为输入
                    mask1 = op_b2d.Mmask_min(test_data)
                    mask2 = op_b2d.Mmask_max(test_data)
                    result = method(mask1, mask2)
                else:
                    # B2D类只需要test_data
                    result = method(test_data)
                
                # 记录测试结果
                shape_tests[f"{op_class_name}.{method_name}"] = result.shape == test_data.shape
                
            except Exception as e:
                print(f"测试 {op_class_name}.{method_name} 时发生错误: {str(e)}")
                shape_tests[f"{op_class_name}.{method_name}"] = False
    
    # 打印测试结果
    print("\n形状测试结果:")
    all_passed = True
    for op_name, result in shape_tests.items():
        print(f"{op_name:20s}: {'通过' if result else '失败'}")
        if not result:
            all_passed = False
    
    print("\n总体测试结果:", "全部通过" if all_passed else "存在失败项")
    
    # 逻辑测试
    # 初始化算子
    op_b2d = OP_B2D()
    op_dd2d = OP_DD2D()
    
    # 获取三个掩码
    mask_min = op_b2d.Mmask_min(test_data)
    mask_middle = op_b2d.Mmask_middle(test_data)
    mask_max = op_b2d.Mmask_max(test_data)
    
    # 测试1：验证三个掩码的并集是否覆盖所有True
    union_all = op_dd2d.Mmask_or(op_dd2d.Mmask_or(mask_min, mask_middle), mask_max)
    test1 = torch.all(union_all)
    print("\n测试1 - 三个掩码的并集是否全为True:")
    print(f"结果: {'通过' if test1 else '失败'}")
    
    # 测试2：验证任意两个掩码的交集是否为False
    intersection_min_middle = op_dd2d.Mmask_and(mask_min, mask_middle)
    intersection_middle_max = op_dd2d.Mmask_and(mask_middle, mask_max)
    intersection_min_max = op_dd2d.Mmask_and(mask_min, mask_max)
    
    test2_1 = not torch.any(intersection_min_middle)
    test2_2 = not torch.any(intersection_middle_max)
    test2_3 = not torch.any(intersection_min_max)
    
    print("\n测试2 - 任意两个掩码的交集是否为False:")
    print(f"min与middle交集: {'通过' if test2_1 else '失败'}")
    print(f"middle与max交集: {'通过' if test2_2 else '失败'}")
    print(f"min与max交集: {'通过' if test2_3 else '失败'}")
    
    # 测试3：验证数值大小关系
    min_values = test_data[mask_min]
    middle_values = test_data[mask_middle]
    max_values = test_data[mask_max]
    
    test3_1 = torch.all(torch.max(min_values) <= torch.max(middle_values))
    test3_2 = torch.all(torch.max(middle_values) <= torch.max(max_values))
    
    print("\n测试3 - 掩码对应值的大小关系:")
    print(f"min最大值 <= middle最小值: {'通过' if test3_1 else '失败'}")
    print(f"middle最大值 <= max最小值: {'通过' if test3_2 else '失败'}")
    
    # 总体结果
    all_tests_passed = test1 and test2_1 and test2_2 and test2_3 and test3_1 and test3_2
    print("\n总体测试结果:", "全部通过" if all_tests_passed else "存在失败项")

    # 获取两个掩码
    mask_plus_std = op_b2d.Mmask_mean_plus_std(test_data)
    mask_sub_std = op_b2d.Mmask_mean_sub_std(test_data)
    
    # 测试1：验证两个掩码的并集是否覆盖所有True
    union_all = op_dd2d.Mmask_or(mask_plus_std, mask_sub_std)
    test1 = torch.sum(union_all).item() / union_all.numel() > 0.9
    print("\n测试1 - 两个掩码的并集是否基本为True:")
    print(f"结果: {'通过' if test1 else '失败'}")
    
    # 测试2：验证两个掩码的交集是否为False
    intersection = op_dd2d.Mmask_and(mask_plus_std, mask_sub_std)
    test2 = not torch.any(intersection)
    
    print("\n测试2 - 两个掩码的交集是否为False:")
    print(f"结果: {'通过' if test2 else '失败'}")
    if not test2:
        print(f"False的比例: {torch.sum(~intersection).item() / intersection.numel():.2%}")
    
    # 测试3：验证数值大小关系
    plus_std_values = test_data[mask_plus_std]
    sub_std_values = test_data[mask_sub_std]
    
    # 注意：这里的大小关系应该是相反的，plus_std应该大于sub_std
    test3 = torch.all(torch.min(plus_std_values) > torch.max(sub_std_values))
    
    print("\n测试3 - 掩码对应值的大小关系:")
    print(f"mean_plus_std最小值 > mean_sub_std最大值: {'通过' if test3 else '失败'}")
    if not test3:
        print(f"mean_plus_std最小值: {torch.min(plus_std_values).item():.4f}")
        print(f"mean_sub_std最大值: {torch.max(sub_std_values).item():.4f}")
    
    # 初始化算子
    op_b2d = OP_B2D()
    op_dd2d = OP_DD2D()
    
    # 获取三个掩码
    mask_open = op_b2d.Mmask_1h_after_open(test_data)
    mask_middle = op_b2d.Mmask_2h_middle(test_data)
    mask_close = op_b2d.Mmask_1h_before_close(test_data)
    
    # 测试1：验证三个掩码的并集是否覆盖所有时间点
    union_all = op_dd2d.Mmask_or(op_dd2d.Mmask_or(mask_open, mask_middle), mask_close)
    test1 = torch.all(union_all)
    print("\n测试1 - 三个掩码的并集是否全为True:")
    print(f"结果: {'通过' if test1 else '失败'}")
    if not test1:
        print(f"True的比例: {torch.sum(union_all).item() / union_all.numel():.2%}")
        print(f"False的位置: {torch.nonzero(~union_all)}")
    
    # 测试2：验证任意两个掩码的交集是否为False
    intersection_open_middle = op_dd2d.Mmask_and(mask_open, mask_middle)
    intersection_middle_close = op_dd2d.Mmask_and(mask_middle, mask_close)
    intersection_open_close = op_dd2d.Mmask_and(mask_open, mask_close)
    
    test2_1 = not torch.any(intersection_open_middle)
    test2_2 = not torch.any(intersection_middle_close)
    test2_3 = not torch.any(intersection_open_close)
    
    print("\n测试2 - 掩码之间是否无交集:")
    print(f"开盘时段与中间时段: {'通过' if test2_1 else '失败'}")
    if not test2_1:
        print(f"交集位置: {torch.nonzero(intersection_open_middle)}")
    
    print(f"中间时段与收盘时段: {'通过' if test2_2 else '失败'}")
    if not test2_2:
        print(f"交集位置: {torch.nonzero(intersection_middle_close)}")
    
    print(f"开盘时段与收盘时段: {'通过' if test2_3 else '失败'}")
    if not test2_3:
        print(f"交集位置: {torch.nonzero(intersection_open_close)}")
    
    # 测试3：验证时间段的连续性
    print("\n测试3 - 时间段的范围验证:")
    print(f"开盘时段范围: 0-59")
    print(f"中间时段范围: 60-180")
    print(f"收盘时段范围: 181-241")
    print(f"开盘时段: {torch.sum(mask_open[0,0]).item()} 分钟")
    print(f"中间时段: {torch.sum(mask_middle[0,0]).item()} 分钟")
    print(f"收盘时段: {torch.sum(mask_close[0,0]).item()} 分钟")
    print(f"总覆盖时长: {torch.sum(union_all[0,0]).item()} 分钟")
    
    # 总体结果
    all_tests_passed = test1 and test2_1 and test2_2 and test2_3
    print("\n总体测试结果:", "全部通过" if all_tests_passed else "存在失败项")
    
    # 测试 Mmask_day_plus 和 Mmask_day_sub
    print("\n测试基于日频数据的掩码算子:")
    
    
    # 获取两个掩码
    mask_day_plus = op_ba2d.Mmask_day_plus(test_data, day_data)
    mask_day_sub = op_ba2d.Mmask_day_sub(test_data, day_data)
    
    # 测试1：验证两个掩码的并集是否覆盖所有时间点
    union_all = op_dd2d.Mmask_or(mask_day_plus, mask_day_sub)
    test1 = torch.sum(union_all).item() / union_all.numel() > 0.9
    print("\n测试1 - 两个掩码的并集是否基本为True:")
    print(f"结果: {'通过' if test1 else '失败'}")
    
    # 测试2：验证两个掩码的交集是否为False
    intersection = op_dd2d.Mmask_and(mask_day_plus, mask_day_sub)
    test2 = not torch.any(intersection)
    print("\n测试2 - 两个掩码是否无交集:")
    print(f"结果: {'通过' if test2 else '失败'}")
    if not test2:
        print(f"False的比例: {torch.sum(~intersection).item() / intersection.numel():.2%}")
    
    # 总体结果
    all_tests_passed = test1 and test2 and test3
    print("\n总体测试结果:", "全部通过" if all_tests_passed else "存在失败项")
    
    # 额外信息：显示掩码覆盖的比例
    print("\n掩码覆盖比例:")
    print(f"大于日频数据的比例: {torch.sum(mask_day_plus).item() / mask_day_plus.numel():.2%}")
    print(f"小于日频数据的比例: {torch.sum(mask_day_sub).item() / mask_day_sub.numel():.2%}")

    # 创建测试数据

    lookback_periods = [1, 2, 3]  # 测试不同的滚动窗口大小
    
    # 初始化算子
    op_bf2d = OP_BF2D()
    op_dd2d = OP_DD2D()
    
    print("\n测试滚动窗口掩码算子:")
    
    for lookback in lookback_periods:
        print(f"\n=== 测试 lookback={lookback} ===")
        
        # 获取两个掩码
        mask_rolling_plus = op_bf2d.Mmask_rolling_plus(test_data, lookback)
        mask_rolling_sub = op_bf2d.Mmask_rolling_sub(test_data, lookback)
        
        # 测试1：验证掩码形状
        shape_test = (mask_rolling_plus.shape == test_data.shape and 
                     mask_rolling_sub.shape == test_data.shape)
        print(f"\n1. 形状测试: {'通过' if shape_test else '失败'}")
        if not shape_test:
            print(f"期望形状: {test_data.shape}")
            print(f"实际形状: {mask_rolling_plus.shape}, {mask_rolling_sub.shape}")
        
        # 测试2：验证无交集
        intersection = op_dd2d.Mmask_and(mask_rolling_plus, mask_rolling_sub)
        no_intersection = not torch.any(intersection)
        print(f"\n2. 无交集测试: {'通过' if no_intersection else '失败'}")
        if not no_intersection:
            print(f"交集占比: {torch.sum(intersection).item() / intersection.numel():.2%}")
        
        # 测试3：验证数值大小关系
        plus_values = test_data[mask_rolling_plus]
        sub_values = test_data[mask_rolling_sub]
        if len(plus_values) > 0 and len(sub_values) > 0:
            value_relation = torch.min(plus_values) > torch.max(sub_values)
            print(f"\n3. 数值大小关系测试: {'通过' if value_relation else '失败'}")
            if not value_relation:
                print(f"rolling_plus最小值: {torch.min(plus_values).item():.4f}")
                print(f"rolling_sub最大值: {torch.max(sub_values).item():.4f}")
        else:
            print("\n3. 数值大小关系测试: 跳过（掩码为空）")
        
        # 测试4：验证滚动效果
        # 对于每一天，验证是否考虑了前lookback天的数据
        day_coverage = torch.zeros(test_data.shape[1], dtype=torch.bool)
        for day in range(lookback, test_data.shape[1]):
            current_day_plus = mask_rolling_plus[:, day, :]
            current_day_sub = mask_rolling_sub[:, day, :]
            has_effect = (torch.any(current_day_plus) or torch.any(current_day_sub))
            day_coverage[day] = has_effect
        
        rolling_test = torch.all(day_coverage[lookback:])
        print(f"\n4. 滚动效果测试: {'通过' if rolling_test else '失败'}")
        if not rolling_test:
            print(f"未覆盖的天数: {torch.sum(~day_coverage[lookback:]).item()}")
        
        # 额外信息
        print("\n掩码覆盖统计:")
        print(f"大于滚动最大值的比例: {torch.sum(mask_rolling_plus).item() / mask_rolling_plus.numel():.2%}")
        print(f"小于滚动最小值的比例: {torch.sum(mask_rolling_sub).item() / mask_rolling_sub.numel():.2%}")
