import torch

OPclass_name_others = ['OP_Closure', 'OP_Basic']


class OP_Closure:
    def __init__(self):
        self.func_list = ["id_industry", "id_int", "id_float", "id_tensor"]

    @staticmethod
    def id_industry(industry):
        return industry

    @staticmethod
    def id_int(int):
        return int

    @staticmethod
    def id_float(thresh):
        return thresh

    @staticmethod
    def id_tensor(tensor):
        return tensor

class OP_Basic:
    def __init__(self):
        self.func_list = ["nanmean", "nanstd", "corrwith", "rank_corrwith", "multi_regress", "regress","PCA"]

    @staticmethod
    def nanmean(tensor, dim=-1):
        return torch.nansum(tensor, dim=dim) / torch.sum(~torch.isnan(tensor), dim=dim)

    @staticmethod
    def nanstd(x, dim=-1):
        mean = OP_Basic.nanmean(x, dim=dim)
        centered_tensor = (x - mean.unsqueeze(dim))
        centered_tensor = centered_tensor.masked_fill(torch.isnan(centered_tensor), 0)
        var = torch.nansum(centered_tensor.pow(2), dim=dim)
        n = torch.sum(~torch.isnan(x), dim=dim)
        return torch.sqrt(var / n)

    @staticmethod
    def corrwith(tensor1, tensor2, dim=-1):
        mask = ~(torch.isnan(tensor1) | torch.isnan(tensor2))
        tensor1 = torch.where(mask, tensor1, float('nan'))
        tensor2 = torch.where(mask, tensor2, float('nan'))

        tensor1_mean = OP_Basic.nanmean(tensor1, dim=dim)
        tensor2_mean = OP_Basic.nanmean(tensor2, dim=dim)
        tensor1_std = OP_Basic.nanstd(tensor1, dim=dim)
        tensor2_std = OP_Basic.nanstd(tensor2, dim=dim)
        centered_tensor1 = tensor1 - tensor1_mean.unsqueeze(dim)
        centered_tensor2 = tensor2 - tensor2_mean.unsqueeze(dim)
        covariance = OP_Basic.nanmean(centered_tensor1 * centered_tensor2, dim=dim)
        correlation = covariance / (tensor1_std * tensor2_std)
        return correlation

    @staticmethod
    def rank_corrwith(tensor1, tensor2, dim=-1):
        nan_mask = torch.isnan(tensor1) | torch.isnan(tensor2)
        tensor1 = tensor1.masked_fill(nan_mask, float('nan'))
        tensor1 = torch.argsort(torch.argsort(tensor1, dim=dim))
        tensor1 = torch.where(~nan_mask, tensor1, float('nan'))
        tensor2 = tensor2.masked_fill(nan_mask, float('nan'))
        tensor2 = torch.argsort(torch.argsort(tensor2, dim=dim))
        tensor2 = torch.where(~nan_mask, tensor2, float('nan'))

        tensor1_mean = OP_Basic.nanmean(tensor1, dim=dim)
        tensor2_mean = OP_Basic.nanmean(tensor2, dim=dim)
        tensor1_std = OP_Basic.nanstd(tensor1, dim=dim)
        tensor2_std = OP_Basic.nanstd(tensor2, dim=dim)
        centered_tensor1 = tensor1 - tensor1_mean.unsqueeze(dim)
        centered_tensor2 = tensor2 - tensor2_mean.unsqueeze(dim)
        covariance = OP_Basic.nanmean(centered_tensor1 * centered_tensor2, dim=dim)
        correlation = covariance / (tensor1_std * tensor2_std)
        return correlation

    @staticmethod
    def multi_regress(y, x_s, dim=-1):
        down_epsilon = 1e-10
        B, N, D = x_s.shape
        M = y.shape[-1]
        if y.dim() == 2:
            y = y.unsqueeze(-1)

        mask_x = ~torch.isnan(x_s).any(dim=-1)  # (B, N)
        mask_y = ~torch.isnan(y).any(dim=-1)  # (B, N)
        mask = mask_x & mask_y  # (B, N)
        y_valid = torch.where(mask.unsqueeze(-1), y, 0.0)  # (B, N, M)
        x_valid = torch.where(mask.unsqueeze(-1), x_s, 0.0)  # (B, N, D)
        const = torch.ones(B, N, 1, device=x_s.device)
        x_const = torch.cat([x_valid, const], dim=-1)  # (B, N, D+1)
        x_const = x_const * mask.unsqueeze(-1)
        
        X_T = x_const.transpose(-2, -1)  # (B, D+1, N)
        W = torch.matmul(X_T, x_const)  # (B, D+1, D+1)
        XTY = torch.matmul(X_T, y_valid)  # (B, D+1, M)
        W_pinv = torch.pinverse(W)
        theta = torch.matmul(W_pinv, XTY)  # (B, D+1, M)

        k = theta[:, :-1, :]  # (B, D, M)
        b = theta[:, -1, :]  # (B, M)

        predict = torch.matmul(x_s, k) + b.unsqueeze(1)
        mask_expanded = mask.unsqueeze(-1).expand_as(y)  # (B, N, M)
        res = torch.where(mask_expanded, y - predict, torch.nan)

        k = torch.where(torch.abs(k) < down_epsilon, 0.0, k)
        b = torch.where(torch.abs(b) < down_epsilon, 0.0, b)
        res = torch.where(torch.abs(res) < down_epsilon, 0.0, res)
        return k.squeeze(-1), b.squeeze(-1), res.squeeze(-1)
    
    @staticmethod
    def regress(y, x, dim=-1):
        # 确保输入的两个张量形状相同
        if y.shape != x.shape:
            raise ValueError("The shapes of y and x must be the same.")

        # 创建mask，确保x和y都为非NaN
        mask = ~torch.isnan(x) & ~torch.isnan(y)

        # 只保留mask为True的部分
        y_valid = torch.where(mask, y, torch.tensor(0.0, device=y.device))
        x_valid = torch.where(mask, x, torch.tensor(0.0, device=x.device))

        # 计算x的均值和方差
        x_mean = torch.sum(x_valid, dim=dim) / torch.sum(mask, dim=dim)
        y_mean = torch.sum(y_valid, dim=dim) / torch.sum(mask, dim=dim)

        # 计算回归系数k和截距b
        xy_cov = torch.sum((x_valid - x_mean.unsqueeze(dim)) * (y_valid - y_mean.unsqueeze(dim)), dim=dim)
        x_var = torch.sum((x_valid - x_mean.unsqueeze(dim)) ** 2, dim=dim)

        k = xy_cov / x_var
        b = y_mean - k * x_mean

        # 计算预测值和残差
        predict = k.unsqueeze(dim) * x + b.unsqueeze(dim)
        res = torch.where(mask, y - predict, float('nan'))

        return k, b, res


    @staticmethod
    def PCA(tensor):
        pass

