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
        self.func_list = ["nanmean", "nanstd", "corrwith", "rank_corrwith", "multi_regress", "regress"]

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
        return torch.sqrt(var / n - 1)

    @staticmethod
    def corrwith(tensor1, tensor2, dim=-1):
        mask = ~(torch.isnan(tensor1) | torch.isnan(tensor2))
        tensor1 = torch.where(mask, tensor1, float('nan'))
        tensor2 = torch.where(mask, tensor2, float('nan'))

        tensor1_mean = OP_Basic.nanmean(tensor1, dim=dim)
        tensor2_mean = OP_Basic.nanmean(tensor2, dim=dim)
        tensor1_std = OP_Basic.nanstd(tensor1, dim=dim, unbiased=False)
        tensor2_std = OP_Basic.nanstd(tensor2, dim=dim, unbiased=False)
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
        tensor1_std = OP_Basic.nanstd(tensor1, dim=dim, unbiased=False)
        tensor2_std = OP_Basic.nanstd(tensor2, dim=dim, unbiased=False)
        centered_tensor1 = tensor1 - tensor1_mean.unsqueeze(dim)
        centered_tensor2 = tensor2 - tensor2_mean.unsqueeze(dim)
        covariance = OP_Basic.nanmean(centered_tensor1 * centered_tensor2, dim=dim)
        correlation = covariance / (tensor1_std * tensor2_std)
        return correlation

    @staticmethod
    def multi_regress(y, x_s, dim=-1):
        down_epsilon = 1e-10
        up_epsilon = 1e20

        mask_x = torch.sum(torch.isnan(x_s), dim=dim) > 0
        mask_y = torch.isnan(y)
        mask = ~(mask_x | mask_y)
        y = torch.where(mask.unsqueeze(-1), y.unsqueeze(-1), 0)
        const = torch.ones(x_s.shape[:dim] + (x_s.shape[dim], 1)).to(x_s.device)
        x_const = torch.cat([x_s, const], dim=-1)
        x_const = torch.where(mask.unsqueeze(-1).expand_as(x_const), x_const, 0)

        x_T = x_const.permute(*range(len(x_const.shape) - 2), -1, -2)
        w = torch.matmul(x_T, x_const)
        det = torch.det(w)

        singularity = ((det < down_epsilon) | (abs(det) >= up_epsilon)).unsqueeze(-1).unsqueeze(-1).expand_as(w)
        theta = torch.matmul(torch.inverse(torch.where(singularity, float('nan'), w)), torch.matmul(x_T, y))
        k = theta[:, :-1]
        b = theta[:, -1]
        predict = torch.sum(k.unsqueeze(dim) * x_s, dim=dim) + b.unsqueeze(-1)
        res = torch.where(mask, y.squeeze(-1), float('nan')) - predict
        k = torch.where(abs(k) < down_epsilon, 0, k)
        b = torch.where(abs(b) < down_epsilon, 0, b)
        res = torch.where(abs(res) < down_epsilon, 0, res)
        return k, b, res

    @staticmethod
    def regress(y, x_s, dim=-1):
        return OP_Basic.multi_regress(y, x_s.unsqueeze(-1), dim=dim)
