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

        mask_x = ~torch.isnan(x_s).any(dim=-1)
        mask_y = ~torch.isnan(y).any(dim=-1)
        mask = mask_x & mask_y

        y_valid = torch.where(mask.unsqueeze(-1), y, torch.tensor(0.0, device=y.device))


        const = torch.ones(B, N, 1, device=x_s.device)
        x_const = torch.cat([x_s, const], dim=-1)

        x_const = torch.where(mask.unsqueeze(-1), x_const, torch.tensor(0.0, device=x_const.device))

        X_T = x_const.transpose(-2, -1)
        W = torch.matmul(X_T, x_const)
        XTY = torch.matmul(X_T, y_valid)

        W_pinv = torch.pinverse(W)
        theta = torch.matmul(W_pinv, XTY)

        k = theta[:, :-1, :]
        b = theta[:, -1, :]

        predict = torch.matmul(x_s, k) + b.unsqueeze(1)

        res = torch.where(mask.unsqueeze(-1), y - predict, torch.tensor(float('nan'), device=y.device))

        k = torch.where(torch.abs(k) < down_epsilon, torch.tensor(0.0, device=k.device), k)
        b = torch.where(torch.abs(b) < down_epsilon, torch.tensor(0.0, device=b.device), b)
        res = torch.where(torch.abs(res) < down_epsilon, torch.tensor(0.0, device=res.device), res)

        return k, b, res

    @staticmethod
    def regress(y, x_s, dim=-1):
        if y.dim() == x_s.dim():
            return OP_Basic.multi_regress(y, x_s, dim=dim)
        elif y.dim() == (x_s.dim() - 1):
            y = y.unsqueeze(-1)
            return OP_Basic.multi_regress(y, x_s, dim=dim)
        else:
            raise ValueError(f"Unsupported dimension mismatch: x_s.dim()={x_s.dim()}, y.dim()={y.dim()}")


    @staticmethod
    def PCA(tensor):
        pass

