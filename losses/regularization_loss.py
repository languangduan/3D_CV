# losses/regularization_loss.py
import torch
from torch import nn


class RegularizationLoss(nn.Module):
    def __init__(self, lambda_weight=0.01, lambda_smooth=0.05):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.lambda_smooth = lambda_smooth

    def forward(self, model):
        """
        计算正则化损失

        Args:
            model: 模型

        Returns:
            loss: 正则化损失
        """
        # 权重衰减（L2正则化）
        weight_loss = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_loss += torch.sum(param ** 2)

        # 特征平滑度正则化
        smooth_loss = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, 'weight') and module.weight is not None:
                    # 计算权重的空间梯度
                    weight = module.weight
                    weight_x_diff = weight[:, :, 1:, :] - weight[:, :, :-1, :]
                    weight_y_diff = weight[:, :, :, 1:] - weight[:, :, :, :-1]

                    # 使用L1范数鼓励稀疏梯度
                    smooth_loss += torch.sum(torch.abs(weight_x_diff)) + torch.sum(torch.abs(weight_y_diff))

        # 总正则化损失
        reg_loss = self.lambda_weight * weight_loss + self.lambda_smooth * smooth_loss

        return reg_loss
