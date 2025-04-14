# losses/regularization_loss.py
import torch
from torch import nn


class RegularizationLoss(nn.Module):
    def __init__(self, lambda_weight=0.001, lambda_smooth=0.01, max_loss=100.0):
        super().__init__()
        self.lambda_weight = lambda_weight  # 降低权重衰减系数
        self.lambda_smooth = lambda_smooth  # 降低平滑度正则化系数
        self.max_loss = max_loss  # 添加最大损失限制

        # 缓存上一次的损失值，用于检测异常变化
        self.last_loss = None

    def forward(self, model):
        """
        计算正则化损失

        Args:
            model: 模型

        Returns:
            loss: 正则化损失
        """
        try:
            # 权重衰减（L2正则化）- 仅应用于特定层
            weight_loss = 0
            weight_count = 0
            for name, param in model.named_parameters():
                # 排除批归一化层的权重和所有偏置项
                if 'weight' in name and not any(bn_name in name for bn_name in ['bn', 'norm', 'BatchNorm']):
                    # 使用更温和的L2正则化
                    weight_loss += torch.sum(param ** 2)
                    weight_count += param.numel()

            # 计算平均权重损失
            if weight_count > 0:
                weight_loss = weight_loss / weight_count

            # 特征平滑度正则化 - 仅应用于特定卷积层
            smooth_loss = 0
            smooth_count = 0
            for name, module in model.named_modules():
                # 仅对特征提取器中的卷积层应用平滑度正则化
                if isinstance(module, nn.Conv2d) and ('feature' in name.lower() or 'extract' in name.lower()):
                    if hasattr(module, 'weight') and module.weight is not None:
                        # 计算权重的空间梯度
                        weight = module.weight
                        if weight.shape[2] > 1:  # 确保有足够的空间维度
                            weight_x_diff = weight[:, :, 1:, :] - weight[:, :, :-1, :]
                            smooth_loss += torch.sum(torch.abs(weight_x_diff))
                            smooth_count += weight_x_diff.numel()

                        if weight.shape[3] > 1:  # 确保有足够的空间维度
                            weight_y_diff = weight[:, :, :, 1:] - weight[:, :, :, :-1]
                            smooth_loss += torch.sum(torch.abs(weight_y_diff))
                            smooth_count += weight_y_diff.numel()

            # 计算平均平滑度损失
            if smooth_count > 0:
                smooth_loss = smooth_loss / smooth_count

            # 总正则化损失
            reg_loss = self.lambda_weight * weight_loss + self.lambda_smooth * smooth_loss

            # 限制最大损失
            reg_loss = torch.clamp(reg_loss, max=self.max_loss)

            # 检测异常变化
            if self.last_loss is not None:
                # 如果损失突然增大10倍以上，可能有问题
                if reg_loss > 10 * self.last_loss:
                    print(f"Warning: Regularization loss increased dramatically: {self.last_loss} -> {reg_loss}")
                    # 平滑处理，使用上一次损失的2倍作为当前损失
                    reg_loss = 2 * self.last_loss

            # 更新上一次损失
            self.last_loss = reg_loss.detach()

            return reg_loss

        except Exception as e:
            print(f"Error in regularization loss: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个小的常数作为后备
            return torch.tensor(1.0, device=next(model.parameters()).device)
