# losses/combined_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.edge_aware_loss import EdgeAwareLoss


class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化权重
        self.lambda_depth = cfg.lambda_depth
        self.lambda_clip = cfg.lambda_clip
        self.lambda_edge = cfg.lambda_edge
        self.lambda_reg = cfg.lambda_reg

        # 边缘感知损失
        self.edge_loss = EdgeAwareLoss(
            lambda_lap=cfg.lambda_lap,
            beta_grad=cfg.beta_grad
        )

        # 温度参数
        self.temperature = cfg.temperature

    def forward(self, outputs, targets):
        """
        计算组合损失
        Args:
            outputs: 模型输出字典
            targets: 目标字典
        """
        # 1. 深度一致性损失
        depth_loss = self._compute_depth_loss(
            outputs['pred_depth'],
            targets['gt_depth'],
            targets['visibility_mask']
        )

        # 2. CLIP对比损失
        clip_loss = self._compute_clip_loss(
            outputs['rendered_images'],
            targets['text_features'],
            outputs['image_features']
        )

        # 3. 边缘感知损失
        edge_loss = self.edge_loss(
            outputs['pred_mesh'],
            targets.get('gt_mesh', None)
        )

        # 4. 正则化损失
        reg_loss = self._compute_regularization(outputs['model_params'])

        # 总损失
        total_loss = (
                self.lambda_depth * depth_loss +
                self.lambda_clip * clip_loss +
                self.lambda_edge * edge_loss +
                self.lambda_reg * reg_loss
        )

        # 返回损失字典用于记录
        loss_dict = {
            'total': total_loss,
            'depth': depth_loss,
            'clip': clip_loss,
            'edge': edge_loss,
            'reg': reg_loss
        }

        return loss_dict

    def _compute_depth_loss(self, pred_depth, gt_depth, mask):
        """计算深度一致性损失"""
        # 应用可见性掩码
        masked_pred = pred_depth * mask
        masked_gt = gt_depth * mask

        # 计算L2损失
        return F.mse_loss(masked_pred, masked_gt)

    def _compute_clip_loss(self, rendered_images, text_features, image_features):
        """计算CLIP对比损失"""
        # 计算相似度
        similarity = torch.matmul(image_features, text_features.T) / self.temperature

        # 对角线元素是正样本
        labels = torch.arange(similarity.shape[0], device=similarity.device)

        # 交叉熵损失
        loss = F.cross_entropy(similarity, labels)

        return loss

    def _compute_regularization(self, params):
        """计算参数正则化损失"""
        # 简单的L2正则化
        reg_loss = sum(p.pow(2).sum() for p in params)
        return reg_loss
