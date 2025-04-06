# losses/combined_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

from losses.edge_aware_loss import EdgeAwareLoss


class CombinedLoss_(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lambda_chamfer = getattr(cfg, 'lambda_chamfer', 1.0)
        self.lambda_density = getattr(cfg, 'lambda_density', 0.1)

    def forward(self, outputs, targets):
        """
        计算点云级别的损失
        Args:
            outputs: 包含 pred_points 和 pred_densities 的字典
            targets: 包含 target_points 的字典
        """
        # Chamfer距离损失
        chamfer_loss = self._compute_chamfer_loss(
            outputs['pred_points'],
            targets['target_points']
        )

        # 密度正则化损失
        density_loss = self._compute_density_loss(
            outputs['pred_densities']
        )

        # 总损失
        total_loss = (
                self.lambda_chamfer * chamfer_loss +
                self.lambda_density * density_loss
        )

        return {
            'total': total_loss,
            'chamfer': chamfer_loss,
            'density': density_loss
        }

    def _compute_chamfer_loss(self, pred_points, target_points):
        # 添加数值检查
        if torch.isnan(pred_points).any() or torch.isnan(target_points).any():
            print("Warning: NaN detected before chamfer distance")
            return torch.tensor(0.0, device=pred_points.device)



        loss, _ = chamfer_distance(pred_points, target_points)

        # 限制损失值范围
        loss = torch.clamp(loss, min=0.0, max=1e6)
        return loss

    def _compute_density_loss(self, densities):
        """密度正则化损失"""
        # 鼓励密度分布更加集中
        return torch.mean(torch.abs(densities))


# losses/combined_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

from losses.edge_aware_loss import EdgeAwareLoss


class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 损失权重
        self.lambda_chamfer = getattr(cfg, 'lambda_chamfer', 1.0)
        self.lambda_density = getattr(cfg, 'lambda_density', 0.1)
        self.lambda_edge = getattr(cfg, 'lambda_edge', 0.2)  # 新增边缘损失权重

        # 边缘感知损失
        self.use_edge_aware = getattr(cfg, 'use_edge_aware', True)  # 是否使用边缘感知损失
        if self.use_edge_aware:
            self.edge_loss = EdgeAwareLoss(
                lambda_lap=getattr(cfg, 'lambda_lap', 0.1),
                beta_grad=getattr(cfg, 'beta_grad', 0.05)
            )

    def forward(self, outputs, targets):
        """
        计算点云级别的损失
        Args:
            outputs: 包含 pred_points 和 pred_densities 的字典
            targets: 包含 target_points 的字典
        """
        # Chamfer距离损失
        chamfer_loss = self._compute_chamfer_loss(
            outputs['pred_points'],
            targets['target_points']
        )

        # 密度正则化损失
        density_loss = self._compute_density_loss(
            outputs['pred_densities']
        )

        # 边缘感知损失
        edge_loss = 0.0
        edge_metrics = {}
        if self.use_edge_aware:
            # 获取深度梯度（如果有）
            depth_gradients = targets.get('depth_gradients', None)

            # 计算边缘感知损失
            edge_loss, edge_metrics = self.edge_loss(
                outputs['pred_points'],
                outputs['pred_densities'],
                depth_gradients
            )

        # 总损失
        total_loss = (
                self.lambda_chamfer * chamfer_loss +
                self.lambda_density * density_loss +
                self.lambda_edge * edge_loss
        )

        return {
            'total': total_loss,
            'chamfer': chamfer_loss,
            'density': density_loss,
            'edge': edge_loss,
            **{f'edge_{k}': v for k, v in edge_metrics.items()}  # 添加边缘损失指标
        }

    def _compute_chamfer_loss(self, pred_points, target_points):
        # 添加数值检查
        if torch.isnan(pred_points).any() or torch.isnan(target_points).any():
            print("Warning: NaN detected before chamfer distance")
            return torch.tensor(0.0, device=pred_points.device)

        loss, _ = chamfer_distance(pred_points, target_points)

        # 限制损失值范围
        loss = torch.clamp(loss, min=0.0, max=1e6)
        return loss

    def _compute_density_loss(self, densities):
        """密度正则化损失"""
        # 鼓励密度分布更加集中
        return torch.mean(torch.abs(densities))
