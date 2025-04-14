# losses/combined_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

from losses.CLIP_Loss import CLIPLoss
from losses.depth_consistency_loss import DepthConsistencyLoss

from losses.edge_aware_loss import EdgeAwareLoss
from losses.regularization_loss import RegularizationLoss

# losses/combined_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

from losses.CLIP_Loss import CLIPLoss
from losses.depth_consistency_loss import DepthConsistencyLoss
from losses.edge_aware_loss import EdgeAwareLoss
from losses.regularization_loss import RegularizationLoss


class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 损失权重
        self.lambda_chamfer = getattr(cfg, 'lambda_chamfer', 0.5)
        self.lambda_density = getattr(cfg, 'lambda_density', 1.0)
        self.lambda_edge = getattr(cfg, 'lambda_edge', 0.5)
        self.lambda_clip = getattr(cfg, 'lambda_clip', 0.2)  # 添加CLIP损失权重
        self.lambda_depth = getattr(cfg, 'lambda_depth', 0.2)  # 添加深度一致性损失权重
        self.lambda_reg = getattr(cfg, 'lambda_reg', 0.01)  # 添加正则化损失权重

        # 边缘感知损失
        self.use_edge_aware = getattr(cfg, 'use_edge_aware', True)
        if self.use_edge_aware:
            self.edge_loss = EdgeAwareLoss(
                lambda_lap=getattr(cfg, 'lambda_lap', 0.1),
                beta_grad=getattr(cfg, 'beta_grad', 0.05)
            )

        # 深度一致性损失
        # TODO:需要修复这一损失。
        self.use_depth = getattr(cfg, 'use_depth', True)
        if self.use_depth:
            self.depth_loss = DepthConsistencyLoss()

        # CLIP损失
        self.use_clip = getattr(cfg, 'use_clip', True)
        if self.use_clip:
            self.clip_loss = CLIPLoss()

        # 正则化损失
        self.use_reg = getattr(cfg, 'use_reg', True)
        if self.use_reg:
            self.reg_loss = RegularizationLoss(
                lambda_weight=0.001,  # 大幅降低权重衰减系数
                lambda_smooth=0.01,   # 大幅降低平滑度正则化系数
                max_loss=50.0          # 设置最大损失限制
            )

    def forward(self, outputs, targets, model=None):
        """
        计算点云级别的损失
        Args:
            outputs: 包含预测结果的字典
            targets: 包含目标值的字典
            model: 可选的模型参数，用于正则化
        """
        loss_dict = {}

        # 确保必要的键存在
        if 'pred_points' not in outputs or 'target_points' not in targets:
            raise KeyError("Missing required keys: 'pred_points' or 'target_points'")

        if 'pred_densities' not in outputs:
            raise KeyError("Missing required key: 'pred_densities'")

        # Chamfer距离损失
        try:
            chamfer_loss = self._compute_chamfer_loss(
                outputs['pred_points'],
                targets['target_points']
            )
            loss_dict['chamfer'] = chamfer_loss
        except Exception as e:
            print(f"Error computing chamfer loss: {e}")
            chamfer_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
            loss_dict['chamfer'] = chamfer_loss

        # 密度正则化损失
        try:
            density_loss = self._compute_density_loss(
                outputs['pred_densities']
            )
            loss_dict['density'] = density_loss
        except Exception as e:
            print(f"Error computing density loss: {e}")
            density_loss = torch.tensor(0.0, device=outputs['pred_densities'].device)
            loss_dict['density'] = density_loss

        # 边缘感知损失
        edge_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
        if self.use_edge_aware:
            try:
                # 获取原始图像（如果有）
                images = outputs.get('original_images', None)

                # 计算边缘感知损失
                edge_loss, edge_metrics = self.edge_loss(
                    outputs['pred_points'],
                    outputs['pred_densities'],
                    images
                )
                loss_dict['edge'] = edge_loss
                # 添加边缘损失指标
                for k, v in edge_metrics.items():
                    loss_dict[f'edge_{k}'] = v
            except Exception as e:
                print(f"Error computing edge loss: {e}")
                edge_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
                loss_dict['edge'] = edge_loss

        # CLIP损失
        clip_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
        if self.use_clip:
            try:
                if 'density_features' in outputs and 'text_prompts' in targets:
                    clip_loss = self.clip_loss(
                        outputs['density_features'],
                        targets['text_prompts'],
                        outputs.get('original_images', None)
                    )
                    loss_dict['clip'] = clip_loss
            except Exception as e:
                print(f"Error computing CLIP loss: {e}")
                clip_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
                loss_dict['clip'] = clip_loss

        # 深度一致性损失
        depth_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
        if self.use_depth:
            try:
                if 'original_images' in outputs:
                    depth_loss = self.depth_loss(
                        outputs['original_images'],
                        outputs['pred_points'],
                        outputs['pred_densities']
                    )
                    depth_loss = torch.clamp(depth_loss, max=1.0)
                    loss_dict['depth'] = depth_loss
            except Exception as e:
                print(f"Error computing depth loss: {e}")
                depth_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
                loss_dict['depth'] = depth_loss

        # 正则化损失
        reg_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
        if self.use_reg and model is not None:
            try:
                reg_loss = self.reg_loss(model)
                loss_dict['reg'] = reg_loss
            except Exception as e:
                print(f"Error computing regularization loss: {e}")
                reg_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
                loss_dict['reg'] = reg_loss

        # 计算总损失
        total_loss = (
                self.lambda_chamfer * chamfer_loss +
                self.lambda_density * density_loss +
                self.lambda_edge * edge_loss +
                self.lambda_clip * clip_loss +
                self.lambda_depth * depth_loss +
                self.lambda_reg * reg_loss
        )

        loss_dict['total'] = total_loss

        return loss_dict

    def _compute_chamfer_loss(self, pred_points, target_points):
        """计算Chamfer距离损失"""
        # 添加数值检查
        if torch.isnan(pred_points).any() or torch.isnan(target_points).any():
            print("Warning: NaN detected before chamfer distance")
            return torch.tensor(0.0, device=pred_points.device)

        # 确保形状正确
        if pred_points.dim() != 3 or target_points.dim() != 3:
            raise ValueError(
                f"Points should be 3D tensors, got pred: {pred_points.shape}, target: {target_points.shape}")

        loss, _ = chamfer_distance(pred_points, target_points)

        # 检查损失是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid chamfer loss: {loss}")
            return torch.tensor(0.0, device=pred_points.device)

        # 限制损失值范围
        loss = torch.clamp(loss, min=0.0, max=1e6)
        return loss

    def _compute_density_loss(self, densities):
        """密度正则化损失"""
        # 检查输入
        if torch.isnan(densities).any():
            print("Warning: NaN detected in densities")
            return torch.tensor(0.0, device=densities.device)

        # 鼓励密度分布更加集中
        loss = torch.mean(torch.abs(densities))

        # 检查损失是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid density loss: {loss}")
            return torch.tensor(0.0, device=densities.device)

        return loss

    # def to(self, device):
    #     """将损失函数的所有相关组件移动到指定设备"""
    #     for attr_name, attr in self.__dict__.items():
    #         if hasattr(attr, 'to'):
    #             try:
    #                 # 检查 to 方法需要的参数
    #                 if callable(attr.to):
    #                     # 尝试调用 to 方法
    #                     setattr(self, attr_name, attr.to(device))
    #             except TypeError:
    #                 # 如果 to 方法需要不同的参数，可能是自定义对象
    #                 print(f"警告: 无法将 {attr_name} 移动到设备 {device}，跳过")
    #                 continue
    #     return self
