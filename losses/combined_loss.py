# losses/combined_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

from losses.edge_aware_loss import EdgeAwareLoss


class CombinedLoss(nn.Module):
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


class CombinedLoss_(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化权重
        self.lambda_depth = getattr(cfg, 'lambda_depth', 0.5)
        self.lambda_clip = getattr(cfg, 'lambda_clip', 0.3)
        self.lambda_edge = getattr(cfg, 'lambda_edge', 0.2)
        self.lambda_reg = getattr(cfg, 'lambda_reg', 0.01)

        # 边缘感知损失
        self.edge_loss = EdgeAwareLoss(
            lambda_lap=getattr(cfg, 'lambda_lap', 0.1),
            beta_grad=getattr(cfg, 'beta_grad', 0.05)
        )

        # 温度参数
        self.temperature = getattr(cfg, 'temperature', 0.07)
        # 添加渲染尺寸属性
        self.render_size = getattr(cfg, 'render_size', 128)

        # 简化版本的标志
        self.simplified = True

    def forward(self, *args):
        """
        计算组合损失 - 支持两种调用方式
        1. forward(outputs, targets) - 完整版本
        2. forward(rendered_images, target_images) - 简化版本
        """
        if len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            # 完整版本调用
            self.simplified = False
            return self._forward_full(args[0], args[1])
        elif len(args) == 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            # 简化版本调用
            self.simplified = True
            return self._forward_simplified(args[0], args[1])
        else:
            raise ValueError(f"Unsupported arguments: {args}")

    def _forward_full(self, outputs, targets):
        """
        计算组合损失 - 完整版本
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

    def _forward_simplified(self, rendered_images, target_images):
        """
        计算组合损失 - 简化版本，只使用RGB重建损失
        Args:
            rendered_images: 渲染的图像张量 [B, 3, H, W]
            target_images: 目标图像张量 [B, 3, H, W]
        """
        # 调整目标图像尺寸以匹配渲染图像
        if target_images.shape != rendered_images.shape:
            target_images = F.interpolate(
                target_images,
                size=(rendered_images.shape[2], rendered_images.shape[3]),
                mode='bilinear',
                align_corners=False
            )

        # 计算RGB重建损失
        rgb_loss = F.mse_loss(rendered_images, target_images)

        # 返回损失字典
        loss_dict = {
            'total': rgb_loss,
            'rgb': rgb_loss
        }

        return loss_dict

    def _compute_depth_loss(self, pred_depth, gt_depth, mask):
        """计算深度一致性损失"""
        if self.simplified:
            return torch.tensor(0.0, device=pred_depth.device)

        # 应用可见性掩码
        masked_pred = pred_depth * mask
        masked_gt = gt_depth * mask

        # 计算L2损失
        return F.mse_loss(masked_pred, masked_gt)

    def _compute_clip_loss(self, rendered_images, text_features, image_features):
        """计算CLIP对比损失"""
        if self.simplified:
            return torch.tensor(0.0, device=rendered_images.device)

        # 计算相似度
        similarity = torch.matmul(image_features, text_features.T) / self.temperature

        # 对角线元素是正样本
        labels = torch.arange(similarity.shape[0], device=similarity.device)

        # 交叉熵损失
        loss = F.cross_entropy(similarity, labels)

        return loss

    def _compute_regularization(self, params):
        """计算参数正则化损失"""
        if self.simplified:
            # 在简化模式下返回零损失
            return torch.tensor(0.0, device=next(iter(params)).device if params else torch.device('cpu'))

        # 简单的L2正则化
        reg_loss = sum(p.pow(2).sum() for p in params)
        return reg_loss


def chamfer_distance_loss(pred_points, target_points, epsilon=1e-8):
    """添加维度检查和自动调整的 Chamfer 距离实现"""
    # 打印输入张量的形状以便调试
    print(f"pred_points shape: {pred_points.shape}")
    print(f"target_points shape: {target_points.shape}")

    # 自动调整维度
    if pred_points.dim() == 2:
        pred_points = pred_points.unsqueeze(0)  # 添加批次维度
    if target_points.dim() == 2:
        target_points = target_points.unsqueeze(0)  # 添加批次维度

    # 确保维度正确
    if pred_points.dim() != 3 or target_points.dim() != 3:
        raise ValueError(
            f"Expected 3D tensors, got pred_points: {pred_points.dim()}D, target_points: {target_points.dim()}D")

    # 确保最后一个维度是3（xyz坐标）
    if pred_points.shape[-1] != 3 or target_points.shape[-1] != 3:
        raise ValueError(
            f"Expected last dimension to be 3, got pred_points: {pred_points.shape[-1]}, target_points: {target_points.shape[-1]}")

    # 计算距离矩阵
    P = pred_points.unsqueeze(2)  # [B,N,1,3]
    T = target_points.unsqueeze(1)  # [B,1,M,3]

    # 计算距离
    dist = torch.sum((P - T) ** 2, dim=-1) + epsilon

    # 计算最近邻距离
    min_dist_p2t = torch.min(dist, dim=2)[0]  # [B,N]
    min_dist_t2p = torch.min(dist, dim=1)[0]  # [B,M]

    # 计算平均距离
    chamfer_dist = torch.mean(min_dist_p2t) + torch.mean(min_dist_t2p)

    return chamfer_dist
