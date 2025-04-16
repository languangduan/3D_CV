# losses/combined_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

from losses.CLIP_Loss import CLIPLoss, ImprovedCLIPLoss
from losses.depth_consistency_loss import DepthConsistencyLoss
from losses.edge_aware_loss import EdgeAwareLoss
from losses.regularization_loss import RegularizationLoss


class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 原始损失权重设置
        self.lambda_chamfer = getattr(cfg, 'lambda_chamfer', 0.5)
        self.lambda_density = getattr(cfg, 'lambda_density', 1.0)
        self.lambda_edge = getattr(cfg, 'lambda_edge', 0.5)
        self.lambda_clip = getattr(cfg, 'lambda_clip', 0.2)
        self.lambda_depth = getattr(cfg, 'lambda_depth', 0.2)
        self.lambda_reg = getattr(cfg, 'lambda_reg', 0.01)

        # 添加动态归一化所需的参数
        self.use_dynamic_weights = getattr(cfg, 'use_dynamic_weights', True)
        self.ema_decay = getattr(cfg, 'ema_decay', 0.99)

        # 使用指数移动平均来跟踪每个损失项的典型值
        self.loss_ema = {
            'chamfer': 0.6,  # 初始估计值
            'density': 1e-5,
            'edge': 0.02,
            'edge_laplacian_loss': 5e-9,
            'edge_gradient_consistency': 0.3,
            'depth': 0.5,
            'reg': 2e-6,
            'shape_prior': 0.08,
            'clip': 0.2
        }

        # 目标损失贡献比例
        self.target_weights = {
            'chamfer': 0.30,  # 30%
            'density': 0.05,  # 5%
            'edge': 0.15,  # 15%
            'depth': 0.30,  # 30%
            'reg': 0.05,  # 5%
            'shape_prior': 0.15,  # 15%
            'clip': 0.10  # 10% (如果使用)
        }

        # 训练步数计数器
        self.steps = 0

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
        self.use_clip = getattr(cfg, 'use_clip', False)
        if self.use_clip:
            self.clip_loss = ImprovedCLIPLoss()

        # 正则化损失
        self.use_reg = getattr(cfg, 'use_reg', True)
        if self.use_reg:
            self.reg_loss = RegularizationLoss(
                lambda_weight=0.001,  # 大幅降低权重衰减系数
                lambda_smooth=0.01,  # 大幅降低平滑度正则化系数
                max_loss=50.0  # 设置最大损失限制
            )


    def update_ema(self, name, value):
        """更新损失项的指数移动平均"""
        if name in self.loss_ema:
            if self.loss_ema[name] == 0:
                self.loss_ema[name] = value
            else:
                self.loss_ema[name] = self.ema_decay * self.loss_ema[name] + (1 - self.ema_decay) * value
        return self.loss_ema.get(name, value)

    def get_normalized_weight(self, name, raw_value):
        """计算归一化权重"""
        ema_value = self.update_ema(name, raw_value)
        if ema_value < 1e-12:  # 防止除零
            return 0.0

        # 计算目标权重
        target_weight = self.target_weights.get(name, 0.1)

        # 归一化权重 = 目标权重 / EMA值
        normalized_weight = target_weight / (ema_value + 1e-12)

        # 限制权重范围，防止数值不稳定
        normalized_weight = min(normalized_weight, 1e6)

        return normalized_weight

    def forward(self, outputs, targets, model=None):
        """
        计算点云级别的损失
        Args:
            outputs: 包含预测结果的字典
            targets: 包含目标值的字典
            model: 可选的模型参数，用于正则化
        """
        self.steps += 1
        loss_dict = {}
        raw_losses = {}

        # 确保必要的键存在
        if 'pred_points' not in outputs or 'target_points' not in targets:
            raise KeyError("Missing required keys: 'pred_points' or 'target_points'")

        if 'pred_densities' not in outputs:
            raise KeyError("Missing required key: 'pred_densities'")

        # Chamfer距离损失
        try:
            # 检查是否有法线信息可用
            pred_normals = outputs.get('pred_normals', None)

            chamfer_loss = self._compute_chamfer_loss(
                outputs['pred_points'],
                targets['target_points'],
                pred_normals  # 传递法线信息
            )
            loss_dict['chamfer'] = chamfer_loss
            raw_losses['chamfer'] = chamfer_loss.detach().item()

        except Exception as e:
            print(f"Error computing chamfer loss: {e}")
            chamfer_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
            loss_dict['chamfer'] = chamfer_loss
            raw_losses['chamfer'] = 0.6  # 使用默认值

        # 密度正则化损失
        try:
            density_loss = self._compute_density_loss(
                outputs['pred_densities']
            )
            loss_dict['density'] = density_loss
            raw_losses['density'] = density_loss.detach().item()
        except Exception as e:
            print(f"Error computing density loss: {e}")
            density_loss = torch.tensor(0.0, device=outputs['pred_densities'].device)
            loss_dict['density'] = density_loss
            raw_losses['density'] = 1e-5  # 使用默认值

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
                raw_losses['edge'] = edge_loss.detach().item()

                # 添加边缘损失指标
                for k, v in edge_metrics.items():
                    loss_dict[f'edge_{k}'] = v
                    raw_losses[f'edge_{k}'] = v.detach().item() if isinstance(v, torch.Tensor) else v
            except Exception as e:
                print(f"Error computing edge loss: {e}")
                edge_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
                loss_dict['edge'] = edge_loss
                raw_losses['edge'] = 0.02  # 使用默认值

        # CLIP损失
        # CLIP损失
        clip_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
        if self.use_clip:
            try:
                if 'text_prompts' in targets:
                    # 使用改进的CLIP损失，直接传递点云和法线
                    clip_loss = self.clip_loss(
                        outputs['pred_points'],
                        outputs['pred_densities'],
                        targets['text_prompts'],
                        outputs.get('pred_normals', None),  # 传递法线信息
                        outputs.get('original_images', None)  # 传递原始图像
                    )
                    loss_dict['clip'] = clip_loss
                    raw_losses['clip'] = clip_loss.detach().item()
            except Exception as e:
                print(f"Error computing CLIP loss: {e}")
                clip_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
                loss_dict['clip'] = clip_loss
                raw_losses['clip'] = 0.2  # 使用默认值

        # clip_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
        # if self.use_clip:
        #     try:
        #         if 'density_features' in outputs and 'text_prompts' in targets:
        #             clip_loss = self.clip_loss(
        #                 outputs['density_features'],
        #                 targets['text_prompts'],
        #                 outputs.get('original_images', None)
        #             )
        #             loss_dict['clip'] = clip_loss
        #             raw_losses['clip'] = clip_loss.detach().item()
        #     except Exception as e:
        #         print(f"Error computing CLIP loss: {e}")
        #         clip_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
        #         loss_dict['clip'] = clip_loss
        #         raw_losses['clip'] = 0.2  # 使用默认值

        # 深度一致性损失
        depth_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
        if self.use_depth:
            try:
                if 'original_images' in outputs:
                    # 传递法线信息到深度一致性损失
                    pred_normals = outputs.get('pred_normals', None)
                    depth_loss = self.depth_loss(
                        outputs['original_images'],
                        outputs['pred_points'],
                        outputs['pred_densities'],
                        pred_normals  # 添加法线参数
                    )
                    depth_loss = torch.clamp(depth_loss, max=1.0)
                    loss_dict['depth'] = depth_loss
                    raw_losses['depth'] = depth_loss.detach().item()
            except Exception as e:
                print(f"Error computing depth loss: {e}")
                depth_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
                loss_dict['depth'] = depth_loss
                raw_losses['depth'] = 0.5  # 使用默认值

        # 正则化损失
        reg_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
        if self.use_reg and model is not None:
            try:
                reg_loss = self.reg_loss(model)
                loss_dict['reg'] = reg_loss
                raw_losses['reg'] = reg_loss.detach().item()
            except Exception as e:
                print(f"Error computing regularization loss: {e}")
                reg_loss = torch.tensor(0.0, device=outputs['pred_points'].device)
                loss_dict['reg'] = reg_loss
                raw_losses['reg'] = 2e-6  # 使用默认值

        # 形状先验损失
        shape_prior_loss = self._compute_shape_prior_loss(outputs['pred_points'])
        loss_dict['shape_prior'] = shape_prior_loss
        raw_losses['shape_prior'] = shape_prior_loss.detach().item()

        # 计算动态权重
        if self.use_dynamic_weights:
            weights = {}
            for name, value in raw_losses.items():
                weights[name] = self.get_normalized_weight(name, value)

            # 应用权重并记录损失
            weighted_losses = {}

            weighted_losses['chamfer'] = chamfer_loss * weights.get('chamfer', self.lambda_chamfer)
            weighted_losses['density'] = density_loss * weights.get('density', self.lambda_density)

            if self.use_edge_aware:
                weighted_losses['edge'] = edge_loss * weights.get('edge', self.lambda_edge)

            if self.use_depth:
                weighted_losses['depth'] = depth_loss * weights.get('depth', self.lambda_depth)

            if self.use_clip:
                weighted_losses['clip'] = clip_loss * weights.get('clip', self.lambda_clip)

            if self.use_reg and model is not None:
                weighted_losses['reg'] = reg_loss * weights.get('reg', self.lambda_reg)

            weighted_losses['shape_prior'] = shape_prior_loss * weights.get('shape_prior', 0.1)

            # 计算总损失
            total_loss = sum(weighted_losses.values())

            # 记录权重信息（用于调试）
            if self.steps % 10 == 0:
                print(f"\nLoss weights: {weights}")
                print(f"Raw losses: {raw_losses}")
                print(f"Weighted losses: {weighted_losses}\n")

            # 记录权重信息到loss_dict
            for name, value in weights.items():
                loss_dict[f'weight_{name}'] = torch.tensor(value, device=total_loss.device)

            for name, value in weighted_losses.items():
                loss_dict[f'weighted_{name}'] = value
        else:
            # 使用固定权重
            total_loss = (
                    self.lambda_chamfer * chamfer_loss +
                    self.lambda_density * density_loss +
                    self.lambda_edge * edge_loss +
                    self.lambda_clip * clip_loss +
                    self.lambda_depth * depth_loss +
                    self.lambda_reg * reg_loss +
                    0.1 * shape_prior_loss
            )

        loss_dict['total'] = total_loss

        return loss_dict

    def _compute_chamfer_loss(self, pred_points, target_points, pred_normals=None):
        """计算Chamfer距离损失，可选择使用法线信息增强"""
        # 添加数值检查
        if torch.isnan(pred_points).any() or torch.isnan(target_points).any():
            print("Warning: NaN detected before chamfer distance")
            return torch.tensor(1.0, device=pred_points.device)  # 返回非零值以避免训练停滞

        # 确保形状正确
        if pred_points.dim() != 3 or target_points.dim() != 3:
            raise ValueError(
                f"Points should be 3D tensors, got pred: {pred_points.shape}, target: {target_points.shape}")

        # 添加点云采样，减少计算量并提高稳定性
        max_points = 5000  # 限制最大点数
        if pred_points.shape[1] > max_points:
            idx = torch.randperm(pred_points.shape[1], device=pred_points.device)[:max_points]
            pred_points = pred_points[:, idx, :]
            if pred_normals is not None:
                pred_normals = pred_normals[:, idx, :]

        if target_points.shape[1] > max_points:
            idx = torch.randperm(target_points.shape[1], device=target_points.device)[:max_points]
            target_points = target_points[:, idx, :]

        # 添加小噪声避免完全重合点
        noise = torch.randn_like(pred_points) * 1e-6
        pred_points = pred_points + noise

        # 标准Chamfer距离计算
        loss, _ = chamfer_distance(
            pred_points,
            target_points,
            point_reduction="mean",
            batch_reduction="mean"
        )

        # 检查损失是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid chamfer loss: {loss}")
            return torch.tensor(1.0, device=pred_points.device)  # 返回非零值

        # 应用平方根变换，减小大值影响
        loss = torch.sqrt(loss + 1e-8)

        # 限制损失值范围，避免异常值
        loss = torch.clamp(loss, min=0.1, max=1.0)

        return loss

    def _compute_shape_prior_loss(self, pred_points):
        """添加形状先验损失，引导点云向合理形状收敛"""
        batch_size = pred_points.shape[0]
        device = pred_points.device

        # 计算每个点云的中心和半径
        centers = torch.mean(pred_points, dim=1, keepdim=True)  # [B, 1, 3]
        centered_points = pred_points - centers

        # 计算点到中心的距离
        distances = torch.norm(centered_points, dim=2)  # [B, N]

        # 计算每个点云的平均半径
        mean_radii = torch.mean(distances, dim=1)  # [B]

        # 创建一个参考球体 - 假设我们希望点云近似球形
        # 可以根据具体应用调整为其他形状
        unit_sphere_loss = torch.mean((distances - mean_radii.unsqueeze(1)).pow(2))

        # 添加体积约束 - 防止点云收缩到一点
        volume_loss = torch.mean(torch.exp(-mean_radii * 10))

        # 添加对称性约束 - 如果适用
        # 这里假设x轴对称
        left_points = pred_points[..., 0] < 0
        right_points = pred_points[..., 0] > 0

        symmetry_loss = torch.tensor(0.0, device=device)
        if torch.any(left_points) and torch.any(right_points):
            # 创建镜像点
            mirrored_points = pred_points.clone()
            mirrored_points[..., 0] = -mirrored_points[..., 0]

            # 计算与最近镜像点的距离
            sym_dist, _ = chamfer_distance(
                pred_points,
                mirrored_points,
                point_reduction="mean",
                batch_reduction="mean"
            )
            symmetry_loss = sym_dist

        # 组合损失
        combined_loss = unit_sphere_loss + 0.1 * volume_loss + 0.2 * symmetry_loss

        return combined_loss

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