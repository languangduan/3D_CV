# losses/depth_consistency_loss.py
import torch
from torch import nn
import torch.nn.functional as F

from utils.depth_estimator import DepthEstimator


class DepthConsistencyLoss(nn.Module):
    def __init__(self, image_size=256, min_depth=0.1, max_depth=10.0):
        super().__init__()
        self.depth_estimator = DepthEstimator()
        self.image_size = image_size
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, images, points, densities):
        """
        计算深度一致性损失

        Args:
            images: 输入图像 [B, 3, H, W]
            points: 点云坐标 [B, N, 3]
            densities: 密度值 [B, N, 1]

        Returns:
            loss: 深度一致性损失
        """
        try:
            # 估计深度图
            with torch.no_grad():  # 不需要梯度
                depth_gt, _ = self.depth_estimator(images)

            # 调整深度图大小以匹配我们的目标分辨率
            B, _, H, W = depth_gt.shape
            if H != self.image_size or W != self.image_size:
                depth_gt = F.interpolate(depth_gt, size=(self.image_size, self.image_size),
                                         mode='bilinear', align_corners=False)

            # 归一化真实深度图
            depth_gt = self.normalize_depth(depth_gt)

            # 将点云投影到图像平面
            projected_depth = self.project_points_to_depth(points, densities)

            # 归一化投影深度图
            projected_depth = self.normalize_depth(projected_depth)

            # 创建有效区域掩码
            pred_mask = (projected_depth > 0).float()  # 预测深度有效区域
            gt_mask = (depth_gt > 0).float()  # 真实深度有效区域
            valid_mask = pred_mask * gt_mask  # 两者都有效的区域

            # 如果有效区域太小，返回零损失
            if valid_mask.sum() < 10:
                return torch.tensor(0.0, device=images.device)

            # 计算深度一致性损失（仅在有效区域）
            # 使用L1损失代替MSE，对异常值更鲁棒
            loss = F.l1_loss(projected_depth * valid_mask, depth_gt * valid_mask, reduction='sum') / (
                        valid_mask.sum() + 1e-8)

            # 添加结构相似性损失
            ssim_loss = self.compute_ssim_loss(projected_depth, depth_gt, valid_mask)

            # 组合损失
            total_loss = 0.8 * loss + 0.2 * ssim_loss

            # 限制损失大小
            total_loss = torch.clamp(total_loss, max=1.0)

            return total_loss

        except Exception as e:
            print(f"Error in depth consistency loss: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=images.device)

    def normalize_depth(self, depth):
        """归一化深度图到[0,1]范围"""
        # 添加小的epsilon避免除零
        valid_mask = (depth > 0).float()

        # 计算每个批次的最小和最大深度（仅考虑有效区域）
        min_depth = torch.zeros_like(depth)
        max_depth = torch.ones_like(depth) * self.max_depth

        for b in range(depth.shape[0]):
            valid_depths = depth[b][valid_mask[b] > 0]
            if valid_depths.numel() > 0:
                min_val = torch.max(torch.min(valid_depths), torch.tensor(self.min_depth, device=depth.device))
                max_val = torch.min(torch.max(valid_depths), torch.tensor(self.max_depth, device=depth.device))
                min_depth[b] = min_val
                max_depth[b] = max_val

        # 归一化
        normalized_depth = (depth - min_depth) / (max_depth - min_depth + 1e-8)
        normalized_depth = normalized_depth * valid_mask  # 保持无效区域为0

        return normalized_depth

    def compute_ssim_loss(self, pred, target, mask=None, window_size=11):
        """计算结构相似性损失"""
        try:
            # 确保输入是4D张量
            if pred.dim() != 4:
                pred = pred.unsqueeze(1)
            if target.dim() != 4:
                target = target.unsqueeze(1)
            if mask is not None and mask.dim() != 4:
                mask = mask.unsqueeze(1)

            # 创建高斯窗口
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            # 使用平均池化作为简化的高斯滤波
            mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
            mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size // 2) - mu1_sq
            sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size // 2) - mu2_sq
            sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            # 应用掩码（如果提供）
            if mask is not None:
                ssim_map = ssim_map * mask
                ssim_loss = 1 - torch.sum(ssim_map) / (torch.sum(mask) + 1e-8)
            else:
                ssim_loss = 1 - torch.mean(ssim_map)

            return ssim_loss

        except Exception as e:
            print(f"Error in SSIM calculation: {e}")
            return torch.tensor(0.0, device=pred.device)

    def project_points_to_depth(self, points, densities):
        """将点云投影为深度图 - 改进版"""
        try:
            B, N, _ = points.shape
            H = W = self.image_size

            # 相机内参（假设标准化坐标）
            fx = fy = 1.0
            cx = cy = 0.5

            # 初始化深度图和权重累积图
            depth_map = torch.zeros(B, 1, H, W, device=points.device)
            weight_map = torch.zeros(B, 1, H, W, device=points.device)

            # 使用sigmoid激活确保权重在[0,1]范围内
            weights = torch.sigmoid(densities).squeeze(-1)  # [B, N]

            for b in range(B):
                # 过滤掉z值太小或为负的点
                valid_mask = points[b, :, 2] > self.min_depth
                if not valid_mask.any():
                    continue

                valid_points = points[b, valid_mask]
                valid_weights = weights[b, valid_mask]

                if valid_points.shape[0] == 0:
                    continue

                # 提取坐标
                x = valid_points[:, 0]
                y = valid_points[:, 1]
                z = valid_points[:, 2]

                # 避免除零
                z_safe = torch.clamp(z, min=self.min_depth)

                # 投影到图像平面
                u = (fx * x / z_safe) + cx
                v = (fy * y / z_safe) + cy

                # 转换为像素坐标，但保持浮点数
                u_px = u * W
                v_px = v * H

                # 限制在有效范围内
                valid_proj = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)

                if not valid_proj.any():
                    continue

                u_px = u_px[valid_proj]
                v_px = v_px[valid_proj]
                z_values = z_safe[valid_proj]
                w_values = valid_weights[valid_proj]

                # 转换为整数坐标
                u_px_int = u_px.long().clamp(0, W - 1)
                v_px_int = v_px.long().clamp(0, H - 1)

                # 使用散点投影，处理深度冲突
                for i in range(len(u_px_int)):
                    ui, vi = u_px_int[i], v_px_int[i]
                    zi, wi = z_values[i], w_values[i]

                    # 累积加权深度
                    depth_map[b, 0, vi, ui] += zi * wi
                    weight_map[b, 0, vi, ui] += wi

            # 计算加权平均深度
            valid_mask = weight_map > 0
            depth_map[valid_mask] = depth_map[valid_mask] / weight_map[valid_mask]

            return depth_map

        except Exception as e:
            print(f"Error in point projection: {e}")
            return torch.zeros(B, 1, H, W, device=points.device)
