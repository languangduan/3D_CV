# losses/depth_consistency_loss.py
import torch
from torch import nn

from utils.depth_estimator import DepthEstimator
import torch.nn.functional as F


class DepthConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_estimator = DepthEstimator()

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
        # 估计深度图
        depth, _ = self.depth_estimator(images)

        # 将点云投影到图像平面
        projected_depth = self.project_points_to_depth(points, densities)

        # 计算深度一致性损失
        mask = (projected_depth > 0)  # 有效深度掩码
        loss = F.mse_loss(projected_depth[mask], depth[mask])

        return loss

    def project_points_to_depth(self, points, densities):
        """将点云投影为深度图"""
        B, N, _ = points.shape
        H = W = 256  # 假设输出深度图分辨率

        # 相机内参（假设标准化坐标）
        fx = fy = 1.0
        cx = cy = 0.5

        # 将3D点投影到图像平面
        x = points[:, :, 0:1]
        y = points[:, :, 1:2]
        z = points[:, :, 2:3]

        # 归一化坐标
        u = (fx * x / z) + cx
        v = (fy * y / z) + cy

        # 转换为像素坐标
        u = (u * W).long().clamp(0, W - 1)
        v = (v * H).long().clamp(0, H - 1)

        # 初始化深度图
        depth = torch.zeros(B, 1, H, W, device=points.device)

        # 使用密度作为权重，累积深度值
        for b in range(B):
            for i in range(N):
                if densities[b, i, 0] > 0.5:  # 只考虑高密度点
                    depth[b, 0, v[b, i, 0], u[b, i, 0]] += z[b, i, 0] * densities[b, i, 0]

        return depth
