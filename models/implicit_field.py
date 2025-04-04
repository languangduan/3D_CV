# models/implicit_field.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImplicitField(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=1, num_layers=3):
        super().__init__()

        self.input_dim = input_dim
        layers = []

        # 第一层
        layers.append(nn.Linear(input_dim + 3, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # 网格配置
        self.grid_size = 32
        self.chunk_size = 8192

    def forward(self, features, points=None):
        B = features.shape[0]

        if len(features.shape) == 4:
            C, H, W = features.shape[1:]
            D = H
            features = features.unsqueeze(2).expand(B, C, D, H, W)

        if points is None:
            # 修改点云生成方式，不设置 requires_grad
            points = self._create_grid_points(B, features.device)

        # 分批处理点云前检查输入
        assert features.requires_grad, "特征张量需要梯度"

        # 分批处理点云
        all_densities = []
        for i in range(0, points.shape[1], self.chunk_size):
            chunk_points = points[:, i:i + self.chunk_size]

            # 特征插值
            point_features = self._sample_features(features, chunk_points)

            # 连接点坐标和特征
            point_input = torch.cat([chunk_points, point_features], dim=-1)

            # 检查输入
            assert point_features.requires_grad, f"特征插值后需要梯度，当前状态: {point_features.requires_grad}"

            # 预测密度
            chunk_densities = self.mlp(point_input)
            all_densities.append(chunk_densities)

        # 合并所有密度预测
        densities = torch.cat(all_densities, dim=1)

        # 最终检查
        assert densities.requires_grad, "密度预测需要梯度"

        return points, densities

    def _create_grid_points(self, batch_size, device):
        # 新增方法：生成固定网格点
        x = torch.linspace(-1, 1, self.grid_size, device=device)
        y = torch.linspace(-1, 1, self.grid_size, device=device)
        z = torch.linspace(-1, 1, self.grid_size, device=device)

        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)
        points = points.unsqueeze(0).expand(batch_size, -1, -1)

        return points.detach()  # 确保网格点是固定的

    def _sample_features(self, features, points):
        # 将points从[-1,1]映射到[0,1]
        grid_points = (points + 1) / 2

        B, C = features.shape[:2]
        N = points.shape[1]

        # 移除手动设置requires_grad
        grid = grid_points.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N, 3]

        sampled_features = F.grid_sample(
            features,
            grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        )

        # 重塑为[B, N, C]
        sampled_features = sampled_features.squeeze(2).squeeze(2).permute(0, 2, 1)

        return sampled_features